"""Document validation and conversion (PDF/DOC/DOCX to Markdown), runnable as a child process.

This module is deliberately light: it imports only markitdown and pdfminer, never ``app.config`` or
``app.llm`` (litellm costs seconds to import). ``app.services.parser`` runs it as
``python -m app.document_conversion <input> <result>`` in a process it can kill when the conversion
exceeds its deadline. A thread running pdfminer cannot be killed, and two stuck threads used to take
every conversion slot until the service restarted (production incident, 2026-10-22 to 2026-11-19).

Child protocol: argv = [input_path, result_path]. Whatever the outcome of the conversion the child
writes one UTF-8 JSON object to ``result_path`` and exits 0::

    {"status": "ok", "text": "..."}
    {"status": "error", "kind": "validation" | "resource_limit" | "other", "type": "...", "message": "..."}

A non-zero exit, or a missing or garbled result file, means the process died (OOM kill, crash).
"""

import array
import io
import json
import sys
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any, BinaryIO, Sequence

from markitdown import MarkItDown
from pdfminer.ascii85 import ascii85decode, asciihexdecode
from pdfminer.ccitt import CCITTFaxDecoder
from pdfminer.lzw import LZWDecoder
from pdfminer.pdfdevice import PDFDevice
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import (
    LITERAL_CRYPT,
    LITERALS_ASCII85_DECODE,
    LITERALS_ASCIIHEX_DECODE,
    LITERALS_CCITTFAX_DECODE,
    LITERALS_DCT_DECODE,
    LITERALS_FLATE_DECODE,
    LITERALS_JBIG2_DECODE,
    LITERALS_JPX_DECODE,
    LITERALS_LZW_DECODE,
    LITERALS_RUNLENGTH_DECODE,
    PDFStream,
    apply_png_predictor,
    apply_tiff_predictor,
    int_value,
)
from pdfminer.psparser import PSKeyword, literal_name

DOCUMENT_IO_CHUNK_SIZE = 64 * 1024
MAX_DOCX_MEMBERS = 1_024
MAX_UNPACKED_DOCUMENT_BYTES = 16 * 1024 * 1024
MAX_EXTRACTED_TEXT_BYTES = 2 * 1024 * 1024
# Decoder row buffers can allocate from dimensions before consuming input.
MAX_PDF_SCANLINE_COLUMNS = 32_768
MAX_PDF_SCANLINE_BYTES = 256 * 1024


class DocumentValidationError(ValueError):
    """Raised when uploaded bytes are not a structurally valid document."""


class DocumentResourceLimitError(ValueError):
    """Raised when a valid document exceeds a bounded processing budget."""


_COMPOUND_FILE_SIGNATURE = bytes.fromhex("D0CF11E0A1B11AE1")


class _PDFDecodeBudget:
    """Track decoded PDF stream bytes across one parser instance."""

    def __init__(self) -> None:
        self.decoded_bytes = 0

    @property
    def remaining(self) -> int:
        """Return the bytes available to the next decoded stream."""
        return MAX_UNPACKED_DOCUMENT_BYTES - self.decoded_bytes

    def consume(self, size: int) -> None:
        """Charge one decoded stream to the document-wide limit."""
        if size > self.remaining:
            raise DocumentResourceLimitError(
                "Document expanded content exceeds the 16MB limit."
            )
        self.decoded_bytes += size


def _append_pdf_output(output: bytearray, chunk: bytes, limit: int) -> None:
    """Append decoded stream bytes without crossing the remaining budget."""
    if len(chunk) > limit - len(output):
        raise DocumentResourceLimitError(
            "Document expanded content exceeds the 16MB limit."
        )
    output.extend(chunk)


def _decode_flate_bounded(data: bytes, limit: int) -> bytes:
    """Incrementally inflate a stream while enforcing its remaining budget."""
    decoder = zlib.decompressobj()
    output = bytearray()
    for offset in range(0, len(data), DOCUMENT_IO_CHUNK_SIZE):
        pending = data[offset : offset + DOCUMENT_IO_CHUNK_SIZE]
        while pending:
            max_output = min(DOCUMENT_IO_CHUNK_SIZE, limit - len(output) + 1)
            chunk = decoder.decompress(pending, max_output)
            _append_pdf_output(output, chunk, limit)
            pending = decoder.unconsumed_tail
    _append_pdf_output(output, decoder.flush(limit - len(output) + 1), limit)
    if not decoder.eof:
        raise ValueError("Truncated FlateDecode stream")
    return bytes(output)


def _decode_lzw_bounded(data: bytes, limit: int) -> bytes:
    """Decode an LZW stream one emitted code at a time under a byte limit."""
    output = bytearray()
    for chunk in LZWDecoder(io.BytesIO(data)).run():
        _append_pdf_output(output, chunk, limit)
    return bytes(output)


def _decode_run_length_bounded(data: bytes, limit: int) -> bytes:
    """Decode Adobe run-length data without first materializing all output."""
    output = bytearray()
    offset = 0
    while offset < len(data):
        length = data[offset]
        offset += 1
        if length == 128:
            break
        if length < 128:
            end = offset + length + 1
            if end > len(data):
                raise ValueError("Truncated run-length literal")
            _append_pdf_output(output, data[offset:end], limit)
            offset = end
            continue
        if offset >= len(data):
            raise ValueError("Truncated run-length repeat")
        _append_pdf_output(output, bytes([data[offset]]) * (257 - length), limit)
        offset += 1
    return bytes(output)


class _BoundedCCITTFaxDecoder(CCITTFaxDecoder):
    """Collect CCITT rows only while they fit the current stream budget."""

    def __init__(
        self,
        width: int,
        *,
        bytealign: bool,
        reversed_bits: bool,
        limit: int,
    ) -> None:
        super().__init__(width, bytealign=bytealign, reversed=reversed_bits)
        self._chunks: list[bytes] = []
        self._decoded_size = 0
        self._limit = limit

    def output_line(self, y: int, bits: Sequence[int]) -> None:
        """Encode one decoded bitmap row after reserving its output bytes."""
        row_size = (len(bits) + 7) // 8
        if row_size > self._limit - self._decoded_size:
            raise DocumentResourceLimitError(
                "Document expanded content exceeds the 16MB limit."
            )
        row = array.array("B", [0] * row_size)
        source_bits = [1 - bit for bit in bits] if self.reversed else bits
        masks = (128, 64, 32, 16, 8, 4, 2, 1)
        for index, bit in enumerate(source_bits):
            if bit:
                row[index // 8] += masks[index % 8]
        encoded = row.tobytes()
        self._chunks.append(encoded)
        self._decoded_size += len(encoded)

    def close(self) -> bytes:
        """Join already bounded rows into the decoded stream."""
        return b"".join(self._chunks)


def _decode_ccitt_bounded(data: bytes, params: dict[str, Any], limit: int) -> bytes:
    """Decode the CCITT variant supported by pdfminer with bounded output."""
    if params.get("K") != -1:
        raise ValueError("Unsupported CCITT encoding")
    width = int_value(params.get("Columns"))
    if width <= 0:
        raise ValueError("CCITT stream has no positive column count")
    if width > MAX_PDF_SCANLINE_COLUMNS or (width + 7) // 8 > limit:
        raise DocumentResourceLimitError(
            "Document expanded content exceeds the 16MB limit."
        )
    decoder = _BoundedCCITTFaxDecoder(
        width,
        bytealign=bool(params.get("EncodedByteAlign")),
        reversed_bits=bool(params.get("BlackIs1")),
        limit=limit,
    )
    decoder.feedbytes(data)
    return decoder.close()


def _apply_pdf_predictor(data: bytes, params: dict[str, Any]) -> bytes:
    """Apply the same predictor transformations as pdfminer."""
    predictor = int_value(params.get("Predictor", 1))
    if predictor == 1:
        return data
    colors = int_value(params.get("Colors", 1))
    columns = int_value(params.get("Columns", 1))
    bits_per_component = int_value(params.get("BitsPerComponent", 8))
    if colors <= 0 or columns <= 0 or bits_per_component <= 0:
        raise ValueError("PDF predictor dimensions must be positive")
    row_bytes = (colors * columns * bits_per_component + 7) // 8
    if columns > MAX_PDF_SCANLINE_COLUMNS or row_bytes > MAX_PDF_SCANLINE_BYTES:
        raise DocumentResourceLimitError(
            "Document decoder row dimensions exceed the processing limit."
        )
    if predictor == 2:
        return apply_tiff_predictor(colors, columns, bits_per_component, data)
    if predictor >= 10:
        return apply_png_predictor(predictor, colors, columns, bits_per_component, data)
    raise ValueError("Unsupported PDF predictor")


def _decode_pdf_stream(
    data: bytes,
    filters: list[tuple[Any, Any]],
    limit: int,
) -> bytes:
    """Decode every PDF filter while bounding each expansion stage."""
    for filter_name, raw_params in filters:
        params = raw_params if isinstance(raw_params, dict) else {}
        if filter_name in LITERALS_FLATE_DECODE:
            data = _decode_flate_bounded(data, limit)
        elif filter_name in LITERALS_LZW_DECODE:
            data = _decode_lzw_bounded(data, limit)
        elif filter_name in LITERALS_ASCII85_DECODE:
            data = ascii85decode(data)
        elif filter_name in LITERALS_ASCIIHEX_DECODE:
            data = asciihexdecode(data)
        elif filter_name in LITERALS_RUNLENGTH_DECODE:
            data = _decode_run_length_bounded(data, limit)
        elif filter_name in LITERALS_CCITTFAX_DECODE:
            data = _decode_ccitt_bounded(data, params, limit)
        elif (
            filter_name in LITERALS_DCT_DECODE
            or filter_name in LITERALS_JBIG2_DECODE
            or filter_name in LITERALS_JPX_DECODE
        ):
            # pdfminer passes already compressed image formats through unchanged.
            pass
        elif filter_name == LITERAL_CRYPT:
            raise ValueError("Encrypted PDF streams are not supported")
        else:
            raise ValueError("Unsupported PDF stream filter")

        if len(data) > limit:
            raise DocumentResourceLimitError(
                "Document expanded content exceeds the 16MB limit."
            )
        if params and "Predictor" in params:
            data = _apply_pdf_predictor(data, params)
            if len(data) > limit:
                raise DocumentResourceLimitError(
                    "Document expanded content exceeds the 16MB limit."
                )
    return data


class _BoundedPDFStream(PDFStream):
    """PDF stream whose decoder charges a request-local shared budget."""

    def __init__(self, stream: PDFStream, budget: _PDFDecodeBudget) -> None:
        super().__init__(stream.attrs, stream.rawdata, stream.decipher)
        self._budget = budget

    def decode(self) -> None:
        """Decode this stream with bounded filter implementations."""
        if self.rawdata is None:
            raise ValueError("PDF stream has no raw data")
        data = self.rawdata
        if self.decipher:
            if self.objid is None or self.genno is None:
                raise ValueError("Encrypted PDF stream is missing an object ID")
            data = self.decipher(self.objid, self.genno, data, self.attrs)
        decoded = _decode_pdf_stream(data, self.get_filters(), self._budget.remaining)
        self._budget.consume(len(decoded))
        self.data = decoded
        self.rawdata = None


class _BoundedPDFParser(PDFParser):
    """Install bounded streams locally without patching pdfminer globals."""

    def __init__(self, stream: BinaryIO, budget: _PDFDecodeBudget) -> None:
        super().__init__(stream)
        self._budget = budget

    def do_keyword(self, pos: int, token: PSKeyword) -> None:
        """Replace each newly parsed stream with its bounded counterpart."""
        super().do_keyword(pos, token)
        if token is self.KEYWORD_STREAM:
            stream_pos, parsed = self.curstack[-1]
            if isinstance(parsed, PDFStream):
                self.curstack[-1] = (
                    stream_pos,
                    _BoundedPDFStream(parsed, self._budget),
                )


def _validate_pdf_container(path: Path) -> None:
    """Require a readable PDF whose decoded streams fit a shared budget."""
    try:
        with path.open("rb") as stream:
            budget = _PDFDecodeBudget()
            document = PDFDocument(_BoundedPDFParser(stream, budget))
            manager = PDFResourceManager(caching=False)
            interpreter = PDFPageInterpreter(manager, PDFDevice(manager))
            has_pages = False
            for page in PDFPage.create_pages(document):
                has_pages = True
                # Follow the real text consumer's references under the same
                # bounded stream decoder. A forged /Image label on a content
                # or font stream cannot bypass preflight expansion limits.
                interpreter.process_page(page)
            if not has_pages:
                raise ValueError("PDF has no pages")
            seen: set[int] = set()
            for xref in document.xrefs:
                for object_id in xref.get_objids():
                    if object_id in seen:
                        continue
                    seen.add(object_id)
                    value = document.getobj(object_id)
                    if isinstance(value, PDFStream):
                        # Text extraction never decodes image XObjects. Their
                        # optional codecs must not reject otherwise readable text.
                        if literal_name(value.attrs.get("Subtype")) != "Image":
                            value.get_data()
    except DocumentResourceLimitError:
        raise
    except Exception as exc:
        raise DocumentValidationError(
            "The uploaded file is not a valid PDF, DOC, or DOCX document."
        ) from exc


def _validate_docx_container(path: Path) -> None:
    """Require a bounded, readable Office Open XML word-processing package."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) > MAX_DOCX_MEMBERS:
                raise DocumentResourceLimitError(
                    f"Document archive contains more than {MAX_DOCX_MEMBERS} entries."
                )

            names = {member.filename for member in members}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise DocumentValidationError(
                    "The uploaded file is not a valid PDF, DOC, or DOCX document."
                )

            declared_size = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise DocumentValidationError(
                        "Encrypted PDF, DOC, or DOCX files are not supported."
                    )
                declared_size += member.file_size
                if declared_size > MAX_UNPACKED_DOCUMENT_BYTES:
                    raise DocumentResourceLimitError(
                        "Document expanded content exceeds the 16MB limit."
                    )

            streamed_size = 0
            for member in members:
                if member.is_dir():
                    continue
                with archive.open(member) as source:
                    while chunk := source.read(DOCUMENT_IO_CHUNK_SIZE):
                        streamed_size += len(chunk)
                        if streamed_size > MAX_UNPACKED_DOCUMENT_BYTES:
                            raise DocumentResourceLimitError(
                                "Document expanded content exceeds the 16MB limit."
                            )
    except DocumentResourceLimitError:
        raise
    except DocumentValidationError:
        raise
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        raise DocumentValidationError(
            "The uploaded file is not a valid PDF, DOC, or DOCX document."
        ) from exc


def _validate_doc_container(path: Path) -> None:
    """Validate the fixed compound-file header used by legacy Word documents."""
    try:
        with path.open("rb") as stream:
            header = stream.read(512)
    except OSError as exc:
        raise DocumentValidationError(
            "The uploaded file is not a valid PDF, DOC, or DOCX document."
        ) from exc

    sector_shift = int.from_bytes(header[30:32], "little") if len(header) >= 32 else -1
    mini_sector_shift = (
        int.from_bytes(header[32:34], "little") if len(header) >= 34 else -1
    )
    if (
        len(header) != 512
        or header[:8] != _COMPOUND_FILE_SIGNATURE
        or header[28:30] != b"\xfe\xff"
        or sector_shift not in (9, 12)
        or mini_sector_shift != 6
        or header[34:40] != b"\x00" * 6
    ):
        raise DocumentValidationError(
            "The uploaded file is not a valid PDF, DOC, or DOCX document."
        )

    major_version = int.from_bytes(header[26:28], "little")
    file_size = path.stat().st_size
    sector_size = 1 << sector_shift
    sector_count = file_size // sector_size - 1
    fat_sector_count = int.from_bytes(header[44:48], "little")
    first_directory_sector = int.from_bytes(header[48:52], "little")
    difat_entries = [
        int.from_bytes(header[offset : offset + 4], "little")
        for offset in range(76, 512, 4)
    ]
    inline_fat_sectors = [entry for entry in difat_entries if entry < 0xFFFFFFFA]
    if (
        major_version not in (3, 4)
        or (major_version == 3 and sector_shift != 9)
        or (major_version == 4 and sector_shift != 12)
        or file_size < sector_size * 3
        or file_size % sector_size != 0
        or fat_sector_count < 1
        or fat_sector_count > sector_count
        or first_directory_sector >= sector_count
        or len(inline_fat_sectors) < min(fat_sector_count, len(difat_entries))
        or any(entry >= sector_count for entry in inline_fat_sectors)
    ):
        raise DocumentValidationError(
            "The uploaded file is not a valid PDF, DOC, or DOCX document."
        )


def _validate_extracted_text(text: str) -> None:
    """Bound prompt-bound UTF-8 text without allocating another full-size copy."""
    extracted_bytes = 0
    for offset in range(0, len(text), DOCUMENT_IO_CHUNK_SIZE):
        chunk = text[offset : offset + DOCUMENT_IO_CHUNK_SIZE]
        extracted_bytes += len(chunk.encode("utf-8"))
        if extracted_bytes > MAX_EXTRACTED_TEXT_BYTES:
            raise DocumentResourceLimitError(
                "Document extracted text exceeds the 2MB processing limit."
            )


def convert_document_file(path: Path) -> str:
    """Validate the container at ``path`` and convert it to Markdown (blocking, in-process).

    The suffix of ``path`` selects the container check and the markitdown converter.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        _validate_pdf_container(path)
    elif suffix == ".doc":
        _validate_doc_container(path)
    elif suffix == ".docx":
        _validate_docx_container(path)
    # No network and no LLM: a bare MarkItDown() only reads the local file.
    md = MarkItDown()
    result = md.convert(str(path))
    text = result.text_content
    if not isinstance(text, str):
        raise DocumentValidationError(
            "The uploaded file is not a valid PDF, DOC, or DOCX document."
        )
    _validate_extracted_text(text)
    return text


def convert_document_bytes(content: bytes, filename: str) -> str:
    """Convert raw bytes in this process through a temporary file (tests and scripts)."""
    suffix = Path(filename).suffix.lower()
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(content)
        return convert_document_file(tmp_path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _error_payload(exc: Exception) -> dict[str, str]:
    """Describe a conversion failure so the parent can raise the equivalent exception."""
    if isinstance(exc, DocumentResourceLimitError):
        kind = "resource_limit"
    elif isinstance(exc, DocumentValidationError):
        kind = "validation"
    else:
        kind = "other"
    return {
        "status": "error",
        "kind": kind,
        "type": type(exc).__name__,
        "message": str(exc),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Child entry point: ``python -m app.document_conversion <input> <result>``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print(
            "usage: python -m app.document_conversion <input> <result>", file=sys.stderr
        )
        return 2
    input_path, result_path = Path(args[0]), Path(args[1])
    try:
        payload: dict[str, str] = {
            "status": "ok",
            "text": convert_document_file(input_path),
        }
    except Exception as exc:  # reported to the parent, which maps it to an HTTP status
        payload = _error_payload(exc)
    result_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
