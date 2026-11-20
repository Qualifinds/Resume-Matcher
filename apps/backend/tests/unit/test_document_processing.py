"""Document container and resource-boundary tests for resume uploads."""

import asyncio
import io
import zipfile
import zlib
from unittest.mock import patch

import pytest
from docx import Document

from app.document_conversion import convert_document_bytes
from app.services.parser import DocumentResourceLimitError, parse_document


def _docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


def _pdf_with_content_streams(
    contents: list[bytes],
    *,
    filter_name: bytes | None = None,
    extra_objects: list[bytes] | None = None,
    stream_attributes: bytes = b"",
) -> bytes:
    """Build one valid PDF whose page references the requested content streams."""
    filter_entry = b" /Filter /" + filter_name if filter_name else b""
    content_references = b" ".join(
        f"{number} 0 R".encode() for number in range(5, 5 + len(contents))
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents ["
            + content_references
            + b"] >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    objects.extend(
        (
            b"<< /Length "
            + str(len(content)).encode()
            + filter_entry
            + stream_attributes
            + b" >>\nstream\n"
            + content
            + b"\nendstream"
        )
        for content in contents
    )
    objects.extend(extra_objects or [])
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(pdf)


def _pdf_bytes(content: bytes, *, flate: bool = False) -> bytes:
    """Build one valid PDF whose page content uses the requested stream bytes."""
    filter_name = b"FlateDecode" if flate else None
    return _pdf_with_content_streams([content], filter_name=filter_name)


def _xref_stream_pdf_bytes(*, decoded_padding: int = 0) -> bytes:
    """Build a valid PDF 1.5 file whose cross-reference table is a Flate stream."""
    page_content = b"BT /F1 12 Tf 72 720 Td (XRef engineer) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            b"<< /Length "
            + str(len(page_content)).encode()
            + b" >>\nstream\n"
            + page_content
            + b"\nendstream"
        ),
    ]
    pdf = bytearray(b"%PDF-1.5\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    offsets.append(xref_offset)
    entries = bytearray(b"\x00\x00\x00\x00\x00\xff\xff")
    for offset in offsets[1:]:
        entries.extend(b"\x01" + offset.to_bytes(4, "big") + b"\x00\x00")
    free_entry = b"\x00\x00\x00\x00\x00\xff\xff"
    extra_entries = (decoded_padding + len(free_entry) - 1) // len(free_entry)
    entries.extend(free_entry * extra_entries)
    object_count = len(entries) // len(free_entry)
    encoded_entries = zlib.compress(entries, level=9)
    pdf.extend(
        b"6 0 obj\n<< /Type /XRef /Size "
        + str(object_count).encode()
        + b" /W [1 4 2] /Index [0 "
        + str(object_count).encode()
        + b"] /Root 1 0 R /Filter /FlateDecode /Length "
        + str(len(encoded_entries)).encode()
        + b" >>\nstream\n"
        + encoded_entries
        + b"\nendstream\nendobj\n"
    )
    pdf.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode())
    return bytes(pdf)


async def test_parse_document_rejects_malformed_pdf_container() -> None:
    """Converter fallback text must not make a corrupt PDF valid."""
    with pytest.raises(ValueError, match="valid PDF, DOC, or DOCX"):
        await parse_document(b"%PDF-1.4 broken", "broken.pdf")


async def test_parse_document_rejects_malformed_docx_container() -> None:
    """A `PK` prefix alone must not make fallback text a DOCX document."""
    with pytest.raises(ValueError, match="valid PDF, DOC, or DOCX"):
        await parse_document(b"PK corrupt", "broken.docx")


async def test_parse_document_rejects_malformed_legacy_doc_container() -> None:
    """Legacy DOC support requires a compound-file container, not fallback text."""
    with pytest.raises(ValueError, match="valid PDF, DOC, or DOCX"):
        await parse_document(b"plain text pretending to be Word", "broken.doc")


async def test_parse_document_rejects_docx_over_unpacked_limit() -> None:
    """A small compressed upload cannot expand beyond the archive budget."""
    compressed = _docx_bytes("A" * (17 * 1024 * 1024))
    assert len(compressed) < 4 * 1024 * 1024

    with pytest.raises(DocumentResourceLimitError, match="expanded content exceeds"):
        await parse_document(compressed, "expanded.docx")


async def test_parse_document_rejects_pdf_over_decoded_stream_limit() -> None:
    """A tiny Flate stream cannot expand past the PDF processing budget."""
    expanded = b"%" + (b"A" * (17 * 1024 * 1024)) + b"\nBT (Synthetic engineer) Tj ET"
    document = _pdf_bytes(zlib.compress(expanded, level=9), flate=True)
    assert len(document) < 4 * 1024 * 1024

    with patch("app.document_conversion.MarkItDown.convert") as convert:
        with pytest.raises(
            DocumentResourceLimitError, match="expanded content exceeds"
        ):
            await asyncio.to_thread(convert_document_bytes, document, "expanded.pdf")

    convert.assert_not_called()


async def test_parse_document_bounds_pdf_xref_stream_during_initialization() -> None:
    """An oversized xref stream is bounded before PDFDocument can inflate it."""
    document = _xref_stream_pdf_bytes(decoded_padding=17 * 1024 * 1024)
    assert len(document) < 4 * 1024 * 1024

    with patch("app.document_conversion.MarkItDown.convert") as convert:
        with pytest.raises(
            DocumentResourceLimitError, match="expanded content exceeds"
        ):
            await asyncio.to_thread(
                convert_document_bytes, document, "expanded-xref.pdf"
            )

    convert.assert_not_called()


async def test_parse_document_applies_one_budget_across_pdf_streams() -> None:
    """Separate streams cannot each consume the full document expansion limit."""
    stream = zlib.compress(b"%" + (b"A" * (9 * 1024 * 1024)), level=9)
    document = _pdf_with_content_streams([stream, stream], filter_name=b"FlateDecode")
    assert len(document) < 4 * 1024 * 1024

    with patch("app.document_conversion.MarkItDown.convert") as convert:
        with pytest.raises(
            DocumentResourceLimitError, match="expanded content exceeds"
        ):
            await asyncio.to_thread(convert_document_bytes, document, "aggregate.pdf")

    convert.assert_not_called()


async def test_parse_document_bounds_run_length_pdf_expansion() -> None:
    """The PDF cap also covers expansion filters other than Flate."""
    encoded = (b"\x81A" * ((17 * 1024 * 1024) // 128)) + b"\x80"
    document = _pdf_with_content_streams([encoded], filter_name=b"RunLengthDecode")
    assert len(document) < 4 * 1024 * 1024

    with patch("app.document_conversion.MarkItDown.convert") as convert:
        with pytest.raises(
            DocumentResourceLimitError, match="expanded content exceeds"
        ):
            await asyncio.to_thread(convert_document_bytes, document, "run-length.pdf")

    convert.assert_not_called()


async def test_parse_document_accepts_normal_flate_pdf() -> None:
    """The PDF resource guard preserves ordinary compressed text documents."""
    content = zlib.compress(b"BT /F1 12 Tf 72 720 Td (Flate engineer) Tj ET")

    assert "Flate engineer" in await parse_document(
        _pdf_bytes(content, flate=True), "normal.pdf"
    )


async def test_parse_document_rejects_too_many_docx_members() -> None:
    """Archive entry count is bounded independently from byte expansion."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<document />")
        for index in range(1_023):
            archive.writestr(f"word/media/{index}.txt", "x")

    with pytest.raises(DocumentResourceLimitError, match="more than 1024 entries"):
        await parse_document(stream.getvalue(), "too-many-members.docx")


async def test_parse_document_rejects_extracted_text_over_prompt_limit() -> None:
    """Extracted text is bounded before it can become an LLM prompt."""
    compressed = _docx_bytes("A" * (3 * 1024 * 1024))

    with pytest.raises(DocumentResourceLimitError, match="extracted text exceeds"):
        await parse_document(compressed, "prompt-too-large.docx")


async def test_real_docx_conversion_keeps_event_loop_responsive() -> None:
    """A competing coroutine advances while real MarkItDown conversion runs."""
    compressed = _docx_bytes("A" * (1024 * 1024))
    ticks = 0
    running = True

    async def heartbeat() -> None:
        nonlocal ticks
        while running:
            ticks += 1
            await asyncio.sleep(0.001)

    pulse = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    extracted = await asyncio.wait_for(
        parse_document(compressed, "responsive.docx"), timeout=10
    )
    running = False
    await pulse

    assert extracted.rstrip() == "A" * (1024 * 1024)
    assert ticks >= 2


@pytest.mark.parametrize("predictor", [2, 12])
def test_pdf_predictor_rejects_huge_dimensions_before_decoder_allocation(
    predictor: int,
) -> None:
    from app.document_conversion import _apply_pdf_predictor

    target = "apply_tiff_predictor" if predictor == 2 else "apply_png_predictor"
    with patch(f"app.document_conversion.{target}", return_value=b"") as decoder:
        with pytest.raises(DocumentResourceLimitError):
            _apply_pdf_predictor(
                b"\x00", {"Predictor": predictor, "Columns": 1_000_000_000}
            )
        decoder.assert_not_called()


def test_pdf_fax_rejects_huge_bitmap_width_before_decoder_allocation() -> None:
    from app.document_conversion import _decode_ccitt_bounded

    with patch("app.document_conversion._BoundedCCITTFaxDecoder") as decoder:
        decoder.return_value.close.return_value = b""
        with pytest.raises(DocumentResourceLimitError):
            _decode_ccitt_bounded(
                b"", {"K": -1, "Columns": 100_000_000}, 16 * 1024 * 1024
            )
        decoder.assert_not_called()


def test_truncated_flate_stream_is_rejected() -> None:
    from app.document_conversion import _decode_flate_bounded

    with pytest.raises(ValueError, match="Truncated"):
        _decode_flate_bounded(zlib.compress(b"resume text")[:-2], 1024)


def test_lzw_stream_decodes_incrementally_with_limit() -> None:
    from app.document_conversion import _decode_lzw_bounded

    codes = [256, *b"ABC", 257]
    bits = "".join(f"{code:09b}" for code in codes)
    encoded = int(bits.ljust((len(bits) + 7) // 8 * 8, "0"), 2).to_bytes(
        (len(bits) + 7) // 8, "big"
    )
    assert _decode_lzw_bounded(encoded, 3) == b"ABC"
    with pytest.raises(DocumentResourceLimitError):
        _decode_lzw_bounded(encoded, 2)


async def test_unused_image_filter_does_not_reject_text_pdf() -> None:
    image = b"<< /Type /XObject /Subtype /Image /Width 1 /Height 1 /Filter /CCITTFaxDecode /DecodeParms << /K 0 >> /Length 1 >>\nstream\nx\nendstream"
    document = _pdf_with_content_streams(
        [b"BT /F1 12 Tf 72 720 Td (Image engineer) Tj ET"], extra_objects=[image]
    )
    assert "Image engineer" in await parse_document(document, "image.pdf")


async def test_image_subtype_cannot_hide_page_content_expansion() -> None:
    encoded = zlib.compress(b"A" * (17 * 1024 * 1024))
    document = _pdf_with_content_streams(
        [encoded], filter_name=b"FlateDecode", stream_attributes=b" /Subtype /Image"
    )
    with pytest.raises(DocumentResourceLimitError):
        await parse_document(document, "forged-image.pdf")
