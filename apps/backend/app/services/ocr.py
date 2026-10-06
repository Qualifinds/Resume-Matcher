"""OCR fallback for scanned (image-only) PDFs.

MarkItDown reads the text layer of a PDF, so a scan comes back empty and the upload used to fail with
422. Here the pages are rendered to PNG with pypdfium2 (wheel with bundled binaries, no apt packages)
and transcribed by the same vision-capable LLM the app already uses for parsing. The output is plain
markdown that feeds the normal parsing step. Any failure returns "" so the caller keeps its 422.
"""

import asyncio
import io
import logging

from app.config import settings
from app.llm import complete_vision, supports_vision

logger = logging.getLogger(__name__)

OCR_SYSTEM_PROMPT = (
    "You are an OCR engine. Transcribe the resume pages exactly as written, in reading order, "
    "as Markdown (headings for sections, '-' for bullets). Keep every word, date, company, "
    "school and number. Do not summarize, translate, correct or invent anything. "
    "Output only the transcription."
)
OCR_USER_PROMPT = "Transcribe these resume pages."
RENDER_SCALE = 2.0  # ~144 dpi: legible for the model without huge images


def render_pdf_pages(
    data: bytes, max_pages: int, scale: float = RENDER_SCALE
) -> list[bytes]:
    """Render the first `max_pages` pages of a PDF to PNG bytes."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        pages: list[bytes] = []
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            try:
                image = page.render(scale=scale).to_pil().convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                pages.append(buffer.getvalue())
            finally:
                page.close()
        return pages
    finally:
        pdf.close()


def needs_ocr(text: str | None) -> bool:
    """True when the extracted text is too short to be a real resume (empty or a junk text layer)."""
    return len("".join((text or "").split())) < settings.ocr_min_chars


async def ocr_pdf(data: bytes) -> str:
    """Transcribe a scanned PDF. Returns "" when OCR is off, unsupported or fails."""
    if not settings.ocr_enabled:
        return ""
    try:
        if not supports_vision():
            logger.warning("OCR skipped: the configured model does not accept images")
            return ""
        pages = await asyncio.to_thread(render_pdf_pages, data, settings.ocr_max_pages)
        if not pages:
            return ""
        text = await asyncio.wait_for(
            complete_vision(OCR_USER_PROMPT, pages, system_prompt=OCR_SYSTEM_PROMPT),
            timeout=settings.ocr_timeout_seconds,
        )
        return text.strip()
    except Exception:
        logger.exception("OCR failed")
        return ""
