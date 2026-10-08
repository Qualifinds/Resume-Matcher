"""OCR fallback for scanned (image-only) PDFs.

MarkItDown reads the text layer of a PDF, so a scan comes back empty and the upload used to fail with
422. Here the pages are rendered to PNG with pypdfium2 (wheel with bundled binaries, no apt packages)
and transcribed by the same vision-capable LLM the app already uses for parsing. The output is plain
markdown that feeds the normal parsing step. Any failure returns "" so the caller keeps its 422.

PDFium is NOT thread-safe: "It is not allowed to call pdfium functions simultaneously across
different threads, not even with different documents. Doing so would crash or corrupt the process"
(pypdfium2 docs). Renders run in worker threads, so every pdfium call goes through `_PDFIUM_LOCK`.
Without it, two PDFs reaching OCR at the same time (the compact-cvs cron plus any batch) could wedge
the whole process: on 2026-10-07 the service stopped answering for ~4h, /docs included, right after
several rejected PDFs hit this path in the same second.
"""

import asyncio
import io
import logging
import threading
import time

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
# Longest side of a rendered page, in pixels. A malformed PDF can declare a page of several meters;
# at scale 2 that is a multi-GB bitmap. 2200 px keeps a letter page at the full 2.0 scale.
MAX_RENDER_PX = 2200

# One pdfium call at a time, process-wide (see module docstring).
_PDFIUM_LOCK = threading.Lock()


class PdfiumBusyError(TimeoutError):
    """The pdfium lock could not be acquired in time (another render is still running)."""


def _render_pages_unlocked(data: bytes, max_pages: int, scale: float) -> list[bytes]:
    """Render up to `max_pages` pages to PNG. Caller MUST hold `_PDFIUM_LOCK`."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        pages: list[bytes] = []
        for index in range(min(len(pdf), max_pages)):
            page = pdf[index]
            try:
                width, height = page.get_size()  # PDF points
                longest = max(width, height, 1.0)
                page_scale = min(scale, MAX_RENDER_PX / longest)
                image = page.render(scale=page_scale).to_pil().convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                pages.append(buffer.getvalue())
            finally:
                page.close()
        return pages
    finally:
        pdf.close()


def render_pdf_pages(
    data: bytes,
    max_pages: int,
    scale: float = RENDER_SCALE,
    lock_timeout: float | None = None,
) -> list[bytes]:
    """Render the first `max_pages` pages of a PDF to PNG bytes, one pdfium call at a time.

    `lock_timeout` (seconds) bounds the wait for a render already in progress; None waits forever.
    Raises `PdfiumBusyError` when the lock is not acquired in time.
    """
    acquired = _PDFIUM_LOCK.acquire(
        timeout=-1 if lock_timeout is None else lock_timeout
    )
    if not acquired:
        raise PdfiumBusyError("pdfium is busy with another render")
    try:
        return _render_pages_unlocked(data, max_pages, scale)
    finally:
        _PDFIUM_LOCK.release()


def needs_ocr(text: str | None) -> bool:
    """True when the extracted text is too short to be a real resume (empty or a junk text layer)."""
    return len("".join((text or "").split())) < settings.ocr_min_chars


async def ocr_pdf(data: bytes) -> str:
    """Transcribe a scanned PDF. Returns "" when OCR is off, unsupported or fails."""
    if not settings.ocr_enabled:
        return ""
    started = time.monotonic()
    try:
        if not supports_vision():
            logger.warning("OCR skipped: the configured model does not accept images")
            return ""
        logger.info("OCR start: %d bytes", len(data))
        budget = settings.ocr_render_timeout_seconds
        # The thread waits at most `budget` for the lock, and the await gives up after waiting for
        # the lock plus rendering. A render that outlives the await keeps the lock until it ends,
        # but later requests stop waiting after `budget` instead of piling up for good.
        pages = await asyncio.wait_for(
            asyncio.to_thread(
                render_pdf_pages, data, settings.ocr_max_pages, RENDER_SCALE, budget
            ),
            timeout=budget * 2,
        )
        if not pages:
            logger.info("OCR end: no pages (%.1fs)", time.monotonic() - started)
            return ""
        text = await asyncio.wait_for(
            complete_vision(OCR_USER_PROMPT, pages, system_prompt=OCR_SYSTEM_PROMPT),
            timeout=settings.ocr_timeout_seconds,
        )
        text = text.strip()
        logger.info(
            "OCR end: %d pages, %d chars (%.1fs)",
            len(pages),
            len(text),
            time.monotonic() - started,
        )
        return text
    except TimeoutError:
        # PdfiumBusyError is a TimeoutError: lock busy, render too slow or vision call too slow.
        logger.warning(
            "OCR gave up: render busy or timed out (%.1fs)", time.monotonic() - started
        )
        return ""
    except Exception:
        logger.exception("OCR failed (%.1fs)", time.monotonic() - started)
        return ""
