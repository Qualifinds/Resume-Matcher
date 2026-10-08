"""OCR fallback for scanned PDFs: threshold, rendering and failure policy."""

from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.services import ocr


def _blank_pdf() -> bytes:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(612, 792)
    pdf.new_page(612, 792)
    import io

    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def test_needs_ocr_threshold() -> None:
    assert ocr.needs_ocr(None)
    assert ocr.needs_ocr("  \n ")
    assert ocr.needs_ocr("Scanned by CamScanner")
    assert not ocr.needs_ocr("word " * 100)


def test_render_pdf_pages_caps_page_count() -> None:
    pages = ocr.render_pdf_pages(_blank_pdf(), max_pages=1)
    assert len(pages) == 1
    assert pages[0].startswith(b"\x89PNG")
    assert len(ocr.render_pdf_pages(_blank_pdf(), max_pages=5)) == 2


@pytest.mark.asyncio
async def test_ocr_pdf_returns_transcription() -> None:
    with (
        patch.object(ocr, "supports_vision", return_value=True),
        patch.object(
            ocr, "complete_vision", new=AsyncMock(return_value="  Jane Doe  ")
        ) as vision,
    ):
        assert await ocr.ocr_pdf(_blank_pdf()) == "Jane Doe"
    assert len(vision.await_args.args[1]) == 2


@pytest.mark.asyncio
async def test_ocr_pdf_is_empty_when_model_has_no_vision() -> None:
    with (
        patch.object(ocr, "supports_vision", return_value=False),
        patch.object(ocr, "complete_vision", new=AsyncMock()) as vision,
    ):
        assert await ocr.ocr_pdf(_blank_pdf()) == ""
    vision.assert_not_awaited()


@pytest.mark.asyncio
async def test_ocr_pdf_swallows_llm_errors() -> None:
    with (
        patch.object(ocr, "supports_vision", return_value=True),
        patch.object(
            ocr, "complete_vision", new=AsyncMock(side_effect=ValueError("boom"))
        ),
    ):
        assert await ocr.ocr_pdf(_blank_pdf()) == ""


@pytest.mark.asyncio
async def test_ocr_pdf_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "ocr_enabled", False)
    with patch.object(ocr, "complete_vision", new=AsyncMock()) as vision:
        assert await ocr.ocr_pdf(_blank_pdf()) == ""
    vision.assert_not_awaited()


def _pdf_with_page(width: float, height: float) -> bytes:
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    pdf.new_page(width, height)
    buffer = io.BytesIO()
    pdf.save(buffer)
    return buffer.getvalue()


def test_renders_never_overlap_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """PDFium is not thread-safe: concurrent OCR requests must render one at a time."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    active = 0
    peak = 0
    guard = threading.Lock()

    def fake_render(data: bytes, max_pages: int, scale: float) -> list[bytes]:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with guard:
            active -= 1
        return [b"png"]

    monkeypatch.setattr(ocr, "_render_pages_unlocked", fake_render)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ocr.render_pdf_pages(b"%PDF", 1), range(16)))

    assert peak == 1
    assert results == [[b"png"]] * 16


def test_render_raises_busy_when_lock_is_held() -> None:
    assert ocr._PDFIUM_LOCK.acquire()
    try:
        with pytest.raises(ocr.PdfiumBusyError):
            ocr.render_pdf_pages(_blank_pdf(), max_pages=1, lock_timeout=0.05)
    finally:
        ocr._PDFIUM_LOCK.release()


@pytest.mark.asyncio
async def test_ocr_pdf_gives_up_instead_of_hanging_when_pdfium_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "ocr_render_timeout_seconds", 0.1)
    data = _blank_pdf()
    assert ocr._PDFIUM_LOCK.acquire()
    try:
        with (
            patch.object(ocr, "supports_vision", return_value=True),
            patch.object(ocr, "complete_vision", new=AsyncMock()) as vision,
        ):
            assert await ocr.ocr_pdf(data) == ""
        vision.assert_not_awaited()
    finally:
        ocr._PDFIUM_LOCK.release()


def test_render_caps_bitmap_size_for_huge_pages() -> None:
    import io

    from PIL import Image

    # 200 x 100 inches: at scale 2.0 that would be a 28800 x 14400 bitmap.
    (png,) = ocr.render_pdf_pages(_pdf_with_page(14400, 7200), max_pages=1)
    width, height = Image.open(io.BytesIO(png)).size
    assert max(width, height) <= ocr.MAX_RENDER_PX


def test_render_keeps_full_scale_for_letter_pages() -> None:
    import io

    from PIL import Image

    (png,) = ocr.render_pdf_pages(_pdf_with_page(612, 792), max_pages=1)
    assert Image.open(io.BytesIO(png)).size == (1224, 1584)
