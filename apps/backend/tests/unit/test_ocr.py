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
