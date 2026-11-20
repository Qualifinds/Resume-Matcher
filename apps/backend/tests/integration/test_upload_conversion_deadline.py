"""POST /resumes/upload keeps its HTTP contract when a conversion process overruns or dies.

Incident 2026-10-22 to 2026-11-19: two uploads whose conversion never finished took both conversion
slots and every later upload answered 504 until the service restarted. Here the converter child is
the fake worker of the unit tests (it hangs or crashes on demand) and ``parse_document`` is real.
"""

import sys
import tempfile
import time
from pathlib import Path

import anyio
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.services import parser

FAKE_WORKER = Path(__file__).parents[1] / "unit" / "_fake_conversion_worker.py"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture
def fake_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Two slots, a 0.5s deadline and the fake worker as the conversion process."""
    workdirs = tmp_path / "workdirs"
    workdirs.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(workdirs))
    monkeypatch.setattr(parser, "_WORKER_COMMAND", [sys.executable, str(FAKE_WORKER)])
    monkeypatch.setattr(
        parser, "_DOCUMENT_CONVERSION_LIMITER", anyio.CapacityLimiter(2)
    )
    monkeypatch.setattr(
        parser, "DOCUMENT_CONVERSION_TIMEOUT_SECONDS", 0.5, raising=False
    )

    def set_mode(mode: str) -> None:
        monkeypatch.setenv("FAKE_CONVERSION_MODE", mode)

    return set_mode, workdirs


async def _upload():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(
            "/api/v1/resumes/upload",
            files={"file": ("resume.docx", b"synthetic bytes", DOCX)},
        )


async def test_repeated_hung_conversions_keep_answering_504_not_stall(
    fake_converter,
) -> None:
    set_mode, workdirs = fake_converter
    set_mode("hang")
    # More hung uploads than slots: before, the third one waited behind two zombies forever.
    for _ in range(4):
        started = time.monotonic()
        response = await _upload()
        assert response.status_code == 504
        assert response.json()["detail"] == (
            "Document conversion timed out. Please try a simpler document."
        )
        assert time.monotonic() - started < 5
    assert list(workdirs.iterdir()) == []


async def test_dead_conversion_process_answers_422_like_a_bad_file(
    fake_converter,
) -> None:
    set_mode, workdirs = fake_converter
    set_mode("crash")
    response = await _upload()
    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Failed to parse document. Please upload a valid PDF, DOC, or DOCX file."
    )
    assert list(workdirs.iterdir()) == []


async def test_invalid_and_oversized_documents_keep_their_status_codes(
    fake_converter,
) -> None:
    set_mode, _ = fake_converter
    set_mode("limit")
    assert (await _upload()).status_code == 413
    set_mode("validation")
    assert (await _upload()).status_code == 422
