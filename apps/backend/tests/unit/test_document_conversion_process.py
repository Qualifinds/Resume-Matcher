"""Process supervision of document conversion: a conversion that overruns is killed.

Incident 2026-10-22 to 2026-11-19: conversion ran in a thread that could not be killed, so two PDFs
that never finished kept both slots and every upload answered 504 until the service restarted. The
conversion now runs in a child process that ``parse_document`` kills at the deadline.

Most tests swap the child for ``_fake_conversion_worker.py`` (hang, crash, hold...) because no real
PDF reliably hangs pdfminer; ``test_real_worker_is_killed_at_the_deadline`` uses the real module.
"""

import asyncio
import io
import logging
import sys
import tempfile
import time
from pathlib import Path

import anyio
import pytest
from docx import Document

from app.config import Settings
from app.document_conversion import DocumentResourceLimitError, DocumentValidationError
from app.services import parser

FAKE_WORKER = Path(__file__).with_name("_fake_conversion_worker.py")
BEAT_INTERVAL = 0.05


@pytest.fixture
def control_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Directory the fake worker reports into; also where conversion work dirs are created."""
    control = tmp_path / "control"
    control.mkdir()
    workdirs = tmp_path / "workdirs"
    workdirs.mkdir()
    monkeypatch.setenv("FAKE_CONVERSION_DIR", str(control))
    monkeypatch.setattr(tempfile, "tempdir", str(workdirs))
    monkeypatch.setattr(parser, "_WORKER_COMMAND", [sys.executable, str(FAKE_WORKER)])
    return control


@pytest.fixture
def fake_mode(monkeypatch: pytest.MonkeyPatch):
    def set_mode(mode: str) -> None:
        monkeypatch.setenv("FAKE_CONVERSION_MODE", mode)

    return set_mode


@pytest.fixture
def slots(monkeypatch: pytest.MonkeyPatch):
    """Replace the process-wide limiter with one of ``n`` slots and set the deadline."""

    def configure(n: int, timeout: float) -> anyio.CapacityLimiter:
        limiter = anyio.CapacityLimiter(n)
        monkeypatch.setattr(parser, "_DOCUMENT_CONVERSION_LIMITER", limiter)
        monkeypatch.setattr(
            parser, "DOCUMENT_CONVERSION_TIMEOUT_SECONDS", timeout, raising=False
        )
        return limiter

    return configure


def _workdirs() -> list[Path]:
    return list(Path(tempfile.tempdir).iterdir())


def _started(control: Path) -> list[Path]:
    return sorted(control.glob("started-*"))


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.02)


async def _assert_workers_dead(control: Path) -> None:
    """Every worker that started has stopped beating, i.e. its process is gone."""
    beats = list(control.glob("beat-*"))
    assert beats, "the fake worker never started beating"
    before = {b: b.read_text() for b in beats}
    await asyncio.sleep(BEAT_INTERVAL * 8)
    assert {b: b.read_text() for b in beats} == before, "a worker is still running"


def _docx_bytes(text: str) -> bytes:
    document = Document()
    document.add_paragraph(text)
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


# ---- the incident: a hung conversion is killed and its slot comes back ----


async def test_hung_conversion_is_killed_at_the_deadline_and_frees_its_slot(
    control_dir: Path, fake_mode, slots
) -> None:
    limiter = slots(1, timeout=0.5)
    fake_mode("hang")
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await parser.parse_document(b"x", "stuck.pdf")
    assert time.monotonic() - started < 5
    await _assert_workers_dead(control_dir)
    assert limiter.borrowed_tokens == 0
    assert _workdirs() == []

    # The same single slot serves the next upload straight away.
    fake_mode("ok")
    started = time.monotonic()
    assert await parser.parse_document(b"x", "next.pdf") == "fake text"
    assert time.monotonic() - started < 5


async def test_every_slot_stuck_at_once_recovers_without_a_restart(
    control_dir: Path, fake_mode, slots
) -> None:
    """The 2026-11-19 state: all slots taken by conversions that never finish."""
    limiter = slots(2, timeout=0.6)
    fake_mode("hang")
    stuck = await asyncio.gather(
        parser.parse_document(b"x", "one.pdf"),
        parser.parse_document(b"x", "two.pdf"),
        return_exceptions=True,
    )
    assert all(isinstance(result, TimeoutError) for result in stuck)
    await _assert_workers_dead(control_dir)
    assert limiter.borrowed_tokens == 0

    fake_mode("ok")
    results = await asyncio.wait_for(
        asyncio.gather(*(parser.parse_document(b"x", f"{i}.pdf") for i in range(4))),
        timeout=10,
    )
    assert results == ["fake text"] * 4


async def test_real_worker_is_killed_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, slots
) -> None:
    """The real ``python -m app.document_conversion`` is still importing at a 0.1s deadline."""
    workdirs = tmp_path / "workdirs"
    workdirs.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(workdirs))
    limiter = slots(1, timeout=0.1)
    with pytest.raises(TimeoutError):
        await parser.parse_document(_docx_bytes("never finishes"), "slow.docx")
    assert limiter.borrowed_tokens == 0
    assert list(workdirs.iterdir()) == []


async def test_deadline_kill_is_logged_with_its_file(
    control_dir: Path, fake_mode, slots, caplog: pytest.LogCaptureFixture
) -> None:
    slots(1, timeout=0.3)
    fake_mode("hang")
    caplog.set_level(logging.WARNING, logger="app.services.parser")
    with pytest.raises(TimeoutError):
        await parser.parse_document(b"x", "slow-cv.pdf")
    assert "killed after exceeding its" in caplog.text
    assert "slow-cv.pdf" in caplog.text


async def test_cancelling_the_request_kills_the_process(
    control_dir: Path, fake_mode, slots
) -> None:
    limiter = slots(1, timeout=30)
    fake_mode("hold")
    task = asyncio.create_task(parser.parse_document(b"x", "cancelled.pdf"))
    await _wait_for(lambda: len(_started(control_dir)) == 1)
    assert len(_workdirs()) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _assert_workers_dead(control_dir)
    assert limiter.borrowed_tokens == 0
    assert _workdirs() == []


async def test_cancelled_queued_conversion_never_starts(
    control_dir: Path, fake_mode, slots
) -> None:
    slots(2, timeout=30)
    fake_mode("hold")
    owners = [
        asyncio.create_task(parser.parse_document(b"x", f"{i}.pdf")) for i in range(2)
    ]
    await _wait_for(lambda: len(_started(control_dir)) == 2)
    queued = asyncio.create_task(parser.parse_document(b"x", "queued.pdf"))
    await asyncio.sleep(0.1)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    (control_dir / "release").touch()
    assert await asyncio.gather(*owners) == ["fake text"] * 2
    assert len(_started(control_dir)) == 2


async def test_at_most_the_configured_slots_run_at_once(
    control_dir: Path, fake_mode, slots
) -> None:
    slots(2, timeout=30)
    fake_mode("hold")
    tasks = [
        asyncio.create_task(parser.parse_document(b"x", f"{i}.pdf")) for i in range(3)
    ]
    await _wait_for(lambda: len(_started(control_dir)) == 2)
    await asyncio.sleep(0.3)
    assert len(_started(control_dir)) == 2, "a third process started past the limit"
    (control_dir / "release").touch()
    assert (
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=10) == ["fake text"] * 3
    )
    assert len(_started(control_dir)) == 3


# ---- same contract as before for the other outcomes ----


async def test_real_worker_converts_a_minimal_pdf_and_a_docx() -> None:
    from tests.unit.test_document_processing import _pdf_bytes

    pdf = _pdf_bytes(b"BT /F1 12 Tf 72 720 Td (Process engineer) Tj ET")
    assert "Process engineer" in await parser.parse_document(pdf, "min.pdf")
    assert (
        await parser.parse_document(_docx_bytes("Word engineer"), "min.docx")
    ).strip() == "Word engineer"


async def test_workdir_is_removed_after_success_and_after_a_real_file_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workdirs = tmp_path / "workdirs"
    workdirs.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(workdirs))
    assert await parser.parse_document(_docx_bytes("clean"), "ok.docx")
    assert list(workdirs.iterdir()) == []
    with pytest.raises(DocumentValidationError, match="valid PDF, DOC, or DOCX"):
        await parser.parse_document(b"PK corrupt", "bad.docx")
    assert list(workdirs.iterdir()) == []


async def test_child_that_dies_is_a_conversion_error_and_cleans_up(
    control_dir: Path, fake_mode, slots
) -> None:
    limiter = slots(1, timeout=10)
    fake_mode("crash")
    with pytest.raises(parser.DocumentConversionError) as caught:
        await parser.parse_document(b"x", "oom.pdf")
    assert "code 3" in str(caught.value)
    assert "fake converter blew up" in str(caught.value)
    assert limiter.borrowed_tokens == 0
    assert _workdirs() == []


@pytest.mark.parametrize("mode", ["silent", "garbage"])
async def test_child_without_a_readable_result_is_a_conversion_error(
    control_dir: Path, fake_mode, slots, mode: str
) -> None:
    slots(1, timeout=10)
    fake_mode(mode)
    with pytest.raises(parser.DocumentConversionError, match="no readable result"):
        await parser.parse_document(b"x", "odd.pdf")
    assert _workdirs() == []


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("validation", DocumentValidationError),
        ("limit", DocumentResourceLimitError),
        ("other", parser.DocumentConversionError),
    ],
)
async def test_errors_reported_by_the_child_keep_their_type(
    control_dir: Path, fake_mode, slots, mode: str, expected: type[Exception]
) -> None:
    slots(1, timeout=10)
    fake_mode(mode)
    with pytest.raises(expected) as caught:
        await parser.parse_document(b"x", "err.pdf")
    assert type(caught.value) is expected
    assert _workdirs() == []


async def test_odd_extension_still_converts(
    control_dir: Path, fake_mode, slots
) -> None:
    slots(1, timeout=10)
    fake_mode("ok")
    assert await parser.parse_document(b"x", "weird.p\x00df") == "fake text"
    started = _started(control_dir)[0].read_text()
    assert Path(started).name == "input"


# ---- configuration ----


def test_conversion_slots_and_deadline_follow_settings() -> None:
    from app.config import settings

    assert parser.DOCUMENT_CONVERSION_WORKERS == settings.document_conversion_workers
    assert (
        parser.DOCUMENT_CONVERSION_TIMEOUT_SECONDS
        == settings.document_conversion_timeout_seconds
    )
    assert (
        parser._DOCUMENT_CONVERSION_LIMITER.total_tokens
        == settings.document_conversion_workers
    )


def test_conversion_settings_default_to_two_slots_and_120_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOCUMENT_CONVERSION_WORKERS", raising=False)
    monkeypatch.delenv("DOCUMENT_CONVERSION_TIMEOUT_SECONDS", raising=False)
    s = Settings(_env_file=None)
    assert s.document_conversion_workers == 2
    assert s.document_conversion_timeout_seconds == 120.0


def test_conversion_settings_accept_valid_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCUMENT_CONVERSION_WORKERS", "4")
    monkeypatch.setenv("DOCUMENT_CONVERSION_TIMEOUT_SECONDS", "45.5")
    s = Settings(_env_file=None)
    assert s.document_conversion_workers == 4
    assert s.document_conversion_timeout_seconds == 45.5


@pytest.mark.parametrize(
    "value", ["", "  ", "abc", "0", "-3", "33", "2.5", "nan", "inf", "1e9"]
)
def test_invalid_conversion_workers_fall_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DOCUMENT_CONVERSION_WORKERS", value)
    assert Settings(_env_file=None).document_conversion_workers == 2


@pytest.mark.parametrize(
    "value", ["", "  ", "abc", "0", "-5", "0.5", "601", "nan", "inf", "1e9"]
)
def test_invalid_conversion_timeout_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DOCUMENT_CONVERSION_TIMEOUT_SECONDS", value)
    assert Settings(_env_file=None).document_conversion_timeout_seconds == 120.0
