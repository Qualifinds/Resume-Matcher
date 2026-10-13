"""PRAGMA busy_timeout comes from settings (SQLITE_BUSY_TIMEOUT_MS)."""

from sqlalchemy import text

from app.config import settings
from app.db_engine import make_sync_engine


def _busy_timeout(path) -> int:
    engine = make_sync_engine(path)
    try:
        with engine.connect() as conn:
            return conn.execute(text("PRAGMA busy_timeout")).scalar_one()
    finally:
        engine.dispose()


def test_default_busy_timeout_is_5000(tmp_path):
    assert settings.sqlite_busy_timeout_ms == 5000
    assert _busy_timeout(tmp_path / "a.db") == 5000


def test_busy_timeout_follows_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "sqlite_busy_timeout_ms", 30000)
    assert _busy_timeout(tmp_path / "b.db") == 30000
