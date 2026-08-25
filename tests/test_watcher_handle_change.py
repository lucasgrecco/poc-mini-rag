"""DB-free tests for CardFileHandler._handle_change error handling.

A JSON file can be created and removed before the watcher gets to read it.
The size check must not escape the try block, otherwise the FileNotFoundError
propagates into watchdog's dispatch thread and kills the whole watcher.
"""

import json
import logging

import pytest

from app.watcher import CardFileHandler


def _boom_session_factory():
    """Session factory that fails loudly if the handler ever reaches the DB."""
    raise AssertionError("session_factory must not be used for unreadable files")


def test_missing_file_returns_quietly():
    handler = CardFileHandler(_boom_session_factory)
    assert handler._handle_change("/path/that/does/not/exist.json") is None


def test_missing_file_logs_no_warning(caplog):
    handler = CardFileHandler(_boom_session_factory)
    with caplog.at_level(logging.DEBUG, logger="app.watcher"):
        handler._handle_change("/path/that/does/not/exist.json")
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_large_file_is_skipped(caplog, tmp_path):
    handler = CardFileHandler(_boom_session_factory)
    big = tmp_path / "1234.json"
    big.write_text("x" * (handler._max_file_size + 1), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="app.watcher"):
        assert handler._handle_change(str(big)) is None
    assert any("Skipping large file" in r.getMessage() for r in caplog.records)


def test_malformed_json_is_logged_and_skipped(caplog, tmp_path):
    handler = CardFileHandler(_boom_session_factory)
    bad = tmp_path / "5678.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="app.watcher"):
        assert handler._handle_change(str(bad)) is None
    assert any("Failed to read" in r.getMessage() for r in caplog.records)


def test_readable_file_reaches_the_session_factory(tmp_path):
    # Guards the happy path: a valid, small file must still be handed to the
    # session factory (which is where upsert_card runs).
    called = []

    def factory():
        called.append(True)
        raise RuntimeError("stop before touching the DB")

    handler = CardFileHandler(factory)
    good = tmp_path / "4007.json"
    good.write_text(json.dumps({"id": 4007, "name": "Test Card"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        handler._handle_change(str(good))
    assert called
