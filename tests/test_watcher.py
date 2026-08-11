"""Unit tests for the file watcher: race-condition safety and size filtering."""

import json
import logging
from unittest import mock

from app.watcher import CardFileHandler


def test_handle_change_survives_file_removed_before_stat(tmp_path, caplog):
    """A file removed between the event and stat() must not crash the handler."""
    path = tmp_path / "race.json"
    path.write_text(
        json.dumps({"id": "race-1", "name": "Race Card"}), encoding="utf-8"
    )
    # Simulate create-then-remove: gone before the handler ever fires, so
    # path.stat() raises FileNotFoundError (an OSError subclass).
    path.unlink()

    session_factory = mock.Mock()
    handler = CardFileHandler(session_factory)

    with caplog.at_level(logging.ERROR, logger="app.watcher"):
        # Must be absorbed by the try/except, not escape into the handler thread.
        handler._handle_change(str(path))

    session_factory.assert_not_called()
    assert any(
        "Failed to read" in record.getMessage() for record in caplog.records
    )


def test_handle_change_still_skips_oversized_file(tmp_path, caplog):
    """Files over the 50 KB limit are skipped without touching the database."""
    path = tmp_path / "big.json"
    path.write_text(
        json.dumps(
            {"id": "big-1", "name": "Big Card", "payload": "x" * (50 * 1024 + 1)}
        ),
        encoding="utf-8",
    )

    session_factory = mock.Mock()
    handler = CardFileHandler(session_factory)

    with caplog.at_level(logging.WARNING, logger="app.watcher"):
        handler._handle_change(str(path))

    session_factory.assert_not_called()
    assert any(
        "Skipping large file" in record.getMessage() for record in caplog.records
    )
