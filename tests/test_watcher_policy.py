"""Unit tests for the watcher's tree-aware move/delete policy.

DB-free: the session factory is mocked and ``app.watcher.upsert_card`` is
patched, so no database or embedding calls are made. A fresh handler is
created per test so the 0.5s debounce state never leaks between tests.
"""

import json
import logging
import types
from unittest import mock

from app.watcher import CardFileHandler


def _write_json(path, card_id):
    path.write_text(
        json.dumps({"id": card_id, "name": f"Card {card_id}"}),
        encoding="utf-8",
    )


def test_in_tree_move_to_subfolder_upserts_dest_no_delete(tmp_path):
    """Move 10000.json into a subfolder: upsert dest, never delete by src."""
    src_dir = tmp_path / "jsons"
    src_dir.mkdir()
    dest_dir = src_dir / "archive"
    dest_dir.mkdir()
    src = src_dir / "10000.json"
    dest = dest_dir / "10000.json"
    _write_json(src, 10000)
    _write_json(dest, 10000)

    handler = CardFileHandler(mock.MagicMock())
    event = types.SimpleNamespace(
        is_directory=False, src_path=str(src), dest_path=str(dest)
    )

    with mock.patch("app.watcher.upsert_card") as mock_upsert, mock.patch.object(
        handler, "_handle_delete"
    ) as mock_delete:
        handler.on_moved(event)

    mock_upsert.assert_called_once()
    mock_delete.assert_not_called()


def test_in_tree_rename_json_to_json_upserts_no_delete(tmp_path):
    """Rename 10000.json -> 10001.json in the same dir: upsert, no delete."""
    src_dir = tmp_path / "jsons"
    src_dir.mkdir()
    src = src_dir / "10000.json"
    dest = src_dir / "10001.json"
    _write_json(src, 10000)
    _write_json(dest, 10001)

    handler = CardFileHandler(mock.MagicMock())
    event = types.SimpleNamespace(
        is_directory=False, src_path=str(src), dest_path=str(dest)
    )

    with mock.patch("app.watcher.upsert_card") as mock_upsert, mock.patch.object(
        handler, "_handle_delete"
    ) as mock_delete:
        handler.on_moved(event)

    mock_upsert.assert_called_once()
    mock_delete.assert_not_called()


def test_rename_json_to_non_json_deletes_src_no_upsert(tmp_path):
    """Rename 10000.json -> notes.txt: delete by src stem, no upsert."""
    src_dir = tmp_path / "jsons"
    src_dir.mkdir()
    src = src_dir / "10000.json"
    dest = src_dir / "notes.txt"
    _write_json(src, 10000)

    handler = CardFileHandler(mock.MagicMock())
    event = types.SimpleNamespace(
        is_directory=False, src_path=str(src), dest_path=str(dest)
    )

    with mock.patch("app.watcher.upsert_card") as mock_upsert, mock.patch.object(
        handler, "_handle_delete"
    ) as mock_delete:
        handler.on_moved(event)

    mock_upsert.assert_not_called()
    mock_delete.assert_called_once_with(str(src))


def test_on_deleted_valid_id_deletes_row(caplog):
    """A FileDeletedEvent for a valid id deletes the row and logs it."""
    session_factory = mock.MagicMock()
    session = session_factory.return_value.__enter__.return_value
    result = mock.MagicMock()
    result.rowcount = 1
    session.execute.return_value = result
    handler = CardFileHandler(session_factory)
    event = types.SimpleNamespace(
        is_directory=False, src_path="/app/jsons/10000.json"
    )

    with caplog.at_level(logging.INFO, logger="app.watcher"):
        handler.on_deleted(event)

    session.execute.assert_called_once()
    session.commit.assert_called_once()
    assert "Deleted card 10000" in caplog.text


def test_move_in_from_outside_created_upserts(tmp_path):
    """A file created in the tree (move-in from outside) upserts it."""
    src_dir = tmp_path / "jsons"
    src_dir.mkdir()
    src = src_dir / "10000.json"
    _write_json(src, 10000)

    handler = CardFileHandler(mock.MagicMock())
    event = types.SimpleNamespace(is_directory=False, src_path=str(src))

    with mock.patch("app.watcher.upsert_card") as mock_upsert:
        handler.on_created(event)

    mock_upsert.assert_called_once()


def test_directory_events_ignored(tmp_path):
    """Directory events never trigger upserts or deletes."""
    src_dir = tmp_path / "jsons"
    src_dir.mkdir()
    src = src_dir / "10000.json"
    _write_json(src, 10000)

    handler = CardFileHandler(mock.MagicMock())
    event = types.SimpleNamespace(
        is_directory=True, src_path=str(src), dest_path=str(src)
    )

    with mock.patch("app.watcher.upsert_card") as mock_upsert, mock.patch.object(
        handler, "_handle_delete"
    ) as mock_delete:
        handler.on_moved(event)
        handler.on_created(event)
        handler.on_deleted(event)

    mock_upsert.assert_not_called()
    mock_delete.assert_not_called()
