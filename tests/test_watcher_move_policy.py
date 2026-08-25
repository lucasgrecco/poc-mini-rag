"""DB-free tests for the watcher's move/delete policy.

Only a real deletion removes a card from the database. A move must never
delete: the observer runs with ``recursive=False``, so reorganizing the card
directory -- archiving a batch, moving cards into a subfolder, mass renaming --
produces move events whose source leaves the watched directory, and deleting on
that signal would drop those cards from the search index.

``on_moved`` still upserts the destination, so a card moved *into* the watched
directory is indexed like a created file.
"""

import logging

import pytest

from app.watcher import CardFileHandler

WATCHED = "/watched/jsons"


class FakeMovedEvent:
    """Minimal stand-in for watchdog's FileMovedEvent / DirMovedEvent."""

    def __init__(
        self, src_path: str, dest_path: str, is_directory: bool = False
    ) -> None:
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = is_directory


class FakeDeletedEvent:
    """Minimal stand-in for watchdog's FileDeletedEvent / DirDeletedEvent."""

    def __init__(self, src_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.dest_path = None
        self.is_directory = is_directory


def _boom_session_factory():
    """Session factory that fails loudly if a test unexpectedly reaches the DB."""
    raise AssertionError("session_factory must not be used by this test")


def _recording_handler():
    """Handler whose DB-touching methods only record the paths they receive.

    Returns ``(handler, changed, deleted)``; ``changed`` and ``deleted`` are the
    lists of file paths handed to ``_handle_change`` / ``_handle_delete``.
    """
    handler = CardFileHandler(_boom_session_factory)
    changed: list[str] = []
    deleted: list[str] = []
    handler._handle_change = changed.append
    handler._handle_delete = deleted.append
    return handler, changed, deleted


def test_move_out_of_watched_dir_does_not_delete():
    # The regression this policy exists for: archiving a card into a subfolder
    # must not remove it from the index.
    handler, _changed, deleted = _recording_handler()
    handler.on_moved(
        FakeMovedEvent(f"{WATCHED}/4101.json", f"{WATCHED}/archive/4101.json")
    )
    assert deleted == []


def test_move_out_to_an_unrelated_directory_does_not_delete():
    handler, _changed, deleted = _recording_handler()
    handler.on_moved(
        FakeMovedEvent(f"{WATCHED}/4102.json", "/somewhere/else/4102.json")
    )
    assert deleted == []


def test_bulk_move_out_keeps_every_row():
    # "Moving 500 cards into jsons/archive/ => 500 rows kept." Distinct paths,
    # so the debounce window cannot be what suppresses the deletes.
    handler, _changed, deleted = _recording_handler()
    for card_id in range(4200, 4300):
        handler.on_moved(
            FakeMovedEvent(
                f"{WATCHED}/{card_id}.json",
                f"{WATCHED}/archive/{card_id}.json",
            )
        )
    assert deleted == []


def test_move_out_is_logged_at_info_naming_the_file(caplog):
    handler, _changed, _deleted = _recording_handler()
    src = f"{WATCHED}/4103.json"
    with caplog.at_level(logging.INFO, logger="app.watcher"):
        handler.on_moved(FakeMovedEvent(src, "/elsewhere/4103.json"))
    messages = [
        r.getMessage() for r in caplog.records if r.levelno == logging.INFO
    ]
    assert any(src in m for m in messages), messages


def test_move_into_watched_dir_upserts_destination():
    handler, changed, deleted = _recording_handler()
    dest = f"{WATCHED}/4104.json"
    handler.on_moved(FakeMovedEvent("/incoming/4104.json", dest))
    assert changed == [dest]
    assert deleted == []


def test_rename_within_directory_upserts_dest_and_keeps_source():
    # Accepted consequence of the policy: the old row survives the rename.
    handler, changed, deleted = _recording_handler()
    src = f"{WATCHED}/4007.json"
    dest = f"{WATCHED}/4008.json"
    handler.on_moved(FakeMovedEvent(src, dest))
    assert changed == [dest]
    assert deleted == []


def test_on_deleted_still_deletes():
    handler, _changed, deleted = _recording_handler()
    src = f"{WATCHED}/4105.json"
    handler.on_deleted(FakeDeletedEvent(src))
    assert deleted == [src]


def test_on_deleted_reaches_the_session_factory():
    # Guards the real delete path: on_deleted must still get as far as the DB
    # layer, not merely call a method that returns early.
    called = []

    def factory():
        called.append(True)
        raise RuntimeError("stop before touching the DB")

    handler = CardFileHandler(factory)
    with pytest.raises(RuntimeError):
        handler.on_deleted(FakeDeletedEvent(f"{WATCHED}/4106.json"))
    assert called


def test_directory_move_is_ignored():
    handler, changed, deleted = _recording_handler()
    handler.on_moved(
        FakeMovedEvent(f"{WATCHED}/subdir", f"{WATCHED}/renamed", is_directory=True)
    )
    assert changed == []
    assert deleted == []


def test_non_json_move_out_is_not_logged(caplog):
    handler, changed, deleted = _recording_handler()
    with caplog.at_level(logging.INFO, logger="app.watcher"):
        handler.on_moved(FakeMovedEvent(f"{WATCHED}/notes.txt", "/elsewhere/notes.txt"))
    assert changed == []
    assert deleted == []
    assert [r.getMessage() for r in caplog.records] == []
