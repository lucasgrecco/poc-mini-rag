"""File watcher: monitors the card JSON directory and auto-updates the database.

When JSON files are created, modified, or deleted in the watched directory,
this module automatically regenerates embeddings and updates the database.

Usage:
    python -m app.watcher                  # Watch default directory
    python -m app.watcher --json-dir ./jsons
    python -m app.watcher --polling        # Use polling (for macOS Docker Desktop)
    python -m app.watcher -v               # Verbose logging
"""

import argparse
import json
import logging
import signal
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL, DEFAULT_JSON_DIR
from app.ingest import upsert_card
from app.models import Base, Card

logger = logging.getLogger(__name__)


class CardFileHandler(FileSystemEventHandler):
    """Handles file system events for card JSON files."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self.session_factory = session_factory
        self._last_event: dict[tuple[str, str], float] = {}
        self._debounce_seconds = 0.5
        self._max_file_size = 50 * 1024  # 50 KB max per JSON file

    def _should_process(self, file_path: str) -> bool:
        """Filter: only process .json files, skip hidden/dotfiles."""
        path = Path(file_path)
        if path.suffix != ".json":
            return False
        if path.name.startswith("."):
            return False
        return True

    def _debounce(self, file_path: str, event_type: str) -> bool:
        """Return True if this event should be processed (not a duplicate).

        Some editors and OSes fire multiple events for a single change.
        This ignores events for the same (file, event_type) within the debounce window.
        """
        now = time.time()
        key = (file_path, event_type)
        last = self._last_event.get(key, 0)
        if now - last < self._debounce_seconds:
            return False
        self._last_event[key] = now
        # Prevent unbounded growth: purge entries older than 10 seconds
        self._last_event = {
            k: v for k, v in self._last_event.items()
            if now - v < 10.0
        }
        return True

    def on_created(self, event: object) -> None:
        if event.is_directory or not self._should_process(event.src_path):
            return
        if not self._debounce(event.src_path, "created"):
            return
        self._handle_change(event.src_path)

    def on_modified(self, event: object) -> None:
        if event.is_directory or not self._should_process(event.src_path):
            return
        if not self._debounce(event.src_path, "modified"):
            return
        self._handle_change(event.src_path)

    def on_moved(self, event: object) -> None:
        """Handle file renames and moves between directories."""
        if event.is_directory:
            return
        # File moved into the watched directory: treat as created
        if self._should_process(event.dest_path):
            if self._debounce(event.dest_path, "created"):
                self._handle_change(event.dest_path)
        # File moved out of the watched directory: treat as deleted
        if self._should_process(event.src_path):
            if self._debounce(event.src_path, "deleted"):
                self._handle_delete(event.src_path)

    def on_deleted(self, event: object) -> None:
        if event.is_directory or not self._should_process(event.src_path):
            return
        self._handle_delete(event.src_path)

    def _handle_change(self, file_path: str) -> None:
        """Read JSON file, generate embedding, and upsert into the database."""
        path = Path(file_path)
        try:
            if path.stat().st_size > self._max_file_size:
                logger.warning("Skipping large file: %s", file_path)
                return
            with open(file_path, "r", encoding="utf-8") as f:
                card_json = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read %s: %s", file_path, e)
            return

        with self.session_factory() as session:
            try:
                upsert_card(card_json, session)
                session.commit()
                logger.info(
                    "Upserted card %s (id=%s)",
                    card_json.get("name"),
                    card_json.get("id"),
                )
            except Exception:
                logger.exception("Failed to upsert card from %s", file_path)
                session.rollback()

    def _handle_delete(self, file_path: str) -> None:
        """Extract card_json_id from filename and delete the corresponding row."""
        try:
            card_id = int(Path(file_path).stem)
        except ValueError:
            logger.warning(
                "Cannot extract card ID from filename: %s", file_path
            )
            return

        with self.session_factory() as session:
            try:
                result = session.execute(
                    delete(Card).where(Card.card_json_id == card_id)
                )
                session.commit()
                if result.rowcount:
                    logger.info(
                        "Deleted card %d (file: %s)", card_id, file_path
                    )
                else:
                    logger.warning(
                        "No card found with card_json_id=%d (file: %s)",
                        card_id,
                        file_path,
                    )
            except Exception:
                logger.exception(
                    "Failed to delete card %d (file: %s)", card_id, file_path
                )
                session.rollback()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch card JSON directory for changes."
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help="Directory containing card JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--polling",
        action="store_true",
        help="Use polling observer instead of inotify (for macOS Docker Desktop)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    json_dir = Path(args.json_dir)
    if not json_dir.is_dir():
        logger.error("Directory not found: %s", json_dir)
        raise SystemExit(1)

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    observer_cls = PollingObserver if args.polling else Observer
    observer = observer_cls()
    handler = CardFileHandler(session_factory)
    observer.schedule(handler, str(json_dir), recursive=False)

    def shutdown(signum: int, frame: object) -> None:
        logger.info("Shutting down watcher (signal %d)...", signum)
        observer.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logger.info(
        "Watching %s (observer=%s)...",
        json_dir,
        "polling" if args.polling else "inotify",
    )
    observer.start()
    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        observer.stop()
        observer.join()
        logger.info("Watcher stopped.")


if __name__ == "__main__":
    main()
