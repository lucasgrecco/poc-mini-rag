"""Unit tests for the ingestion pipeline's per-batch commit and resume logic.

DB-free: a FakeSession stands in for SQLAlchemy's Session and the embedding
and upsert functions are monkeypatched.
"""

import json
import logging
from unittest import mock

from app import ingest
from app.ingest import process_jsons


class FakeSession:
    """Minimal stand-in for sqlalchemy.orm.Session used by process_jsons."""

    def __init__(self, existing_ids=()):
        self.existing_ids = set(existing_ids)
        self.commits = 0
        self.rollbacks = 0
        self.scalars_calls = 0
        self.commit_failures = {}  # commit number (1-based) -> exception to raise

    def scalars(self, stmt):
        self.scalars_calls += 1
        return list(self.existing_ids)

    def commit(self):
        self.commits += 1
        exc = self.commit_failures.get(self.commits)
        if exc is not None:
            raise exc

    def rollback(self):
        self.rollbacks += 1

    def execute(self, stmt):
        pass


def _write_cards(tmp_path, n, start=1000):
    """Write n JSON card files with ids start..start+n-1.

    Ids are 4-digit so that lexicographic filename order matches numeric
    id order, giving deterministic batch boundaries.
    """
    for i in range(start, start + n):
        (tmp_path / f"{i}.json").write_text(
            json.dumps({"id": i, "name": f"Card {i}"}), encoding="utf-8"
        )


def _mock_batch_embeddings(monkeypatch):
    """Mock get_both_embeddings_batch to return None vectors (no API calls)."""
    embed = mock.Mock(
        side_effect=lambda texts: ([None] * len(texts), [None] * len(texts))
    )
    monkeypatch.setattr(ingest, "get_both_embeddings_batch", embed)
    return embed


def _total_embedded(embed_mock):
    """Total number of texts passed to the batch embedding mock."""
    return sum(len(call.args[0]) for call in embed_mock.call_args_list)


def test_commit_once_per_batch(tmp_path, monkeypatch):
    """250 files -> 3 batches -> exactly 3 commits, all 250 ingested."""
    _write_cards(tmp_path, 250)
    session = FakeSession()
    _mock_batch_embeddings(monkeypatch)
    upsert = mock.Mock()
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", upsert)

    count = process_jsons(str(tmp_path), session)

    assert count == 250
    assert session.commits == 3
    assert session.rollbacks == 0
    assert upsert.call_count == 250


def test_rollback_on_batch_failure_continues(tmp_path, monkeypatch):
    """A batch-2 upsert failure rolls back batch 2 and continues to batch 3."""
    _write_cards(tmp_path, 300)
    session = FakeSession()
    _mock_batch_embeddings(monkeypatch)

    def _upsert(card_json, openai_vec, local_vec, session):
        if 1100 <= card_json["id"] <= 1199:
            raise RuntimeError("injected upsert failure")

    monkeypatch.setattr(ingest, "upsert_card_with_vectors", _upsert)

    count = process_jsons(str(tmp_path), session)

    assert count == 200
    assert session.commits == 2
    assert session.rollbacks == 1


def test_commit_failure_rolls_back(tmp_path, monkeypatch):
    """A commit failure on the first batch rolls back and the run continues."""
    _write_cards(tmp_path, 300)
    session = FakeSession()
    session.commit_failures = {1: RuntimeError("injected commit failure")}
    _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", mock.Mock())

    count = process_jsons(str(tmp_path), session)

    assert count == 200
    assert session.commits == 3
    assert session.rollbacks == 1


def test_fallback_commits_per_batch(tmp_path, monkeypatch):
    """Batch embedding failure falls back to per-card upserts, one commit per batch."""
    _write_cards(tmp_path, 250)
    session = FakeSession()
    monkeypatch.setattr(
        ingest,
        "get_both_embeddings_batch",
        mock.Mock(side_effect=RuntimeError("batch embedding down")),
    )
    monkeypatch.setattr(ingest, "upsert_card", mock.Mock())
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", mock.Mock())

    count = process_jsons(str(tmp_path), session)

    assert count == 250
    assert session.commits == 3
    assert session.rollbacks == 0


def test_resume_skips_present(tmp_path, monkeypatch):
    """50 of 150 files already present -> only 100 are embedded."""
    _write_cards(tmp_path, 150)
    session = FakeSession(existing_ids=range(1000, 1050))
    embed = _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", mock.Mock())

    count = process_jsons(str(tmp_path), session)

    assert count == 100
    assert session.commits == 1
    assert _total_embedded(embed) == 100


def test_force_bypasses_skip(tmp_path, monkeypatch):
    """--force re-embeds all files even when every id is already present."""
    _write_cards(tmp_path, 150)
    session = FakeSession(existing_ids=range(1000, 1150))
    embed = _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", mock.Mock())

    count = process_jsons(str(tmp_path), session, force=True)

    assert count == 150
    assert session.scalars_calls == 0
    assert _total_embedded(embed) == 150


def test_summary_log_counts(tmp_path, monkeypatch, caplog):
    """The summary log reports skipped and ingested counts."""
    _write_cards(tmp_path, 150)
    session = FakeSession(existing_ids=range(1000, 1050))
    _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", mock.Mock())

    with caplog.at_level(logging.INFO, logger="app.ingest"):
        count = process_jsons(str(tmp_path), session)

    assert count == 100
    assert any(
        "Ingestion summary: 50 files skipped (already in cards), 100 ingested this run"
        in record.getMessage()
        for record in caplog.records
    )


def test_nothing_pending_returns_zero(tmp_path, monkeypatch, caplog):
    """When every file is already present, the run returns 0 immediately."""
    _write_cards(tmp_path, 50)
    session = FakeSession(existing_ids=range(1000, 1050))
    embed = _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", mock.Mock())

    with caplog.at_level(logging.INFO, logger="app.ingest"):
        count = process_jsons(str(tmp_path), session)

    assert count == 0
    assert session.commits == 0
    assert embed.call_count == 0
    assert any(
        "Ingestion summary: 50 files skipped (already in cards), 0 ingested this run"
        in record.getMessage()
        for record in caplog.records
    )
