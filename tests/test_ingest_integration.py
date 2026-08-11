"""Integration tests for the ingestion pipeline against docker postgres.

These tests require the ``db`` service from docker-compose to be reachable
and skip cleanly when it is not. They use a reserved ``card_json_id`` range
(>= 9_000_000) that is deleted at fixture start and end, so the real card
corpus is never touched. Embeddings are mocked: no real OpenAI or local
model calls are made.
"""

import json
import os
from unittest import mock

import pytest
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session

from app import ingest
from app.config import DATABASE_URL as CONFIG_DATABASE_URL
from app.ingest import process_jsons
from app.models import Card

BASE_ID = 9_000_000
BATCH_SIZE = 100


def _database_url():
    """Return the first reachable database URL, or None.

    Candidates: the DATABASE_URL environment variable, the app's configured
    default (``db`` hostname, reachable inside the compose network), and the
    localhost mapping used when the db port is published on the host.
    """
    candidates = [
        os.getenv("DATABASE_URL"),
        CONFIG_DATABASE_URL,
        "postgresql+psycopg2://admin:admin@localhost:5432/rag_db",
    ]
    for url in candidates:
        if not url:
            continue
        try:
            engine = create_engine(url, connect_args={"connect_timeout": 3})
            with engine.connect() as conn:
                conn.execute(select(1))
            engine.dispose()
            return url
        except Exception:
            continue
    return None


@pytest.fixture(scope="module")
def engine():
    url = _database_url()
    if url is None:
        pytest.skip("docker postgres not reachable")
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_reserved_range(engine):
    """Delete the reserved card_json_id range before and after each test."""
    with Session(engine) as session:
        session.execute(delete(Card).where(Card.card_json_id >= BASE_ID))
        session.commit()
    yield
    with Session(engine) as session:
        session.execute(delete(Card).where(Card.card_json_id >= BASE_ID))
        session.commit()


def _write_cards(tmp_path, n, start=BASE_ID):
    """Write n JSON card files with ids start..start+n-1."""
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


def _insert_cards(engine, ids):
    """Insert bare Card rows directly (no embeddings)."""
    with Session(engine) as session:
        for i in ids:
            session.add(
                Card(card_json_id=i, name=f"Card {i}", content=f"Name: Card {i}")
            )
        session.commit()


def _count_reserved(engine):
    with Session(engine) as session:
        return session.scalar(
            select(func.count()).select_from(Card).where(Card.card_json_id >= BASE_ID)
        )


def _ids_present(engine):
    with Session(engine) as session:
        return set(
            session.scalars(
                select(Card.card_json_id).where(Card.card_json_id >= BASE_ID)
            )
        )


def _failing_upsert_wrapper(real_upsert):
    """Wrap upsert_card_with_vectors, raising mid-batch for the second
    batch's ids so the rollback undoes real work."""

    def _wrapper(card_json, openai_vec, local_vec, session):
        if BASE_ID + 149 <= card_json["id"] < BASE_ID + 200:
            raise RuntimeError("injected upsert failure")
        real_upsert(card_json, openai_vec, local_vec, session)

    return _wrapper


def test_batch_failure_rolls_back_and_continues(tmp_path, monkeypatch, engine):
    """Batch 2 fails -> rolled back; batches 1 and 3 are committed."""
    _write_cards(tmp_path, 300)
    _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(
        ingest,
        "upsert_card_with_vectors",
        _failing_upsert_wrapper(ingest.upsert_card_with_vectors),
    )

    with Session(engine) as session:
        count = process_jsons(str(tmp_path), session)

    assert count == 200
    present = _ids_present(engine)
    assert {BASE_ID + i for i in range(100)} <= present  # batch 1
    assert not ({BASE_ID + i for i in range(100, 200)} & present)  # batch 2
    assert {BASE_ID + i for i in range(200, 300)} <= present  # batch 3


def test_poisoned_transaction_rolls_back_batch(tmp_path, monkeypatch, engine):
    """A DB-level failure mid-batch aborts the transaction; the batch is
    rolled back and later batches still commit.

    The poisoned card has a non-integer id, which violates the integer
    column constraint on ``card_json_id``. The failing statement aborts the
    transaction (PG's 'current transaction is aborted' mechanism), so the
    batch cannot commit; the code rolls the batch back and continues with
    the next batch.
    """
    _write_cards(tmp_path, 300)
    # Poison a card in the second half of batch 2 (id BASE_ID + 149).
    poisoned_id = BASE_ID + 149
    (tmp_path / f"{poisoned_id}.json").write_text(
        json.dumps({"id": "not-an-integer", "name": "Poisoned card"}),
        encoding="utf-8",
    )
    _mock_batch_embeddings(monkeypatch)

    with Session(engine) as session:
        count = process_jsons(str(tmp_path), session)

    assert count == 200
    present = _ids_present(engine)
    assert {BASE_ID + i for i in range(100)} <= present  # batch 1
    assert not ({BASE_ID + i for i in range(100, 200)} & present)  # batch 2
    assert {BASE_ID + i for i in range(200, 300)} <= present  # batch 3


def test_resume_skips_present(tmp_path, monkeypatch, engine):
    """50 of 100 files already present -> only 50 are embedded."""
    _write_cards(tmp_path, 100)
    _insert_cards(engine, range(BASE_ID, BASE_ID + 50))
    embed = _mock_batch_embeddings(monkeypatch)

    with Session(engine) as session:
        count = process_jsons(str(tmp_path), session)

    assert count == 50
    assert _total_embedded(embed) == 50
    assert _count_reserved(engine) == 100


def test_resume_after_mid_run_failure(tmp_path, monkeypatch, engine):
    """A second run after a mid-run failure embeds only the missing files."""
    _write_cards(tmp_path, 300)
    real_upsert = ingest.upsert_card_with_vectors
    _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(
        ingest, "upsert_card_with_vectors", _failing_upsert_wrapper(real_upsert)
    )

    with Session(engine) as session:
        count1 = process_jsons(str(tmp_path), session)
    assert count1 == 200

    # Run 2: no failure injection; only the missing batch-2 files are embedded.
    embed2 = _mock_batch_embeddings(monkeypatch)
    monkeypatch.setattr(ingest, "upsert_card_with_vectors", real_upsert)

    with Session(engine) as session:
        count2 = process_jsons(str(tmp_path), session)

    assert count2 == 100
    assert _total_embedded(embed2) == 100
    assert _count_reserved(engine) == 300


def test_force_reembeds_everything(tmp_path, monkeypatch, engine):
    """--force re-embeds all files even when every id is already present."""
    _write_cards(tmp_path, 100)
    _insert_cards(engine, range(BASE_ID, BASE_ID + 100))
    embed = _mock_batch_embeddings(monkeypatch)

    with Session(engine) as session:
        count = process_jsons(str(tmp_path), session, force=True)

    assert count == 100
    assert _total_embedded(embed) == 100
    assert _count_reserved(engine) == 100


def test_fallback_commits_against_db(tmp_path, monkeypatch, engine):
    """Batch embedding failure falls back to per-card upserts against the DB."""
    _write_cards(tmp_path, 250)
    monkeypatch.setattr(
        ingest,
        "get_both_embeddings_batch",
        mock.Mock(side_effect=RuntimeError("batch embedding down")),
    )
    monkeypatch.setattr(
        ingest, "get_both_embeddings", mock.Mock(return_value=(None, None))
    )

    with Session(engine) as session:
        count = process_jsons(str(tmp_path), session)

    assert count == 250
    assert _count_reserved(engine) == 250
