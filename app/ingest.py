"""Ingestion pipeline: reads Yu-Gi-Oh! JSON cards, generates embeddings,
and stores them in the PostgreSQL database with pgvector support.
"""

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from app.config import DATABASE_URL, DEFAULT_JSON_DIR
from app.embeddings import get_both_embeddings, get_both_embeddings_batch
from app.models import Base, Card

logger = logging.getLogger(__name__)


def clean_int(value: object) -> int | None:
    """Safely cast a value to int; returns None on failure."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None


def build_card_content(card_json: dict) -> str:
    """Build a concatenated text representation of a card for embedding.

    Args:
        card_json: Raw card data from a JSON file.

    Returns:
        A pipe-delimited string of the card's key fields.
    """
    parts = [f"Name: {card_json.get('name')}"]

    attribute = card_json.get("englishAttribute")
    if attribute:
        parts.append(f"Attribute: {attribute}")

    atk = clean_int(card_json.get("atk"))
    defense = clean_int(card_json.get("def"))
    level = clean_int(card_json.get("level"))

    if level is not None:
        parts.append(f"Level: {level}")
    if atk is not None:
        parts.append(f"ATK: {atk}")
    if defense is not None:
        parts.append(f"DEF: {defense}")

    properties = card_json.get("properties")
    if properties is not None:
        parts.append(f"PROPERTIES: {properties}")

    full_content = " | ".join(parts)

    effect = card_json.get("effectText", "")
    if effect:
        full_content += f" | Description: {effect}"

    return full_content


BATCH_SIZE = 100  # Safe margin below OpenAI's 2048 input limit


class IngestStats(NamedTuple):
    """Outcome of an ingestion run.

    Attributes:
        ingested: Cards embedded and upserted in committed batches.
        skipped: Cards left untouched because their content was unchanged.
    """

    ingested: int
    skipped: int


def compute_content_hash(content: str) -> str:
    """Return the sha256 hex digest of a card's embedded content text.

    Hashes exactly the string produced by :func:`build_card_content`, so the
    digest tracks what actually gets embedded: any edit to a card's JSON that
    changes its content changes the hash, and the card is re-ingested.

    Args:
        content: The content string to hash.

    Returns:
        A 64-character lowercase hex digest.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def should_skip_card(
    card_json_id: int | None,
    content_hash: str,
    existing_hashes: Mapping[int, str | None],
) -> bool:
    """Decide whether a card can be skipped without embedding or upserting.

    A card is skipped only when a row already exists for its ``card_json_id``
    *and* the hash stored on that row matches the freshly computed one. A card
    with no row yet, a row whose stored hash is ``None`` (ingested before the
    column existed), or any change in content all mean "not skippable".

    Args:
        card_json_id: The card's ``id`` field, or None if absent.
        content_hash: Freshly computed hash of the card's current content.
        existing_hashes: Mapping of card_json_id to stored hash, loaded once
            up front by :func:`load_existing_hashes`.

    Returns:
        True if the card is already stored unchanged and can be skipped.
    """
    if card_json_id is None:
        return False
    stored = existing_hashes.get(card_json_id)
    return stored is not None and stored == content_hash


def load_existing_hashes(session: Session) -> dict[int, str | None]:
    """Load every stored (card_json_id, content_hash) pair in a single query.

    One round trip for the whole table keeps the resume check in memory; the
    ingest loop must never issue a SELECT per card.

    Args:
        session: An active SQLAlchemy session.

    Returns:
        Mapping of card_json_id to its stored content hash (None if unset).
    """
    rows = session.execute(select(Card.card_json_id, Card.content_hash)).all()
    return {card_json_id: content_hash for card_json_id, content_hash in rows}


def upsert_card_with_vectors(
    card_json: dict,
    openai_vector: list[float] | None,
    local_vector: list[float] | None,
    session: Session,
) -> None:
    """Upsert a single card with pre-computed embedding vectors. Does NOT commit.

    Args:
        card_json: Raw card data from a JSON file.
        openai_vector: Pre-computed OpenAI embedding (or None).
        local_vector: Pre-computed local embedding (or None).
        session: An active SQLAlchemy session.

    Raises:
        ValueError: If card_json has no 'id' field.
    """
    card_id = card_json.get("id")
    if card_id is None:
        raise ValueError("Card JSON missing 'id' field")

    content = build_card_content(card_json)

    values = {
        "card_json_id": card_id,
        "name": card_json.get("name"),
        "level": clean_int(card_json.get("level")),
        "atk": clean_int(card_json.get("atk")),
        "def_": clean_int(card_json.get("def")),
        "english_attribute": card_json.get("englishAttribute"),
        "properties": card_json.get("properties"),
        "content": content,
        "content_hash": compute_content_hash(content),
    }
    if openai_vector is not None:
        values["embedding"] = openai_vector
    if local_vector is not None:
        values["embedding_local"] = local_vector

    stmt = pg_insert(Card).values(**values).on_conflict_do_update(
        index_elements=["card_json_id"],
        set_=values,
    )
    session.execute(stmt)


def upsert_card(card_json: dict, session: Session) -> None:
    """Upsert a single card by card_json_id. Does NOT commit.

    Generates BOTH OpenAI and local embeddings, then delegates to
    :func:`upsert_card_with_vectors`.

    Args:
        card_json: Raw card data from a JSON file.
        session: An active SQLAlchemy session.

    Raises:
        ValueError: If card_json has no 'id' field.
    """
    content = build_card_content(card_json)
    openai_vec, local_vec = get_both_embeddings(content)
    upsert_card_with_vectors(card_json, openai_vec, local_vec, session)


def process_jsons(json_dir: str, session: Session) -> IngestStats:
    """Ingest all JSON card files from the given directory using batch embeddings.

    Cards are processed in batches of BATCH_SIZE to minimize OpenAI API calls.
    If a batch embedding call fails, falls back to individual processing.

    Each batch is committed on its own, so a crash leaves every completed
    batch durably in the database instead of discarding the whole run. A batch
    that fails to commit is rolled back alone, without losing the batches
    already committed before it.

    Cards whose stored content_hash still matches their JSON are skipped
    outright -- no embedding call, no upsert -- so a resumed run only pays for
    what actually changed. An edited JSON hashes differently and is
    re-ingested.

    Args:
        json_dir: Path to the directory containing ``.json`` card files.
        session: An active SQLAlchemy session.

    Returns:
        An :class:`IngestStats` with the ingested and skipped card counts.
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", json_dir)
        return IngestStats(ingested=0, skipped=0)

    # One query for the whole table; the per-card check below is in memory.
    existing_hashes = load_existing_hashes(session)

    ingested = 0
    skipped = 0
    total = len(json_files)

    # Process in batches
    for i in tqdm(range(0, total, BATCH_SIZE), desc="Ingesting cards", unit="batch"):
        batch_files = json_files[i:i + BATCH_SIZE]
        batch_cards: list[dict] = []
        batch_texts: list[str] = []

        # Read all JSONs in this batch, dropping the ones already up to date
        for json_file in batch_files:
            with open(json_file, "r", encoding="utf-8") as f:
                card_json = json.load(f)
            content = build_card_content(card_json)
            if should_skip_card(
                card_json.get("id"),
                compute_content_hash(content),
                existing_hashes,
            ):
                skipped += 1
                continue
            batch_cards.append(card_json)
            batch_texts.append(content)

        # Every card in this batch was unchanged: nothing to embed or commit.
        if not batch_cards:
            continue

        batch_count = 0

        # Single API call for the whole batch (both providers)
        try:
            openai_vecs, local_vecs = get_both_embeddings_batch(batch_texts)
        except Exception:
            logger.exception(
                "Batch embedding failed for batch %d, falling back to individual",
                i // BATCH_SIZE,
            )
            # Fallback: process one by one
            for card_json in batch_cards:
                try:
                    upsert_card(card_json, session)
                    batch_count += 1
                except Exception:
                    logger.exception("Failed: %s", card_json.get("name"))
        else:
            # Upsert each card with both embeddings
            for j, card_json in enumerate(batch_cards):
                try:
                    openai_vec = openai_vecs[j] if openai_vecs else None
                    local_vec = local_vecs[j] if local_vecs else None
                    upsert_card_with_vectors(
                        card_json, openai_vec, local_vec, session
                    )
                    batch_count += 1
                except Exception:
                    logger.exception("Failed: %s", card_json.get("name"))

        # Commit this batch on its own, so the work already done survives a
        # later failure. A batch that cannot be committed is rolled back by
        # itself; the batches committed before it are untouched.
        try:
            session.commit()
            ingested += batch_count
        except Exception:
            logger.exception(
                "Commit failed for batch %d, rolling back that batch",
                i // BATCH_SIZE,
            )
            session.rollback()

    return IngestStats(ingested=ingested, skipped=skipped)


def show_stats(session: Session) -> None:
    """Print current card count in the database."""
    total = session.scalar(select(func.count()).select_from(Card))
    print(f"Database contains {total} cards.")


def main() -> None:
    """Entry point for the ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description="Ingest Yu-Gi-Oh! card JSONs into the RAG database."
    )
    parser.add_argument(
        "--json-dir",
        default=DEFAULT_JSON_DIR,
        help=f"Directory containing .json card files (default: {DEFAULT_JSON_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logger.info("Starting card ingestion from %s ...", args.json_dir)

    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as session:
        stats = process_jsons(args.json_dir, session)

    logger.info(
        "Ingestion complete! %d cards ingested, %d skipped as unchanged.",
        stats.ingested,
        stats.skipped,
    )
    with session_factory() as session:
        show_stats(session)


if __name__ == "__main__":
    main()
