"""Ingestion pipeline: reads Yu-Gi-Oh! JSON cards, generates embeddings,
and stores them in the PostgreSQL database with pgvector support.
"""

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from app.config import DATABASE_URL, DEFAULT_JSON_DIR
from app.embeddings import get_embedding, get_both_embeddings, get_both_embeddings_batch
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


def upsert_card_with_vector(
    card_json: dict, vector: list[float], session: Session
) -> None:
    """Upsert a single card with a pre-computed embedding vector. Does NOT commit.

    Uses the auto-detected provider to decide which column to populate.

    Args:
        card_json: Raw card data from a JSON file.
        vector: Pre-computed embedding vector.
        session: An active SQLAlchemy session.

    Raises:
        ValueError: If card_json has no 'id' field.
    """
    from app.config import EMBEDDING_PROVIDER

    if EMBEDDING_PROVIDER == "openai":
        upsert_card_with_vectors(card_json, vector, None, session)
    else:
        upsert_card_with_vectors(card_json, None, vector, session)


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


def process_jsons(json_dir: str, session: Session, force: bool = False) -> int:
    """Ingest JSON card files from the given directory using batch embeddings.

    Cards are processed in batches of BATCH_SIZE to minimize OpenAI API calls.
    Each batch is committed as a unit: any failure (embedding, upsert, or
    commit) rolls the batch back and processing continues with the next batch.
    If a batch embedding call fails, falls back to individual processing
    (which also commits once per batch).

    Unless ``force`` is set, files whose ``id`` already exists in the cards
    table are skipped — the cards table is the resume record.

    Args:
        json_dir: Path to the directory containing ``.json`` card files.
        session: An active SQLAlchemy session.
        force: Re-embed and re-upsert all files, bypassing the resume skip.

    Returns:
        The number of cards newly ingested this run.
    """
    json_path = Path(json_dir)
    json_files = sorted(json_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", json_dir)
        return 0

    # Phase 1: resume filter — skip files already present in the cards table.
    skipped = 0
    if force:
        pending_files = json_files
    else:
        existing_ids = set(session.scalars(select(Card.card_json_id)))
        pending_files = []
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                card_json = json.load(f)
            if card_json.get("id") in existing_ids:
                skipped += 1
            else:
                pending_files.append(json_file)

    if not pending_files:
        logger.info(
            "Ingestion summary: %d files skipped (already in cards), %d ingested this run",
            skipped,
            0,
        )
        return 0

    # Phase 2: batch loop — one commit per batch, rollback on any failure.
    count = 0
    total = len(pending_files)

    with tqdm(total=total, desc="Ingesting cards", unit="file") as progress:
        for i in range(0, total, BATCH_SIZE):
            batch_files = pending_files[i:i + BATCH_SIZE]
            batch_cards: list[dict] = []
            batch_texts: list[str] = []

            # Read all JSONs in this batch
            for json_file in batch_files:
                with open(json_file, "r", encoding="utf-8") as f:
                    card_json = json.load(f)
                content = build_card_content(card_json)
                batch_cards.append(card_json)
                batch_texts.append(content)

            try:
                # Single API call for the whole batch (both providers)
                try:
                    openai_vecs, local_vecs = get_both_embeddings_batch(batch_texts)
                except Exception:
                    logger.exception(
                        "Batch embedding failed for batch %d, falling back to individual",
                        i // BATCH_SIZE,
                    )
                    # Fallback: process one by one
                    batch_count = 0
                    for card_json in batch_cards:
                        try:
                            upsert_card(card_json, session)
                            batch_count += 1
                        except Exception:
                            logger.exception("Failed: %s", card_json.get("name"))
                    session.commit()
                    count += batch_count
                else:
                    # Upsert each card with both embeddings
                    for j, card_json in enumerate(batch_cards):
                        openai_vec = openai_vecs[j] if openai_vecs else None
                        local_vec = local_vecs[j] if local_vecs else None
                        upsert_card_with_vectors(
                            card_json, openai_vec, local_vec, session
                        )
                    session.commit()
                    count += len(batch_cards)
            except Exception:
                session.rollback()
                logger.exception("Batch %d failed, rolling back", i // BATCH_SIZE)
            progress.update(len(batch_files))

    # Phase 3: summary log
    logger.info(
        "Ingestion summary: %d files skipped (already in cards), %d ingested this run",
        skipped,
        count,
    )
    return count


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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed and re-upsert all files, bypassing the resume skip",
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
        count = process_jsons(args.json_dir, session, force=args.force)
        show_stats(session)

    logger.info("Ingestion complete! %d cards ingested.", count)


if __name__ == "__main__":
    main()
