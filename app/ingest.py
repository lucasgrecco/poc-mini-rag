"""Ingestion pipeline: reads Yu-Gi-Oh! JSON cards, generates embeddings,
and stores them in the PostgreSQL database with pgvector support.
"""

import argparse
import json
import logging
from pathlib import Path

from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

from app.config import DATABASE_URL, DEFAULT_JSON_DIR
from app.embeddings import get_embedding
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


def process_jsons(json_dir: str, session: Session) -> int:
    """Ingest all JSON card files from the given directory.

    Args:
        json_dir: Path to the directory containing ``.json`` card files.
        session: An active SQLAlchemy session.

    Returns:
        The number of cards ingested.
    """
    json_path = Path(json_dir)
    json_files = list(json_path.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", json_dir)
        return 0

    count = 0

    for json_file in tqdm(json_files, desc="Ingesting cards", unit="card"):
        with open(json_file, "r", encoding="utf-8") as f:
            card_json = json.load(f)

        content = build_card_content(card_json)

        name = card_json.get("name")
        logger.debug("Processing: %s", name)

        try:
            vector = get_embedding(content)
        except Exception:
            logger.exception("Failed to generate embedding for card: %s", name)
            continue

        new_card = Card(
            name=name,
            level=clean_int(card_json.get("level")),
            atk=clean_int(card_json.get("atk")),
            def_=clean_int(card_json.get("def")),
            english_attribute=card_json.get("englishAttribute"),
            properties=card_json.get("properties"),
            content=content,
            embedding=vector,
        )

        session.add(new_card)
        count += 1

    session.commit()
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
        count = process_jsons(args.json_dir, session)

    logger.info("Ingestion complete! %d cards ingested.", count)
    show_stats(session_factory())


if __name__ == "__main__":
    main()
