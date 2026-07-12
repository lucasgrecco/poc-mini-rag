"""SQLAlchemy ORM models for the poc-rag application.

Defines the ``cards`` table schema, which stores Yu-Gi-Oh! card data
alongside pre-computed embedding vectors for semantic search.
"""

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, ARRAY
from pgvector.sqlalchemy import Vector

from app.config import EMBEDDING_DIMENSIONS, EMBEDDING_LOCAL_DIMENSIONS


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


class Card(Base):
    """A Yu-Gi-Oh! card stored with its embedding vector.

    Attributes:
        id: Unique card identifier.
        level: Card level or rank.
        name: Card name.
        atk: Attack points.
        def_: Defense points.
        english_attribute: Card attribute (e.g. ``wind``, ``fire``, ``light``).
        content: Concatenated text representation used to generate the embedding.
        properties: List of card properties (e.g. ``["Dragon", "Normal"]``).
        embedding: 1536-dimensional float vector from OpenAI embeddings.
        embedding_local: 1024-dimensional float vector from local mxbai-embed-large-v1.
    """

    __tablename__ = "cards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    card_json_id: Mapped[int] = mapped_column(
        Integer, unique=True, nullable=False, index=True
    )
    level: Mapped[int | None] = mapped_column(Integer)
    name: Mapped[str | None] = mapped_column(String)
    atk: Mapped[int | None] = mapped_column(Integer)
    def_: Mapped[int | None] = mapped_column(Integer)
    english_attribute: Mapped[str | None] = mapped_column(String)
    content: Mapped[str | None] = mapped_column(Text)
    properties: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS)
    )
    embedding_local: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_LOCAL_DIMENSIONS)
    )
