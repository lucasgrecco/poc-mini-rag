"""SQLAlchemy ORM models for the poc-rag application.

Defines the ``cards`` table schema, which stores Yu-Gi-Oh! card data
alongside pre-computed embedding vectors for semantic search.
"""

from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import Integer, String, Text, ARRAY
from pgvector.sqlalchemy import Vector


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
    """

    __tablename__ = "cards"

    id: int = mapped_column(Integer, primary_key=True)
    level: int | None = mapped_column(Integer)
    name: str | None = mapped_column(String)
    atk: int | None = mapped_column(Integer)
    def_: int | None = mapped_column(Integer)
    english_attribute: str | None = mapped_column(String)
    content: str | None = mapped_column(Text)
    properties: list[str] | None = mapped_column(ARRAY(String))
    embedding: list[float] | None = mapped_column(Vector(1536))
