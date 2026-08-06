"""add_hnsw_indexes_on_embeddings

Revision ID: 8b2d0e4f6a3c
Revises: 7a1c9d3e5f2b
Create Date: 2026-07-13 00:00:00.000000

Adds HNSW (vector_cosine_ops) ANN indexes on both embedding columns.
Every prior search was a sequential scan; at 13.5K+ rows this matters.
Requires pgvector >= 0.8.0 for `hnsw.iterative_scan` support, used by
app/retrieval.py to keep recall correct when combined with structured
WHERE filters.

`embedding` was changed to a dimension-less `vector` type in
4d8b3f9c2e1a to allow switching providers, but pgvector requires a fixed
dimension to build an ANN index ("column does not have dimensions"). In
practice this column only ever stores 1536-dim OpenAI vectors (local
vectors go into embedding_local, which is already vector(1024)), so it's
restored to a fixed dimension here before indexing.
"""
from typing import Sequence, Union

from alembic import op

revision: str = '8b2d0e4f6a3c'
down_revision: Union[str, Sequence[str], None] = '7a1c9d3e5f2b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Fix embedding's dimension, then add HNSW indexes on both columns."""
    op.execute("ALTER TABLE cards ALTER COLUMN embedding TYPE vector(1536)")
    op.execute(
        "CREATE INDEX ix_cards_embedding_hnsw ON cards "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute(
        "CREATE INDEX ix_cards_embedding_local_hnsw ON cards "
        "USING hnsw (embedding_local vector_cosine_ops)"
    )


def downgrade() -> None:
    """Drop the HNSW indexes and restore embedding to a dimension-less vector."""
    op.execute("DROP INDEX IF EXISTS ix_cards_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_cards_embedding_local_hnsw")
    op.execute("ALTER TABLE cards ALTER COLUMN embedding TYPE vector")
