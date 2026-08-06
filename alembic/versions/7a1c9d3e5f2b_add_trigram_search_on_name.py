"""add_trigram_search_on_name

Revision ID: 7a1c9d3e5f2b
Revises: 5e9c4d0f1a2b
Create Date: 2026-07-13 00:00:00.000000

Enables pg_trgm and adds a GiST trigram index on cards.name, needed for the
hybrid lexical+vector retrieval fusion (`name <-> :query` KNN ordering and
`name % :query` similarity filtering). GiST is required here rather than
GIN because GIN's trgm opclass does not support the `<->` KNN distance
operator used to rank lexical matches.
"""
from typing import Sequence, Union

from alembic import op

revision: str = '7a1c9d3e5f2b'
down_revision: Union[str, Sequence[str], None] = '5e9c4d0f1a2b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable pg_trgm and add a GiST trigram index on cards.name."""
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute(
        "CREATE INDEX ix_cards_name_trgm ON cards USING gist (name gist_trgm_ops)"
    )


def downgrade() -> None:
    """Drop the trigram index and extension."""
    op.execute("DROP INDEX IF EXISTS ix_cards_name_trgm")
    op.execute("DROP EXTENSION IF EXISTS pg_trgm")
