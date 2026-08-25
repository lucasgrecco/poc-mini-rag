"""add_content_hash_to_cards

Revision ID: a4f7b2c9d0e1
Revises: 9c3e1f5a7b4d
Create Date: 2026-08-25 00:00:00.000000

Adds content_hash column (sha256 hex digest of the embedded content text,
as built by app/ingest.py:build_card_content). The ingestion pipeline reads
these hashes up front and skips cards whose content is unchanged, avoiding a
repeat embedding call on resume. Nullable on purpose: rows ingested before
this column existed carry NULL and are treated as stale, so they are
re-ingested once and then acquire a hash.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'a4f7b2c9d0e1'
down_revision: Union[str, Sequence[str], None] = '9c3e1f5a7b4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add nullable content_hash column for incremental re-ingestion."""
    op.add_column(
        'cards',
        sa.Column('content_hash', sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Remove content_hash column."""
    op.drop_column('cards', 'content_hash')
