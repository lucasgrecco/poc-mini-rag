"""add_embedding_local_to_cards

Revision ID: 5e9c4d0f1a2b
Revises: 4d8b3f9c2e1a
Create Date: 2026-07-12 00:00:00.000000

Adds embedding_local column (Vector(1024)) for local mxbai-embed-large-v1 embeddings.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import pgvector

revision: str = '5e9c4d0f1a2b'
down_revision: Union[str, Sequence[str], None] = '4d8b3f9c2e1a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Add embedding_local column for local model embeddings."""
    op.add_column(
        'cards',
        sa.Column('embedding_local', pgvector.sqlalchemy.Vector(1024), nullable=True),
    )

def downgrade() -> None:
    """Remove embedding_local column."""
    op.drop_column('cards', 'embedding_local')
