"""make_embedding_column_variable_dim

Revision ID: 4d8b3f9c2e1a
Revises: 3c7a2e8f1b4d
Create Date: 2026-07-12 00:00:00.000000

Changes the embedding column from fixed Vector(1536) to variable-dim Vector
to support switching between embedding providers (OpenAI 1536 vs local 1024).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '4d8b3f9c2e1a'
down_revision: Union[str, Sequence[str], None] = '3c7a2e8f1b4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Change embedding column to variable-dimension vector."""
    op.execute("ALTER TABLE cards ALTER COLUMN embedding TYPE vector")

def downgrade() -> None:
    """Restore embedding column to fixed 1536-dim vector."""
    op.execute("ALTER TABLE cards ALTER COLUMN embedding TYPE vector(1536)")
