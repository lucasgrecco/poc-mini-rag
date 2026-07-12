"""add_card_json_id_to_cards

Revision ID: 3c7a2e8f1b4d
Revises: 16bd3be62cd9
Create Date: 2026-07-12 00:00:00.000000

Adds card_json_id column (unique, indexed) to cards table.
Existing rows must be truncated and re-ingested after this migration
since the old ingestion did not store the JSON id.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '3c7a2e8f1b4d'
down_revision: Union[str, Sequence[str], None] = '16bd3be62cd9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('cards', sa.Column('card_json_id', sa.Integer(), nullable=True))
    op.create_unique_constraint('uq_cards_card_json_id', 'cards', ['card_json_id'])
    op.create_index('ix_cards_card_json_id', 'cards', ['card_json_id'])

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_cards_card_json_id', table_name='cards')
    op.drop_constraint('uq_cards_card_json_id', 'cards', type_='unique')
    op.drop_column('cards', 'card_json_id')
