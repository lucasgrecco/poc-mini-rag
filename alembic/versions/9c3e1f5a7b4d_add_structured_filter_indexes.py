"""add_structured_filter_indexes

Revision ID: 9c3e1f5a7b4d
Revises: 8b2d0e4f6a3c
Create Date: 2026-07-13 00:00:00.000000

B-tree indexes for the structured retrieval filters (atk/def_/level/attribute)
and a GIN index on properties for `@>` containment lookups (card_type filter
in app/retrieval.py:build_filter_clause). Keeps the structured filter cheap
independent of the vector/lexical ranking path.
"""
from typing import Sequence, Union

from alembic import op

revision: str = '9c3e1f5a7b4d'
down_revision: Union[str, Sequence[str], None] = '8b2d0e4f6a3c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add filter indexes for atk/def_/level/english_attribute/properties."""
    op.create_index('ix_cards_atk', 'cards', ['atk'])
    op.create_index('ix_cards_def_', 'cards', ['def_'])
    op.create_index('ix_cards_level', 'cards', ['level'])
    op.create_index('ix_cards_english_attribute', 'cards', ['english_attribute'])
    op.execute("CREATE INDEX ix_cards_properties_gin ON cards USING gin (properties)")


def downgrade() -> None:
    """Drop the filter indexes."""
    op.execute("DROP INDEX IF EXISTS ix_cards_properties_gin")
    op.drop_index('ix_cards_english_attribute', table_name='cards')
    op.drop_index('ix_cards_level', table_name='cards')
    op.drop_index('ix_cards_def_', table_name='cards')
    op.drop_index('ix_cards_atk', table_name='cards')
