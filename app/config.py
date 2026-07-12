"""Centralized configuration for the poc-rag application.

All environment-specific settings are loaded from environment variables
with sensible defaults for local Docker development.
"""

import os

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://admin:admin@db/rag_db",
)

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.4-mini"
# CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_DIMENSIONS = 1536

# Search defaults
SEARCH_LIMIT = 10

# Ingestion defaults
DEFAULT_JSON_DIR = os.getenv("CARD_JSON_DIR", "jsons")
