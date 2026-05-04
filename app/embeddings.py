"""Shared embedding utilities.

Provides a single, reusable interface for generating text embeddings
via the OpenAI API. Used by both the ingestion and search pipelines.
"""

import logging
from openai import OpenAI
from app.config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Return a lazily-initialized OpenAI client singleton."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Create a .env file with your API key."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def get_embedding(text: str) -> list[float]:
    """Generate a 1536-dimensional embedding vector for the given text.

    Args:
        text: The input text to embed.

    Returns:
        A list of 1536 floats representing the embedding.

    Raises:
        RuntimeError: If OPENAI_API_KEY is not set.
        openai.APIError: If the OpenAI API request fails.
    """
    logger.debug("Generating embedding for text (%d chars)", len(text))
    response = _get_client().embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding
