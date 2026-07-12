"""MCP server exposing Yu-Gi-Oh! card RAG search as tools.

Provides semantic card search and card lookup via the Model Context Protocol.
Compatible with Pi, Claude Code, and any MCP-compatible host.

Usage:
    uv run ygo-search          # stdio transport (default)
    python -m app.mcp_server   # alternative entry point
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP, Context
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    SEARCH_LIMIT,
)
from app.embeddings import get_embedding
from app.search import get_model_answer

logger = logging.getLogger(__name__)

# Convert sync DB URL to async (psycopg2 → asyncpg)
ASYNC_DATABASE_URL = (
    DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    .replace("postgresql+psycopg2://", "postgresql+asyncpg://", 1)
)


@dataclass
class ServerContext:
    """Application context shared across tool calls."""
    session_factory: async_sessionmaker


@asynccontextmanager
async def lifespan(server: FastMCP):
    """Manage database connection pool lifecycle.

    Creates an async SQLAlchemy engine with connection pooling on startup
    and disposes it cleanly on shutdown.
    """
    engine = create_async_engine(
        ASYNC_DATABASE_URL,
        pool_size=3,
        max_overflow=0,
        pool_pre_ping=True,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    logger.info("Database connection pool created")
    try:
        yield ServerContext(session_factory=session_factory)
    finally:
        await engine.dispose()
        logger.info("Database connection pool closed")


mcp = FastMCP(
    "ygo-search",
    lifespan=lifespan,
    json_response=True,
)


@mcp.tool()
async def search_cards(
    query: str,
    limit: int = SEARCH_LIMIT,
    verbose: bool = False,
    ctx: Context = None,
) -> str:
    """Semantic search over Yu-Gi-Oh! cards using RAG (Retrieval-Augmented Generation).

    Retrieves the most semantically similar cards to your query using
    pgvector cosine similarity on OpenAI embeddings, then generates an
    AI-powered answer grounded in the retrieved card data.

    Use this when you need to find cards matching a natural language
    description, e.g. "dragons with more than 2500 ATK" or
    "spell cards that destroy monsters".

    Args:
        query: Natural language query describing what cards you want.
        limit: Number of cards to retrieve (1-50, default 10).
        verbose: If true, include full card details in the response.

    Returns:
        AI-generated answer with the top matching cards listed.
    """
    limit = max(1, min(limit, 50))

    await ctx.info(f"Searching cards for: {query}")

    # Generate query embedding (blocking call → run in thread)
    try:
        query_vector = await asyncio.to_thread(get_embedding, query)
    except Exception as e:
        return f"Failed to generate embedding: {e}"

    # Search via pgvector cosine distance
    server_ctx = ctx.request_context.lifespan_context
    try:
        async with server_ctx.session_factory() as session:
            # Convert embedding list to pgvector-compatible string for asyncpg
            vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
            sql = text("""
                SELECT name, content
                FROM cards
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :limit
            """)
            result = await session.execute(
                sql, {"query_vec": vector_str, "limit": limit}
            )
            rows = result.mappings().all()
            rows = [dict(r) for r in rows]
    except Exception as e:
        return f"Database query failed: {e}"

    if not rows:
        return "No cards found matching your query."

    # Format results for the LLM
    results_db = [
        {"name": row["name"], "content": row["content"]}
        for row in rows
    ]

    # Generate AI answer (blocking call → run in thread)
    await ctx.info("Generating AI answer...")
    try:
        answer = await asyncio.to_thread(get_model_answer, query, results_db)
    except Exception as e:
        answer = f"(AI answer unavailable: {e})"

    # Build response
    lines = [answer, "", "---", f"Top {len(rows)} Results:", ""]
    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {row['name']}")
        if verbose:
            lines.append(f"   {row['content']}")
            lines.append("")

    return "\n".join(lines)


@mcp.tool()
async def get_card(
    card_id: int,
    ctx: Context = None,
) -> str:
    """Retrieve a specific Yu-Gi-Oh! card by its ID.

    Returns full card details including name, type, ATK, DEF, level,
    attribute, effect text, and properties.

    Args:
        card_id: The card's JSON ID (e.g., 4007 for Blue-Eyes White Dragon,
                 10000 for Ten Thousand Dragon).

    Returns:
        Full card details formatted as text, or an error if not found.
    """
    server_ctx = ctx.request_context.lifespan_context
    try:
        async with server_ctx.session_factory() as session:
            sql = text("""
                SELECT name, content, level, atk, def_,
                       english_attribute, properties
                FROM cards
                WHERE card_json_id = :card_id
            """)
            result = await session.execute(sql, {"card_id": card_id})
            row = result.mappings().first()
            row = dict(row) if row else None
    except Exception as e:
        return f"Database query failed: {e}"

    if not row:
        return f"Card with ID {card_id} not found."

    return (
        f"Name: {row['name']}\n"
        f"Content: {row['content']}\n"
        f"Level: {row['level']} | ATK: {row['atk']} | DEF: {row['def_']}\n"
        f"Attribute: {row['english_attribute']}\n"
        f"Properties: {row['properties']}"
    )


def main():
    """Entry point for the MCP server (stdio transport)."""
    mcp.run()


if __name__ == "__main__":
    main()
