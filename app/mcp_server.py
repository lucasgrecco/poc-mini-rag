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
    SEARCH_LIMIT,
    CANDIDATE_POOL_SIZE,
)
from app.query_parser import prepare_search_query
from app.retrieval import hybrid_search_async, rerank

# MCP always uses local embeddings — no API key needed, pure retrieval
EMBEDDING_COLUMN = "embedding_local"

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
)


@mcp.tool()
async def search_cards(
    query: str,
    limit: int = SEARCH_LIMIT,
    min_atk: int | None = None,
    max_atk: int | None = None,
    min_def: int | None = None,
    max_def: int | None = None,
    level: int | None = None,
    attribute: str | None = None,
    card_type: str | None = None,
    ctx: Context = None,
) -> str:
    """Hybrid search over Yu-Gi-Oh! cards: vector + lexical fusion + reranking.

    Combines pgvector cosine similarity, pg_trgm lexical name matching (via
    Reciprocal Rank Fusion), and cross-encoder reranking. Returns raw card
    data — the host AI (Pi, Claude Code) does its own reasoning on the
    results.

    IMPORTANT: ATK constraints written in the query text ARE now detected
    and enforced automatically — no need to extract them yourself. Examples:
    "more than 2500 ATK", "2500+", ">= 2500", "between 1000 and 2000",
    "exactly 2500 ATK" (PT-BR equivalents like "mais de 2500 de ataque" and
    "entre 1000 e 2000" work too). The explicit min_atk/max_atk parameters
    are still accepted and merge (tighten) with the parsed bounds: effective
    min = max of the two, effective max = min of the two.

    DEF, level, attribute and card_type are still NOT parsed from the query
    text — pass them explicitly (e.g. card_type="Dragon", level=8).

    Example: user asks "dragões fortes com mais de 2500 de ataque" ->
        search_cards(query="dragões fortes com mais de 2500 de ataque", card_type="Dragon")
        (ATK >= 2500 is enforced automatically)
    Example: user asks for a specific card by name ->
        search_cards(query="Blue-Eyes White Dragon")  # lexical fusion ranks
        the exact name match to the top even if the embedding alone wouldn't.

    Args:
        query: Natural language / descriptive part of the query.
        limit: Number of cards to return (1-50, default 10).
        min_atk: Minimum ATK (inclusive).
        max_atk: Maximum ATK (inclusive).
        min_def: Minimum DEF (inclusive).
        max_def: Maximum DEF (inclusive).
        level: Exact card level/rank.
        attribute: Exact attribute (e.g. "light", "dark", "wind").
        card_type: A value that must appear in the card's properties/type
            list (e.g. "Dragon", "Spell", "Xyz", "Effect").

    Returns:
        Raw card data for the top matching cards.
    """
    limit = max(1, min(limit, 50))
    candidate_pool_size = max(CANDIDATE_POOL_SIZE, limit * 2)

    search_query, eff_min_atk, eff_max_atk = prepare_search_query(query, min_atk, max_atk)
    logger.info(
        "Parsed ATK bounds from query: eff_min_atk=%s eff_max_atk=%s (remainder=%r)",
        eff_min_atk,
        eff_max_atk,
        search_query,
    )

    await ctx.info(f"Searching cards for: {query}")

    # Generate query embedding (blocking call → run in thread)
    from app.embeddings import get_local_embedding  # Lazy import: avoids 4s sentence-transformers load at startup
    try:
        query_vector = await asyncio.to_thread(get_local_embedding, search_query)
    except Exception as e:
        return f"Failed to generate embedding: {e}"

    # Hybrid vector + lexical candidate retrieval, then rerank
    server_ctx = ctx.request_context.lifespan_context
    filters = {
        "min_atk": eff_min_atk,
        "max_atk": eff_max_atk,
        "min_def": min_def,
        "max_def": max_def,
        "level": level,
        "attribute": attribute,
        "card_type": card_type,
    }
    try:
        async with server_ctx.session_factory() as session:
            candidates = await hybrid_search_async(
                session,
                search_query,
                query_vector,
                EMBEDDING_COLUMN,
                filters=filters,
                candidate_pool_size=candidate_pool_size,
            )
    except Exception as e:
        return f"Database query failed: {e}"

    if not candidates:
        return "No cards found matching your query."

    rows = await asyncio.to_thread(rerank, search_query, candidates, limit)

    # Return raw results — host AI does its own reasoning
    lines = [f"Top {len(rows)} results for: {query}", ""]
    for i, row in enumerate(rows, 1):
        lines.append(f"{i}. {row['name']}")
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
    import asyncio
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
