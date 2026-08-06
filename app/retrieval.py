"""Shared retrieval core for search.py and mcp_server.py.

Combines structured filtering (exact ATK/DEF/level/attribute/type constraints),
hybrid vector + lexical fusion via Reciprocal Rank Fusion (RRF), and
cross-encoder reranking. Both the CLI (sync/psycopg2) and the MCP server
(async/asyncpg) build on the same SQL templates so retrieval behavior never
drifts between the two entry points.
"""

import logging

from sqlalchemy import text

from app.config import CANDIDATE_POOL_SIZE, RERANK_MODEL, RRF_K

logger = logging.getLogger(__name__)

_reranker = None


def build_filter_clause(
    min_atk: int | None = None,
    max_atk: int | None = None,
    min_def: int | None = None,
    max_def: int | None = None,
    level: int | None = None,
    attribute: str | None = None,
    card_type: str | None = None,
) -> tuple[str, dict]:
    """Build a parameterized SQL boolean expression from structured filters.

    Returns (sql_fragment, params). sql_fragment is "" when no filters are
    set — callers must treat that as "no restriction", not append a bare WHERE.
    All values are bound via SQLAlchemy params; nothing here is ever
    interpolated raw into the query string.
    """
    clauses = []
    params: dict = {}

    if min_atk is not None:
        clauses.append("atk >= :min_atk")
        params["min_atk"] = min_atk
    if max_atk is not None:
        clauses.append("atk <= :max_atk")
        params["max_atk"] = max_atk
    if min_def is not None:
        clauses.append("def_ >= :min_def")
        params["min_def"] = min_def
    if max_def is not None:
        clauses.append("def_ <= :max_def")
        params["max_def"] = max_def
    if level is not None:
        clauses.append("level = :level")
        params["level"] = level
    if attribute is not None:
        clauses.append("english_attribute = :attribute")
        params["attribute"] = attribute
    if card_type is not None:
        # Containment (`@>`) rather than `:card_type = ANY(properties)` so the
        # GIN index on `properties` can actually be used.
        clauses.append("properties @> ARRAY[:card_type]::varchar[]")
        params["card_type"] = card_type

    return " AND ".join(clauses), params


def _hybrid_search_sql(embedding_column: str, filter_sql: str):
    """Build the hybrid RRF candidate query for the given embedding column.

    vec_top and lex_top each do a plain `ORDER BY ... LIMIT` (not a window
    function over the whole table), so pgvector's HNSW index and the pg_trgm
    GiST index can actually accelerate them. `scope_clause` restricts both to
    the structurally-filtered id set; `hnsw.iterative_scan` (set by the caller
    for this transaction) keeps the HNSW walk from under-returning once that
    filter is applied. Card fields are re-selected via a plain id join, not
    from vec_top itself, since vec_top/lex_top only carry ids + ranks.
    """
    scope_clause = filter_sql if filter_sql else "TRUE"

    sql = f"""
        WITH vec_top AS (
            SELECT id, ROW_NUMBER() OVER (
                ORDER BY {embedding_column} <=> CAST(:vector AS vector)
            ) AS vec_rank
            FROM cards
            WHERE {scope_clause}
            ORDER BY {embedding_column} <=> CAST(:vector AS vector)
            LIMIT :candidate_pool_size
        ),
        lex_top AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY name <-> :query_text) AS lex_rank
            FROM cards
            WHERE ({scope_clause}) AND name % :query_text
            ORDER BY name <-> :query_text
            LIMIT :candidate_pool_size
        )
        SELECT c.id, c.name, c.content, c.level, c.atk, c.def_,
               c.english_attribute, c.properties,
               (1.0 / (:rrf_k + COALESCE(v.vec_rank, 1000000))
                + 1.0 / (:rrf_k + COALESCE(l.lex_rank, 1000000))) AS rrf_score
        FROM cards c
        JOIN vec_top v ON v.id = c.id
        LEFT JOIN lex_top l ON l.id = c.id
        ORDER BY rrf_score DESC
        LIMIT :candidate_pool_size
    """
    return text(sql)


def _build_params(
    query_text: str,
    query_vector: list[float],
    candidate_pool_size: int,
    rrf_k: int,
    filter_params: dict,
) -> dict:
    """Build the bind-parameter dict for ``_hybrid_search_sql``.

    The ``query_vector`` is a list of floats, but the SQL binds it via
    ``CAST(:vector AS vector)`` which expects the pgvector literal string
    format ``[v1,v2,...]``. This function serializes the vector into that
    string form so pgvector can parse it on the server side.

    The ``filter_params`` produced by ``build_filter_clause`` are merged into
    the returned dict via ``**filter_params`` so the filter bind parameters are
    available alongside the vector, query text, pool size, and RRF constant.

    Args:
        query_text: The natural-language query string for lexical search.
        query_vector: Embedding vector (list of floats) to be serialized into
            the pgvector ``[v1,v2,...]`` string format for ``CAST(:vector AS
            vector)``.
        candidate_pool_size: Row limit for the candidate pool pulled by the
            RRF query.
        rrf_k: RRF constant used in the reciprocal-rank-fusion scoring.
        filter_params: Bind parameters produced by ``build_filter_clause``;
            merged into the result via ``**filter_params``.

    Returns:
        A dict of bind parameters (vector string, query_text,
        candidate_pool_size, rrf_k, and all filter_params entries) suitable
        for ``_hybrid_search_sql``.
    """
    vector_str = "[" + ",".join(str(v) for v in query_vector) + "]"
    return {
        "vector": vector_str,
        "query_text": query_text,
        "candidate_pool_size": candidate_pool_size,
        "rrf_k": rrf_k,
        **filter_params,
    }


def hybrid_search_sync(
    conn,
    query_text: str,
    query_vector: list[float],
    embedding_column: str,
    filters: dict | None = None,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Run the hybrid RRF candidate query on a sync SQLAlchemy connection."""
    filter_sql, filter_params = build_filter_clause(**(filters or {}))
    sql = _hybrid_search_sql(embedding_column, filter_sql)
    params = _build_params(query_text, query_vector, candidate_pool_size, rrf_k, filter_params)

    conn.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
    # At this table size the planner underestimates HNSW's cost vs a plain
    # Seq Scan + Sort (measured ~95ms seq scan vs ~57ms via HNSW on 13.6K
    # rows) even after ANALYZE. Forcing it off is scoped to this transaction.
    conn.execute(text("SET LOCAL enable_seqscan = off"))
    rows = conn.execute(sql, params).mappings().all()
    return [dict(r) for r in rows]


async def hybrid_search_async(
    session,
    query_text: str,
    query_vector: list[float],
    embedding_column: str,
    filters: dict | None = None,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Run the hybrid RRF candidate query on an async SQLAlchemy session."""
    filter_sql, filter_params = build_filter_clause(**(filters or {}))
    sql = _hybrid_search_sql(embedding_column, filter_sql)
    params = _build_params(query_text, query_vector, candidate_pool_size, rrf_k, filter_params)

    await session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
    await session.execute(text("SET LOCAL enable_seqscan = off"))
    result = await session.execute(sql, params)
    rows = result.mappings().all()
    return [dict(r) for r in rows]


def _get_reranker():
    """Return a lazily-initialized CrossEncoder reranker singleton.

    Mirrors the CUDA-OOM-falls-back-to-CPU pattern used for the local
    embedding model in app/embeddings.py.
    """
    global _reranker
    if _reranker is None:
        import torch
        from sentence_transformers import CrossEncoder

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading reranker model: %s (device=%s)", RERANK_MODEL, device)
        try:
            _reranker = CrossEncoder(RERANK_MODEL, device=device)
            _reranker.predict([("test", "test")])
        except RuntimeError:
            logger.warning("CUDA OOM loading reranker, falling back to CPU")
            _reranker = CrossEncoder(RERANK_MODEL, device="cpu")
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Re-score candidates with a cross-encoder and return the top_k, best first.

    Args:
        query: The original natural language query.
        candidates: Rows from hybrid_search_sync/async (must have a "content" key).
        top_k: Number of results to keep after reranking.

    Returns:
        candidates sorted by rerank score (descending), truncated to top_k.
    """
    if not candidates:
        return []

    model = _get_reranker()
    pairs = [(query, c["content"] or "") for c in candidates]
    scores = model.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return candidates[:top_k]
