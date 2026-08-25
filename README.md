# 🃏 yugioh-cards-rag — Yu-Gi-Oh! Semantic Search (RAG Proof of Concept)

**Retrieval-Augmented Generation (RAG)** applied to Yu-Gi-Oh! card data.
Ask natural language questions and get AI-powered answers grounded in real card data.

```
Query: "dragons with more than 2500 ATK"
  → Structured filter (card_type="Dragon", min_atk=2500) narrows the candidate set
  → Hybrid vector + lexical retrieval (RRF) ranks candidates within that set
  → Cross-encoder reranks the pool
  → LLM generates a summarized answer from those cards
```

---

## Architecture

```
┌───────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  jsons/*.json │────▶│  app/ingest.py       │────▶│  PostgreSQL  │
│  (card data)  │     │  + dual embeddings   │     │  + pgvector  │
└───────────────┘     └──────────────────────┘     └──────┬───────┘
                      ┌─────────────┐                     │
                      │   Watcher   │ (auto-reindex)      │
                      └─────────────┘                     │
                                                          ▼
                                            ┌───────────────────────┐
                                            │  app/retrieval.py     │
                                            │  filter + RRF fusion  │
                                            │  + cross-encoder      │
                                            └───────────┬───────────┘
                     ┌────────────────────┐             │
  User query ───────▶│  app/search.py     │◀────────────┤
                     │  + gpt-5.4-mini    │             │
                     │  (OpenAI embed.)   │             │
                     └────────────────────┘             │
                     ┌────────────────────┐             │
  MCP host ─────────▶│  app/mcp_server.py │◀────────────┘
  (Pi, Claude Code)  │  (local embed.)    │
                     │  hybrid retrieval  │
                     └────────────────────┘
```

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Database | PostgreSQL + pgvector |
| OpenAI Embeddings | `text-embedding-3-small` (1536 dims) |
| Local Embeddings | `mxbai-embed-large-v1` (1024 dims, GPU when present, else CPU) |
| LLM | OpenAI `gpt-5.4-mini` |
| Reranker | `BAAI/bge-reranker-v2-m3` (cross-encoder, GPU when present, else CPU) |
| MCP Server | FastMCP (stdio transport, hybrid retrieval) |
| Observability | LangSmith tracing |
| Infrastructure | Docker Compose (GPU is opt-in, via overlay file) |

---

## Dual Embedding Provider

The system stores **both** OpenAI and local embeddings in the same row:

| Column | Provider | Dimensions | Used by |
|---|---|---|---|
| `embedding` | OpenAI `text-embedding-3-small` | 1536 | CLI search (`app.search`) |
| `embedding_local` | Local `mxbai-embed-large-v1` | 1024 | MCP server (`app.mcp_server`) |

- **CLI search**: auto-detects — uses OpenAI when `OPENAI_API_KEY` is set, else local
- **MCP server**: always uses local embeddings (no API key needed, pure retrieval)
- **Ingestion**: always generates **both** embeddings in a single pass
- **Switching providers**: instant — no re-ingestion needed

The local model uses the GPU when torch sees one and falls back to CPU otherwise —
slower, never fatal. Same selection as the reranker.

**GPU is opt-in**, via `GPU=` and a compose overlay. The base `docker-compose.yml`
reserves no device, so it comes up on any host:

```bash
make run              # CPU — any machine
make run GPU=nvidia   # NVIDIA (requires nvidia-container-toolkit on the host)
make run GPU=rocm     # AMD via ROCm
make gpu-check        # what torch sees from inside the container
```

Keeping the NVIDIA reservation in the base file made `up` fail outright on any
machine without the toolkit — `could not select device driver "nvidia" with
capabilities: [[gpu]]`, raised before a single container starts. That is why it
lives in `docker-compose.nvidia.yml`.

### AMD (ROCm)

`GPU=rocm` passes `/dev/kfd` and `/dev/dri` into the container and reinstalls torch
from the ROCm index on top of `uv sync`. Three things make this smaller than it
looks:

- **`device="cuda"` is correct on AMD.** ROCm exposes HIP through torch's CUDA API,
  so `torch.cuda.is_available()` returns `True` on a Radeon. No branch in the
  application code — `make gpu-check` should print `cuda True` with `hip` populated
  and `cuda-build None`.
- **No `HSA_OVERRIDE_GFX_VERSION`.** The RX 7900 XTX is Navi 31 / `gfx1100`, on
  ROCm's official support list. That override exists for cards outside it; setting
  it when it isn't needed masks real errors.
- **No ROCm install on the host.** The PyTorch ROCm wheels bundle the ROCm runtime;
  the host needs only the `amdgpu` kernel driver.

The wheel index is pinned in the Makefile as `ROCM_INDEX` and must match the ROCm
generation you are on — check <https://pytorch.org/get-started/locally/> and
override if needed:

```bash
make run GPU=rocm ROCM_INDEX=https://download.pytorch.org/whl/rocm6.4
```

**Two cards are one card, as written.** `sentence-transformers` loads the model onto
a single device, so a second GPU sits idle. Pin which one with `HIP_VISIBLE_DEVICES`
in `docker-compose.rocm.yml`. Using both would mean
`SentenceTransformer.start_multi_process_pool()` in the ingest path — worth it only
if batch ingestion becomes the bottleneck.

---

## Retrieval Pipeline

`app/retrieval.py` is the shared retrieval core used by both `app.search` and
`app.mcp_server` — neither writes raw SQL directly anymore.

**Structured filters** — semantic embeddings capture meaning, not exact
numeric/categorical constraints. Pass those separately instead of hoping the
embedding encodes them:

| Filter | Matches |
|---|---|
| `min_atk` / `max_atk` | `atk >= / <=` |
| `min_def` / `max_def` | `def_ >= / <=` |
| `level` | exact `level` |
| `attribute` | exact `english_attribute` |
| `card_type` | containment in `properties` (e.g. `"Dragon"`, `"Xyz"`) |

Example: `card_type="Dragon", min_atk=2500` — only Dragon-type cards with
ATK ≥ 2500 enter the ranking at all, instead of relying on the embedding to
infer "dragon" and "strong" from a free-text query.

> **ATK auto-extraction:** ATK expressions written inside the query text are
> auto-extracted and enforced as the same structured `min_atk`/`max_atk`
> filters — in EN and PT-BR. Thresholds ("more than 4000 ATK", "at least
> 2500", "mais de 4000 de ataque", "pelo menos 2500"), less-than ("less
> than 2500", "menos de 2500"), equality ("exactly 2500 ATK", "2500 ATK"),
> ranges ("between 1000 and 2000", "entre 1000 e 2000", "1000-2000"),
> operators (">= 2500", "<= 2500") and postfix forms ("2500+", "2500 or
> less") are all recognized. Parsed bounds merge with explicit
> `--min-atk`/`--max-atk` (effective min = the larger, effective max = the
> smaller — AND semantics). Only ATK is auto-parsed; DEF, level, attribute
> and card type remain explicit.

**Hybrid vector + lexical fusion** — candidates are ranked by combining two
independent signals via Reciprocal Rank Fusion (RRF):
- pgvector cosine distance on `embedding` / `embedding_local`
- pg_trgm lexical similarity of the query against `name`

This means searching for an exact card name (e.g. `"Blue-Eyes White Dragon"`)
reliably surfaces that card even when the embedding alone wouldn't rank it
first.

**Reranking** — the fused candidate pool is re-scored by a local cross-encoder
(`BAAI/bge-reranker-v2-m3`, multilingual — queries arrive in any language,
card text is in English) before truncating to the requested `limit`. Same
lazy-load + GPU→CPU fallback pattern as the local embedding model.

**Indexes** (see `alembic/versions/`): HNSW (`vector_cosine_ops`) on
`embedding` and `embedding_local`; GiST trigram on `name`; GIN on `properties`;
B-tree on `atk`/`def_`/`level`/`english_attribute`.

> Requires pgvector **≥ 0.8.0** for `hnsw.iterative_scan` (keeps HNSW recall
> correct when combined with a structured filter). `docker-compose.yml` uses
> `pgvector/pgvector:pg15`, which ships a current pgvector version.

---

## MCP Server

The MCP server exposes two tools as a **retrieval-only** interface — no LLM
call happens server-side. It returns raw card data (already filtered, fused,
and reranked per [Retrieval Pipeline](#retrieval-pipeline)), and the host AI
(Pi, Claude Code) does its own reasoning over the results.

| Tool | Description |
|---|---|
| `search_cards` | Hybrid search (vector + lexical + rerank) with optional structured filters |
| `get_card` | Retrieve a specific card by its JSON ID |

`search_cards` takes `query`, `limit`, and the structured filters described in
[Retrieval Pipeline](#retrieval-pipeline) (`min_atk`, `max_atk`, `min_def`,
`max_def`, `level`, `attribute`, `card_type`). ATK expressions inside the
query text are auto-extracted and enforced (EN + PT-BR: thresholds,
at-least, less-than, equality, ranges, ">= 2500", "2500+") and merge with
explicit `min_atk`/`max_atk` (effective min = the larger, effective max =
the smaller — AND semantics). Only ATK is auto-parsed; DEF, level,
attribute and card type must still be extracted by the host AI and passed
via their params:

```python
search_cards(query="strong dragons", card_type="Dragon", min_atk=2500)
```

ATK-only example (no explicit params needed):

```python
search_cards(query="dragons with more than 4000 ATK")  # enforces atk >= 4000
```

```json
// .mcp.json
{
  "mcpServers": {
    "ygo-search": {
      "command": "docker",
      "args": ["compose", "exec", "cli", "uv", "run", "ygo-search"],
      "type": "stdio"
    }
  }
}
```

The MCP server always uses local embeddings (`embedding_local` column), regardless
of whether `OPENAI_API_KEY` is set. This means zero API cost and zero latency for
MCP-based searches.

---

## Quick Start

### One-command demo (for presentations / non-developers)

```bash
make demo
```

Answer the prompts — the script sets up everything and launches the interactive search.

### Developer setup

```bash
make run                                    # Start containers, deps, migrations
docker compose exec cli uv run python -m app.ingest  # Ingest cards
docker compose exec cli uv run python -m app.search  # Launch search
```

> An OpenAI API key is optional for CLI search: without one, search runs on local
> embeddings and prints the ranked cards, but not the generated AI answer.
> Copy `.env.example` to `.env` and fill in the values (or let `make demo` prompt
> you for them).
> The MCP server works without any API key (uses local embeddings).

---

## Example Output

```
🏆 Top 10 Results:

1. Dragon Master Lords
   Name: Dragon Master Lords | Attribute: dark | ATK: 5000 | DEF: 5000
   PROPERTIES: ['Dragon', 'Xyz', 'Effect']
   Description: 2 Level 12 monsters...
------------------------------------------------------------
2. Dragon of Pride and Soul
   Name: Dragon of Pride and Soul | Attribute: dark | Level: 8 | ATK: 2500
   DEF: 2500 | PROPERTIES: ['Dragon', 'Effect']
   Description: Cannot be Normal Summoned/Set...
------------------------------------------------------------
...

💡 AI Answer:
Here are the Dragon monsters from the list with more than 2500 ATK:
- Dragon Master Lords — 5000 ATK / 5000 DEF — powerful Dragon/Xyz/Effect...
- Ultimate Dragon of Pride and Soul — 4500 ATK / 4500 DEF...
```

---

## CLI Reference

```bash
# Ingestion with custom directory and verbose logging
docker compose exec cli uv run python -m app.ingest --json-dir ./jsons -v

# Single query (non-interactive)
docker compose exec cli uv run python -m app.search --query "spell cards that destroy monsters"

# Single query with structured filters
docker compose exec cli uv run python -m app.search --query "strong dragons" --card-type Dragon --min-atk 2500

# Interactive mode (REPL)
docker compose exec cli uv run python -m app.search

# Start the file watcher (auto-reindex on card JSON changes)
make watch
```

**The watcher treats `jsons/` as the index.** A card JSON added or edited in the
watched directory is upserted; a card JSON that leaves it — deleted, archived,
or moved into a subfolder — has its row removed. The observer is non-recursive,
and watchdog reports a move out of the watched directory as a plain deletion, so
archiving a batch of cards does drop them from the search index. That is the
contract, not an accident: if it is not in `jsons/`, it is not searchable. Batch
ingestion (`python -m app.ingest`) never deletes anything, so re-adding the files
and re-ingesting restores them. Renaming a file in place is the one exception —
it upserts the new card ID and leaves the old row alone.

| Command | Action |
|---|---|
| `make init` | Start containers + install dependencies (minimal setup) |
| `make run` | Full dev setup: containers + deps + migrations |
| `make demo` | Interactive onboarding → full setup → launches search |
| `make watch` | Start file watcher for auto-reindexing |
| `make reset` | Full teardown + recreate |
| `make gpu-check` | Print what torch sees inside the container (device, HIP/CUDA build) |
| `make exec CMD="..."` | Run arbitrary command in container |

---

## Project Structure

```
yugioh-cards-rag/
├── app/
│   ├── config.py          # Centralized config (env vars, provider auto-detect)
│   ├── embeddings.py      # Dual embedding provider (OpenAI + local GPU/CPU)
│   ├── models.py          # SQLAlchemy ORM (Card model, dual vectors)
│   ├── ingest.py          # Ingestion pipeline (batch + dual embeddings)
│   ├── search.py          # Interactive RAG search (CLI, OpenAI)
│   ├── mcp_server.py      # MCP server (hybrid retrieval, local embeddings)
│   ├── query_parser.py    # ATK constraint extraction from query text (EN/PT-BR)
│   ├── retrieval.py       # Shared retrieval core: filters, RRF fusion, reranking
│   └── watcher.py         # File watcher for auto-reindexing
├── jsons/                 # Card data (one JSON file per card)
├── alembic/               # Database migrations
├── tests/                 # DB-free unit tests (query parser, filter builder)
├── docker-compose.yml     # PostgreSQL + pgvector (no GPU reservation)
├── docker-compose.nvidia.yml  # Optional overlay: reserves the NVIDIA GPU
├── docker-compose.rocm.yml    # Optional overlay: /dev/kfd + /dev/dri (AMD)
├── Dockerfile
├── demo.sh                # One-command setup driven by `make demo`
├── .env.example           # Environment variables template
├── .mcp.json              # MCP server config for Claude Code / Pi
├── Makefile
├── pyproject.toml
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | No | OpenAI API key. When set, CLI search uses OpenAI embeddings and generates the AI answer. When unset, CLI search still runs on local embeddings, minus the AI answer. Never needed for MCP. |
| `CARD_JSON_DIR` | No | Path to card JSON directory (default: `jsons`) |
| `DATABASE_URL` | No | SQLAlchemy connection URL (default: `postgresql+psycopg2://admin:admin@db/rag_db`) |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

Copy `.env.example` to `.env` and fill in the values. `.env` is git-ignored and never committed.

**Provider auto-detection:** when `OPENAI_API_KEY` is set, the CLI uses OpenAI
embeddings (1536 dims) and generates the AI answer with `CHAT_MODEL`. When unset,
the CLI falls back to local embeddings (1024 dims) and still runs the full
retrieval pipeline — query embedding, structured ATK filters, hybrid RRF and
cross-encoder reranking are all local — but the generated answer is not
available: the CLI prints the ranked cards followed by a one-line notice in
place of the AI answer. The MCP server always uses local embeddings regardless.

---

## Data Model

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Unique card ID (auto-increment) |
| `card_json_id` | Integer (unique, indexed) | Card ID from JSON file (for upserts) |
| `name` | String | Card name |
| `level` | Integer | Card level/rank |
| `atk` | Integer | Attack points |
| `def_` | Integer | Defense points |
| `english_attribute` | String | Attribute (wind, fire, light, etc.) |
| `properties` | Array | Card properties (Dragon, Normal, etc.) |
| `content` | Text | Text representation for embedding |
| `embedding` | Vector(1536) | OpenAI `text-embedding-3-small` embedding |
| `embedding_local` | Vector(1024) | Local `mxbai-embed-large-v1` embedding |

### Indexes

| Index | Type | On | Purpose |
|---|---|---|---|
| `ix_cards_embedding_hnsw` | HNSW (`vector_cosine_ops`) | `embedding` | ANN search for OpenAI embeddings |
| `ix_cards_embedding_local_hnsw` | HNSW (`vector_cosine_ops`) | `embedding_local` | ANN search for local embeddings |
| `ix_cards_name_trgm` | GiST (`gist_trgm_ops`) | `name` | Lexical similarity ranking (RRF) |
| `ix_cards_properties_gin` | GIN | `properties` | `card_type` containment filter |
| `ix_cards_atk` / `ix_cards_def_` / `ix_cards_level` / `ix_cards_english_attribute` | B-tree | respective column | Structured filters |
| `ix_cards_card_json_id` / `uq_cards_card_json_id` | B-tree (unique) | `card_json_id` | Idempotent upserts |