# 🃏 poc-rag — Yu-Gi-Oh! Semantic Search (RAG Proof of Concept)

**Retrieval-Augmented Generation (RAG)** applied to Yu-Gi-Oh! card data.
Ask natural language questions and get AI-powered answers grounded in real card data.

```
Query: "dragons with more than 2500 ATK"
  → Retrieves top 10 semantically similar cards via pgvector
  → LLM generates a summarized answer from those cards
```

---

## Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  cards/*.json │────▶│  app/ingest.py       │────▶│  PostgreSQL  │
│  (card data)  │     │  + dual embeddings   │     │  + pgvector  │
└──────────────┘     └──────────────────────┘     └──────┬───────┘
                         ┌─────────┐                   │
                         │ Watcher │ (auto-reindex)    │
                         └─────────┘                   │
                     ┌─────────────────┐                │
                     │  app/search.py  │◀───────────────┘
  User query ────────▶│  + GPT-5.4-mini │
                     │  (OpenAI emb.)  │
                     └─────────────────┘
                     ┌─────────────────┐
                     │  app/mcp_server  │◀───────────────┘
  MCP host ──────────▶│  (local emb.)   │
  (Pi, Claude Code)   │  pure retrieval  │
                     └─────────────────┘
```

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Database | PostgreSQL + pgvector |
| OpenAI Embeddings | `text-embedding-3-small` (1536 dims) |
| Local Embeddings | `mxbai-embed-large-v1` (1024 dims, GPU/CUDA) |
| LLM | OpenAI `gpt-5.4-mini` |
| MCP Server | FastMCP (stdio transport, pure retrieval) |
| Observability | LangSmith tracing |
| Infrastructure | Docker Compose (with NVIDIA GPU support) |

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

The local model runs on **NVIDIA GPU (CUDA)**. The Docker container is configured
with GPU access via `deploy.resources.reservations.devices`. CUDA is required
for local embeddings — there is no CPU fallback.

---

## MCP Server

The MCP server exposes two tools as a **pure retrieval** interface — it returns
raw card data, and the host AI (Pi, Claude Code) does its own reasoning.

| Tool | Description |
|---|---|
| `search_cards` | Semantic search via pgvector cosine similarity |
| `get_card` | Retrieve a specific card by its JSON ID |

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

> Requires an OpenAI API key for CLI search. Copy `.env.example` to `.env` and
> fill in the values (or let `make demo` prompt you for them).
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

# Interactive mode (REPL)
docker compose exec cli uv run python -m app.search

# Start the file watcher (auto-reindex on card JSON changes)
make watch
```

| Command | Action |
|---|---|
| `make init` | Start containers + install dependencies (minimal setup) |
| `make run` | Full dev setup: containers + deps + migrations |
| `make demo` | Interactive onboarding → full setup → launches search |
| `make watch` | Start file watcher for auto-reindexing |
| `make reset` | Full teardown + recreate |
| `make exec CMD="..."` | Run arbitrary command in container |

---

## Project Structure

```
poc-rag/
├── app/
│   ├── config.py          # Centralized config (env vars, provider auto-detect)
│   ├── embeddings.py      # Dual embedding provider (OpenAI + local CUDA)
│   ├── models.py          # SQLAlchemy ORM (Card model, dual vectors)
│   ├── ingest.py          # Ingestion pipeline (batch + dual embeddings)
│   ├── search.py          # Interactive RAG search (CLI, OpenAI)
│   ├── mcp_server.py      # MCP server (pure retrieval, local embeddings)
│   └── watcher.py         # File watcher for auto-reindexing
├── cards/                 # External card data (bind-mounted, read-only)
├── alembic/               # Database migrations
├── docker-compose.yml     # PostgreSQL + pgvector + GPU support
├── Dockerfile
├── .env.example           # Environment variables template
├── .mcp.json              # MCP server config for Claude Code / Pi
├── Makefile
├── pyproject.toml
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | CLI search | OpenAI API key for embeddings and chat. Not needed for MCP. |
| `CARD_JSON_DIR` | No | Path to card JSON directory (default: `jsons`) |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

Copy `.env.example` to `.env` and fill in the values. `.env` is git-ignored and never committed.

**Provider auto-detection:** when `OPENAI_API_KEY` is set, the CLI uses OpenAI
embeddings (1536 dims). When unset, it falls back to local embeddings (1024 dims).
The MCP server always uses local embeddings regardless.

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