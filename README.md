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
┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  jsons/*.json │────▶│  app/ingest.py  │────▶│  PostgreSQL  │
│  (card data)  │     │  + OpenAI emb.  │     │  + pgvector  │
└──────────────┘     └─────────────────┘     └──────┬───────┘
                                                     │
┌──────────────┐     ┌─────────────────┐             │
│  User query  │────▶│  app/search.py  │◀────────────┘
│  (natural lg) │     │  + GPT-4o-mini  │
└──────────────┘     └─────────────────┘
```

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Database | PostgreSQL + pgvector |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| LLM | OpenAI `gpt-4o-mini` |
| Observability | LangSmith tracing |
| Infrastructure | Docker Compose |

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
docker compose exec cli uv run app/ingest.py  # Ingest cards
docker compose exec cli uv run app/search.py  # Launch search
```

> Requires an OpenAI API key. Copy `.env.example` to `.env` and fill in the values
> (or let `make demo` prompt you for them).

---

## Example Output

```
🏆 Top 10 Results:

1. Blue-Eyes White Dragon
   Name: Blue-Eyes White Dragon | Attribute: light | Level: 8 | ATK: 3000 | DEF: 2500
   ------------------------------------------------------------
2. Red-Eyes Black Dragon
   Name: Red-Eyes Black Dragon | Attribute: dark | Level: 7 | ATK: 2400 | DEF: 2000
   ------------------------------------------------------------
...

💡 AI Answer:
The top dragons with high attack in the database are Blue-Eyes White Dragon
(3000 ATK) and Red-Eyes Black Dragon (2400 ATK). Both are Normal monsters
with no special effects.
```

---

## CLI Reference

```bash
# Ingestion with custom directory and verbose logging
docker compose exec cli uv run app/ingest.py --json-dir ./jsons -v

# Single query (non-interactive)
docker compose exec cli uv run app/search.py --query "spell cards that destroy monsters"

# Interactive mode (REPL)
docker compose exec cli uv run app/search.py
```

| Command | Action |
|---|---|
| `make init` | Start containers + install dependencies (minimal setup) |
| `make run` | Full dev setup: containers + deps + migrations |
| `make demo` | Interactive onboarding → full setup → launches search |
| `make reset` | Full teardown + recreate |
| `make exec CMD="..."` | Run arbitrary command in container |

---

## Project Structure

```
poc-rag/
├── app/
│   ├── config.py          # Centralized config (env vars, defaults)
│   ├── embeddings.py      # Shared OpenAI embedding client
│   ├── models.py          # SQLAlchemy ORM (Card model)
│   ├── ingest.py          # Ingestion pipeline
│   └── search.py          # Interactive RAG search
├── jsons/                 # Raw card data
├── alembic/               # Database migrations
├── docker-compose.yml
├── Dockerfile
├── .env.example          # Environment variables template
├── Makefile
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key for embeddings and chat |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

Copy `.env.example` to `.env` and fill in the values. `.env` is git-ignored and never committed.

---

## Data Model

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Unique card ID |
| `name` | String | Card name |
| `level` | Integer | Card level/rank |
| `atk` | Integer | Attack points |
| `def_` | Integer | Defense points |
| `english_attribute` | String | Attribute (wind, fire, light, etc.) |
| `properties` | Array | Card properties (Dragon, Normal, etc.) |
| `content` | Text | Text representation for embedding |
| `embedding` | Vector(1536) | Float vector from OpenAI |
