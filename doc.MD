# poc-rag — Yu-Gi-Oh! Semantic Search (RAG Proof of Concept)

## Overview

A Retrieval-Augmented Generation (RAG) proof of concept that enables semantic search over Yu-Gi-Oh! card data. Cards are ingested from JSON files, converted to vector embeddings via OpenAI, and stored in PostgreSQL with pgvector. A search interface accepts natural language queries and returns the most semantically similar cards.

**Tech stack:**

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package manager | uv |
| Database | PostgreSQL + pgvector |
| ORM | SQLAlchemy 2.0 |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Migrations | Alembic |
| Containers | Docker Compose |

---

## Architecture

### Ingestion pipeline

```
jsons/*.json
     |
     v
app/ingest.py
  - Reads card fields (name, level, atk, def, attribute, effectText)
  - Builds concatenated text: "Name: X | Attribute: Y | ATK: Z | ..."
     |
     v
OpenAI API (text-embedding-3-small)
  - Returns 1536-dimensional float vector
     |
     v
PostgreSQL / cards table
  - Stores card data + embedding vector
```

### Search pipeline

```
User query (stdin)
     |
     v
app/search.py
  - Generates embedding for the query
     |
     v
PostgreSQL (pgvector)
  - ORDER BY embedding <=> query_vector  (cosine distance)
  - LIMIT 10
     |
     v
Top 10 semantically similar cards (stdout)
```

---

## Project Structure

```
poc-rag/
├── app/
│   ├── models.py          # SQLAlchemy ORM — Card model and Base
│   ├── ingest.py          # Ingestion pipeline: JSON → embeddings → DB
│   └── search.py          # Interactive semantic search via pgvector
├── alembic/
│   ├── env.py             # Alembic environment config (imports Base.metadata)
│   └── versions/
│       ├── 1bf1dff56c54_criacao_tabela_cards.py  # Creates cards table + vector extension
│       └── 2f134ab196da_card_migration.py         # Placeholder (no-op)
├── jsons/                 # Raw Yu-Gi-Oh! card data (one JSON file per card)
├── docker-compose.yml     # Defines db and cli services
├── Dockerfile             # python:3.13-slim + uv
├── Makefile               # Convenience command shortcuts
├── alembic.ini            # Alembic config (DB URL, script location)
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependency tree
└── .env                   # Local environment variables (not committed)
```

---

## Infrastructure

### Docker Compose services

| Service | Image | Purpose |
|---|---|---|
| `db` | `ankane/pgvector:latest` | PostgreSQL with pgvector extension |
| `cli` | `./Dockerfile` (python:3.13-slim + uv) | Python runtime for running scripts |

### Ports and volumes

| Service | Port | Volume |
|---|---|---|
| `db` | `5432:5432` | `db_data:/var/lib/postgresql/data` (persistent) |
| `cli` | — | `.:/app` (project root bind-mounted) |

### Environment setup

Create a `.env` file in the project root (never commit this file):

```
OPENAI_API_KEY=sk-...
```

The `cli` service loads this file automatically via `env_file: .env` in `docker-compose.yml`.

---

## Data Model

### `cards` table

| Column | Type | Description |
|---|---|---|
| `id` | Integer (PK) | Unique card identifier |
| `name` | String | Card name |
| `level` | Integer | Card level or rank |
| `atk` | Integer | Attack points |
| `def_` | Integer | Defense points |
| `english_attribute` | String | Card attribute (e.g. `wind`, `fire`, `light`) |
| `content` | Text | Concatenated text used to generate the embedding |
| `embedding` | Vector(1536) | Float vector from OpenAI embeddings |

### Input JSON format

Each file in `jsons/` represents a single card:

```json
{
  "id": 4007,
  "type": "monster",
  "name": "Blue-Eyes White Dragon",
  "englishAttribute": "light",
  "localizedAttribute": "LIGHT",
  "effectText": "This legendary dragon is a powerful engine of destruction...",
  "level": 8,
  "atk": 3000,
  "def": 2500,
  "properties": ["Dragon", "Normal"]
}
```

Fields used during ingestion: `name`, `level`, `atk`, `def`, `englishAttribute`, `effectText`.
`properties`, `localizedAttribute`, `type`, and `id` are ignored.

---

## Modules

### `app/models.py`

Defines the SQLAlchemy ORM schema.

- **`Base`** — Declarative base class (`DeclarativeBase`). All models inherit from it. Required by Alembic's `env.py` for autogenerate.
- **`Card`** — Maps to the `cards` table. All columns correspond 1:1 to the data model above.

### `app/ingest.py`

Ingestion pipeline. Run once to populate the database.

| Function | Purpose |
|---|---|
| `clean_int(value)` | Safely converts any value to int; returns `None` on failure |
| `get_embedding(text)` | Calls OpenAI `text-embedding-3-small`, returns 1536-dim list |
| `process_jsons()` | Main loop: reads all `.json` files, builds content string, generates embedding, saves `Card` to DB |

The `content` field is built as:
```
Name: {name} | Attribute: {attr} | Level: {level} | ATK: {atk} | DEF: {def} | Description: {effectText}
```
Only non-null fields are included.

### `app/search.py`

Interactive semantic search. Accepts a natural language query and returns the 10 most similar cards using pgvector's cosine distance operator `<=>`.

| Function | Purpose |
|---|---|
| `get_embedding(query_text)` | Same as ingest — generates query embedding |
| `realizar_busca()` | Prompts for input, runs similarity query, prints results |

SQL used internally:
```sql
SELECT name, content FROM cards
ORDER BY embedding <=> :vetor
LIMIT :limite
```

---

## Database Migrations

Alembic is configured in `alembic/env.py` to use `Base.metadata` from `app.models` for autogenerate support.

### Migration history

| Revision | Description |
|---|---|
| `1bf1dff56c54` | Creates `vector` PostgreSQL extension and `cards` table |
| `2f134ab196da` | Empty migration (no schema changes) |

### Workflow for new migrations

```bash
# 1. Apply existing migrations first
docker compose exec cli uv run alembic upgrade head

# 2. Modify app/models.py with schema changes

# 3. Generate migration
docker compose exec cli uv run alembic revision --autogenerate -m "description"

# 4. Apply
docker compose exec cli uv run alembic upgrade head
```

---

## Commands Reference

### Makefile shortcuts

| Command | Action |
|---|---|
| `make init` | Start containers and run `uv sync` |
| `make reset` | Tear down volumes, recreate containers, run `uv sync` |
| `make exec CMD="..."` | Run arbitrary command inside the `cli` container |

### Full command reference

| Action | Command |
|---|---|
| Start containers | `docker compose up -d` |
| Stop containers | `docker compose stop` |
| Remove containers | `docker compose down` |
| Remove containers + data | `docker compose down -v` |
| Open shell in cli | `docker compose exec cli bash` |
| Install dependencies | `docker compose exec cli uv sync` |
| Add a package | `docker compose exec cli uv add <package>` |
| Apply migrations | `docker compose exec cli uv run alembic upgrade head` |
| Generate migration | `docker compose exec cli uv run alembic revision --autogenerate -m "<msg>"` |
| Run ingestion | `docker compose exec cli uv run app/ingest.py` |
| Run search | `docker compose exec cli uv run app/search.py` |

### Verify ingestion

```bash
# Row count
docker compose exec db psql -U admin -d rag_db -c "SELECT COUNT(*) FROM cards;"

# Sample rows
docker compose exec db psql -U admin -d rag_db -c "SELECT id, name, atk, def_ FROM cards LIMIT 10;"

# Confirm embeddings were generated
docker compose exec db psql -U admin -d rag_db -c "SELECT name, embedding IS NOT NULL AS has_embedding FROM cards LIMIT 10;"
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key used for embedding generation in both `ingest.py` and `search.py` |

Set in `.env` at the project root. This file is listed in `.gitignore` and must never be committed.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `SyntaxError: Expected one or more names after 'import'` | Incomplete import in `alembic/env.py` | Change `from app.models import` to `from app.models import Base` |
| `NameError: name 'pgvector' is not defined` | Missing import in migration file | Add `import pgvector.sqlalchemy` to the migration |
| `type "vector" does not exist` | pgvector extension not enabled in DB | Add `op.execute("CREATE EXTENSION IF NOT EXISTS vector")` in migration `upgrade()` |
| `ModuleNotFoundError: No module named 'app'` | Script run from inside `app/` without proper path | Use `from models import` instead of `from app.models import` in scripts under `app/` |
| `Target database is not up to date` | Tried to autogenerate before applying existing migrations | Run `alembic upgrade head` before `alembic revision --autogenerate` |
| `OPENAI_API_KEY must be set` | Key missing from environment | Add `OPENAI_API_KEY=sk-...` to `.env` and restart containers |
