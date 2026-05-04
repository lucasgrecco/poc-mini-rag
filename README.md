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

### Prerequisites
- Docker + Docker Compose
- OpenAI API key

### 1. Configure

Create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
```

### 2. Start

```bash
make init
```

### 3. Ingest cards

```bash
docker compose exec cli uv run app/ingest.py
```

### 4. Search

```bash
docker compose exec cli uv run app/search.py
```

```
Query: dragons with high attack and special effects
```

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
| `make init` | Start containers + install dependencies |
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
├── Makefile
└── doc.MD                 # Developer documentation
```

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
