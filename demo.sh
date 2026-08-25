#!/usr/bin/env bash
set -euo pipefail

# ── demo.sh ──
# One-command setup for the Yu-Gi-Oh! RAG demo.
# Run via: make demo

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

divider() {
  echo -e "${CYAN}── $1 ─────────────────────────────${NC}"
}

success() { echo -e "  ${GREEN}✓${NC} $1"; }
warn()   { echo -e "  ${YELLOW}⚠${NC}  $1"; }
fail()   { echo -e "  ${RED}✗${NC} $1"; }

cleanup() {
  echo -e "\n${YELLOW}Setup cancelled.${NC}"
  exit 0
}
trap cleanup INT TERM

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   🃏  Yu-Gi-Oh! RAG — Semantic Search   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── 0. Prerequisites ──────────────────────

if ! command -v docker &>/dev/null; then
  fail "Docker is not installed."
  echo ""
  echo "  Please install Docker Desktop from:"
  echo "  https://www.docker.com/products/docker-desktop"
  exit 1
fi

if ! docker compose version &>/dev/null; then
  fail "Docker Compose is not available."
  echo ""
  echo "  Please install Docker Desktop (includes Compose):"
  echo "  https://www.docker.com/products/docker-desktop"
  exit 1
fi

# ── 1. Environment setup ─────────────────

setup_env() {
  divider "API Keys"

  # Skip if .env already has a non-empty OPENAI_API_KEY
  if [[ -f .env ]] && grep -qE '^OPENAI_API_KEY=.+' .env 2>/dev/null; then
    success ".env already configured"
    return
  fi

  echo ""
  echo "  An OpenAI API key is optional. Without one, search still runs on"
  echo "  local embeddings and prints the ranked cards, but not the AI answer."
  echo ""
  read -r -p "  OpenAI API key (press Enter to skip): " openai_key
  echo ""

  if [[ -z "$openai_key" ]]; then
    warn "No API key provided. Search will run in retrieval-only mode."
  fi

  read -r -p "  Enable LangSmith tracing? [y/N]: " enable_ls
  echo ""

  ls_tracing="false"
  ls_endpoint=""
  ls_api_key=""
  ls_project="poc-rag"

  if [[ "$enable_ls" =~ ^[Yy]$ ]]; then
    ls_tracing="true"
    read -r -p "  LangSmith endpoint [https://eu.api.smith.langchain.com]: " input_endpoint
    ls_endpoint="${input_endpoint:-https://eu.api.smith.langchain.com}"
    read -r -p "  LangSmith API key: " ls_api_key
    read -r -p "  LangSmith project [poc-rag]: " input_project
    ls_project="${input_project:-poc-rag}"
  fi

  {
    echo "OPENAI_API_KEY=${openai_key}"
    echo "LANGSMITH_TRACING=${ls_tracing}"
    echo "LANGSMITH_ENDPOINT=${ls_endpoint}"
    echo "LANGSMITH_API_KEY=${ls_api_key}"
    echo "LANGSMITH_PROJECT=${ls_project}"
  } > .env

  success ".env written"
}

# ── 2. Start services ────────────────────

start_services() {
  divider "Starting services"

  docker compose up -d

  # Wait for database to accept connections
  echo -n "  Waiting for database..."
  for i in $(seq 1 30); do
    if docker compose exec -T db pg_isready -U admin -d rag_db &>/dev/null; then
      echo ""
      break
    fi
    sleep 2
    echo -n "."
  done
  if ! docker compose exec -T db pg_isready -U admin -d rag_db &>/dev/null; then
    echo ""
    fail "Database did not become ready within 60 seconds."
    exit 1
  fi
  success "Database ready"

  # Wait for CLI container to be running
  echo -n "  Waiting for CLI container..."
  for i in $(seq 1 15); do
    if docker compose exec -T cli true &>/dev/null; then
      echo ""
      break
    fi
    sleep 2
    echo -n "."
  done
  if ! docker compose exec -T cli true &>/dev/null; then
    echo ""
    fail "CLI container is not running. Check: docker compose logs cli"
    exit 1
  fi
  success "CLI ready"
}

# ── 3. Install dependencies ──────────────

install_deps() {
  divider "Dependencies"
  docker compose exec -T cli uv sync
  # uv sync reconciles the venv with uv.lock, which pins the CUDA torch, so it undoes
  # the ROCm swap. COMPOSE_FILE is exported by `make demo GPU=rocm`; a bare
  # `bash demo.sh` leaves it unset and stays on the base files, i.e. CPU.
  case "${COMPOSE_FILE:-}" in
    *rocm*) docker compose exec -T cli sh scripts/rocm-swap.sh ;;
  esac
  success "Dependencies installed"
}

# ── 4. Run migrations ────────────────────

run_migrations() {
  divider "Migrations"
  docker compose exec -T cli uv run alembic upgrade head
  success "Migrations applied"
}

# ── 5. Ingest cards ──────────────────────

ingest_cards() {
  divider "Card ingestion"

  local card_count
  card_count=$(docker compose exec -T db psql -U admin -d rag_db -tAc \
    "SELECT COUNT(*) FROM cards;" 2>/dev/null || echo "0")

  if [[ "$card_count" -gt 0 ]]; then
    echo ""
    read -r -p "  Cards already in database (${card_count}). Re-ingest? [y/N]: " reingest
    echo ""
    if [[ ! "$reingest" =~ ^[Yy]$ ]]; then
      success "Skipping ingestion (${card_count} cards already present)"
      return
    fi
  fi

  if ! docker compose exec -T cli uv run python -m app.ingest; then
    warn "Ingestion had errors. Launching search with available cards..."
  fi
}

# ── 6. Launch search ─────────────────────

launch_search() {
  divider "Launching search"
  echo ""
  docker compose exec -it cli uv run python -m app.search
}

# ── Main ─────────────────────────────────

setup_env
start_services
install_deps
run_migrations
ingest_cards
launch_search
