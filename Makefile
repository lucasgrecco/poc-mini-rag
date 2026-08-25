# GPU= picks the compose overlay. Without GPU=, it comes up on any host and torch
# falls back to CPU on its own.
#   make run              → CPU, any machine
#   make run GPU=nvidia   → reserves the NVIDIA card (requires nvidia-container-toolkit)
#   make run GPU=rocm     → hands the AMD cards over via /dev/kfd + /dev/dri
COMPOSE := docker compose $(if $(GPU),-f docker-compose.yml -f docker-compose.$(GPU).yml)

# The default PyPI wheel is a CUDA build and ignores the Radeon. Under GPU=rocm torch
# is reinstalled from the ROCm index on top of uv sync — dependencies live in the .venv
# of the bind mount, not in the image, so the swap happens here and not in the Dockerfile.
# Check the index matching the machine's ROCm generation at
# https://pytorch.org/get-started/locally/ before the first run.
ROCM_INDEX ?= https://download.pytorch.org/whl/rocm6.3
SYNC_EXTRA := $(if $(filter rocm,$(GPU)),$(COMPOSE) exec cli uv pip install --index-url $(ROCM_INDEX) torch,true)

init:
	$(COMPOSE) up -d
	$(COMPOSE) exec cli uv sync
	$(SYNC_EXTRA)

reset:
	$(COMPOSE) down -v
	$(COMPOSE) up -d
	$(COMPOSE) exec cli uv sync
	$(SYNC_EXTRA)

run:
	$(COMPOSE) up -d
	$(COMPOSE) exec cli uv sync
	$(SYNC_EXTRA)
	$(COMPOSE) exec cli uv run alembic upgrade head
	@echo ""
	@echo "✓ Ready."
	@$(MAKE) --no-print-directory env-status
	@echo ""
	@echo "  Next: docker compose exec cli uv run python -m app.ingest"
	@echo "        docker compose exec cli uv run python -m app.search"

# Both integrations are optional and a fresh clone has neither. Reported after
# `make run` so it is obvious which mode the stack came up in, rather than the
# absence of keys being discovered later as missing output.
env-status:
	@echo ""
	@echo "  Integrations (both optional):"
	@if [ -f .env ] && grep -qE '^OPENAI_API_KEY=.+' .env; then \
	  echo "    OpenAI     key found  -> OpenAI embeddings + generated answers"; \
	else \
	  echo "    OpenAI     no key     -> local embeddings on this machine,"; \
	  echo "                             retrieval only, no generated answer"; \
	fi
	@if [ -f .env ] && grep -qE '^LANGSMITH_API_KEY=.+' .env; then \
	  echo "    LangSmith  key found  -> tracing enabled"; \
	else \
	  echo "    LangSmith  no key     -> no tracing, no observability"; \
	fi
	@if [ ! -f .env ]; then \
	  echo ""; \
	  echo "  To enable either: cp .env.example .env, fill in OPENAI_API_KEY"; \
	  echo "  and/or the LANGSMITH_* values, then run make run again."; \
	elif ! grep -qE '^OPENAI_API_KEY=.+' .env || ! grep -qE '^LANGSMITH_API_KEY=.+' .env; then \
	  echo ""; \
	  echo "  To enable the rest: fill in the missing values in your existing"; \
	  echo "  .env, then run make run again."; \
	fi

# Reports which device torch sees from inside the container. On a Radeon with ROCm
# the expected output is `cuda True` — HIP answers through the CUDA API.
gpu-check:
	$(COMPOSE) exec cli uv run python -c "import torch; print('cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count(), '| hip', torch.version.hip, '| cuda-build', torch.version.cuda)"

watch:
	$(COMPOSE) exec cli uv run python -m app.watcher

demo:
	bash demo.sh

exec:
	$(COMPOSE) exec cli $(CMD)
