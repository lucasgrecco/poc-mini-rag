# GPU= picks the compose overlay. Without GPU=, it comes up on any host and torch
# falls back to CPU on its own.
#   make run              → CPU, any machine
#   make run GPU=nvidia   → reserves the NVIDIA card (requires nvidia-container-toolkit)
#   make run GPU=rocm     → hands the AMD card over via /dev/kfd + /dev/dri
#
# The file set travels as an exported COMPOSE_FILE (":"-separated) rather than as -f
# flags, so demo.sh's own `docker compose` calls join the same project and overlay.
# Without it, `make demo GPU=rocm` would run demo.sh's `up -d` against the base file
# alone, changing the config hash and recreating cli *without* /dev/kfd, group_add or
# ipc: host.
COMPOSE_FILE := docker-compose.yml$(if $(GPU),:docker-compose.$(GPU).yml)
export COMPOSE_FILE
COMPOSE := docker compose

# ROCm: docker compose resolves group_add *names* against the container image's
# /etc/group (python:3.13-slim has no "render"), and the GID that owns the ROCm
# nodes varies per host. Derive it from /dev/kfd and pass the number in.
export ROCM_DEV_GID := $(shell stat -c '%g' /dev/kfd 2>/dev/null)

# The default PyPI wheel is a CUDA build and ignores the Radeon. Under GPU=rocm torch
# is reinstalled from the ROCm index on top of uv sync — dependencies live in the .venv
# of the bind mount, not in the image, so the swap happens here and not in the Dockerfile.
# The index default lives in scripts/rocm-swap.sh; set ROCM_INDEX to override it, and
# check the generation matching the machine at https://pytorch.org/get-started/locally/.
ROCM_INDEX ?=
SYNC_EXTRA := $(if $(filter rocm,$(GPU)),$(COMPOSE) exec -e ROCM_INDEX="$(ROCM_INDEX)" cli sh scripts/rocm-swap.sh,true)

init:
	$(COMPOSE) up -d
	$(COMPOSE) exec cli uv sync
	$(SYNC_EXTRA)

reset:
	$(COMPOSE) down -v
	$(COMPOSE) up -d
	$(COMPOSE) exec cli uv sync
	$(SYNC_EXTRA)

# The swap goes last: alembic needs no GPU, and migrating before a ~2.5 GB download
# means a failed download still leaves the database migrated.
run:
	$(COMPOSE) up -d
	$(COMPOSE) exec cli uv sync
	$(COMPOSE) exec cli uv run alembic upgrade head
	$(SYNC_EXTRA)
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

# `uv add` under GPU=rocm updates pyproject.toml/uv.lock but installs nothing, because
# the overlay sets UV_NO_SYNC=1. Sync explicitly here, then re-apply the swap that the
# sync just undid.
sync:
	$(COMPOSE) exec cli uv sync
	$(SYNC_EXTRA)

# Reports which device torch sees from inside the container. On a Radeon with ROCm
# the expected output is `cuda True` — HIP answers through the CUDA API.
gpu-check:
	$(COMPOSE) exec cli uv run python -c "import torch; print('cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count(), '| hip', torch.version.hip, '| cuda-build', torch.version.cuda)"

watch:
	$(COMPOSE) exec cli uv run python -m app.watcher

# DB-free unit tests. pytest ships in the `dev` dependency group, which uv syncs by
# default. Pass extra flags with ARGS, e.g. make test ARGS="-k query_parser -v".
test:
	$(COMPOSE) exec cli uv run python -m pytest tests/ $(ARGS)

demo:
	bash demo.sh

exec:
	$(COMPOSE) exec cli $(CMD)
