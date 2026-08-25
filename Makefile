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
	@echo "✓ Ready. Run: docker compose exec cli uv run python -m app.ingest"
	@echo "            docker compose exec cli uv run python -m app.search"

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
