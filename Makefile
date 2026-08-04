# GPU= escolhe a sobreposição de compose. Sem GPU=, sobe em qualquer host e o torch
# cai para CPU sozinho.
#   make run              → CPU, qualquer máquina
#   make run GPU=nvidia   → reserva a placa NVIDIA (exige nvidia-container-toolkit)
#   make run GPU=rocm     → entrega as placas AMD via /dev/kfd + /dev/dri
COMPOSE := docker compose $(if $(GPU),-f docker-compose.yml -f docker-compose.$(GPU).yml)

# O wheel padrão do PyPI é build CUDA e ignora a Radeon. Em GPU=rocm o torch é
# reinstalado do índice ROCm por cima do uv sync — as dependências moram no .venv do
# bind mount, não na imagem, então é aqui que a troca acontece e não no Dockerfile.
# Conferir o índice correspondente à ROCm da máquina em
# https://pytorch.org/get-started/locally/ antes da primeira subida.
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

# Diz qual device o torch enxerga de dentro do container. Numa Radeon com ROCm o
# esperado é `cuda True` — HIP responde pela API CUDA.
gpu-check:
	$(COMPOSE) exec cli uv run python -c "import torch; print('cuda', torch.cuda.is_available(), '| devices', torch.cuda.device_count(), '| hip', torch.version.hip, '| cuda-build', torch.version.cuda)"

watch:
	$(COMPOSE) exec cli uv run python -m app.watcher

demo:
	bash demo.sh

exec:
	$(COMPOSE) exec cli $(CMD)
