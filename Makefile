init:
	docker compose up -d
	docker compose exec cli uv sync

reset:
	docker compose down -v
	docker compose up -d
	docker compose exec cli uv sync

run:
	docker compose up -d
	docker compose exec cli uv sync
	docker compose exec cli uv run alembic upgrade head
	@echo "✓ Ready. Run ingest or search manually."

demo:
	bash demo.sh

exec:
	docker compose exec cli $(CMD)
