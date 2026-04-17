init:
	docker compose up -d
	docker compose exec cli uv sync

reset:
	docker compose down -v
	docker compose up -d
	docker compose exec cli uv sync

exec:
	docker compose exec cli $(CMD)
