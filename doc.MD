# Yu-Gi-Oh! Micro RAG - Documentação da Fundação

Este documento registra a configuração da infraestrutura e do banco de dados vetorial.

## Infraestrutura (Docker Compose)

O ambiente roda em containers Docker gerenciados diretamente pelo Docker Compose (migrado do Docksal):

- **db**: PostgreSQL com a extensão `pgvector` para busca semântica.
- **cli**: Ambiente Python 3.13-slim com `uv` pré-instalado.

### Arquivos de configuração

| Arquivo | Propósito |
|---|---|
| `docker-compose.yml` | Define os serviços `db` e `cli` |
| `Dockerfile` | Imagem do `cli` com Python 3.13 e `uv` |
| `.env` | Variáveis de ambiente locais (não versionado) |
| `Makefile` | Atalhos para os comandos mais comuns |

### Variáveis de ambiente (`.env`)

```
OPENAI_API_KEY=sk-...
```

## Ambiente Python (uv)

Utilizamos o `uv` para gerenciamento de pacotes e ambientes virtuais:

- `sqlalchemy`: ORM para mapeamento de dados.
- `alembic`: Controle de versões (migrações) do banco de dados.
- `psycopg2-binary`: Driver de conexão com PostgreSQL.
- `pgvector`: Suporte a vetores no SQLAlchemy.
- `openai`: Cliente da API OpenAI para geração de embeddings.

## Modelo de Dados (SQLAlchemy)

A tabela principal chama-se `cards` e possui a seguinte estrutura:

| Coluna | Tipo | Descrição |
|---|---|---|
| `id` | Integer (PK) | Identificador único |
| `name` | String | Nome da carta |
| `level` | Integer | Nível/Rank |
| `atk` | Integer | Pontos de ataque |
| `def_` | Integer | Pontos de defesa |
| `english_attribute` | String | Atributo em inglês |
| `content` | Text | Texto concatenado para a IA ler |
| `embedding` | Vector(1536) | Vetor gerado pela OpenAI |

## Comandos principais

### Containers

| Ação | Docker Compose (atual) | Docksal (antigo) |
|---|---|---|
| Inicializar projeto | `make init` | `fin project start` |
| Reiniciar do zero | `make reset` | `fin project remove && fin project start` |
| Parar containers | `docker compose stop` | `fin stop` |
| Remover containers | `docker compose down` | `fin rm` |
| Abrir terminal no cli | `docker compose exec cli bash` | `fin bash cli` |
| Executar comando no cli | `make exec CMD="<comando>"` | `fin exec <comando>` |

### Dependências Python

| Ação | Comando |
|---|---|
| Instalar dependências | `docker compose exec cli uv sync` |
| Adicionar pacote | `docker compose exec cli uv add <pacote>` |

### Banco de dados e migrações

| Ação | Comando |
|---|---|
| Inicializar alembic | `docker compose exec cli uv run alembic init alembic` |
| Gerar migração | `docker compose exec cli uv run alembic revision --autogenerate -m "<mensagem>"` |
| Aplicar migrações | `docker compose exec cli uv run alembic upgrade head` |
| Executar ingest | `docker compose exec cli uv run app/ingest.py` |


### Iniciar DB + migração
- docker compose exec cli uv run alembic upgrade head                                                                         
- docker compose exec cli uv run alembic revision --autogenerate -m "card_migration"
