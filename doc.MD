# Yu-Gi-Oh! Micro RAG - Documentação da Fundação ⚙️

Este documento registra a configuração da infraestrutura e do banco de dados vetorial.

## 🏗️ Infraestrutura (Docksal)
O ambiente roda em containers Docker gerenciados pelo Docksal:
- **db**: PostgreSQL com a extensão `pgvector` para busca semântica.
- **cli**: Ambiente Python 3.12-slim isolado.

## 🐍 Ambiente Python (UV)
Utilizamos o `uv` para gerenciamento de pacotes e ambientes virtuais:
- `sqlalchemy`: ORM para mapeamento de dados.
- `alembic`: Controle de versões (migrações) do banco de dados.
- `psycopg2-binary`: Driver de conexão com PostgreSQL.
- `pgvector`: Suporte a vetores no SQLAlchemy.

## 🗄️ Modelo de Dados (SQLAlchemy)
A tabela principal chama-se `cards` e possui a seguinte estrutura:
- `id`: Integer (Primary Key)
- `name`: String
- `level`: Integer
- `atk`: Integer
- `def_`: Integer
- `english_attribute`: String
- `content`: Text (Texto concatenado para a IA ler)
- `embedding`: Vector(1536) (Vetor gerado pela OpenAI)

## 📜 Comandos Principais
1. **Inicializar banco**: `uv run alembic init alembic`
2. **Gerar migração**: `uv run alembic revision --autogenerate -m "create_cards_table"`
3. **Aplicar tabelas**: `uv run alembic upgrade head`


Com certeza! Vamos organizar tudo o que fizemos na nossa ⚙️ Fundação para que você tenha esse histórico à mão.

Aqui estão os comandos, divididos por objetivo:
🚢 Infraestrutura e Containers

    fin project start: Inicia os serviços (PostgreSQL e Python) definidos no docksal.yml.

    fin bash cli: Abre o terminal dentro do container de aplicação onde rodamos o Python.

    fin project remove: Para e remove os containers do projeto.

📦 Gerenciamento de Pacotes (dentro do container)

    pip install uv: Instala o gerenciador de pacotes ultra rápido.

    uv init: Inicializa a estrutura básica do projeto Python.

    uv add sqlalchemy alembic psycopg2-binary pgvector: Adiciona as bibliotecas de banco de dados, migrações e vetores ao projeto.

🗄️ Banco de Dados e Migrações

    uv run alembic init alembic: Cria a pasta de configuração das migrações.

    uv run alembic revision --autogenerate -m "create_cards_table": Gera o script Python que descreve a criação da tabela no banco.

    uv run alembic upgrade head: Executa o script no PostgreSQL, criando as tabelas de fato.