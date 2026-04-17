import os

from langsmith import traceable
from sqlalchemy import create_engine, text
from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import langsmith

# 1. Configurações
DATABASE_URL = "postgresql+psycopg2://admin:admin@db/rag_db"
engine = create_engine(DATABASE_URL)
client = wrap_openai(OpenAI(api_key=os.environ.get("OPENAI_API_KEY")))


@traceable(run_type="tool", name="Retrieve embedding")
def get_embedding(query_text):
    client_ls = langsmith.Client()
    print("LangSmith conectado:", client_ls.api_url)
    print("Tracing ativo:", langsmith.utils.tracing_is_enabled())
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

@traceable(name="Chat Pipeline")
def realizar_busca():
    print("\n--- 🔍 Yu-Gi-Oh! Semantic Search ---\n")
    pergunta = input("O que você procura? ")

    print("\nGerando vetor da pergunta... 🧠")
    vetor_pergunta = get_embedding(pergunta)

    print("Buscando no banco de dados... 🗄️")
    # O pgvector lê a lista de números nativamente se a passarmos como string: '[0.1, 0.2, ...]'
    vetor_str = str(vetor_pergunta)

    query = text("""
        SELECT name, content 
        FROM cards 
        ORDER BY embedding <=> :vetor 
        LIMIT :limite
    """)

    # 3. Execução e Exibição
    with engine.connect() as conn:
        resultados = conn.execute(query, {"vetor": vetor_str, "limite": 10}).fetchall()

    print("\n🏆 Resultados Encontrados:\n")
    for i, row in enumerate(resultados, 1):
        print(f"{i}. {row.name}")
        print(f"   {row.content}")
        print("-" * 60)

# 4. Ponto de partida
if __name__ == "__main__":
    realizar_busca()