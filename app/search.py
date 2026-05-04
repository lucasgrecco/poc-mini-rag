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


@traceable(run_type="tool", name="Chat with model")
def get_model_answer(question, results_db):
    card_text = []
    for row in results_db:
        card_text.append(f"- {row.name}: {row.content}")

    context = "\n".join(card_text)

    prompt = f"""Você é um especialista em cartas de Yu-Gi-Oh!.
Sua missão é responder à pergunta do usuário utilizando APENAS as cartas fornecidas no contexto abaixo.
Se a resposta não puder ser encontrada nas cartas abaixo, diga simplesmente: "Não encontrei cartas que correspondam a isso na minha base."
Não invente informações ou efeitos que não estejam no contexto.

CONTEXTO DE CARTAS ENCONTRADAS:
{context}
"""
    print("Gerando resposta com a IA... 💬\n")
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Rápido, barato e muito inteligente
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2  # Temperatura baixa para ele ser mais factual e menos "criativo"
    )

    return response.choices[0].message.content


@traceable(name="Chat Pipeline")
def search_card():
    print("\n--- 🔍 Yu-Gi-Oh! Semantic Search ---\n")
    question = input("O que você procura? ")

    print("\nGerando vetor da pergunta... 🧠")
    vetor_pergunta = get_embedding(question)

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
        results = conn.execute(query, {"vetor": vetor_str, "limite": 10}).fetchall()
        model_answer = get_model_answer(question, results);
    print("\n🏆 Resultados Encontrados:\n")
    for i, row in enumerate(results, 1):
        print(f"{i}. {row.name}")
        print(f"   {row.content}")
        print("-" * 60)
        print(f"Model Answer: {model_answer}")


# 4. Ponto de partida
if __name__ == "__main__":
    search_card()
