"""Interactive semantic search over Yu-Gi-Oh! cards using RAG.

Accepts natural language queries, retrieves the most semantically similar
cards via pgvector cosine distance, and generates a natural language answer
using an LLM.
"""

import argparse
import logging

from langsmith import traceable
from sqlalchemy import create_engine, text
from openai import OpenAI
from langsmith.wrappers import wrap_openai

from app.config import DATABASE_URL, CHAT_MODEL, SEARCH_LIMIT
from app.embeddings import get_embedding

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL)
_client = wrap_openai(OpenAI())


@traceable(run_type="tool", name="Chat with model")
def get_model_answer(question: str, results_db: list) -> str:
    """Generate a natural language answer from retrieved card results.

    Args:
        question: The user's original query.
        results_db: List of rows from the pgvector similarity query.

    Returns:
        A natural language answer grounded in the retrieved cards.
    """
    card_text = []
    for row in results_db:
        card_text.append(f"- {row.name}: {row.content}")

    context = "\n".join(card_text)

    prompt = f"""You are a Yu-Gi-Oh! card expert.
The user asked: "{question}"

Below are cards retrieved from the database. You MUST always describe at least the most relevant cards from this list — never refuse to answer when cards are present.
If the user's query is generic (e.g., "any card", "a card"), simply list and describe the top cards found.
For spell and trap cards, always describe their effects using the card description. DO NOT IGNORE THE CONTEXT PROVIDED.
Only if the list below is COMPLETELY EMPTY (0 cards), say: "I couldn't find relevant cards for that query."
Do not invent effects, stats, or lore that are not in the cards below.

CARDS FOUND:
{context}
"""

    logger.debug("=== PROMPT SENT TO MODEL ===\n%s\n=== END PROMPT ===", prompt)

    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,  # Low temperature keeps responses factual
    )

    return response.choices[0].message.content


@traceable(name="Chat Pipeline")
def search_card(query: str) -> None:
    """Execute a single semantic search query.

    Args:
        query: The user's natural language search query.
    """
    logger.info("Generating query embedding...")
    query_vector = get_embedding(query)

    logger.info("Searching database with pgvector...")
    vector_str = str(query_vector)

    sql = text("""
        SELECT name, content
        FROM cards
        ORDER BY embedding <=> :vector
        LIMIT :limit
    """)

    with engine.connect() as conn:
        results = conn.execute(
            sql, {"vector": vector_str, "limit": SEARCH_LIMIT}
        ).fetchall()
        model_answer = get_model_answer(query, results)

    print(f"\n🏆 Top {len(results)} Results:\n")
    for i, row in enumerate(results, 1):
        print(f"{i}. {row.name}")
        print(f"   {row.content}")
        print("-" * 60)

    print(f"\n💡 AI Answer:\n{model_answer}\n")


def interactive_loop() -> None:
    """Run the search in an interactive REPL-style loop."""
    print("\n--- 🔍 Yu-Gi-Oh! Semantic Search ---\n")
    print("Type your query or 'quit' / 'exit' to leave.\n")

    while True:
        try:
            question = input("Query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit"):
            print("Goodbye! 👋")
            break

        try:
            search_card(question)
        except Exception:
            logger.exception("Search failed for query: %s", question)
            print("An error occurred. Please try again.\n")


def main() -> None:
    """Entry point for the search CLI."""
    parser = argparse.ArgumentParser(
        description="Semantic search over Yu-Gi-Oh! cards using RAG."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.query:
        search_card(args.query)
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
