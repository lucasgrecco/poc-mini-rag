import json
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Card
from openai import OpenAI

# Configuração do Banco de Dados
DATABASE_URL = "postgresql+psycopg2://admin:admin@db/rag_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def clean_int(value):
    """Converte para inteiro ou retorna None se não for um número."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

# O Python busca automaticamente a variável do sistema

def get_embedding(text):
    # Esta função será responsável por converter o texto em números
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def process_jsons():
    json_path = Path("jsons")
    # Percorre todos os arquivos .json na pasta
    for json_file in json_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            card_json = json.load(f)

            atk = clean_int(card_json.get('atk'))
            defense = clean_int(card_json.get('def'))
            level = clean_int(card_json.get('level'))

            parts = [f"Name: {card_json.get('name')}"]

            attribute = card_json.get('englishAttribute')
            if attribute: parts.append(f"Attribute: {attribute}")

            if level: parts.append(f"Level: {level}")
            if atk is not None: parts.append(f"ATK: {atk}")
            if defense is not None: parts.append(f"DEF: {defense}")

            full_content = " | ".join(parts)
            effect = card_json.get('effectText', '')
            if effect:
                full_content += f" | Description: {effect}"

            print(f"Processando: {card_json.get('name')}")
            vector = get_embedding(full_content)

            new_card = Card(
                name=card_json.get('name'),
                level=level,
                atk=atk,
                def_=defense,
                english_attribute=attribute,
                content=full_content,
                embedding=vector
            )

            session.add(new_card)

    session.commit()

# ... fim do arquivo ...

if __name__ == "__main__":
    print("Iniciando a ingestão de cartas... ⚙️")
    process_jsons()
    print("Ingestão concluída com sucesso! 🃏")