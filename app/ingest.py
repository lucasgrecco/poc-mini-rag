import json
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Card  # Importamos nosso molde

# Configuração do Banco de Dados
DATABASE_URL = "postgresql+psycopg2://admin:admin@db/rag_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
def clean_int(value):
    """Converte para inteiro ou retorna None se não for um número."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

def process_jsons():
    json_path = Path("jsons")
    # Percorre todos os arquivos .json na pasta
    for json_file in json_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            cards_data = json.load(f)
            for card_json in cards_data:
                # 1. Limpeza dos dados numéricos
                atk = clean_int(card_json.get('atk'))
                defense = clean_int(card_json.get('def'))

                props = card_json.get('properties', {})
                level = clean_int(props.get('level'))

                # 2. Montagem do 'content' (apenas com o que existe)
                parts = [f"Name: {card_json.get('Name')}"]
                if level: parts.append(f"Level: {level}")
                if atk is not None: parts.append(f"ATK: {atk}")
                # ... e assim por diante ...

                full_content = " | ".join(parts)

                # 3. Criação do objeto Card
                new_card = Card(
                    name=card_json.get('Name'),
                    level=level,
                    atk=atk,
                    def_=defense,
                    english_attribute=card_json.get('englishAttribute'),
                    content=full_content
                    # embedding = ??? (Onde a mágica acontece)
                )

                session.add(new_card)

            session.commit()  # Salva as cartas do arquivo no banco de dados