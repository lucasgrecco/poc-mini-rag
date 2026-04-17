from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import Integer, String, Text
from pgvector.sqlalchemy import Vector

# Importaremos o vetor e o array depois

class Base(DeclarativeBase):
    pass


class Card(Base):
    __tablename__ = "cards"
    id = mapped_column(Integer, primary_key=True)
    level = mapped_column(Integer)
    name = mapped_column(String)
    atk = mapped_column(Integer)
    def_ = mapped_column(Integer)
    english_attribute = mapped_column(String)
    content = mapped_column(Text)
    embedding = mapped_column(Vector(1536))

