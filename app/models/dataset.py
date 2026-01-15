"""
Model para armazenar informações dos datasets uploadados.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON

from app.database import Base


class Dataset(Base):
    """
    Representa um dataset carregado na plataforma.

    Armazena metadados do arquivo, informações sobre colunas,
    tipos de dados e estatísticas básicas.
    """
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)

    num_rows = Column(Integer, nullable=False)
    num_columns = Column(Integer, nullable=False)

    # Armazena informações das colunas: nome, tipo, valores únicos, nulos
    columns_info = Column(JSON, nullable=False)

    # Estatísticas básicas do dataset
    statistics = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
