"""
Model para armazenar informações dos modelos treinados.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship

from app.database import Base


class TrainedModel(Base):
    """
    Representa um modelo de machine learning treinado.

    Armazena o algoritmo utilizado, métricas de avaliação,
    caminho do artefato e configurações de deploy.
    """
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)

    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    experiment = relationship("Experiment", back_populates="trained_models")

    algorithm_name = Column(String(100), nullable=False)
    model_path = Column(String(500), nullable=False)

    # Métricas de avaliação
    metrics = Column(JSON, nullable=False)

    # Métrica principal para ranking (AUC para classificação, RMSE para regressão)
    primary_metric_value = Column(Float, nullable=False)

    # Ranking do modelo dentro do experimento (1 = melhor)
    rank = Column(Integer, nullable=True)

    # Configurações de deploy
    is_deployed_api = Column(Boolean, default=False)
    is_deployed_batch = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)
