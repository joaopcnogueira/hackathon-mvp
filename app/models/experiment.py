"""
Model para armazenar informações dos experimentos de treinamento.
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
import enum

from app.database import Base


class ProblemType(enum.Enum):
    """Tipo de problema de machine learning."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ExperimentStatus(enum.Enum):
    """Status do experimento."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"


class Experiment(Base):
    """
    Representa um experimento de treinamento de modelos.

    Um experimento contém as configurações de treinamento,
    referência ao dataset, e os modelos treinados.
    """
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)

    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    dataset = relationship("Dataset")

    target_column = Column(String(255), nullable=False)
    feature_columns = Column(JSON, nullable=False)

    problem_type = Column(Enum(ProblemType), nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING)

    # Informações do pré-processamento
    preprocessing_info = Column(JSON, nullable=True)

    # Caminho do dataset processado
    processed_data_path = Column(String(500), nullable=True)

    # Informações do pipeline de pré-processamento para usar em predições
    preprocessing_pipeline_path = Column(String(500), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    trained_models = relationship("TrainedModel", back_populates="experiment")
