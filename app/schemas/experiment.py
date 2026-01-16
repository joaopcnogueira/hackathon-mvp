"""
Schemas Pydantic para validação de dados de experimentos.
"""
from datetime import datetime
from typing import Any
from pydantic import BaseModel


class ExperimentCreate(BaseModel):
    """Schema para criação de experimento."""
    name: str
    dataset_id: int
    target_column: str
    feature_columns: list[str]
    problem_type: str  # "classification", "regression" ou "time_series"
    date_column: str | None = None  # Obrigatório para time_series
    id_column: str | None = None  # Opcional: identificador de múltiplas séries


class TrainedModelResponse(BaseModel):
    """Resposta com informações de um modelo treinado."""
    id: int
    algorithm_name: str
    metrics: dict[str, float]
    primary_metric_value: float
    rank: int | None
    is_deployed_api: bool
    is_deployed_batch: bool
    created_at: datetime

    class Config:
        from_attributes = True


class ExperimentResponse(BaseModel):
    """Resposta com informações completas do experimento."""
    id: int
    name: str
    dataset_id: int
    target_column: str
    feature_columns: list[str]
    problem_type: str
    status: str
    preprocessing_info: dict[str, Any] | None
    date_column: str | None = None
    id_column: str | None = None
    created_at: datetime
    trained_models: list[TrainedModelResponse] = []

    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    """Request para predição via API."""
    data: list[dict[str, Any]]


class PredictionResponse(BaseModel):
    """Resposta de predição."""
    predictions: list[Any]
    model_id: int
    algorithm: str
