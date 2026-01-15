"""
Schemas Pydantic para validação de dados de datasets.
"""
from datetime import datetime
from typing import Any
from pydantic import BaseModel


class ColumnInfo(BaseModel):
    """Informações de uma coluna do dataset."""
    name: str
    dtype: str
    unique_count: int
    null_count: int
    null_percentage: float
    sample_values: list[Any]


class DatasetCreate(BaseModel):
    """Schema para criação de dataset via API."""
    name: str


class DatasetPreview(BaseModel):
    """Preview do dataset com primeiras linhas."""
    columns: list[ColumnInfo]
    preview_data: list[dict[str, Any]]
    num_rows: int
    num_columns: int


class DatasetResponse(BaseModel):
    """Resposta com informações completas do dataset."""
    id: int
    name: str
    filename: str
    num_rows: int
    num_columns: int
    columns_info: list[ColumnInfo]
    created_at: datetime

    class Config:
        from_attributes = True
