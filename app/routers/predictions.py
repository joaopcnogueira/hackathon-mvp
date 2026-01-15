"""
Rotas para predições via API e batch.
"""
from pathlib import Path
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd

from app.database import get_db
from app.config import UPLOADS_DIR
from app.models.trained_model import TrainedModel
from app.models.experiment import Experiment, ProblemType
from app.services.prediction import prediction_service

router = APIRouter(tags=["predictions"])


class PredictionRequest(BaseModel):
    """Request para predição via API."""
    data: list[dict[str, Any]]
    return_probabilities: bool = False


class PredictionResponse(BaseModel):
    """Resposta de predição."""
    predictions: list[Any]
    probabilities: list[dict[str, float]] | None = None
    model_id: int
    algorithm: str


@router.post("/api/v1/predict/{model_id}", response_model=PredictionResponse)
def predict(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Faz predições usando um modelo deployado via API.

    Parâmetros:
        model_id: ID do modelo a ser usado.
        request: Dados para predição no formato {data: [{col1: val1, ...}, ...]}.

    Retorna:
        Lista de predições para cada registro.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    if not model.is_deployed_api:
        raise HTTPException(
            status_code=403,
            detail="Este modelo não está deployado para uso via API"
        )

    experiment = model.experiment

    if not experiment.preprocessing_pipeline_path:
        raise HTTPException(
            status_code=500,
            detail="Pipeline de pré-processamento não encontrado"
        )

    try:
        predictions = prediction_service.predict(
            model_path=model.model_path,
            preprocessing_pipeline_path=experiment.preprocessing_pipeline_path,
            feature_columns=experiment.feature_columns,
            data=request.data
        )

        # Retorna probabilidades se solicitado e for classificação
        probabilities = None
        if request.return_probabilities and experiment.problem_type == ProblemType.CLASSIFICATION:
            probabilities = prediction_service.predict_proba(
                model_path=model.model_path,
                preprocessing_pipeline_path=experiment.preprocessing_pipeline_path,
                feature_columns=experiment.feature_columns,
                data=request.data
            )

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_id=model.id,
            algorithm=model.algorithm_name
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@router.post("/api/v1/predict/{model_id}/proba")
def predict_proba(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Faz predições com probabilidades (para modelos de classificação).

    Parâmetros:
        model_id: ID do modelo.
        request: Dados para predição.

    Retorna:
        Probabilidades para cada classe.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    if not model.is_deployed_api:
        raise HTTPException(
            status_code=403,
            detail="Este modelo não está deployado para uso via API"
        )

    experiment = model.experiment

    if experiment.problem_type != ProblemType.CLASSIFICATION:
        raise HTTPException(
            status_code=400,
            detail="Probabilidades disponíveis apenas para modelos de classificação"
        )

    try:
        probabilities = prediction_service.predict_proba(
            model_path=model.model_path,
            preprocessing_pipeline_path=experiment.preprocessing_pipeline_path,
            feature_columns=experiment.feature_columns,
            data=request.data
        )

        return {
            "probabilities": probabilities,
            "model_id": model.id,
            "algorithm": model.algorithm_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@router.post("/api/models/{model_id}/predict-batch")
async def predict_batch(
    model_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Faz predições em lote a partir de um arquivo.

    Parâmetros:
        model_id: ID do modelo.
        file: Arquivo CSV ou Excel com os dados.

    Retorna:
        ID do resultado e preview das predições.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    if not model.is_deployed_batch:
        raise HTTPException(
            status_code=403,
            detail="Este modelo não está configurado para predição em batch"
        )

    experiment = model.experiment

    # Salva arquivo temporário
    file_ext = Path(file.filename).suffix.lower()
    temp_path = UPLOADS_DIR / f"batch_input_{model_id}{file_ext}"

    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    # Valida se o arquivo tem as colunas esperadas
    try:
        if file_ext == ".csv":
            df_check = pd.read_csv(temp_path, nrows=1)
        else:
            df_check = pd.read_excel(temp_path, nrows=1)

        # Verifica colunas (case-insensitive)
        uploaded_columns_lower = {col.lower().strip() for col in df_check.columns}
        expected_columns_lower = {col.lower().strip() for col in experiment.feature_columns}

        matched_columns = uploaded_columns_lower & expected_columns_lower
        match_ratio = len(matched_columns) / len(expected_columns_lower) if expected_columns_lower else 0

        # Exige pelo menos 50% das colunas esperadas
        if match_ratio < 0.5:
            temp_path.unlink(missing_ok=True)
            missing = [col for col in experiment.feature_columns
                       if col.lower().strip() not in uploaded_columns_lower]
            raise HTTPException(
                status_code=400,
                detail=f"Arquivo incompatível. Colunas esperadas não encontradas: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}"
            )
    except HTTPException:
        raise
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {str(e)}")

    try:
        is_classification = experiment.problem_type == ProblemType.CLASSIFICATION

        result_id, df = prediction_service.predict_batch(
            model_path=model.model_path,
            preprocessing_pipeline_path=experiment.preprocessing_pipeline_path,
            feature_columns=experiment.feature_columns,
            input_file=temp_path,
            is_classification=is_classification
        )

        # Remove arquivo temporário
        temp_path.unlink(missing_ok=True)

        # Usa o preview do serviço que já trata valores NaN
        batch_result = prediction_service.get_batch_result(result_id)

        return {
            "result_id": result_id,
            "num_predictions": len(df),
            "preview": batch_result["preview"] if batch_result else [],
            "download_url": f"/api/batch-results/{result_id}/download"
        }

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@router.get("/api/batch-results/{result_id}/download")
def download_batch_result(result_id: str):
    """
    Faz download do resultado de predição em batch.

    Parâmetros:
        result_id: ID do resultado.

    Retorna:
        Arquivo CSV com as predições.
    """
    result_path = prediction_service.get_batch_result_path(result_id)

    if not result_path or not result_path.exists():
        raise HTTPException(status_code=404, detail="Resultado não encontrado")

    return FileResponse(
        path=result_path,
        filename=f"predictions_{result_id}.csv",
        media_type="text/csv"
    )
