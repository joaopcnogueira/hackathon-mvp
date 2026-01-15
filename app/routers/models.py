"""
Rotas para gerenciamento de modelos treinados.
"""
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.models.trained_model import TrainedModel

router = APIRouter(prefix="/api/models", tags=["models"])


class DeployConfig(BaseModel):
    """Configuração de deploy do modelo."""
    deploy_api: bool = True
    deploy_batch: bool = True


@router.get("/{model_id}")
def get_model(model_id: int, db: Session = Depends(get_db)):
    """
    Obtém informações de um modelo específico.

    Parâmetros:
        model_id: ID do modelo.

    Retorna:
        Informações do modelo incluindo métricas e status de deploy.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    return {
        "id": model.id,
        "experiment_id": model.experiment_id,
        "algorithm_name": model.algorithm_name,
        "metrics": model.metrics,
        "primary_metric_value": model.primary_metric_value,
        "rank": model.rank,
        "is_deployed_api": model.is_deployed_api,
        "is_deployed_batch": model.is_deployed_batch,
        "created_at": model.created_at.isoformat()
    }


@router.post("/{model_id}/deploy")
def deploy_model(
    model_id: int,
    config: DeployConfig,
    db: Session = Depends(get_db)
):
    """
    Configura o deploy de um modelo.

    Parâmetros:
        model_id: ID do modelo.
        config: Configuração de deploy (API, batch ou ambos).

    Retorna:
        Confirmação do deploy com endpoints disponíveis.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    # Remove deploy de outros modelos do mesmo experimento
    other_models = (
        db.query(TrainedModel)
        .filter(TrainedModel.experiment_id == model.experiment_id)
        .filter(TrainedModel.id != model_id)
        .all()
    )

    for other in other_models:
        if config.deploy_api:
            other.is_deployed_api = False
        if config.deploy_batch:
            other.is_deployed_batch = False

    # Configura o deploy do modelo selecionado
    model.is_deployed_api = config.deploy_api
    model.is_deployed_batch = config.deploy_batch

    db.commit()

    response = {
        "message": "Deploy configurado com sucesso",
        "model_id": model.id,
        "algorithm": model.algorithm_name
    }

    if config.deploy_api:
        response["api_endpoint"] = f"/api/v1/predict/{model.id}"
        response["api_docs"] = "/docs#/predictions"

    if config.deploy_batch:
        response["batch_endpoint"] = f"/api/models/{model.id}/predict-batch"

    return response


@router.get("/{model_id}/download")
def download_model(model_id: int, db: Session = Depends(get_db)):
    """
    Faz download do arquivo do modelo (.pkl).

    Parâmetros:
        model_id: ID do modelo.

    Retorna:
        Arquivo do modelo para download.
    """
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    model_path = Path(model.model_path)

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Arquivo do modelo não encontrado")

    filename = f"{model.algorithm_name.lower().replace(' ', '_')}_{model.id}.pkl"

    return FileResponse(
        path=model_path,
        filename=filename,
        media_type="application/octet-stream"
    )
