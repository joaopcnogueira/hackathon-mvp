"""
Rotas para gerenciamento de datasets.
"""
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import UPLOADS_DIR, ALLOWED_EXTENSIONS
from app.models.dataset import Dataset
from app.models.experiment import Experiment
from app.models.trained_model import TrainedModel
from app.services.preprocessing import PreprocessingService

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/upload")
async def upload_dataset(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Faz upload de um dataset (CSV ou Excel).

    Parâmetros:
        name: Nome descritivo do dataset.
        file: Arquivo CSV, XLSX ou XLS.

    Retorna:
        Informações do dataset criado com preview das colunas.
    """
    # Valida extensão
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado. Use: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Salva o arquivo
    filepath = UPLOADS_DIR / file.filename
    content = await file.read()

    with open(filepath, "wb") as f:
        f.write(content)

    # Lê o arquivo para análise
    try:
        if file_ext == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Erro ao ler arquivo: {str(e)}")

    # Analisa o dataset
    preprocessor = PreprocessingService()
    analysis = preprocessor.analyze_dataframe(df)

    # Cria registro no banco
    dataset = Dataset(
        name=name,
        filename=file.filename,
        filepath=str(filepath),
        num_rows=analysis["num_rows"],
        num_columns=analysis["num_columns"],
        columns_info=analysis["columns_info"]
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "id": dataset.id,
        "name": dataset.name,
        "filename": dataset.filename,
        "num_rows": dataset.num_rows,
        "num_columns": dataset.num_columns,
        "columns_info": dataset.columns_info
    }


@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    """
    Lista todos os datasets cadastrados.

    Retorna:
        Lista de datasets com informações básicas.
    """
    datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
    return [
        {
            "id": ds.id,
            "name": ds.name,
            "filename": ds.filename,
            "num_rows": ds.num_rows,
            "num_columns": ds.num_columns,
            "created_at": ds.created_at.isoformat()
        }
        for ds in datasets
    ]


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Obtém detalhes de um dataset específico.

    Parâmetros:
        dataset_id: ID do dataset.

    Retorna:
        Informações completas do dataset incluindo preview.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")

    # Lê preview dos dados
    try:
        filepath = Path(dataset.filepath)
        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath, nrows=10)
        else:
            df = pd.read_excel(filepath, nrows=10)

        # Converte para dict e trata valores NaN (não são JSON serializáveis)
        preview_data = df.to_dict(orient="records")
        for record in preview_data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
    except Exception:
        preview_data = []

    # Sanitiza columns_info para garantir que não há NaN
    columns_info = dataset.columns_info
    if columns_info:
        for col_info in columns_info:
            for key, value in col_info.items():
                if isinstance(value, float) and pd.isna(value):
                    col_info[key] = None

    return {
        "id": dataset.id,
        "name": dataset.name,
        "filename": dataset.filename,
        "num_rows": dataset.num_rows,
        "num_columns": dataset.num_columns,
        "columns_info": columns_info,
        "preview_data": preview_data,
        "created_at": dataset.created_at.isoformat()
    }


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """
    Exclui um dataset e todos os experimentos/modelos associados (cascade).

    Parâmetros:
        dataset_id: ID do dataset a ser excluído.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")

    # Busca experimentos vinculados ao dataset
    experiments = db.query(Experiment).filter(Experiment.dataset_id == dataset_id).all()

    # Remove cada experimento e seus modelos (cascade manual)
    for experiment in experiments:
        # Remove artefatos físicos dos modelos treinados
        trained_models = db.query(TrainedModel).filter(
            TrainedModel.experiment_id == experiment.id
        ).all()

        for model in trained_models:
            try:
                Path(model.model_path).unlink(missing_ok=True)
            except Exception:
                pass
            db.delete(model)

        # Remove artefatos do experimento
        if experiment.preprocessing_pipeline_path:
            try:
                Path(experiment.preprocessing_pipeline_path).unlink(missing_ok=True)
            except Exception:
                pass

        db.delete(experiment)

    # Remove arquivo físico do dataset
    try:
        filepath = Path(dataset.filepath)
        filepath.unlink(missing_ok=True)
    except Exception:
        pass

    db.delete(dataset)
    db.commit()

    return {"message": "Dataset excluído com sucesso"}
