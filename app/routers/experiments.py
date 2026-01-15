"""
Rotas para gerenciamento de experimentos de treinamento.

O pipeline de treinamento segue a ordem correta para evitar data leakage:
1. Split treino/teste nos dados brutos
2. Fit do pré-processamento apenas nos dados de treino
3. Transform dos dados de teste usando parâmetros do treino
4. Treinamento e avaliação
"""
import pandas as pd
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from app.database import get_db
from app.config import ARTIFACTS_DIR
from app.models.dataset import Dataset
from app.models.experiment import Experiment, ProblemType, ExperimentStatus
from app.models.trained_model import TrainedModel
from app.services.preprocessing import PreprocessingService
from app.services.training import TrainingService

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    """Schema para criação de experimento."""
    name: str
    dataset_id: int
    target_column: str
    feature_columns: list[str]
    problem_type: str


def run_experiment_pipeline(experiment_id: int, db_url: str):
    """
    Executa o pipeline completo de pré-processamento e treinamento.
    Função executada em background.

    Parâmetros:
        experiment_id: ID do experimento a ser processado.
        db_url: URL de conexão com o banco de dados.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
        if not experiment:
            return

        # Atualiza status para preprocessing
        experiment.status = ExperimentStatus.PREPROCESSING
        db.commit()

        # Carrega o dataset
        dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
        filepath = Path(dataset.filepath)

        if filepath.suffix == ".csv":
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        # Separa features e target
        X = df[experiment.feature_columns].copy()
        y = df[experiment.target_column].copy()

        # Remove linhas onde o target é nulo
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        is_classification = experiment.problem_type == ProblemType.CLASSIFICATION

        # IMPORTANTE: Split ANTES do pré-processamento para evitar data leakage
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pré-processamento: fit apenas nos dados de TREINO
        preprocessor = PreprocessingService()
        X_train, preprocessing_info = preprocessor.fit_transform(X_train_raw, y_train)

        # Transforma dados de teste usando parâmetros do treino (sem vazamento)
        X_test = preprocessor.transform_test(X_test_raw)

        # Salva pipeline de pré-processamento
        pipeline_path = ARTIFACTS_DIR / f"pipeline_{experiment.id}.pkl"
        preprocessor.save_pipeline(pipeline_path)

        experiment.preprocessing_info = preprocessing_info
        experiment.preprocessing_pipeline_path = str(pipeline_path)

        # Atualiza status para training
        experiment.status = ExperimentStatus.TRAINING
        db.commit()

        # Executa treinamento com dados já divididos
        trainer = TrainingService()
        results = trainer.train_all_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            is_classification=is_classification,
            artifacts_dir=ARTIFACTS_DIR,
            experiment_id=experiment.id
        )

        # Salva os modelos treinados no banco
        for result in results:
            trained_model = TrainedModel(
                experiment_id=experiment.id,
                algorithm_name=result["algorithm_name"],
                model_path=result["model_path"],
                metrics=result["metrics"],
                primary_metric_value=result["primary_metric_value"],
                rank=result["rank"]
            )
            db.add(trained_model)

        # Atualiza status para completed
        experiment.status = ExperimentStatus.COMPLETED
        db.commit()

    except Exception as e:
        experiment.status = ExperimentStatus.FAILED
        experiment.preprocessing_info = {"error": str(e)}
        db.commit()
        raise

    finally:
        db.close()


@router.post("")
def create_experiment(
    data: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Cria um novo experimento e inicia o pipeline de treinamento.

    Parâmetros:
        data: Configurações do experimento.

    Retorna:
        Informações do experimento criado.
    """
    # Valida dataset
    dataset = db.query(Dataset).filter(Dataset.id == data.dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset não encontrado")

    # Valida tipo de problema
    try:
        problem_type = ProblemType(data.problem_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Tipo de problema inválido. Use 'classification' ou 'regression'"
        )

    # Cria o experimento
    experiment = Experiment(
        name=data.name,
        dataset_id=data.dataset_id,
        target_column=data.target_column,
        feature_columns=data.feature_columns,
        problem_type=problem_type,
        status=ExperimentStatus.PENDING
    )

    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    # Inicia pipeline em background
    from app.config import DATABASE_URL
    background_tasks.add_task(run_experiment_pipeline, experiment.id, DATABASE_URL)

    return {
        "id": experiment.id,
        "name": experiment.name,
        "status": experiment.status.value,
        "message": "Experimento criado. Processamento iniciado em background."
    }


@router.get("")
def list_experiments(db: Session = Depends(get_db)):
    """
    Lista todos os experimentos.

    Retorna:
        Lista de experimentos com informações básicas.
    """
    experiments = (
        db.query(Experiment)
        .order_by(Experiment.created_at.desc())
        .all()
    )

    return [
        {
            "id": exp.id,
            "name": exp.name,
            "dataset_id": exp.dataset_id,
            "dataset_name": exp.dataset.name if exp.dataset else None,
            "problem_type": exp.problem_type.value,
            "status": exp.status.value,
            "created_at": exp.created_at.isoformat()
        }
        for exp in experiments
    ]


@router.get("/{experiment_id}")
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """
    Obtém detalhes de um experimento específico.

    Parâmetros:
        experiment_id: ID do experimento.

    Retorna:
        Informações completas do experimento incluindo modelos treinados.
    """
    experiment = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id)
        .first()
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    # Ordena modelos por rank
    trained_models = sorted(
        experiment.trained_models,
        key=lambda m: m.rank if m.rank else 999
    )

    return {
        "id": experiment.id,
        "name": experiment.name,
        "dataset_id": experiment.dataset_id,
        "dataset_name": experiment.dataset.name if experiment.dataset else None,
        "target_column": experiment.target_column,
        "feature_columns": experiment.feature_columns,
        "problem_type": experiment.problem_type.value,
        "status": experiment.status.value,
        "preprocessing_info": experiment.preprocessing_info,
        "created_at": experiment.created_at.isoformat(),
        "trained_models": [
            {
                "id": m.id,
                "algorithm_name": m.algorithm_name,
                "metrics": m.metrics,
                "primary_metric_value": m.primary_metric_value,
                "rank": m.rank,
                "is_deployed_api": m.is_deployed_api,
                "is_deployed_batch": m.is_deployed_batch
            }
            for m in trained_models
        ]
    }


@router.delete("/{experiment_id}")
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    """
    Exclui um experimento e seus artefatos.

    Parâmetros:
        experiment_id: ID do experimento a ser excluído.
    """
    experiment = (
        db.query(Experiment)
        .filter(Experiment.id == experiment_id)
        .first()
    )

    if not experiment:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    # Remove artefatos físicos dos modelos
    for model in experiment.trained_models:
        try:
            Path(model.model_path).unlink(missing_ok=True)
        except Exception:
            pass

    if experiment.preprocessing_pipeline_path:
        try:
            Path(experiment.preprocessing_pipeline_path).unlink(missing_ok=True)
        except Exception:
            pass

    # Remove modelos treinados do banco primeiro (FK constraint)
    for model in experiment.trained_models:
        db.delete(model)

    # Remove o experimento
    db.delete(experiment)
    db.commit()

    return {"message": "Experimento excluído com sucesso"}
