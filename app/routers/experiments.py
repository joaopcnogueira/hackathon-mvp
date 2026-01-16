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
from app.services.timeseries_preprocessing import TimeSeriesPreprocessingService
from app.services.classical_timeseries import ClassicalTimeSeriesService
from app.services.training import TrainingService

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class ExperimentCreate(BaseModel):
    """Schema para criação de experimento."""
    name: str
    dataset_id: int
    target_column: str
    feature_columns: list[str]
    problem_type: str
    date_column: str | None = None  # Obrigatório para time_series
    id_column: str | None = None  # Opcional: identificador de múltiplas séries
    forecast_horizon: int | None = None  # Períodos à frente para prever
    frequency: str | None = None  # Frequência da série (D, W, M, etc.)


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

        is_classification = experiment.problem_type == ProblemType.CLASSIFICATION
        is_time_series = experiment.problem_type == ProblemType.TIME_SERIES

        if is_time_series:
            # Pipeline específico para séries temporais
            _run_timeseries_pipeline(experiment, df, db)
        else:
            # Pipeline padrão para classificação/regressão
            _run_standard_pipeline(experiment, df, is_classification, db)

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


def _run_standard_pipeline(
    experiment: Experiment,
    df: pd.DataFrame,
    is_classification: bool,
    db
):
    """
    Executa o pipeline padrão para classificação e regressão.
    """
    # Separa features e target
    X = df[experiment.feature_columns].copy()
    y = df[experiment.target_column].copy()

    # Remove linhas onde o target é nulo
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]

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


def _run_timeseries_pipeline(
    experiment: Experiment,
    df: pd.DataFrame,
    db
):
    """
    Executa o pipeline específico para séries temporais.

    Detecta a cardinalidade do ID para decidir qual tipo de algoritmo usar:
    - Múltiplas séries (>1 ID): ML algorithms (Ridge, RF, GB)
    - Série única (≤1 ID): Algoritmos clássicos (ARIMA, ETS, Prophet)
    """
    # Detecta cardinalidade do ID
    if experiment.id_column and experiment.id_column in df.columns:
        num_unique_ids = df[experiment.id_column].nunique()
    else:
        num_unique_ids = 1

    # Define horizonte de previsão (usa valor informado ou default de 7)
    forecast_horizon = experiment.forecast_horizon or 7

    # Detecta ou usa frequência informada
    frequency = experiment.frequency

    if num_unique_ids > 1:
        # Múltiplas séries → Algoritmos ML com features
        _run_ml_timeseries_pipeline(experiment, df, db, num_unique_ids)
    else:
        # Série única → Algoritmos clássicos
        _run_classical_timeseries_pipeline(
            experiment, df, db, forecast_horizon, frequency
        )


def _run_ml_timeseries_pipeline(
    experiment: Experiment,
    df: pd.DataFrame,
    db,
    num_unique_ids: int
):
    """
    Executa o pipeline de ML para múltiplas séries temporais.

    Usa Ridge, Random Forest e Gradient Boosting com features de lag,
    rolling statistics e datetime.
    """
    # Pré-processamento de séries temporais
    ts_preprocessor = TimeSeriesPreprocessingService()

    # Aplica pré-processamento (cria features de lag, rolling, datetime)
    df_processed, preprocessing_info = ts_preprocessor.fit_transform(
        df=df,
        date_column=experiment.date_column,
        target_column=experiment.target_column,
        id_column=experiment.id_column
    )

    preprocessing_info["algorithm_type"] = "ml"
    preprocessing_info["num_unique_ids"] = num_unique_ids
    preprocessing_info["algorithms"] = ["Ridge Regression", "Random Forest", "Gradient Boosting"]

    # Salva pipeline de pré-processamento
    pipeline_path = ARTIFACTS_DIR / f"pipeline_{experiment.id}.pkl"
    ts_preprocessor.save_pipeline(pipeline_path)

    experiment.preprocessing_info = preprocessing_info
    experiment.preprocessing_pipeline_path = str(pipeline_path)

    # Atualiza status para training
    experiment.status = ExperimentStatus.TRAINING
    db.commit()

    # Split temporal (não aleatório)
    X_train, X_test, y_train, y_test = ts_preprocessor.prepare_train_test_split(
        df_processed, test_size=0.2
    )

    # Executa treinamento
    trainer = TrainingService()
    results = trainer.train_all_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        is_classification=False,
        artifacts_dir=ARTIFACTS_DIR,
        experiment_id=experiment.id,
        problem_type="time_series"
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


def _run_classical_timeseries_pipeline(
    experiment: Experiment,
    df: pd.DataFrame,
    db,
    forecast_horizon: int,
    frequency: str | None
):
    """
    Executa o pipeline de algoritmos clássicos para série única.

    Usa Auto ARIMA, Exponential Smoothing e Prophet.
    """
    # Auto-detecta frequência se não informada
    if not frequency:
        df_temp = df.copy()
        df_temp[experiment.date_column] = pd.to_datetime(df_temp[experiment.date_column])
        df_temp = df_temp.sort_values(experiment.date_column)
        detected_freq = pd.infer_freq(df_temp[experiment.date_column])
        frequency = detected_freq if detected_freq else "D"

    preprocessing_info = {
        "algorithm_type": "classical",
        "num_unique_ids": 1,
        "algorithms": ["Auto ARIMA", "Exponential Smoothing", "Prophet"],
        "forecast_horizon": forecast_horizon,
        "frequency": frequency,
        "transformations": ["Série única detectada - usando algoritmos clássicos"]
    }

    experiment.preprocessing_info = preprocessing_info
    experiment.frequency = frequency

    # Atualiza status para training
    experiment.status = ExperimentStatus.TRAINING
    db.commit()

    # Treina modelos clássicos
    classical_service = ClassicalTimeSeriesService()
    results = classical_service.train_all_models(
        df=df,
        date_column=experiment.date_column,
        target_column=experiment.target_column,
        forecast_horizon=forecast_horizon,
        frequency=frequency,
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
            detail="Tipo de problema inválido. Use 'classification', 'regression' ou 'time_series'"
        )

    # Valida campos obrigatórios para séries temporais
    if problem_type == ProblemType.TIME_SERIES:
        if not data.date_column:
            raise HTTPException(
                status_code=400,
                detail="Campo 'date_column' é obrigatório para séries temporais"
            )

        # Valida se a coluna de data pode ser convertida para datetime
        filepath = Path(dataset.filepath)
        if filepath.suffix == ".csv":
            df_sample = pd.read_csv(filepath, nrows=100)
        else:
            df_sample = pd.read_excel(filepath, nrows=100)

        try:
            pd.to_datetime(df_sample[data.date_column])
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Coluna '{data.date_column}' não pode ser convertida para data. "
                       f"Verifique o formato dos valores. Erro: {str(e)}"
            )

    # Cria o experimento
    experiment = Experiment(
        name=data.name,
        dataset_id=data.dataset_id,
        target_column=data.target_column,
        feature_columns=data.feature_columns,
        problem_type=problem_type,
        date_column=data.date_column,
        id_column=data.id_column,
        forecast_horizon=data.forecast_horizon,
        frequency=data.frequency,
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
        "date_column": experiment.date_column,
        "id_column": experiment.id_column,
        "forecast_horizon": experiment.forecast_horizon,
        "frequency": experiment.frequency,
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


@router.get("/{experiment_id}/forecast-chart")
def get_forecast_chart_data(
    experiment_id: int,
    model_id: int | None = None,
    series_id: str | None = None,
    db: Session = Depends(get_db)
):
    """
    Retorna dados para o gráfico de previsão de séries temporais.

    Inclui dados históricos e previsões para o horizonte definido.

    Parâmetros:
        experiment_id: ID do experimento.
        model_id: ID do modelo a usar (se não informado, usa o melhor).
        series_id: ID da série específica (para múltiplas séries).

    Retorna:
        Dados históricos, previsões e lista de séries disponíveis.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    if experiment.problem_type != ProblemType.TIME_SERIES:
        raise HTTPException(status_code=400, detail="Experimento não é de séries temporais")

    if experiment.status != ExperimentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Experimento ainda não foi concluído")

    # Carrega o dataset
    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    filepath = Path(dataset.filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Converte coluna de data
    df[experiment.date_column] = pd.to_datetime(df[experiment.date_column])
    df = df.sort_values(experiment.date_column)

    # Identifica séries disponíveis
    available_series = []
    if experiment.id_column and experiment.id_column in df.columns:
        available_series = df[experiment.id_column].unique().tolist()

    # Filtra por série se especificado
    if series_id and experiment.id_column:
        df = df[df[experiment.id_column] == series_id]
    elif available_series:
        # Usa a primeira série como padrão
        series_id = available_series[0]
        df = df[df[experiment.id_column] == series_id]

    # Seleciona o modelo
    if model_id:
        model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    else:
        # Usa o melhor modelo (rank 1)
        model = (
            db.query(TrainedModel)
            .filter(TrainedModel.experiment_id == experiment_id)
            .order_by(TrainedModel.rank)
            .first()
        )

    if not model:
        raise HTTPException(status_code=404, detail="Modelo não encontrado")

    # Prepara dados históricos para o gráfico
    historical_data = df[[experiment.date_column, experiment.target_column]].copy()
    historical_data = historical_data.tail(60)  # Últimos 60 pontos

    # Gera previsões
    forecast_horizon = experiment.forecast_horizon or 7
    forecast_data = _generate_forecast(
        model=model,
        experiment=experiment,
        df=df,
        forecast_horizon=forecast_horizon
    )

    return {
        "historical": {
            "dates": historical_data[experiment.date_column].dt.strftime("%Y-%m-%d").tolist(),
            "values": historical_data[experiment.target_column].tolist()
        },
        "forecast": forecast_data,
        "available_series": available_series,
        "current_series": series_id,
        "model_name": model.algorithm_name,
        "model_id": model.id,
        "forecast_horizon": forecast_horizon
    }


def _generate_forecast(
    model: TrainedModel,
    experiment: Experiment,
    df: pd.DataFrame,
    forecast_horizon: int
) -> dict:
    """
    Gera previsões usando o modelo treinado.

    Parâmetros:
        model: Modelo treinado.
        experiment: Experimento.
        df: DataFrame com dados históricos.
        forecast_horizon: Número de períodos para prever.

    Retorna:
        Dicionário com datas e valores previstos.
    """
    import joblib
    import numpy as np

    algorithm_type = experiment.preprocessing_info.get("algorithm_type", "ml")
    last_date = df[experiment.date_column].max()

    # Gera datas futuras
    frequency = experiment.frequency or "D"
    freq_map = {"H": "h", "D": "D", "W": "W", "M": "MS", "Q": "QS", "Y": "YS"}
    pd_freq = freq_map.get(frequency, "D")

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_horizon,
        freq=pd_freq
    )

    if algorithm_type == "classical":
        # Usa o serviço de séries temporais clássicas
        classical_service = ClassicalTimeSeriesService()
        forecast_df = classical_service.forecast(
            model_path=model.model_path,
            algorithm_name=model.algorithm_name,
            last_data=df,
            periods=forecast_horizon
        )

        if "date" in forecast_df.columns:
            dates = forecast_df["date"].dt.strftime("%Y-%m-%d").tolist()
        else:
            dates = future_dates.strftime("%Y-%m-%d").tolist()

        values = forecast_df["forecast"].tolist()

        # Intervalos de confiança se disponíveis
        lower_bound = forecast_df.get("lower_bound", pd.Series([None] * len(values))).tolist()
        upper_bound = forecast_df.get("upper_bound", pd.Series([None] * len(values))).tolist()

    else:
        # Algoritmos ML - precisa criar features para previsão
        ts_preprocessor = TimeSeriesPreprocessingService()
        ts_preprocessor.load_pipeline(Path(experiment.preprocessing_pipeline_path))

        # Carrega o modelo (salvo diretamente, não como dicionário)
        ml_model = joblib.load(model.model_path)

        # Prepara dados para previsão iterativa
        values = []
        current_df = df.copy()

        for i in range(forecast_horizon):
            # Cria linha para próxima previsão
            next_date = future_dates[i]
            next_row = pd.DataFrame({experiment.date_column: [next_date]})

            if experiment.id_column and experiment.id_column in df.columns:
                next_row[experiment.id_column] = df[experiment.id_column].iloc[0]

            next_row[experiment.target_column] = np.nan

            # Combina com histórico e transforma
            combined = pd.concat([current_df, next_row], ignore_index=True)
            features = ts_preprocessor.transform_for_prediction(
                next_row, history_df=current_df
            )

            # Faz previsão
            pred_normalized = ml_model.predict(features)[0]
            pred = ts_preprocessor.denormalize_predictions(np.array([pred_normalized]))[0]
            values.append(float(pred))

            # Atualiza histórico com a previsão
            next_row[experiment.target_column] = pred
            current_df = pd.concat([current_df, next_row], ignore_index=True)

        dates = future_dates.strftime("%Y-%m-%d").tolist()
        lower_bound = [None] * len(values)
        upper_bound = [None] * len(values)

    return {
        "dates": dates,
        "values": values,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound
    }


@router.get("/{experiment_id}/forecast-chart-all")
def get_forecast_chart_all_models(
    experiment_id: int,
    series_id: str | None = None,
    db: Session = Depends(get_db)
):
    """
    Retorna dados para o gráfico de previsão com todos os modelos.

    Inclui dados históricos e previsões de cada modelo treinado.

    Parâmetros:
        experiment_id: ID do experimento.
        series_id: ID da série específica (para múltiplas séries).

    Retorna:
        Dados históricos e previsões de todos os modelos.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    if experiment.problem_type != ProblemType.TIME_SERIES:
        raise HTTPException(status_code=400, detail="Experimento não é de séries temporais")

    if experiment.status != ExperimentStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Experimento ainda não foi concluído")

    # Carrega o dataset
    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    filepath = Path(dataset.filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # Converte coluna de data
    df[experiment.date_column] = pd.to_datetime(df[experiment.date_column])
    df = df.sort_values(experiment.date_column)

    # Identifica séries disponíveis
    available_series = []
    if experiment.id_column and experiment.id_column in df.columns:
        available_series = df[experiment.id_column].unique().tolist()

    # Filtra por série se especificado
    if series_id and experiment.id_column:
        df = df[df[experiment.id_column] == series_id]
    elif available_series:
        series_id = available_series[0]
        df = df[df[experiment.id_column] == series_id]

    # Prepara dados históricos
    historical_data = df[[experiment.date_column, experiment.target_column]].copy()
    historical_data = historical_data.tail(60)

    # Gera previsões para cada modelo
    forecast_horizon = experiment.forecast_horizon or 7
    models_forecasts = []

    trained_models = (
        db.query(TrainedModel)
        .filter(TrainedModel.experiment_id == experiment_id)
        .order_by(TrainedModel.rank)
        .all()
    )

    for model in trained_models:
        try:
            forecast_data = _generate_forecast(
                model=model,
                experiment=experiment,
                df=df,
                forecast_horizon=forecast_horizon
            )
            models_forecasts.append({
                "model_id": model.id,
                "model_name": model.algorithm_name,
                "rank": model.rank,
                "mape": model.metrics.get("mape", 0),
                "forecast": forecast_data
            })
        except Exception as e:
            print(f"Erro ao gerar previsão para {model.algorithm_name}: {e}")

    return {
        "historical": {
            "dates": historical_data[experiment.date_column].dt.strftime("%Y-%m-%d").tolist(),
            "values": historical_data[experiment.target_column].tolist()
        },
        "models_forecasts": models_forecasts,
        "available_series": available_series,
        "current_series": series_id,
        "forecast_horizon": forecast_horizon
    }


@router.get("/{experiment_id}/series")
def get_available_series(experiment_id: int, db: Session = Depends(get_db)):
    """
    Retorna a lista de séries disponíveis para um experimento de séries temporais.

    Parâmetros:
        experiment_id: ID do experimento.

    Retorna:
        Lista de IDs de séries disponíveis.
    """
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    if not experiment:
        raise HTTPException(status_code=404, detail="Experimento não encontrado")

    if not experiment.id_column:
        return {"series": []}

    # Carrega o dataset
    dataset = db.query(Dataset).filter(Dataset.id == experiment.dataset_id).first()
    filepath = Path(dataset.filepath)

    if filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    if experiment.id_column not in df.columns:
        return {"series": []}

    series_list = df[experiment.id_column].unique().tolist()

    return {"series": series_list}
