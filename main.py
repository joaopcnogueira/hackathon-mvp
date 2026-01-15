"""
Aplicação principal da plataforma AutoML.

Inicializa o servidor FastAPI com todas as rotas e templates.
"""
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pathlib import Path

from app.database import init_db, get_db, SessionLocal
from app.routers import (
    datasets_router,
    experiments_router,
    models_router,
    predictions_router
)
from app.models.dataset import Dataset
from app.models.experiment import Experiment

app = FastAPI(
    title="AutoML Platform",
    description="Plataforma para treinamento automático de modelos de Machine Learning",
    version="1.0.0"
)

# Monta arquivos estáticos
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Configura templates
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# Inicializa banco de dados na startup
@app.on_event("startup")
def startup_event():
    """Inicializa o banco de dados ao iniciar a aplicação."""
    init_db()


# Registra routers da API
app.include_router(datasets_router)
app.include_router(experiments_router)
app.include_router(models_router)
app.include_router(predictions_router)


# Rotas de páginas (templates)

@app.get("/")
def home(request: Request):
    """
    Página inicial com dashboard.
    Exibe datasets e experimentos recentes.
    """
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).limit(5).all()
        experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).limit(5).all()

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "datasets": datasets,
                "experiments": experiments
            }
        )
    finally:
        db.close()


@app.get("/datasets")
def datasets_page(request: Request):
    """
    Página de listagem de datasets.
    """
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
        return templates.TemplateResponse(
            "datasets.html",
            {"request": request, "datasets": datasets}
        )
    finally:
        db.close()


@app.get("/datasets/upload")
def upload_page(request: Request):
    """
    Página de upload de dataset.
    """
    return templates.TemplateResponse("upload.html", {"request": request})


@app.get("/datasets/{dataset_id}")
def dataset_detail_page(request: Request, dataset_id: int):
    """
    Página de detalhes de um dataset.
    """
    import pandas as pd
    db = SessionLocal()
    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if not dataset:
            return RedirectResponse(url="/datasets", status_code=302)

        # Conta colunas numéricas e categóricas
        columns_info = dataset.columns_info
        numeric_cols = sum(
            1 for col in columns_info
            if "int" in col["dtype"] or "float" in col["dtype"]
        )
        categorical_cols = len(columns_info) - numeric_cols

        # Carrega preview dos dados
        filepath = Path(dataset.filepath)
        try:
            if filepath.suffix == ".csv":
                df = pd.read_csv(filepath, nrows=10)
            else:
                df = pd.read_excel(filepath, nrows=10)
            preview_data = df.to_dict(orient="records")
        except Exception:
            preview_data = []

        return templates.TemplateResponse(
            "dataset_detail.html",
            {
                "request": request,
                "dataset": dataset,
                "columns_info": columns_info,
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "preview_data": preview_data
            }
        )
    finally:
        db.close()


@app.get("/experiments")
def experiments_page(request: Request):
    """
    Página de listagem de experimentos.
    """
    db = SessionLocal()
    try:
        experiments = db.query(Experiment).order_by(Experiment.created_at.desc()).all()
        return templates.TemplateResponse(
            "experiments.html",
            {"request": request, "experiments": experiments}
        )
    finally:
        db.close()


@app.get("/experiments/new")
def new_experiment_page(request: Request, dataset_id: int = None):
    """
    Página de criação de novo experimento.
    """
    db = SessionLocal()
    try:
        datasets = db.query(Dataset).order_by(Dataset.created_at.desc()).all()
        return templates.TemplateResponse(
            "experiment_new.html",
            {
                "request": request,
                "datasets": datasets,
                "selected_dataset_id": dataset_id
            }
        )
    finally:
        db.close()


@app.get("/experiments/{experiment_id}")
def experiment_detail_page(request: Request, experiment_id: int):
    """
    Página de detalhes de um experimento com resultados.
    """
    db = SessionLocal()
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

        if not experiment:
            return RedirectResponse(url="/experiments", status_code=302)

        return templates.TemplateResponse(
            "experiment_detail.html",
            {"request": request, "experiment": experiment}
        )
    finally:
        db.close()


@app.get("/experiments/{experiment_id}/batch")
def batch_prediction_page(request: Request, experiment_id: int):
    """
    Página de predição em batch.
    """
    db = SessionLocal()
    try:
        experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

        if not experiment:
            return RedirectResponse(url="/experiments", status_code=302)

        # Encontra modelo configurado para batch
        batch_model = None
        for model in experiment.trained_models:
            if model.is_deployed_batch:
                batch_model = model
                break

        if not batch_model:
            return RedirectResponse(
                url=f"/experiments/{experiment_id}",
                status_code=302
            )

        return templates.TemplateResponse(
            "batch.html",
            {
                "request": request,
                "experiment": experiment,
                "model": batch_model
            }
        )
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
