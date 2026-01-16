# Tutorial: Construindo uma Plataforma AutoML do Zero

Este tutorial simula o desenvolvimento completo de uma plataforma de Machine Learning automatizada, partindo da experimentação até a API de produção. O público-alvo são cientistas de dados que dominam modelagem e ML, mas querem aprender a estruturar um projeto completo.

## Índice

1. [Configuração do Ambiente](#1-configuração-do-ambiente)
2. [Experimentação: Preprocessing](#2-experimentação-preprocessing)
3. [Experimentação: Training](#3-experimentação-training)
4. [Experimentação: Prediction](#4-experimentação-prediction)
5. [Estruturando o Projeto](#5-estruturando-o-projeto)
6. [Camada de Dados: Models e Database](#6-camada-de-dados-models-e-database)
7. [Validação: Schemas](#7-validação-schemas)
8. [API: Routers](#8-api-routers)
9. [Frontend](#9-frontend)
10. [Executando a Aplicação](#10-executando-a-aplicação)

---

## 1. Configuração do Ambiente

### 1.1 Instalando o uv

O [uv](https://github.com/astral-sh/uv) é um gerenciador de pacotes Python extremamente rápido. Instale-o:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 1.2 Criando o Projeto

```bash
# Cria o diretório do projeto
mkdir automl-platform
cd automl-platform

# Inicializa o projeto Python com uv
uv init
```

### 1.3 Configurando o pyproject.toml

Edite o `pyproject.toml` para adicionar as dependências necessárias:

```toml
[project]
name = "automl-platform"
version = "0.1.0"
description = "Plataforma AutoML para treinamento e deploy de modelos de machine learning"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "python-multipart>=0.0.12",
    "sqlalchemy>=2.0.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.5.0",
    "jinja2>=3.1.0",
    "joblib>=1.4.0",
    "openpyxl>=3.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["app"]
```

### 1.4 Instalando Dependências

```bash
uv sync
```

### 1.5 Configurando Jupyter para Prototipação

Para experimentar as funções antes de integrá-las ao projeto:

```bash
# Adiciona jupyter como dependência de desenvolvimento
uv add --dev jupyter ipykernel

# Inicia o Jupyter
uv run jupyter notebook
```

### 1.6 Estrutura Inicial de Diretórios

```bash
mkdir -p app/services app/models app/schemas app/routers
mkdir -p templates static data uploads artifacts
touch app/__init__.py app/services/__init__.py app/models/__init__.py
touch app/schemas/__init__.py app/routers/__init__.py
```

Estrutura resultante:

```
automl-platform/
├── pyproject.toml
├── app/
│   ├── __init__.py
│   ├── services/          # Lógica de negócio (ML)
│   ├── models/            # ORM (SQLAlchemy)
│   ├── schemas/           # Validação (Pydantic)
│   └── routers/           # Endpoints API
├── templates/             # HTML (Jinja2)
├── static/                # CSS/JS
├── data/                  # Datasets de exemplo
├── uploads/               # Arquivos enviados
└── artifacts/             # Modelos salvos
```

---

## 2. Experimentação: Preprocessing

Antes de construir a API, vamos desenvolver e testar a lógica de ML em um notebook. Comece criando um notebook `experiments/preprocessing.ipynb`.

### 2.1 Entendendo o Problema

O pré-processamento precisa:
1. Tratar valores nulos (imputação)
2. Codificar variáveis categóricas
3. Normalizar variáveis numéricas
4. **Evitar data leakage** (ajustar transformações apenas no treino)

### 2.2 Experimentando no Notebook

```python
import pandas as pd
import numpy as np

# Carrega dataset de exemplo
df = pd.read_csv("data/titanic.csv")
print(f"Shape: {df.shape}")
df.head()
```

```python
# Análise exploratória
def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Analisa o DataFrame e retorna informações sobre as colunas."""
    columns_info = []

    for col in df.columns:
        dtype = str(df[col].dtype)
        unique_count = df[col].nunique()
        null_count = df[col].isnull().sum()
        null_percentage = (null_count / len(df)) * 100 if len(df) > 0 else 0
        sample_values = df[col].dropna().head(5).tolist()

        columns_info.append({
            "name": col,
            "dtype": dtype,
            "unique_count": int(unique_count),
            "null_count": int(null_count),
            "null_percentage": float(null_percentage),
            "sample_values": [str(v) for v in sample_values]
        })

    return {
        "columns_info": columns_info,
        "num_rows": len(df),
        "num_columns": len(df.columns)
    }

analyze_dataframe(df)
```

### 2.3 Identificando Tipos de Colunas

```python
def identify_column_types(df: pd.DataFrame) -> tuple[list, list]:
    """Separa colunas numéricas e categóricas."""
    numeric_columns = []
    categorical_columns = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)
        else:
            categorical_columns.append(col)

    return numeric_columns, categorical_columns

numeric_cols, categorical_cols = identify_column_types(df)
print(f"Numéricas: {numeric_cols}")
print(f"Categóricas: {categorical_cols}")
```

### 2.4 Tratamento de Valores Nulos

```python
def handle_missing_values(
    df: pd.DataFrame,
    numeric_columns: list,
    categorical_columns: list
) -> tuple[pd.DataFrame, dict]:
    """
    Trata valores nulos.
    Numéricas: mediana
    Categóricas: moda (valor mais frequente)

    Retorna o DataFrame tratado e os valores usados para imputação.
    """
    imputation_values = {}
    df = df.copy()

    for col in numeric_columns:
        if col not in df.columns:
            continue
        median_value = df[col].median()
        imputation_values[col] = float(median_value) if pd.notna(median_value) else 0.0
        df[col] = df[col].fillna(median_value)

    for col in categorical_columns:
        if col not in df.columns:
            continue
        mode_value = df[col].mode()
        impute_val = mode_value[0] if len(mode_value) > 0 else "UNKNOWN"
        imputation_values[col] = impute_val
        df[col] = df[col].fillna(impute_val)

    return df, imputation_values

# Teste
df_filled, impute_vals = handle_missing_values(df, numeric_cols, categorical_cols)
print(f"Valores de imputação: {impute_vals}")
print(f"Nulos restantes: {df_filled.isnull().sum().sum()}")
```

### 2.5 Target Encoding

Escolhemos Target Encoding em vez de One-Hot Encoding porque:
- Não aumenta a dimensionalidade (evita "explosão" de colunas)
- Captura relação entre categoria e target
- Funciona bem com modelos baseados em árvores

```python
def encode_categorical(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_columns: list
) -> tuple[pd.DataFrame, dict]:
    """
    Aplica Target Encoding nas colunas categóricas.
    Substitui cada categoria pela média do target para aquela categoria.
    """
    X = X.copy()
    target_encodings = {}
    global_mean = y.mean() if pd.api.types.is_numeric_dtype(y) else 0

    for col in categorical_columns:
        if col not in X.columns:
            continue

        encoding_map = {}
        for category in X[col].unique():
            mask = X[col] == category
            if pd.api.types.is_numeric_dtype(y):
                encoding_map[category] = y[mask].mean()
            else:
                # Para classificação com target não numérico
                encoding_map[category] = mask.sum() / len(mask)

        target_encodings[col] = encoding_map
        X[col] = X[col].map(encoding_map).fillna(global_mean)

    return X, target_encodings

# Teste
y = df["Survived"]
X = df[["Sex", "Embarked"]].copy()
X_encoded, encodings = encode_categorical(X, y, ["Sex", "Embarked"])
print("Encodings:", encodings)
X_encoded.head()
```

### 2.6 Normalização

```python
def normalize_numeric(
    df: pd.DataFrame,
    columns: list
) -> tuple[pd.DataFrame, dict]:
    """
    Normaliza colunas numéricas usando StandardScaler (média 0, std 1).
    """
    df = df.copy()
    numeric_stats = {}

    for col in columns:
        if col not in df.columns:
            continue

        mean_val = df[col].mean()
        std_val = df[col].std()

        numeric_stats[col] = {
            "mean": float(mean_val),
            "std": float(std_val) if std_val > 0 else 1.0
        }

        if std_val > 0:
            df[col] = (df[col] - mean_val) / std_val
        else:
            df[col] = 0

    return df, numeric_stats

# Teste
X_norm, stats = normalize_numeric(df[["Age", "Fare"]], ["Age", "Fare"])
print("Estatísticas:", stats)
print(f"Média após normalização: {X_norm['Age'].mean():.4f}")
print(f"Std após normalização: {X_norm['Age'].std():.4f}")
```

### 2.7 Consolidando em uma Classe

Após validar cada função, consolidamos em uma classe reutilizável:

```python
# app/services/preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Any


class PreprocessingService:
    """
    Serviço responsável pelo pré-processamento automático de datasets.
    """

    def __init__(self):
        self.transformations: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.target_encodings: dict[str, dict[Any, float]] = {}
        self.numeric_stats: dict[str, dict[str, float]] = {}
        self.imputation_values: dict[str, Any] = {}
        self.nulls_filled: int = 0

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
        """
        Ajusta o pré-processamento nos dados de TREINO e os transforma.
        Para dados de teste, use transform_test().
        """
        # ... implementação ...
        pass

    def transform_test(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma dados de teste usando parâmetros ajustados no treino.
        """
        # ... implementação ...
        pass

    def save_pipeline(self, filepath: Path) -> None:
        """Salva o pipeline para uso em predições."""
        pipeline = {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "target_encodings": self.target_encodings,
            "numeric_stats": self.numeric_stats,
            "imputation_values": self.imputation_values
        }
        joblib.dump(pipeline, filepath)

    def load_pipeline(self, filepath: Path) -> None:
        """Carrega um pipeline salvo."""
        pipeline = joblib.load(filepath)
        self.numeric_columns = pipeline["numeric_columns"]
        self.categorical_columns = pipeline["categorical_columns"]
        self.target_encodings = pipeline["target_encodings"]
        self.numeric_stats = pipeline["numeric_stats"]
        self.imputation_values = pipeline.get("imputation_values", {})
```

> **Ponto-chave sobre Data Leakage**: O método `fit_transform()` aprende os parâmetros (médias, encodings) APENAS nos dados de treino. O método `transform_test()` aplica esses parâmetros nos dados de teste sem recalcular. Isso evita vazamento de informação do teste para o treino.

---

## 3. Experimentação: Training

Agora vamos desenvolver a lógica de treinamento. Crie `experiments/training.ipynb`.

### 3.1 Definindo os Algoritmos

```python
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

CLASSIFICATION_ALGORITHMS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}

REGRESSION_ALGORITHMS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}
```

### 3.2 Métricas de Classificação

```python
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score
)

def calculate_classification_metrics(model, X_test, y_test, y_pred) -> dict:
    """Calcula métricas de classificação."""
    metrics = {}

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

    # AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] == 2:
            metrics["auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            metrics["auc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
    else:
        metrics["auc"] = metrics["accuracy"]

    # Precision, Recall, F1
    unique_classes = len(np.unique(y_test))
    average = "binary" if unique_classes == 2 else "weighted"

    metrics["precision"] = float(precision_score(y_test, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, y_pred, average=average, zero_division=0))

    return metrics
```

### 3.3 Métricas de Regressão

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_regression_metrics(y_test, y_pred) -> dict:
    """Calcula métricas de regressão."""
    metrics = {}

    # RMSE
    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    # MAE
    metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

    # R²
    metrics["r2"] = float(r2_score(y_test, y_pred))

    # MAPE
    mask = y_test != 0
    if mask.any():
        mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
        metrics["mape"] = float(mape)
    else:
        metrics["mape"] = 0.0

    return metrics
```

### 3.4 Treinando e Avaliando

```python
from sklearn.model_selection import train_test_split
import joblib

def train_and_evaluate(X, y, is_classification=True, artifacts_dir="artifacts", experiment_id=1):
    """Pipeline completo de treino e avaliação."""

    # 1. Split ANTES do pré-processamento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Pré-processamento (fit no treino, transform no teste)
    preprocessor = PreprocessingService()
    X_train_processed, _ = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform_test(X_test)

    # 3. Salva o pipeline
    preprocessor.save_pipeline(Path(f"{artifacts_dir}/pipeline_{experiment_id}.pkl"))

    # 4. Treina todos os modelos
    algorithms = CLASSIFICATION_ALGORITHMS if is_classification else REGRESSION_ALGORITHMS
    results = []

    for name, model in algorithms.items():
        # Treina
        model.fit(X_train_processed, y_train)
        y_pred = model.predict(X_test_processed)

        # Avalia
        if is_classification:
            metrics = calculate_classification_metrics(model, X_test_processed, y_test, y_pred)
            primary_metric = metrics["auc"]
        else:
            metrics = calculate_regression_metrics(y_test, y_pred)
            primary_metric = -metrics["rmse"]  # Negativo para ranking (menor é melhor)

        # Salva modelo
        model_path = f"{artifacts_dir}/model_{experiment_id}_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_path)

        results.append({
            "algorithm_name": name,
            "model_path": model_path,
            "metrics": metrics,
            "primary_metric_value": primary_metric
        })

    # 5. Ranking
    results = sorted(results, key=lambda x: x["primary_metric_value"], reverse=True)
    for i, result in enumerate(results):
        result["rank"] = i + 1

    return results

# Teste com Titanic
df = pd.read_csv("data/titanic.csv")
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = df[feature_cols]
y = df["Survived"]

results = train_and_evaluate(X, y, is_classification=True)
for r in results:
    print(f"#{r['rank']} {r['algorithm_name']}: AUC={r['metrics']['auc']:.4f}")
```

### 3.5 Consolidando a Classe TrainingService

```python
# app/services/training.py
class TrainingService:
    """
    Serviço responsável pelo treinamento de modelos de ML.
    """

    CLASSIFICATION_ALGORITHMS = { ... }
    REGRESSION_ALGORITHMS = { ... }

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        is_classification: bool,
        artifacts_dir: Path,
        experiment_id: int
    ) -> list[dict]:
        """Treina todos os modelos e retorna resultados com ranking."""
        # ... implementação ...
        pass
```

---

## 4. Experimentação: Prediction

A última peça do pipeline de ML é fazer predições com modelos treinados.

### 4.1 Predição Simples

```python
def predict(model_path: str, pipeline_path: str, feature_columns: list, data: list[dict]) -> list:
    """Faz predições usando modelo e pipeline salvos."""

    # Carrega modelo
    model = joblib.load(model_path)

    # Carrega e aplica pipeline
    preprocessor = PreprocessingService()
    preprocessor.load_pipeline(Path(pipeline_path))

    # Converte para DataFrame e transforma
    df = pd.DataFrame(data)
    X = preprocessor.transform(df, feature_columns)

    # Predição
    predictions = model.predict(X)
    return predictions.tolist()

# Teste
new_passenger = [{
    "Pclass": 1,
    "Sex": "female",
    "Age": 30,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 100,
    "Embarked": "C"
}]

prediction = predict(
    "artifacts/model_1_random_forest.pkl",
    "artifacts/pipeline_1.pkl",
    feature_cols,
    new_passenger
)
print(f"Survived: {prediction[0]}")
```

### 4.2 Predição com Probabilidades

```python
def predict_proba(model_path: str, pipeline_path: str, feature_columns: list, data: list[dict]) -> list:
    """Faz predições com probabilidades."""
    model = joblib.load(model_path)
    preprocessor = PreprocessingService()
    preprocessor.load_pipeline(Path(pipeline_path))

    df = pd.DataFrame(data)
    X = preprocessor.transform(df, feature_columns)

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        classes = model.classes_
        return [{str(cls): float(p) for cls, p in zip(classes, proba)} for proba in probas]
    else:
        predictions = model.predict(X)
        return [{str(pred): 1.0} for pred in predictions]

# Teste
proba = predict_proba(
    "artifacts/model_1_random_forest.pkl",
    "artifacts/pipeline_1.pkl",
    feature_cols,
    new_passenger
)
print(f"Probabilidades: {proba}")
```

### 4.3 Predição em Batch

```python
import uuid

def predict_batch(
    model_path: str,
    pipeline_path: str,
    feature_columns: list,
    input_file: Path,
    is_classification: bool
) -> tuple[str, pd.DataFrame]:
    """Faz predições em lote a partir de arquivo."""

    # Lê arquivo
    if input_file.suffix == ".csv":
        df_original = pd.read_csv(input_file)
    else:
        df_original = pd.read_excel(input_file)

    # Carrega modelo e pipeline
    model = joblib.load(model_path)
    preprocessor = PreprocessingService()
    preprocessor.load_pipeline(Path(pipeline_path))

    # Transforma
    X = preprocessor.transform(df_original, feature_columns)

    # Predição
    df_result = df_original.copy()

    if is_classification and hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        classes = model.classes_

        if len(classes) == 2:
            # Binário: probabilidade da classe positiva
            df_result["prediction"] = probas[:, 1]
        else:
            # Multiclasse
            df_result["prediction"] = model.predict(X)
    else:
        df_result["prediction"] = model.predict(X)

    # Salva resultado
    result_id = str(uuid.uuid4())
    result_path = Path(f"artifacts/batch_result_{result_id}.csv")
    df_result.to_csv(result_path, index=False)

    return result_id, df_result

# Teste
result_id, df_result = predict_batch(
    "artifacts/model_1_random_forest.pkl",
    "artifacts/pipeline_1.pkl",
    feature_cols,
    Path("data/titanic.csv"),
    is_classification=True
)
print(f"Result ID: {result_id}")
df_result[["PassengerId", "Name", "prediction"]].head()
```

---

## 5. Estruturando o Projeto

Com a lógica de ML validada, vamos estruturar o projeto para produção.

### 5.1 Configuração Centralizada

```python
# app/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = f"sqlite:///{BASE_DIR}/automl.db"
UPLOADS_DIR = BASE_DIR / "uploads"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 100

# Cria diretórios se não existirem
UPLOADS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)
```

### 5.2 Estrutura Final

```
automl-platform/
├── main.py                 # Entrypoint FastAPI
├── app/
│   ├── __init__.py
│   ├── config.py          # Configurações
│   ├── database.py        # Setup SQLite/SQLAlchemy
│   ├── services/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   ├── training.py
│   │   └── prediction.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── experiment.py
│   │   └── trained_model.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── routers/
│       ├── __init__.py
│       ├── datasets.py
│       ├── experiments.py
│       ├── models.py
│       └── predictions.py
├── templates/
├── static/
├── data/
├── uploads/
└── artifacts/
```

---

## 6. Camada de Dados: Models e Database

### 6.1 Configuração do SQLAlchemy

```python
# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency para injetar sessão do banco."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Cria as tabelas no banco."""
    Base.metadata.create_all(bind=engine)
```

### 6.2 Model: Dataset

```python
# app/models/dataset.py
from sqlalchemy import Column, Integer, String, DateTime, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    num_rows = Column(Integer)
    num_columns = Column(Integer)
    columns_info = Column(JSON)  # Informações detalhadas das colunas
    statistics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relacionamento: um dataset tem muitos experimentos
    experiments = relationship("Experiment", back_populates="dataset", cascade="all, delete-orphan")
```

### 6.3 Model: Experiment

```python
# app/models/experiment.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.database import Base

class ExperimentStatus(enum.Enum):
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class ProblemType(enum.Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    target_column = Column(String, nullable=False)
    feature_columns = Column(JSON, nullable=False)
    problem_type = Column(Enum(ProblemType), nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING)
    preprocessing_info = Column(JSON)
    preprocessing_pipeline_path = Column(String)
    error_message = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="experiments")
    trained_models = relationship("TrainedModel", back_populates="experiment", cascade="all, delete-orphan")
```

### 6.4 Model: TrainedModel

```python
# app/models/trained_model.py
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=False)
    algorithm_name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)
    metrics = Column(JSON)
    primary_metric_value = Column(Float)
    rank = Column(Integer)
    is_deployed_api = Column(Boolean, default=False)
    is_deployed_batch = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    experiment = relationship("Experiment", back_populates="trained_models")
```

---

## 7. Validação: Schemas

Os schemas Pydantic validam entrada/saída da API.

```python
# app/schemas/schemas.py
from pydantic import BaseModel
from datetime import datetime
from typing import Any

# Dataset
class DatasetBase(BaseModel):
    name: str

class DatasetResponse(DatasetBase):
    id: int
    filename: str
    num_rows: int | None
    num_columns: int | None
    columns_info: list[dict] | None
    created_at: datetime

    class Config:
        from_attributes = True

# Experiment
class ExperimentCreate(BaseModel):
    name: str
    dataset_id: int
    target_column: str
    feature_columns: list[str]
    problem_type: str  # "classification" ou "regression"

class ExperimentResponse(BaseModel):
    id: int
    name: str
    dataset_id: int
    target_column: str
    feature_columns: list[str]
    problem_type: str
    status: str
    preprocessing_info: dict | None
    created_at: datetime

    class Config:
        from_attributes = True

# Model
class TrainedModelResponse(BaseModel):
    id: int
    experiment_id: int
    algorithm_name: str
    metrics: dict | None
    primary_metric_value: float | None
    rank: int | None
    is_deployed_api: bool
    is_deployed_batch: bool

    class Config:
        from_attributes = True

# Prediction
class PredictionRequest(BaseModel):
    data: list[dict[str, Any]]
    return_probabilities: bool = False

class PredictionResponse(BaseModel):
    predictions: list[Any]
    probabilities: list[dict] | None = None
```

---

## 8. API: Routers

### 8.1 Router: Datasets

```python
# app/routers/datasets.py
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from pathlib import Path

from app.database import get_db
from app.models.dataset import Dataset
from app.services.preprocessing import PreprocessingService
from app.config import UPLOADS_DIR, ALLOWED_EXTENSIONS

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload e análise automática de dataset."""

    # Valida extensão
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Extensão não permitida. Use: {ALLOWED_EXTENSIONS}")

    # Salva arquivo
    filepath = UPLOADS_DIR / file.filename
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Lê e analisa
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    preprocessor = PreprocessingService()
    analysis = preprocessor.analyze_dataframe(df)

    # Salva no banco
    dataset = Dataset(
        name=file.filename.rsplit(".", 1)[0],
        filename=file.filename,
        filepath=str(filepath),
        num_rows=analysis["num_rows"],
        num_columns=analysis["num_columns"],
        columns_info=analysis["columns_info"]
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {"id": dataset.id, "message": "Dataset uploaded successfully"}

@router.get("")
def list_datasets(db: Session = Depends(get_db)):
    """Lista todos os datasets."""
    return db.query(Dataset).all()

@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Retorna detalhes de um dataset."""
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")
    return dataset
```

### 8.2 Router: Experiments

```python
# app/routers/experiments.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from app.database import get_db
from app.models.dataset import Dataset
from app.models.experiment import Experiment, ExperimentStatus, ProblemType
from app.models.trained_model import TrainedModel
from app.schemas.schemas import ExperimentCreate
from app.services.preprocessing import PreprocessingService
from app.services.training import TrainingService
from app.config import ARTIFACTS_DIR

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

def run_experiment_pipeline(experiment_id: int, db: Session):
    """Background task para executar o pipeline de ML."""
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()

    try:
        # Atualiza status
        experiment.status = ExperimentStatus.PREPROCESSING
        db.commit()

        # Carrega dataset
        dataset = experiment.dataset
        df = pd.read_csv(dataset.filepath) if dataset.filepath.endswith(".csv") else pd.read_excel(dataset.filepath)

        # Separa features e target
        X = df[experiment.feature_columns]
        y = df[experiment.target_column]

        # Remove linhas com target nulo
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        # Split ANTES do pré-processamento (evita data leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pré-processamento
        preprocessor = PreprocessingService()
        X_train_processed, preprocessing_info = preprocessor.fit_transform(X_train, y_train)
        X_test_processed = preprocessor.transform_test(X_test)

        # Salva pipeline
        pipeline_path = ARTIFACTS_DIR / f"pipeline_{experiment.id}.pkl"
        preprocessor.save_pipeline(pipeline_path)

        experiment.preprocessing_info = preprocessing_info
        experiment.preprocessing_pipeline_path = str(pipeline_path)
        experiment.status = ExperimentStatus.TRAINING
        db.commit()

        # Treinamento
        training_service = TrainingService()
        is_classification = experiment.problem_type == ProblemType.CLASSIFICATION

        results = training_service.train_all_models(
            X_train_processed, X_test_processed,
            y_train, y_test,
            is_classification, ARTIFACTS_DIR, experiment.id
        )

        # Salva modelos no banco
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

        experiment.status = ExperimentStatus.COMPLETED
        db.commit()

    except Exception as e:
        experiment.status = ExperimentStatus.FAILED
        experiment.error_message = str(e)
        db.commit()

@router.post("")
def create_experiment(
    experiment_data: ExperimentCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Cria um novo experimento e inicia o treinamento."""

    # Valida dataset
    dataset = db.query(Dataset).filter(Dataset.id == experiment_data.dataset_id).first()
    if not dataset:
        raise HTTPException(404, "Dataset not found")

    # Cria experimento
    experiment = Experiment(
        name=experiment_data.name,
        dataset_id=experiment_data.dataset_id,
        target_column=experiment_data.target_column,
        feature_columns=experiment_data.feature_columns,
        problem_type=ProblemType(experiment_data.problem_type)
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)

    # Inicia pipeline em background
    background_tasks.add_task(run_experiment_pipeline, experiment.id, db)

    return {"id": experiment.id, "status": "pending"}
```

### 8.3 Router: Predictions

```python
# app/routers/predictions.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.trained_model import TrainedModel
from app.models.experiment import Experiment
from app.schemas.schemas import PredictionRequest, PredictionResponse
from app.services.prediction import prediction_service

router = APIRouter(prefix="/api/v1/predict", tags=["predictions"])

@router.post("/{model_id}")
def predict(
    model_id: int,
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Faz predições usando o modelo especificado."""

    # Busca modelo
    model = db.query(TrainedModel).filter(TrainedModel.id == model_id).first()
    if not model:
        raise HTTPException(404, "Model not found")

    if not model.is_deployed_api:
        raise HTTPException(403, "Model not deployed for API predictions")

    # Busca experimento para obter feature_columns
    experiment = model.experiment

    # Faz predição
    predictions = prediction_service.predict(
        model.model_path,
        experiment.preprocessing_pipeline_path,
        experiment.feature_columns,
        request.data
    )

    response = {"predictions": predictions}

    # Se solicitado, retorna probabilidades
    if request.return_probabilities:
        probabilities = prediction_service.predict_proba(
            model.model_path,
            experiment.preprocessing_pipeline_path,
            experiment.feature_columns,
            request.data
        )
        response["probabilities"] = probabilities

    return response
```

---

## 9. Frontend

O frontend usa Jinja2 templates com Bootstrap 5.

### 9.1 Template Base

```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}AutoML Platform{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/style.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">AutoML Platform</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/datasets">Datasets</a>
                <a class="nav-link" href="/experiments">Experiments</a>
            </div>
        </div>
    </nav>

    <main class="container mt-4">
        {% block content %}{% endblock %}
    </main>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/app.js"></script>
</body>
</html>
```

### 9.2 Página de Upload

```html
<!-- templates/upload.html -->
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h5>Upload Dataset</h5>
            </div>
            <div class="card-body">
                <form action="/api/datasets/upload" method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="file" class="form-label">Arquivo (CSV ou Excel)</label>
                        <input type="file" class="form-control" id="file" name="file"
                               accept=".csv,.xlsx,.xls" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Upload</button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
```

### 9.3 Página de Resultados

```html
<!-- templates/experiment_detail.html -->
{% extends "base.html" %}

{% block content %}
<h2>{{ experiment.name }}</h2>
<p>Status: <span class="badge bg-{{ 'success' if experiment.status == 'completed' else 'warning' }}">
    {{ experiment.status }}
</span></p>

{% if experiment.status == 'completed' %}
<h4>Modelos Treinados</h4>
<table class="table">
    <thead>
        <tr>
            <th>Rank</th>
            <th>Algoritmo</th>
            <th>{{ 'AUC' if experiment.problem_type == 'classification' else 'RMSE' }}</th>
            <th>Ações</th>
        </tr>
    </thead>
    <tbody>
        {% for model in models %}
        <tr>
            <td>#{{ model.rank }}</td>
            <td>{{ model.algorithm_name }}</td>
            <td>{{ "%.4f"|format(model.primary_metric_value) }}</td>
            <td>
                <button class="btn btn-sm btn-primary"
                        onclick="deployModel({{ model.id }})">
                    Deploy
                </button>
            </td>
        </tr>
        {% endfor %}
    </tbody>
</table>
{% endif %}
{% endblock %}
```

---

## 10. Executando a Aplicação

### 10.1 Entrypoint Principal

```python
# main.py
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from app.database import init_db
from app.routers import datasets, experiments, models, predictions

app = FastAPI(
    title="AutoML Platform",
    description="Plataforma para treinamento automático de modelos de ML",
    version="0.1.0"
)

# Static files e templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Routers
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(models.router)
app.include_router(predictions.router)

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

### 10.2 Executando

```bash
# Desenvolvimento (com hot reload)
uv run python main.py

# Ou diretamente com uvicorn
uv run uvicorn main:app --reload --port 8000
```

Acesse `http://localhost:8000` no navegador.

### 10.3 Testando a API

```bash
# Upload de dataset
curl -X POST "http://localhost:8000/api/datasets/upload" \
  -F "file=@data/titanic.csv"

# Criar experimento
curl -X POST "http://localhost:8000/api/experiments" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Titanic Survival",
    "dataset_id": 1,
    "target_column": "Survived",
    "feature_columns": ["Pclass", "Sex", "Age", "Fare", "Embarked"],
    "problem_type": "classification"
  }'

# Fazer predição
curl -X POST "http://localhost:8000/api/v1/predict/1" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [{"Pclass": 1, "Sex": "female", "Age": 30, "Fare": 100, "Embarked": "C"}],
    "return_probabilities": true
  }'
```

---

## Resumo

Neste tutorial, construímos uma plataforma AutoML completa seguindo uma abordagem bottom-up:

1. **Configuração** - Ambiente com uv e Jupyter para prototipação
2. **Experimentação** - Desenvolvimento e validação das funções de ML
3. **Services** - Consolidação da lógica em classes reutilizáveis
4. **Models** - Persistência com SQLAlchemy
5. **Schemas** - Validação com Pydantic
6. **Routers** - Endpoints REST com FastAPI
7. **Frontend** - Interface web com Jinja2 e Bootstrap

O ponto-chave é começar pela experimentação, validar a lógica de ML em notebooks, e só depois estruturar o projeto para produção. Isso garante que a lógica core esteja correta antes de adicionar camadas de complexidade.
