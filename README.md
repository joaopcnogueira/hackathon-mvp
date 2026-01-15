# AutoML Platform

Plataforma web para treinamento automático de modelos de Machine Learning. Permite upload de datasets, pré-processamento automático, treinamento de múltiplos algoritmos e deploy de modelos para predições via API ou batch.

## Funcionalidades

- **Upload de Datasets**: Suporte para CSV e Excel (xlsx/xls)
- **Pré-processamento Automático**: Tratamento de valores nulos, encoding de variáveis categóricas (Target Encoding), normalização
- **Treinamento de Modelos**: 3 algoritmos para classificação e 3 para regressão
- **Avaliação e Ranking**: Métricas automáticas com ranking por performance
- **Deploy Flexível**: Predições via API REST ou processamento em batch
- **Download de Artefatos**: Exportar modelos treinados (.pkl)

## Algoritmos Suportados

### Classificação
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Métricas**: AUC-ROC (principal), Accuracy, Precision, Recall, F1-Score

### Regressão
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Métricas**: RMSE (principal), MAE, R²

## Estrutura do Projeto

```
hackathon-mvp/
├── main.py                 # Entrypoint FastAPI
├── app/
│   ├── config.py          # Configurações
│   ├── database.py        # Setup SQLite
│   ├── models/            # SQLAlchemy models
│   ├── schemas/           # Pydantic schemas
│   ├── services/          # Lógica de negócio
│   │   ├── preprocessing.py
│   │   ├── training.py
│   │   └── prediction.py
│   └── routers/           # Endpoints API
├── templates/             # Jinja2 templates
├── static/                # CSS/JS
├── artifacts/             # Modelos salvos
├── uploads/               # Datasets uploadados
└── data/                  # Datasets exemplo
```

## Requisitos

- Python 3.11+
- uv (gerenciador de pacotes)

## Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/hackathon-mvp.git
cd hackathon-mvp

# Instale as dependências com uv
uv sync

# Execute o servidor
uv run python main.py
```

O servidor estará disponível em `http://localhost:8000`

## Uso

### 1. Upload de Dataset
- Acesse a página inicial
- Clique em "Upload Dataset"
- Selecione um arquivo CSV ou Excel
- O sistema detectará automaticamente os tipos das colunas

### 2. Criar Experimento
- Na página do dataset, clique em "Iniciar Experimento"
- Selecione a coluna alvo (target)
- Selecione as colunas de features
- Escolha o tipo de problema (classificação ou regressão)
- Inicie o treinamento

### 3. Visualizar Resultados
- Acompanhe o status do experimento
- Veja o ranking dos modelos por performance
- Compare métricas entre algoritmos

### 4. Deploy do Modelo
- Selecione o melhor modelo
- Ative deploy via API e/ou Batch
- Use o endpoint `/api/predictions/{model_id}` para predições via API
- Use a página de Batch para processar arquivos

## API Endpoints

### Datasets
- `POST /api/datasets/upload` - Upload de dataset
- `GET /api/datasets` - Listar datasets
- `GET /api/datasets/{id}` - Detalhes do dataset
- `DELETE /api/datasets/{id}` - Excluir dataset (cascade)

### Experiments
- `POST /api/experiments` - Criar experimento
- `GET /api/experiments` - Listar experimentos
- `GET /api/experiments/{id}` - Detalhes do experimento
- `DELETE /api/experiments/{id}` - Excluir experimento

### Models
- `GET /api/models/{id}` - Detalhes do modelo
- `GET /api/models/{id}/download` - Download do modelo (.pkl)
- `PUT /api/models/{id}/deploy` - Configurar deploy

### Predictions
- `POST /api/predictions/{model_id}` - Predição via API
- `POST /api/predictions/{model_id}/batch` - Predição em batch

## Tecnologias

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Jinja2, Bootstrap 5
- **ML**: scikit-learn, pandas, numpy
- **Serialização**: joblib

## Licença

MIT
