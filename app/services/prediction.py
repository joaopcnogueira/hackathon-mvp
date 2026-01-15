"""
Serviço de predição para modelos treinados.

Suporta predições via API (individual) e em batch (arquivo).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Any
import uuid

from app.services.preprocessing import PreprocessingService
from app.config import ARTIFACTS_DIR


class PredictionService:
    """
    Serviço responsável por fazer predições usando modelos treinados.

    Suporta predições individuais (API) e em lote (batch).
    """

    def __init__(self):
        self.batch_results: dict[str, dict] = {}

    def predict(
        self,
        model_path: str,
        preprocessing_pipeline_path: str,
        feature_columns: list[str],
        data: list[dict[str, Any]]
    ) -> list[Any]:
        """
        Faz predições usando o modelo e pipeline especificados.

        Parâmetros:
            model_path: Caminho para o modelo treinado.
            preprocessing_pipeline_path: Caminho para o pipeline de pré-processamento.
            feature_columns: Lista de colunas de features.
            data: Lista de dicionários com os dados para predição.

        Retorna:
            Lista com as predições.
        """
        # Carrega o modelo
        model = joblib.load(model_path)

        # Carrega e aplica o pipeline de pré-processamento
        preprocessor = PreprocessingService()
        preprocessor.load_pipeline(Path(preprocessing_pipeline_path))

        # Converte dados para DataFrame
        df = pd.DataFrame(data)

        # Aplica pré-processamento
        X = preprocessor.transform(df, feature_columns)

        # Faz predições
        predictions = model.predict(X)

        return predictions.tolist()

    def predict_proba(
        self,
        model_path: str,
        preprocessing_pipeline_path: str,
        feature_columns: list[str],
        data: list[dict[str, Any]]
    ) -> list[dict[str, float]]:
        """
        Faz predições com probabilidades (para classificação).

        Parâmetros:
            model_path: Caminho para o modelo.
            preprocessing_pipeline_path: Caminho para o pipeline.
            feature_columns: Colunas de features.
            data: Dados para predição.

        Retorna:
            Lista de dicionários com probabilidades por classe.
        """
        model = joblib.load(model_path)
        preprocessor = PreprocessingService()
        preprocessor.load_pipeline(Path(preprocessing_pipeline_path))

        df = pd.DataFrame(data)
        X = preprocessor.transform(df, feature_columns)

        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            classes = model.classes_

            results = []
            for proba in probas:
                result = {str(cls): float(p) for cls, p in zip(classes, proba)}
                results.append(result)

            return results
        else:
            # Modelo não suporta probabilidades, retorna predição como 100%
            predictions = model.predict(X)
            return [{str(pred): 1.0} for pred in predictions]

    def predict_batch(
        self,
        model_path: str,
        preprocessing_pipeline_path: str,
        feature_columns: list[str],
        input_file: Path,
        is_classification: bool
    ) -> tuple[str, pd.DataFrame]:
        """
        Faz predições em lote a partir de um arquivo.

        Parâmetros:
            model_path: Caminho para o modelo.
            preprocessing_pipeline_path: Caminho para o pipeline.
            feature_columns: Colunas de features esperadas pelo modelo.
            input_file: Arquivo de entrada (CSV ou Excel).
            is_classification: Se é um problema de classificação.

        Retorna:
            Tupla com (ID do resultado, DataFrame com predições).

        Observação:
            O arquivo pode conter colunas extras (ex: IDs) que serão preservadas
            no resultado. Colunas faltantes serão preenchidas com valor padrão.
        """
        # Lê o arquivo mantendo todas as colunas originais
        if input_file.suffix == ".csv":
            df_original = pd.read_csv(input_file)
        else:
            df_original = pd.read_excel(input_file)

        # Carrega modelo e pipeline
        model = joblib.load(model_path)
        preprocessor = PreprocessingService()
        preprocessor.load_pipeline(Path(preprocessing_pipeline_path))

        # Aplica pré-processamento apenas nas colunas de features
        # Colunas extras são ignoradas, colunas faltantes são preenchidas
        X = preprocessor.transform(df_original, feature_columns)

        # Cria DataFrame de resultado com todas as colunas originais
        df_result = df_original.copy()

        # Para classificação, usa probabilidade como predição principal
        if is_classification and hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            classes = model.classes_

            if len(classes) == 2:
                # Binário: predição é a probabilidade da classe positiva (0 a 1)
                df_result["prediction"] = probas[:, 1]
            else:
                # Multiclasse: predição é a classe com maior probabilidade
                predictions = model.predict(X)
                df_result["prediction"] = predictions
                # Adiciona probabilidade de cada classe
                for i, cls in enumerate(classes):
                    df_result[f"prob_{cls}"] = probas[:, i]
        else:
            # Regressão ou modelo sem probabilidades: usa predict diretamente
            predictions = model.predict(X)
            df_result["prediction"] = predictions

        # Salva o resultado
        result_id = str(uuid.uuid4())
        result_path = ARTIFACTS_DIR / f"batch_result_{result_id}.csv"
        df_result.to_csv(result_path, index=False)

        # Converte preview para JSON-safe, tratando todos os tipos de valores nulos
        preview_records = df_result.head(10).to_dict(orient="records")
        for record in preview_records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None

        self.batch_results[result_id] = {
            "path": str(result_path),
            "num_rows": len(df_result),
            "preview": preview_records
        }

        return result_id, df_result

    def get_batch_result(self, result_id: str) -> dict | None:
        """
        Recupera informações de um resultado batch.

        Parâmetros:
            result_id: ID do resultado.

        Retorna:
            Dicionário com informações ou None se não encontrado.
        """
        return self.batch_results.get(result_id)

    def get_batch_result_path(self, result_id: str) -> Path | None:
        """
        Recupera o caminho do arquivo de resultado batch.

        Parâmetros:
            result_id: ID do resultado.

        Retorna:
            Path do arquivo ou None se não encontrado.
        """
        result = self.batch_results.get(result_id)
        if result:
            return Path(result["path"])
        return None


# Instância global para manter os resultados em memória
prediction_service = PredictionService()
