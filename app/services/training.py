"""
Serviço de treinamento de modelos de machine learning.

Treina múltiplos algoritmos e avalia usando métricas apropriadas
para classificação (AUC), regressão (RMSE) e séries temporais (MAPE).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)


class TrainingService:
    """
    Serviço responsável pelo treinamento de modelos de ML.

    Treina 3 algoritmos para cada tipo de problema e retorna
    métricas de avaliação para ranking.
    """

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

    TIME_SERIES_ALGORITHMS = {
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        is_classification: bool,
        artifacts_dir: Path,
        experiment_id: int,
        problem_type: str = None
    ) -> list[dict[str, Any]]:
        """
        Treina todos os modelos para o tipo de problema especificado.

        O split treino/teste deve ser feito ANTES de chamar este método,
        garantindo que o pré-processamento seja ajustado apenas nos dados
        de treino (evita data leakage).

        Parâmetros:
            X_train: Features de treino (já pré-processadas).
            X_test: Features de teste (já pré-processadas).
            y_train: Target de treino.
            y_test: Target de teste.
            is_classification: Se é classificação ou regressão.
            artifacts_dir: Diretório para salvar os modelos.
            experiment_id: ID do experimento.
            problem_type: Tipo do problema (classification, regression, time_series).

        Retorna:
            Lista com informações de cada modelo treinado.
        """
        if problem_type == "time_series":
            algorithms = self.TIME_SERIES_ALGORITHMS
        elif is_classification:
            algorithms = self.CLASSIFICATION_ALGORITHMS
        else:
            algorithms = self.REGRESSION_ALGORITHMS

        results = []

        for name, model in algorithms.items():
            model_result = self._train_single_model(
                model=model,
                name=name,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                is_classification=is_classification,
                artifacts_dir=artifacts_dir,
                experiment_id=experiment_id,
                problem_type=problem_type
            )
            results.append(model_result)

        # Calcula ranking baseado na métrica principal
        results = self._calculate_ranking(results, is_classification, problem_type)

        return results

    def _train_single_model(
        self,
        model: Any,
        name: str,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        is_classification: bool,
        artifacts_dir: Path,
        experiment_id: int,
        problem_type: str = None
    ) -> dict[str, Any]:
        """
        Treina um único modelo e calcula suas métricas.

        Parâmetros:
            model: Instância do modelo sklearn.
            name: Nome do algoritmo.
            X_train, X_test: Dados de treino/teste.
            y_train, y_test: Targets de treino/teste.
            is_classification: Tipo de problema.
            artifacts_dir: Diretório de artefatos.
            experiment_id: ID do experimento.
            problem_type: Tipo do problema (classification, regression, time_series).

        Retorna:
            Dicionário com informações do modelo treinado.
        """
        # Treina o modelo
        model.fit(X_train, y_train)

        # Faz predições
        y_pred = model.predict(X_test)

        # Calcula métricas baseado no tipo de problema
        if problem_type == "time_series":
            metrics = self._calculate_timeseries_metrics(y_test, y_pred)
            # Para MAPE, menor é melhor, então invertemos para ranking
            primary_metric = -metrics.get("mape", float("inf"))
        elif is_classification:
            metrics = self._calculate_classification_metrics(
                model, X_test, y_test, y_pred
            )
            primary_metric = metrics.get("auc", 0)
        else:
            metrics = self._calculate_regression_metrics(y_test, y_pred)
            # Para RMSE, menor é melhor, então invertemos para ranking
            primary_metric = -metrics.get("rmse", float("inf"))

        # Salva o modelo
        model_filename = f"model_{experiment_id}_{name.lower().replace(' ', '_')}.pkl"
        model_path = artifacts_dir / model_filename
        joblib.dump(model, model_path)

        return {
            "algorithm_name": name,
            "model_path": str(model_path),
            "metrics": metrics,
            "primary_metric_value": primary_metric
        }

    def _calculate_classification_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calcula métricas de classificação.

        Retorna:
            Dicionário com AUC, accuracy, precision, recall e F1.
        """
        metrics = {}

        # Accuracy
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

        # Para AUC e outras métricas, precisamos de probabilidades
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)

                # Para binário, usa a segunda coluna
                if y_proba.shape[1] == 2:
                    metrics["auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    # Multiclasse: usa OvR
                    metrics["auc"] = float(
                        roc_auc_score(y_test, y_proba, multi_class="ovr")
                    )
            else:
                metrics["auc"] = metrics["accuracy"]
        except Exception:
            metrics["auc"] = metrics["accuracy"]

        # Precision, Recall, F1
        # Para multiclasse, usa average='weighted'
        try:
            unique_classes = len(np.unique(y_test))
            average = "binary" if unique_classes == 2 else "weighted"
            pos_label = 1 if unique_classes == 2 else None

            metrics["precision"] = float(
                precision_score(y_test, y_pred, average=average, pos_label=pos_label, zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_test, y_pred, average=average, pos_label=pos_label, zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_test, y_pred, average=average, pos_label=pos_label, zero_division=0)
            )
        except Exception:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0

        return metrics

    def _calculate_regression_metrics(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calcula métricas de regressão.

        Retorna:
            Dicionário com RMSE, MAE, R² e MAPE.
        """
        metrics = {}

        # RMSE
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))

        # MAE
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))

        # R²
        metrics["r2"] = float(r2_score(y_test, y_pred))

        # MAPE (Mean Absolute Percentage Error)
        # Evita divisão por zero
        mask = y_test != 0
        if mask.any():
            mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = 0.0

        return metrics

    def _calculate_timeseries_metrics(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calcula métricas específicas para séries temporais.

        Retorna:
            Dicionário com MAPE, RMSE, MAE, R² e SMAPE.
        """
        y_test_arr = np.array(y_test)
        metrics = {}

        # RMSE
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test_arr, y_pred)))

        # MAE
        metrics["mae"] = float(mean_absolute_error(y_test_arr, y_pred))

        # R²
        metrics["r2"] = float(r2_score(y_test_arr, y_pred))

        # MAPE (Mean Absolute Percentage Error) - métrica principal
        mask = y_test_arr != 0
        if mask.any():
            mape = np.mean(np.abs((y_test_arr[mask] - y_pred[mask]) / y_test_arr[mask])) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = 0.0

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = (np.abs(y_test_arr) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.any():
            smape = np.mean(np.abs(y_test_arr[mask] - y_pred[mask]) / denominator[mask]) * 100
            metrics["smape"] = float(smape)
        else:
            metrics["smape"] = 0.0

        return metrics

    def _calculate_ranking(
        self,
        results: list[dict[str, Any]],
        is_classification: bool,
        problem_type: str = None
    ) -> list[dict[str, Any]]:
        """
        Calcula o ranking dos modelos baseado na métrica principal.

        Para classificação: maior AUC = melhor
        Para regressão: menor RMSE = melhor (já invertido)
        Para séries temporais: menor MAPE = melhor (já invertido)

        Parâmetros:
            results: Lista de resultados dos modelos.
            is_classification: Tipo de problema.
            problem_type: Tipo do problema (classification, regression, time_series).

        Retorna:
            Lista ordenada com ranking adicionado.
        """
        # Ordena por métrica principal (decrescente)
        sorted_results = sorted(
            results,
            key=lambda x: x["primary_metric_value"],
            reverse=True
        )

        # Atribui ranking
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1

            # Ajusta o valor da métrica principal para exibição
            if problem_type == "time_series":
                # Desfaz a inversão do MAPE
                result["primary_metric_value"] = -result["primary_metric_value"]
            elif not is_classification:
                # Desfaz a inversão do RMSE
                result["primary_metric_value"] = -result["primary_metric_value"]

        return sorted_results
