"""
Serviço para treinamento de modelos clássicos de séries temporais.

Implementa algoritmos estatísticos para séries temporais univariadas:
- Auto ARIMA: seleção automática de parâmetros (p,d,q)
- Exponential Smoothing: Holt-Winters com tendência e sazonalidade
- Prophet: modelo aditivo do Facebook para séries com sazonalidade

Usado quando há no máximo 1 ID distinto na série temporal.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Any
import warnings

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


class ClassicalTimeSeriesService:
    """
    Serviço para treinar modelos clássicos de séries temporais.

    Treina Auto ARIMA, Exponential Smoothing e Prophet em paralelo,
    avalia cada um e retorna ranking baseado em MAPE.
    """

    FREQUENCY_MAP = {
        "H": {"period": 24, "seasonal_periods": 24},      # Horária
        "D": {"period": 7, "seasonal_periods": 7},        # Diária
        "W": {"period": 52, "seasonal_periods": 52},      # Semanal
        "M": {"period": 12, "seasonal_periods": 12},      # Mensal
        "Q": {"period": 4, "seasonal_periods": 4},        # Trimestral
        "Y": {"period": 1, "seasonal_periods": None},     # Anual
    }

    def __init__(self):
        self.frequency: str = "D"
        self.forecast_horizon: int = 7
        self.date_column: str = ""
        self.target_column: str = ""

    def train_all_models(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        forecast_horizon: int,
        frequency: str,
        artifacts_dir: Path,
        experiment_id: int
    ) -> list[dict[str, Any]]:
        """
        Treina todos os modelos clássicos de séries temporais.

        Parâmetros:
            df: DataFrame com os dados da série temporal.
            date_column: Nome da coluna de data.
            target_column: Nome da coluna com valores da série.
            forecast_horizon: Número de períodos para prever.
            frequency: Frequência da série (D, W, M, etc.).
            artifacts_dir: Diretório para salvar modelos.
            experiment_id: ID do experimento.

        Retorna:
            Lista com informações de cada modelo treinado.
        """
        self.date_column = date_column
        self.target_column = target_column
        self.forecast_horizon = forecast_horizon
        self.frequency = frequency if frequency else "D"

        # Prepara os dados
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

        # Remove valores nulos
        df = df.dropna(subset=[target_column])

        # Split temporal: 80% treino, 20% teste
        n = len(df)
        split_idx = int(n * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        y_train = train_df[target_column].values
        y_test = test_df[target_column].values
        dates_train = train_df[date_column]
        dates_test = test_df[date_column]

        results = []

        # Treina cada modelo
        arima_result = self._train_auto_arima(
            y_train, y_test, dates_train, dates_test,
            artifacts_dir, experiment_id
        )
        if arima_result:
            results.append(arima_result)

        ets_result = self._train_exponential_smoothing(
            y_train, y_test, dates_train, dates_test,
            artifacts_dir, experiment_id
        )
        if ets_result:
            results.append(ets_result)

        prophet_result = self._train_prophet(
            train_df, test_df, dates_test,
            artifacts_dir, experiment_id
        )
        if prophet_result:
            results.append(prophet_result)

        # Calcula ranking baseado em MAPE (menor é melhor)
        results = self._calculate_ranking(results)

        return results

    def _train_auto_arima(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        dates_train: pd.Series,
        dates_test: pd.Series,
        artifacts_dir: Path,
        experiment_id: int
    ) -> dict[str, Any] | None:
        """
        Treina modelo Auto ARIMA usando pmdarima.

        Seleciona automaticamente os melhores parâmetros (p, d, q)
        usando critério de informação (AIC).
        """
        try:
            from pmdarima import auto_arima

            freq_info = self.FREQUENCY_MAP.get(self.frequency, {"period": 7})
            seasonal_period = freq_info.get("seasonal_periods", 7)

            # Auto ARIMA com seleção automática de parâmetros
            model = auto_arima(
                y_train,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                m=seasonal_period if seasonal_period and seasonal_period > 1 else 1,
                seasonal=seasonal_period is not None and seasonal_period > 1,
                d=None,  # Auto-detecta diferenciação
                D=None,  # Auto-detecta diferenciação sazonal
                trace=False,
                error_action="ignore",
                suppress_warnings=True,
                stepwise=True,
                random_state=42
            )

            # Faz previsão para o período de teste
            y_pred = model.predict(n_periods=len(y_test))

            # Calcula métricas
            metrics = self._calculate_metrics(y_test, y_pred)

            # Salva o modelo
            model_path = str(artifacts_dir / f"arima_{experiment_id}.pkl")
            joblib.dump({
                "model": model,
                "order": model.order,
                "seasonal_order": model.seasonal_order,
                "frequency": self.frequency,
                "forecast_horizon": self.forecast_horizon
            }, model_path)

            return {
                "algorithm_name": "Auto ARIMA",
                "model_path": model_path,
                "metrics": metrics,
                "primary_metric_value": -metrics["mape"],  # Negativo para ranking
                "extra_info": {
                    "order": model.order,
                    "seasonal_order": model.seasonal_order,
                    "aic": float(model.aic())
                }
            }

        except Exception as e:
            print(f"Erro ao treinar Auto ARIMA: {e}")
            return None

    def _train_exponential_smoothing(
        self,
        y_train: np.ndarray,
        y_test: np.ndarray,
        dates_train: pd.Series,
        dates_test: pd.Series,
        artifacts_dir: Path,
        experiment_id: int
    ) -> dict[str, Any] | None:
        """
        Treina modelo Exponential Smoothing (Holt-Winters).

        Captura tendência e sazonalidade usando suavização exponencial.
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            freq_info = self.FREQUENCY_MAP.get(self.frequency, {"seasonal_periods": None})
            seasonal_periods = freq_info.get("seasonal_periods")

            # Configura sazonalidade baseado na frequência
            use_seasonal = seasonal_periods is not None and seasonal_periods > 1
            use_seasonal = use_seasonal and len(y_train) >= 2 * seasonal_periods

            if use_seasonal:
                model = ExponentialSmoothing(
                    y_train,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(
                    y_train,
                    trend="add",
                    seasonal=None
                )

            fitted_model = model.fit(optimized=True)

            # Faz previsão para o período de teste
            y_pred = fitted_model.forecast(len(y_test))

            # Calcula métricas
            metrics = self._calculate_metrics(y_test, y_pred)

            # Salva o modelo
            model_path = str(artifacts_dir / f"ets_{experiment_id}.pkl")
            joblib.dump({
                "model": fitted_model,
                "seasonal_periods": seasonal_periods,
                "use_seasonal": use_seasonal,
                "frequency": self.frequency,
                "forecast_horizon": self.forecast_horizon
            }, model_path)

            return {
                "algorithm_name": "Exponential Smoothing",
                "model_path": model_path,
                "metrics": metrics,
                "primary_metric_value": -metrics["mape"],  # Negativo para ranking
                "extra_info": {
                    "seasonal_periods": seasonal_periods,
                    "use_seasonal": use_seasonal,
                    "aic": float(fitted_model.aic) if hasattr(fitted_model, "aic") else None
                }
            }

        except Exception as e:
            print(f"Erro ao treinar Exponential Smoothing: {e}")
            return None

    def _train_prophet(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dates_test: pd.Series,
        artifacts_dir: Path,
        experiment_id: int
    ) -> dict[str, Any] | None:
        """
        Treina modelo Prophet do Facebook.

        Captura múltiplas sazonalidades (diária, semanal, anual)
        e feriados automaticamente.
        """
        try:
            from prophet import Prophet

            # Prophet requer colunas 'ds' e 'y'
            prophet_train = pd.DataFrame({
                "ds": train_df[self.date_column],
                "y": train_df[self.target_column]
            })

            prophet_test = pd.DataFrame({
                "ds": test_df[self.date_column]
            })

            # Configura Prophet baseado na frequência
            freq_info = self.FREQUENCY_MAP.get(self.frequency, {})

            model = Prophet(
                yearly_seasonality=self.frequency in ["D", "W", "M"],
                weekly_seasonality=self.frequency in ["D", "H"],
                daily_seasonality=self.frequency == "H",
                interval_width=0.95
            )

            model.fit(prophet_train)

            # Faz previsão para o período de teste
            forecast = model.predict(prophet_test)
            y_pred = forecast["yhat"].values
            y_test = test_df[self.target_column].values

            # Calcula métricas
            metrics = self._calculate_metrics(y_test, y_pred)

            # Salva o modelo
            model_path = str(artifacts_dir / f"prophet_{experiment_id}.pkl")
            joblib.dump({
                "model": model,
                "frequency": self.frequency,
                "forecast_horizon": self.forecast_horizon,
                "date_column": self.date_column,
                "target_column": self.target_column
            }, model_path)

            return {
                "algorithm_name": "Prophet",
                "model_path": model_path,
                "metrics": metrics,
                "primary_metric_value": -metrics["mape"],  # Negativo para ranking
                "extra_info": {
                    "yearly_seasonality": self.frequency in ["D", "W", "M"],
                    "weekly_seasonality": self.frequency in ["D", "H"]
                }
            }

        except Exception as e:
            print(f"Erro ao treinar Prophet: {e}")
            return None

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Calcula métricas de avaliação para séries temporais.

        Retorna:
            Dicionário com MAPE, SMAPE, RMSE, MAE e R².
        """
        metrics = {}

        # RMSE
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

        # MAE
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

        # R²
        metrics["r2"] = float(r2_score(y_true, y_pred))

        # MAPE (Mean Absolute Percentage Error) - métrica principal
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            metrics["mape"] = float(mape)
        else:
            metrics["mape"] = 0.0

        # SMAPE (Symmetric Mean Absolute Percentage Error)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if mask.any():
            smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
            metrics["smape"] = float(smape)
        else:
            metrics["smape"] = 0.0

        return metrics

    def _calculate_ranking(
        self,
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Calcula o ranking dos modelos baseado em MAPE.

        Menor MAPE = melhor posição no ranking.
        """
        # Ordena por métrica principal (já é negativo, então maior = menor MAPE)
        sorted_results = sorted(
            results,
            key=lambda x: x["primary_metric_value"],
            reverse=True
        )

        # Atribui ranking e desfaz inversão
        for i, result in enumerate(sorted_results):
            result["rank"] = i + 1
            result["primary_metric_value"] = -result["primary_metric_value"]

        return sorted_results

    def forecast(
        self,
        model_path: str,
        algorithm_name: str,
        last_data: pd.DataFrame,
        periods: int
    ) -> pd.DataFrame:
        """
        Gera previsões usando um modelo treinado.

        Parâmetros:
            model_path: Caminho do modelo salvo.
            algorithm_name: Nome do algoritmo (Auto ARIMA, Exponential Smoothing, Prophet).
            last_data: Dados recentes para contexto (necessário para alguns modelos).
            periods: Número de períodos para prever.

        Retorna:
            DataFrame com datas e previsões.
        """
        model_data = joblib.load(model_path)

        if algorithm_name == "Auto ARIMA":
            return self._forecast_arima(model_data, periods)
        elif algorithm_name == "Exponential Smoothing":
            return self._forecast_ets(model_data, periods)
        elif algorithm_name == "Prophet":
            return self._forecast_prophet(model_data, last_data, periods)
        else:
            raise ValueError(f"Algoritmo desconhecido: {algorithm_name}")

    def _forecast_arima(
        self,
        model_data: dict,
        periods: int
    ) -> pd.DataFrame:
        """
        Gera previsões com modelo ARIMA.
        """
        model = model_data["model"]
        predictions = model.predict(n_periods=periods)

        return pd.DataFrame({
            "period": range(1, periods + 1),
            "forecast": predictions
        })

    def _forecast_ets(
        self,
        model_data: dict,
        periods: int
    ) -> pd.DataFrame:
        """
        Gera previsões com modelo Exponential Smoothing.
        """
        model = model_data["model"]
        predictions = model.forecast(periods)

        return pd.DataFrame({
            "period": range(1, periods + 1),
            "forecast": predictions
        })

    def _forecast_prophet(
        self,
        model_data: dict,
        last_data: pd.DataFrame,
        periods: int
    ) -> pd.DataFrame:
        """
        Gera previsões com modelo Prophet.
        """
        model = model_data["model"]
        date_column = model_data["date_column"]

        # Cria dataframe com datas futuras
        last_date = pd.to_datetime(last_data[date_column]).max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=periods,
            freq=model_data["frequency"]
        )

        future_df = pd.DataFrame({"ds": future_dates})
        forecast = model.predict(future_df)

        return pd.DataFrame({
            "date": future_dates,
            "forecast": forecast["yhat"].values,
            "lower_bound": forecast["yhat_lower"].values,
            "upper_bound": forecast["yhat_upper"].values
        })
