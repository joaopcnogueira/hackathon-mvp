"""
Serviço de pré-processamento para séries temporais.

Prepara dados temporais para treinamento de modelos de forecasting:
- Parsing e validação da coluna de data
- Ordenação temporal
- Criação de features baseadas em lags
- Suporte a múltiplas séries (agrupadas por id_column)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Any


class TimeSeriesPreprocessingService:
    """
    Serviço responsável pelo pré-processamento de dados de séries temporais.

    Transforma dados tabulares com colunas de data, id e valor em
    estruturas prontas para modelos de forecasting.
    """

    DEFAULT_LAGS = [1, 2, 3, 7, 14]
    ROLLING_WINDOWS = [7, 14, 30]

    def __init__(self):
        self.date_column: str = ""
        self.id_column: str | None = None
        self.target_column: str = ""
        self.frequency: str | None = None
        self.transformations: list[str] = []
        self.normalization_stats: dict[str, dict[str, float]] = {}
        self.lags_used: list[int] = []
        self.has_multiple_series: bool = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        date_column: str,
        target_column: str,
        id_column: str | None = None,
        lags: list[int] | None = None
    ) -> tuple[pd.DataFrame, dict]:
        """
        Ajusta e transforma dados de séries temporais.

        Parâmetros:
            df: DataFrame com os dados brutos.
            date_column: Nome da coluna de data.
            target_column: Nome da coluna com o valor da série.
            id_column: Nome da coluna de identificação (opcional, para múltiplas séries).
            lags: Lista de lags a serem criados. Se None, usa DEFAULT_LAGS.

        Retorna:
            Tupla com (DataFrame transformado, informações do pré-processamento).
        """
        self.date_column = date_column
        self.target_column = target_column
        self.id_column = id_column
        self.transformations = []
        self.lags_used = lags if lags else self.DEFAULT_LAGS

        df = df.copy()

        # Valida e converte coluna de data
        df = self._parse_date_column(df)

        # Verifica se há múltiplas séries
        self.has_multiple_series = id_column is not None and id_column in df.columns
        if self.has_multiple_series:
            num_series = df[id_column].nunique()
            self.transformations.append(f"Identificadas {num_series} séries distintas")

        # Ordena por data (e id, se houver)
        df = self._sort_by_date(df)

        # Remove linhas com target nulo
        original_len = len(df)
        df = df.dropna(subset=[target_column])
        removed = original_len - len(df)
        if removed > 0:
            self.transformations.append(f"Removidas {removed} linhas com target nulo")

        # Detecta frequência da série
        self._detect_frequency(df)

        # Cria features de lag
        df = self._create_lag_features(df)

        # Cria features de rolling statistics
        df = self._create_rolling_features(df)

        # Cria features temporais (dia da semana, mês, etc.)
        df = self._create_datetime_features(df)

        # Normaliza o target para melhor convergência
        df = self._normalize_target(df)

        # Remove linhas com NaN gerados pelos lags
        df = self._remove_nan_rows(df)

        preprocessing_info = {
            "transformations": self.transformations,
            "original_rows": original_len,
            "processed_rows": len(df),
            "frequency": self.frequency,
            "lags_used": self.lags_used,
            "has_multiple_series": self.has_multiple_series,
            "date_column": self.date_column,
            "target_column": self.target_column,
            "id_column": self.id_column
        }

        return df, preprocessing_info

    def _parse_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converte a coluna de data para datetime.
        """
        try:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
            self.transformations.append(f"Coluna '{self.date_column}' convertida para datetime")
        except Exception as e:
            raise ValueError(f"Erro ao converter coluna de data: {e}")
        return df

    def _sort_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ordena o DataFrame por data (e id, se aplicável).
        """
        sort_columns = [self.date_column]
        if self.has_multiple_series:
            sort_columns = [self.id_column, self.date_column]

        df = df.sort_values(sort_columns).reset_index(drop=True)
        self.transformations.append("Dados ordenados cronologicamente")
        return df

    def _detect_frequency(self, df: pd.DataFrame) -> None:
        """
        Detecta a frequência da série temporal.
        """
        if self.has_multiple_series:
            # Usa a primeira série para detectar frequência
            first_id = df[self.id_column].iloc[0]
            sample_dates = df[df[self.id_column] == first_id][self.date_column]
        else:
            sample_dates = df[self.date_column]

        if len(sample_dates) >= 2:
            freq = pd.infer_freq(sample_dates)
            self.frequency = freq if freq else "irregular"
        else:
            self.frequency = "unknown"

        self.transformations.append(f"Frequência detectada: {self.frequency}")

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de lag (valores anteriores).
        """
        if self.has_multiple_series:
            for lag in self.lags_used:
                df[f"lag_{lag}"] = df.groupby(self.id_column)[self.target_column].shift(lag)
        else:
            for lag in self.lags_used:
                df[f"lag_{lag}"] = df[self.target_column].shift(lag)

        self.transformations.append(f"Criados {len(self.lags_used)} features de lag")
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de estatísticas móveis (média e desvio padrão).
        """
        for window in self.ROLLING_WINDOWS:
            if self.has_multiple_series:
                df[f"rolling_mean_{window}"] = df.groupby(self.id_column)[self.target_column].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                df[f"rolling_std_{window}"] = df.groupby(self.id_column)[self.target_column].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
            else:
                df[f"rolling_mean_{window}"] = df[self.target_column].shift(1).rolling(
                    window=window, min_periods=1
                ).mean()
                df[f"rolling_std_{window}"] = df[self.target_column].shift(1).rolling(
                    window=window, min_periods=1
                ).std()

        self.transformations.append(f"Criados features de rolling statistics (janelas: {self.ROLLING_WINDOWS})")
        return df

    def _create_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas na data (dia da semana, mês, etc.).
        """
        df["day_of_week"] = df[self.date_column].dt.dayofweek
        df["day_of_month"] = df[self.date_column].dt.day
        df["month"] = df[self.date_column].dt.month
        df["quarter"] = df[self.date_column].dt.quarter
        df["is_weekend"] = (df[self.date_column].dt.dayofweek >= 5).astype(int)

        self.transformations.append("Criados features temporais (dia_semana, dia_mes, mes, trimestre, is_weekend)")
        return df

    def _normalize_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza o target usando StandardScaler.
        Salva os parâmetros para desnormalizar as predições.
        """
        mean_val = df[self.target_column].mean()
        std_val = df[self.target_column].std()

        self.normalization_stats["target"] = {
            "mean": float(mean_val),
            "std": float(std_val) if std_val > 0 else 1.0
        }

        if std_val > 0:
            df[f"{self.target_column}_normalized"] = (df[self.target_column] - mean_val) / std_val
        else:
            df[f"{self.target_column}_normalized"] = 0

        self.transformations.append("Target normalizado (StandardScaler)")
        return df

    def _remove_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas com NaN gerados pelos lags e rolling.
        """
        original_len = len(df)
        df = df.dropna()
        removed = original_len - len(df)
        if removed > 0:
            self.transformations.append(f"Removidas {removed} linhas iniciais (warm-up dos lags)")
        return df

    def get_feature_columns(self) -> list[str]:
        """
        Retorna a lista de colunas de features geradas.
        """
        features = []

        # Lag features
        for lag in self.lags_used:
            features.append(f"lag_{lag}")

        # Rolling features
        for window in self.ROLLING_WINDOWS:
            features.append(f"rolling_mean_{window}")
            features.append(f"rolling_std_{window}")

        # Datetime features
        features.extend(["day_of_week", "day_of_month", "month", "quarter", "is_weekend"])

        return features

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide os dados em treino e teste de forma temporal (não aleatória).

        Para séries temporais, o split deve ser cronológico para evitar data leakage.

        Parâmetros:
            df: DataFrame já pré-processado.
            test_size: Proporção do conjunto de teste.

        Retorna:
            Tupla com (X_train, X_test, y_train, y_test).
        """
        feature_columns = self.get_feature_columns()
        target_col = f"{self.target_column}_normalized"

        if self.has_multiple_series:
            # Split temporal por série
            train_dfs = []
            test_dfs = []

            for series_id in df[self.id_column].unique():
                series_df = df[df[self.id_column] == series_id].copy()
                n = len(series_df)
                split_idx = int(n * (1 - test_size))

                train_dfs.append(series_df.iloc[:split_idx])
                test_dfs.append(series_df.iloc[split_idx:])

            train_df = pd.concat(train_dfs, ignore_index=True)
            test_df = pd.concat(test_dfs, ignore_index=True)
        else:
            n = len(df)
            split_idx = int(n * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

        X_train = train_df[feature_columns]
        X_test = test_df[feature_columns]
        y_train = train_df[target_col]
        y_test = test_df[target_col]

        return X_train, X_test, y_train, y_test

    def denormalize_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Desnormaliza as predições para a escala original.

        Parâmetros:
            predictions: Array com predições normalizadas.

        Retorna:
            Array com predições na escala original.
        """
        stats = self.normalization_stats.get("target", {"mean": 0, "std": 1})
        return predictions * stats["std"] + stats["mean"]

    def save_pipeline(self, filepath: Path) -> None:
        """
        Salva o pipeline de pré-processamento para uso em predições.

        Parâmetros:
            filepath: Caminho para salvar o pipeline.
        """
        pipeline = {
            "date_column": self.date_column,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "frequency": self.frequency,
            "lags_used": self.lags_used,
            "rolling_windows": self.ROLLING_WINDOWS,
            "normalization_stats": self.normalization_stats,
            "has_multiple_series": self.has_multiple_series
        }
        joblib.dump(pipeline, filepath)

    def load_pipeline(self, filepath: Path) -> None:
        """
        Carrega um pipeline de pré-processamento salvo.

        Parâmetros:
            filepath: Caminho do pipeline salvo.
        """
        pipeline = joblib.load(filepath)
        self.date_column = pipeline["date_column"]
        self.target_column = pipeline["target_column"]
        self.id_column = pipeline["id_column"]
        self.frequency = pipeline["frequency"]
        self.lags_used = pipeline["lags_used"]
        self.ROLLING_WINDOWS = pipeline.get("rolling_windows", self.ROLLING_WINDOWS)
        self.normalization_stats = pipeline["normalization_stats"]
        self.has_multiple_series = pipeline["has_multiple_series"]

    def transform_for_prediction(
        self,
        df: pd.DataFrame,
        history_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Transforma novos dados para predição, usando histórico se necessário.

        Para calcular lags e rolling stats, precisamos do histórico recente.

        Parâmetros:
            df: DataFrame com dados para predição (pode ser só as datas futuras).
            history_df: Dados históricos para calcular lags (opcional).

        Retorna:
            DataFrame com features prontas para predição.
        """
        df = df.copy()

        # Converte data
        df[self.date_column] = pd.to_datetime(df[self.date_column])

        if history_df is not None:
            history_df = history_df.copy()
            history_df[self.date_column] = pd.to_datetime(history_df[self.date_column])
            combined = pd.concat([history_df, df], ignore_index=True)
        else:
            combined = df

        # Ordena
        if self.has_multiple_series and self.id_column in combined.columns:
            combined = combined.sort_values([self.id_column, self.date_column])
        else:
            combined = combined.sort_values(self.date_column)

        # Cria features de lag
        if self.has_multiple_series:
            for lag in self.lags_used:
                combined[f"lag_{lag}"] = combined.groupby(self.id_column)[self.target_column].shift(lag)
        else:
            for lag in self.lags_used:
                combined[f"lag_{lag}"] = combined[self.target_column].shift(lag)

        # Cria rolling features
        for window in self.ROLLING_WINDOWS:
            if self.has_multiple_series:
                combined[f"rolling_mean_{window}"] = combined.groupby(self.id_column)[self.target_column].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
                combined[f"rolling_std_{window}"] = combined.groupby(self.id_column)[self.target_column].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
            else:
                combined[f"rolling_mean_{window}"] = combined[self.target_column].shift(1).rolling(
                    window=window, min_periods=1
                ).mean()
                combined[f"rolling_std_{window}"] = combined[self.target_column].shift(1).rolling(
                    window=window, min_periods=1
                ).std()

        # Cria datetime features
        combined["day_of_week"] = combined[self.date_column].dt.dayofweek
        combined["day_of_month"] = combined[self.date_column].dt.day
        combined["month"] = combined[self.date_column].dt.month
        combined["quarter"] = combined[self.date_column].dt.quarter
        combined["is_weekend"] = (combined[self.date_column].dt.dayofweek >= 5).astype(int)

        # Filtra apenas as linhas originais do df
        if history_df is not None:
            result = combined.tail(len(df)).reset_index(drop=True)
        else:
            result = combined

        # Preenche NaNs restantes com 0
        result = result.fillna(0)

        return result[self.get_feature_columns()]
