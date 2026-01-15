"""
Serviço de pré-processamento automático de dados.

Aplica transformações conservadoras:
- Imputação de valores nulos (mediana/moda)
- Target Encoding para variáveis categóricas
- Normalização de variáveis numéricas
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from typing import Any


class PreprocessingService:
    """
    Serviço responsável pelo pré-processamento automático de datasets.

    Aplica transformações conservadoras que preservam os dados
    enquanto preparam para treinamento de modelos.
    """

    def __init__(self):
        self.transformations: list[str] = []
        self.numeric_columns: list[str] = []
        self.categorical_columns: list[str] = []
        self.target_encodings: dict[str, dict[Any, float]] = {}
        self.numeric_stats: dict[str, dict[str, float]] = {}
        self.nulls_filled: int = 0

    def analyze_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Analisa o DataFrame e retorna informações sobre as colunas.

        Parâmetros:
            df: DataFrame a ser analisado.

        Retorna:
            Dicionário com informações das colunas.
        """
        columns_info = []

        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            null_percentage = (null_count / len(df)) * 100 if len(df) > 0 else 0

            sample_values = df[col].dropna().head(5).tolist()
            sample_values = [str(v) for v in sample_values]

            columns_info.append({
                "name": col,
                "dtype": dtype,
                "unique_count": int(unique_count),
                "null_count": int(null_count),
                "null_percentage": float(null_percentage),
                "sample_values": sample_values
            })

        return {
            "columns_info": columns_info,
            "num_rows": len(df),
            "num_columns": len(df.columns)
        }

    def preprocess(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        is_classification: bool
    ) -> tuple[pd.DataFrame, pd.Series, dict]:
        """
        Executa o pré-processamento completo do dataset.

        Parâmetros:
            df: DataFrame com os dados.
            target_column: Nome da coluna target.
            feature_columns: Lista de colunas de features.
            is_classification: Se é um problema de classificação.

        Retorna:
            Tupla com (X processado, y, informações do pré-processamento).
        """
        self.transformations = []
        self.nulls_filled = 0
        original_rows = len(df)

        # Separa features e target
        X = df[feature_columns].copy()
        y = df[target_column].copy()

        # Remove linhas onde o target é nulo
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < original_rows:
            removed = original_rows - len(y)
            self.transformations.append(f"Removidas {removed} linhas com target nulo")

        # Identifica tipos de colunas
        self._identify_column_types(X)

        # Processa valores nulos
        X = self._handle_missing_values(X)

        # Processa colunas categóricas com Target Encoding
        X = self._encode_categorical(X, y)

        # Normaliza colunas numéricas
        X = self._normalize_numeric(X)

        # Converte tipos numpy para tipos nativos Python (JSON serializable)
        preprocessing_info = {
            "transformations": self.transformations,
            "original_rows": int(original_rows),
            "processed_rows": int(len(X)),
            "num_features": int(len(X.columns)),
            "nulls_filled": int(self.nulls_filled),
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns)
        }

        return X, y, preprocessing_info

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """
        Identifica quais colunas são numéricas e quais são categóricas.
        """
        self.numeric_columns = []
        self.categorical_columns = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_columns.append(col)
            else:
                self.categorical_columns.append(col)

        if self.categorical_columns:
            self.transformations.append(
                f"Identificadas {len(self.categorical_columns)} colunas categóricas"
            )
        if self.numeric_columns:
            self.transformations.append(
                f"Identificadas {len(self.numeric_columns)} colunas numéricas"
            )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trata valores nulos com estratégia conservadora.
        Numéricas: mediana
        Categóricas: moda (valor mais frequente)
        """
        for col in self.numeric_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                self.nulls_filled += null_count

        for col in self.categorical_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                mode_value = df[col].mode()
                if len(mode_value) > 0:
                    df[col] = df[col].fillna(mode_value[0])
                else:
                    df[col] = df[col].fillna("UNKNOWN")
                self.nulls_filled += null_count

        if self.nulls_filled > 0:
            self.transformations.append(
                f"Preenchidos {self.nulls_filled} valores nulos (mediana/moda)"
            )

        return df

    def _encode_categorical(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Aplica Target Encoding nas colunas categóricas.
        Substitui cada categoria pela média do target para aquela categoria.
        """
        if not self.categorical_columns:
            return X

        global_mean = y.mean() if pd.api.types.is_numeric_dtype(y) else 0

        for col in self.categorical_columns:
            if col not in X.columns:
                continue

            encoding_map = {}
            for category in X[col].unique():
                mask = X[col] == category
                if pd.api.types.is_numeric_dtype(y):
                    encoding_map[category] = y[mask].mean()
                else:
                    # Para classificação com target não numérico, usa frequência
                    encoding_map[category] = mask.sum() / len(mask)

            self.target_encodings[col] = encoding_map
            X[col] = X[col].map(encoding_map).fillna(global_mean)

        self.transformations.append(
            f"Aplicado Target Encoding em {len(self.categorical_columns)} colunas"
        )

        return X

    def _normalize_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza colunas numéricas usando StandardScaler (média 0, std 1).
        """
        for col in self.numeric_columns:
            if col not in df.columns:
                continue

            mean_val = df[col].mean()
            std_val = df[col].std()

            self.numeric_stats[col] = {
                "mean": float(mean_val),
                "std": float(std_val) if std_val > 0 else 1.0
            }

            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
            else:
                df[col] = 0

        # Normaliza também as colunas que foram target encoded
        for col in self.categorical_columns:
            if col not in df.columns:
                continue

            mean_val = df[col].mean()
            std_val = df[col].std()

            self.numeric_stats[col] = {
                "mean": float(mean_val),
                "std": float(std_val) if std_val > 0 else 1.0
            }

            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
            else:
                df[col] = 0

        self.transformations.append("Normalização aplicada (StandardScaler)")

        return df

    def save_pipeline(self, filepath: Path) -> None:
        """
        Salva o pipeline de pré-processamento para uso em predições.

        Parâmetros:
            filepath: Caminho para salvar o pipeline.
        """
        pipeline = {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "target_encodings": self.target_encodings,
            "numeric_stats": self.numeric_stats
        }
        joblib.dump(pipeline, filepath)

    def load_pipeline(self, filepath: Path) -> None:
        """
        Carrega um pipeline de pré-processamento salvo.

        Parâmetros:
            filepath: Caminho do pipeline salvo.
        """
        pipeline = joblib.load(filepath)
        self.numeric_columns = pipeline["numeric_columns"]
        self.categorical_columns = pipeline["categorical_columns"]
        self.target_encodings = pipeline["target_encodings"]
        self.numeric_stats = pipeline["numeric_stats"]

    def transform(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        """
        Aplica as transformações em novos dados (para predição).
        Usa os parâmetros aprendidos durante o fit.

        Parâmetros:
            df: DataFrame com novos dados.
            feature_columns: Colunas de features esperadas.

        Retorna:
            DataFrame transformado.

        Observação:
            Colunas extras no DataFrame são ignoradas.
            Colunas faltantes são preenchidas com valor padrão (0 normalizado).
        """
        # Cria mapeamento case-insensitive das colunas do DataFrame
        df_columns_lower = {col.lower().strip(): col for col in df.columns}

        # Mapeia colunas esperadas para colunas reais do DataFrame
        column_mapping = {}
        for expected_col in feature_columns:
            expected_lower = expected_col.lower().strip()
            if expected_lower in df_columns_lower:
                column_mapping[expected_col] = df_columns_lower[expected_lower]

        available_columns = list(column_mapping.keys())
        missing_columns = [col for col in feature_columns if col not in column_mapping]

        # Cria DataFrame com colunas renomeadas para o nome esperado
        X = pd.DataFrame()
        for expected_col, actual_col in column_mapping.items():
            X[expected_col] = df[actual_col].copy()

        # Adiciona colunas faltantes com valor padrão (0 = valor normalizado médio)
        for col in missing_columns:
            X[col] = 0.0

        # Reordena para manter a ordem original das features
        X = X[feature_columns]

        # Preenche nulos em numéricas com 0 (já normalizado)
        for col in self.numeric_columns:
            if col in X.columns:
                X[col] = X[col].fillna(0)

        # Aplica target encoding nas categóricas
        for col in self.categorical_columns:
            if col not in X.columns:
                continue

            if col in self.target_encodings:
                encoding_map = self.target_encodings[col]
                global_mean = np.mean(list(encoding_map.values()))
                X[col] = X[col].map(encoding_map).fillna(global_mean)

        # Normaliza todas as colunas
        for col in feature_columns:
            if col in self.numeric_stats:
                stats = self.numeric_stats[col]
                X[col] = (X[col] - stats["mean"]) / stats["std"]

        return X
