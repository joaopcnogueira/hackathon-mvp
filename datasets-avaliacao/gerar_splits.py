"""
Script para gerar os splits dos datasets de avaliacao do hackathon.

Gera as seguintes divisoes:
- Classificacao (Adult Census): 70% funcionalidade, 30% performance (stratified)
- Regressao (California Housing): 70% funcionalidade, 30% performance
- Series Temporais (Air Passengers): primeiros 132 meses (func), ultimos 12 meses (perf)

NOTA: Os arquivos originais (adult_census.csv, california_housing.csv, air_passengers.csv)
sao necessarios apenas na primeira execucao. Apos gerar os splits, os originais podem ser
removidos. O script verifica se os splits ja existem antes de tentar gera-los.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def split_adult_census():
    """
    Divide o dataset Adult Census em funcionalidade (70%) e performance (30%).
    Usa stratified split para manter a proporcao das classes.
    """
    print("Processando Adult Census...")

    func_path = Path('classificacao/adult_census_funcionalidade.csv')
    perf_path = Path('classificacao/adult_census_performance.csv')

    if func_path.exists() and perf_path.exists():
        print("  Splits ja existem, pulando...")
        df_func = pd.read_csv(func_path)
        df_perf = pd.read_csv(perf_path)
        print(f"  Funcionalidade: {len(df_func)} registros")
        print(f"  Performance: {len(df_perf)} registros")
        return

    df = pd.read_csv('classificacao/adult_census.csv')
    print(f"  Total de registros: {len(df)}")

    df_func, df_perf = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df['income']
    )

    df_func.to_csv('classificacao/adult_census_funcionalidade.csv', index=False)
    df_perf.to_csv('classificacao/adult_census_performance.csv', index=False)

    print(f"  Funcionalidade: {len(df_func)} registros ({len(df_func)/len(df)*100:.1f}%)")
    print(f"  Performance: {len(df_perf)} registros ({len(df_perf)/len(df)*100:.1f}%)")
    print(f"  Distribuicao target (func): {df_func['income'].value_counts().to_dict()}")
    print(f"  Distribuicao target (perf): {df_perf['income'].value_counts().to_dict()}")


def split_california_housing():
    """
    Divide o dataset California Housing em funcionalidade (70%) e performance (30%).
    """
    print("\nProcessando California Housing...")

    func_path = Path('regressao/california_housing_funcionalidade.csv')
    perf_path = Path('regressao/california_housing_performance.csv')

    if func_path.exists() and perf_path.exists():
        print("  Splits ja existem, pulando...")
        df_func = pd.read_csv(func_path)
        df_perf = pd.read_csv(perf_path)
        print(f"  Funcionalidade: {len(df_func)} registros")
        print(f"  Performance: {len(df_perf)} registros")
        return

    df = pd.read_csv('regressao/california_housing.csv')
    print(f"  Total de registros: {len(df)}")

    df_func, df_perf = train_test_split(
        df,
        test_size=0.3,
        random_state=42
    )

    df_func.to_csv('regressao/california_housing_funcionalidade.csv', index=False)
    df_perf.to_csv('regressao/california_housing_performance.csv', index=False)

    print(f"  Funcionalidade: {len(df_func)} registros ({len(df_func)/len(df)*100:.1f}%)")
    print(f"  Performance: {len(df_perf)} registros ({len(df_perf)/len(df)*100:.1f}%)")


def split_air_passengers():
    """
    Divide o dataset Air Passengers temporalmente:
    - Funcionalidade: primeiros 132 meses (1949-01 a 1959-12)
    - Performance: ultimos 12 meses (1960-01 a 1960-12)
    """
    print("\nProcessando Air Passengers...")

    func_path = Path('series_temporais/air_passengers_funcionalidade.csv')
    perf_path = Path('series_temporais/air_passengers_performance.csv')

    if func_path.exists() and perf_path.exists():
        print("  Splits ja existem, pulando...")
        df_func = pd.read_csv(func_path)
        df_perf = pd.read_csv(perf_path)
        print(f"  Funcionalidade: {len(df_func)} meses ({df_func['Month'].iloc[0]} a {df_func['Month'].iloc[-1]})")
        print(f"  Performance: {len(df_perf)} meses ({df_perf['Month'].iloc[0]} a {df_perf['Month'].iloc[-1]})")
        return

    df = pd.read_csv('series_temporais/air_passengers.csv')
    print(f"  Total de registros: {len(df)}")
    print(f"  Periodo completo: {df['Month'].iloc[0]} a {df['Month'].iloc[-1]}")

    # Ultimos 12 meses para performance, resto para funcionalidade
    df_func = df.iloc[:-12].copy()
    df_perf = df.iloc[-12:].copy()

    df_func.to_csv('series_temporais/air_passengers_funcionalidade.csv', index=False)
    df_perf.to_csv('series_temporais/air_passengers_performance.csv', index=False)

    print(f"  Funcionalidade: {len(df_func)} meses ({df_func['Month'].iloc[0]} a {df_func['Month'].iloc[-1]})")
    print(f"  Performance: {len(df_perf)} meses ({df_perf['Month'].iloc[0]} a {df_perf['Month'].iloc[-1]})")


def main():
    print("=" * 60)
    print("Gerando splits dos datasets de avaliacao")
    print("=" * 60)

    split_adult_census()
    split_california_housing()
    split_air_passengers()

    print("\n" + "=" * 60)
    print("Splits gerados com sucesso!")
    print("=" * 60)
    print("\nArquivos criados:")
    print("  - classificacao/adult_census_funcionalidade.csv")
    print("  - classificacao/adult_census_performance.csv")
    print("  - regressao/california_housing_funcionalidade.csv")
    print("  - regressao/california_housing_performance.csv")
    print("  - series_temporais/air_passengers_funcionalidade.csv")
    print("  - series_temporais/air_passengers_performance.csv")


if __name__ == "__main__":
    main()
