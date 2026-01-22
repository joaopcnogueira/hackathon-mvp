"""
Script para download dos datasets de avaliação do Hackathon.

Datasets utilizados:
- Classificação: Online Shoppers (dev), Adult Census (avaliação)
- Regressão: Insurance (dev), California Housing (avaliação)
- Séries Temporais: Daily Temperature (dev), Air Passengers (avaliação)

Uso:
    python download_datasets.py
"""

import os
import pandas as pd


def create_directories():
    """
    Cria a estrutura de diretórios para armazenar os datasets.
    """
    directories = ['classificacao', 'regressao', 'series_temporais']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Diretórios criados.\n")


def download_classification_datasets():
    """
    Baixa os datasets de classificação: Online Shoppers e Adult Census.
    """
    print("=== CLASSIFICAÇÃO ===\n")

    # Online Shoppers Purchasing Intention (Desenvolvimento)
    print("Baixando Online Shoppers (desenvolvimento)...")
    url_shoppers = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
    df_shoppers = pd.read_csv(url_shoppers, compression='zip')
    df_shoppers.to_csv('classificacao/online_shoppers.csv', index=False)
    print(f"  ✓ Salvo: classificacao/online_shoppers.csv ({len(df_shoppers)} registros)\n")

    # Adult Census Income (Avaliação)
    print("Baixando Adult Census (avaliação)...")
    url_adult = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv"
    columns_adult = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
        'hours_per_week', 'native_country', 'income'
    ]
    df_adult = pd.read_csv(url_adult, header=None, names=columns_adult, na_values=' ?')
    df_adult.to_csv('classificacao/adult_census.csv', index=False)
    print(f"  ✓ Salvo: classificacao/adult_census.csv ({len(df_adult)} registros)\n")


def download_regression_datasets():
    """
    Baixa os datasets de regressão: Insurance e California Housing.
    """
    print("=== REGRESSÃO ===\n")

    # Insurance (Desenvolvimento)
    print("Baixando Insurance (desenvolvimento)...")
    url_insurance = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df_insurance = pd.read_csv(url_insurance)
    df_insurance.to_csv('regressao/insurance.csv', index=False)
    print(f"  ✓ Salvo: regressao/insurance.csv ({len(df_insurance)} registros)\n")

    # California Housing (Avaliação)
    print("Baixando California Housing (avaliação)...")
    url_housing = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    df_housing = pd.read_csv(url_housing)
    df_housing.to_csv('regressao/california_housing.csv', index=False)
    print(f"  ✓ Salvo: regressao/california_housing.csv ({len(df_housing)} registros)\n")


def download_time_series_datasets():
    """
    Baixa os datasets de séries temporais: Daily Temperature e Air Passengers.
    """
    print("=== SÉRIES TEMPORAIS ===\n")

    # Daily Temperature (Desenvolvimento)
    print("Baixando Daily Temperature (desenvolvimento)...")
    url_temp = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    df_temp = pd.read_csv(url_temp)
    df_temp.columns = ['Date', 'Temperature']
    df_temp.to_csv('series_temporais/daily_temperature.csv', index=False)
    print(f"  ✓ Salvo: series_temporais/daily_temperature.csv ({len(df_temp)} registros)\n")

    # Air Passengers (Avaliação)
    print("Baixando Air Passengers (avaliação)...")
    url_air = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df_air = pd.read_csv(url_air)
    df_air.columns = ['Month', 'Passengers']
    df_air.to_csv('series_temporais/air_passengers.csv', index=False)
    print(f"  ✓ Salvo: series_temporais/air_passengers.csv ({len(df_air)} registros)\n")


def main():
    """
    Função principal que executa o download de todos os datasets.
    """
    print("\n" + "=" * 50)
    print("  DOWNLOAD DOS DATASETS DE AVALIAÇÃO")
    print("=" * 50 + "\n")

    create_directories()
    download_classification_datasets()
    download_regression_datasets()
    download_time_series_datasets()

    print("=" * 50)
    print("  ✓ Download concluído com sucesso!")
    print("=" * 50)
    print("\nPróximo passo: execute 'python gerar_splits.py' para criar os splits de avaliação.\n")


if __name__ == "__main__":
    main()
