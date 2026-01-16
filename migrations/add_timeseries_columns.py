"""
Migração para adicionar colunas de séries temporais à tabela experiments.

Execute este script para atualizar um banco de dados existente:
    python migrations/add_timeseries_columns.py
"""
import sqlite3
from pathlib import Path

DATABASE_PATH = Path(__file__).parent.parent / "automl.db"


def migrate():
    """
    Adiciona as colunas de séries temporais à tabela experiments:
    - date_column: coluna de data
    - id_column: coluna de ID (múltiplas séries)
    - forecast_horizon: horizonte de previsão
    - frequency: frequência da série
    """
    if not DATABASE_PATH.exists():
        print("Banco de dados não encontrado. Será criado automaticamente ao iniciar a aplicação.")
        return

    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Verifica se as colunas já existem
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [col[1] for col in cursor.fetchall()]

    if "date_column" not in columns:
        print("Adicionando coluna 'date_column'...")
        cursor.execute("ALTER TABLE experiments ADD COLUMN date_column VARCHAR(255)")
    else:
        print("Coluna 'date_column' já existe.")

    if "id_column" not in columns:
        print("Adicionando coluna 'id_column'...")
        cursor.execute("ALTER TABLE experiments ADD COLUMN id_column VARCHAR(255)")
    else:
        print("Coluna 'id_column' já existe.")

    if "forecast_horizon" not in columns:
        print("Adicionando coluna 'forecast_horizon'...")
        cursor.execute("ALTER TABLE experiments ADD COLUMN forecast_horizon INTEGER")
    else:
        print("Coluna 'forecast_horizon' já existe.")

    if "frequency" not in columns:
        print("Adicionando coluna 'frequency'...")
        cursor.execute("ALTER TABLE experiments ADD COLUMN frequency VARCHAR(50)")
    else:
        print("Coluna 'frequency' já existe.")

    conn.commit()
    conn.close()
    print("Migração concluída com sucesso!")


if __name__ == "__main__":
    migrate()
