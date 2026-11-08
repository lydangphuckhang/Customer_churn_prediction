import pandas as pd
import sqlite3
import logging

logger = logging.getLogger(__name__)

def load_csv(csv_path: str) -> pd.DataFrame:
    """Loads raw data from a CSV file."""
    logger.info(f"Loading CSV data from {csv_path}")
    return pd.read_csv(csv_path)

def save_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "customers") -> None:
    """Saves DataFrame to SQLite database."""
    logger.info(f"Saving data to SQLite table '{table_name}' at {db_path}")
    with sqlite3.connect(db_path) as conn:
        df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)

def load_from_sqlite(db_path: str, query: str) -> pd.DataFrame:
    """Retrieves data from SQLite using a SQL query."""
    logger.info(f"Executing query: {query}")
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
