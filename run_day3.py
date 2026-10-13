import pandas as pd
import os
import logging
from src.data.loader import DataLoader
from src.features.processor import DataProcessor
import sqlite3
# Configurer logging pour main.py
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

def run_day3_pipeline():
    # Define file paths
    csv_file = os.path.join("data","raw","telco_churn_raw.csv")
    db_file = os.path.join("data","processed","telco.db")

    # Initialize DataLoader and retrieve raw data
    loader = DataLoader(csv_path=csv_file, db_path= db_file)
    query= "SELECT * FROM customers"
    df_raw = loader.get_data_via_sql(query)
    logging.info(f"Raw data shape: {df_raw.shape}")

    # Initialize DataProcessor and run the pipeline 
    processor = DataProcessor(df_raw)
    df_clean= processor.process_all()
    logging.info(f"Processed data shape: {df_clean.shape}")

    print("\n---CHECKING DATA TYPES (All columns should be numeric) ---")
    df_clean.info()

    print("\n" + "="*50 + "\n")


    # Save the cleaned data to the DATABASE
    logging.info("Transmission of cleaned data on process...")
    conn=sqlite3.connect(db_file)
    df_clean.to_sql(name= 'customers_cleaned', con= conn, if_exists= 'replace', index= False)
    conn.close()
    logging.info("Saved")

if __name__ == "__main__":
    run_day3_pipeline()