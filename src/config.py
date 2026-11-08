import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "telco_churn_raw.csv")
DB_PATH = os.path.join(DATA_DIR, "processed", "telco.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")