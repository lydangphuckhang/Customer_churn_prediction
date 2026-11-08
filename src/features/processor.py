import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values, data types, and drops unnecessary columns."""
    logger.info("Cleaning data: fixing types and missing values...")
    df = df.copy()
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    return df

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical variables into numeric formats."""
    logger.info("Encoding categorical features...")
    df = df.copy()
    
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Executes the full preprocessing pipeline."""
    df_cleaned = clean_data(df)
    df_encoded = encode_features(df_cleaned)
    return df_encoded