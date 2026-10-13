import logging
import os
from typing import Dict, Any, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

class ModelTrainer:
    """
    Handles the Machine Learning model training process.
    Includes data splitting, addressing class imbalance (SMOTE), training, and model serialization.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if none exists
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
        self.model = None

    def split_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and testing sets.
        
        Args:
            df: The cleaned DataFrame (customers_cleaned).
            target_col: The name of the target column (Churn).
            test_size: Proportion of the dataset to include in the test split.
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        self.logger.info(f"Starting data split (Test size: {test_size})...")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        self.logger.info(f"Data splitting completed. Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Applies SMOTE to handle imbalanced data.
        CRITICAL: Apply ONLY to the training set to prevent data leakage.
        """
        self.logger.info("Applying SMOTE to balance the training data...")
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        self.logger.info(f"Shape after SMOTE: {X_train_resampled.shape}. Target distribution:\n{y_train_resampled.value_counts()}")
        return X_train_resampled, y_train_resampled

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'rf', params: Dict[str, Any] = None) -> Any:
        """
        Trains the machine learning model. Supports Random Forest ('rf') and XGBoost ('xgb').
        """
        self.logger.info(f"Starting model training: {model_type.upper()}...")
        
        if params is None:
            params = {}

        if model_type == 'rf':
            rf_params = {'n_estimators': 100, 'random_state': self.random_state, **params}
            self.model = RandomForestClassifier(**rf_params)
        elif model_type == 'xgb':
            xgb_params = {'use_label_encoder': False, 'eval_metric': 'logloss', 'random_state': self.random_state, **params}
            self.model = XGBClassifier(**xgb_params)
        else:
            raise ValueError("Unsupported model_type. Please use 'rf' (Random Forest) or 'xgb' (XGBoost).")

        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed successfully.")
        return self.model

    def save_model(self, model_dir: str = 'models', filename: str = 'model.joblib') -> str:
        """
        Serializes and saves the trained model to the specified directory.
        """
        if self.model is None:
            raise ValueError("No trained model found. Please call train_model() first.")
            
        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model successfully saved at: {filepath}")
        return filepath