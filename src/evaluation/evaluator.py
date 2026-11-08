import logging
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> None:
    """Calculates and logs model performance metrics."""
    logger.info("Evaluating model performance on test set...")
    
    # In ra báo cáo phân loại (Precision, Recall, F1-Score)
    report = classification_report(y_true, y_pred)
    logger.info(f"\nClassification Report:\n{report}")
    
    # In ra Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\nConfusion Matrix:\n{cm}")