import logging
from src.data.loader import DataLoader
from src.models.trainer import ModelTrainer

def main():
    # Setup Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=== STARTING DAY 4 PIPELINE: MODEL TRAINING ===")

    # Initialize modules
    db_path = 'data/processed/telco.db'
    csv_path = 'data/raw/telco_churn.csv' 
    loader = DataLoader(csv_path=csv_path, db_path=db_path)
    trainer = ModelTrainer(random_state=42)

    # Retrieve clean data from SQLite
    logger.info("Reading 'customers_cleaned' data from SQLite...")
    query = "SELECT * FROM customers_cleaned"
    df_cleaned = loader.get_data_via_sql(query)
    
    if df_cleaned.empty:
        logger.error("No data found. Ensure you have run the Feature Engineering pipeline (Day 3)!")
        return

    # Train/Test Split
    X_train, X_test, y_train, y_test = trainer.split_data(df=df_cleaned, target_col='Churn', test_size=0.2)

    # Handle Imbalanced Data using SMOTE (ONLY ON TRAIN SET)
    X_train_resampled, y_train_resampled = trainer.apply_smote(X_train, y_train)

    # Train the model (Example: Random Forest)
    rf_params = {
        'n_estimators': 200,
        'max_depth': 10
    }
    model = trainer.train_model(
        X_train=X_train_resampled, 
        y_train=y_train_resampled, 
        model_type='rf', 
        params=rf_params
    )

    # Save the trained model
    trainer.save_model(model_dir='models', filename='rf_smote_model.joblib')

if __name__ == "__main__":
    main()