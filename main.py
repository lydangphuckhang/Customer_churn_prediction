import logging
import os
from src import config
from src.data import loader
from src.features import processor
from src.models import trainer
from src.evaluation import evaluator

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== STARTING TELCO CHURN PIPELINE ===")
    
   
    logger.info("DATA PROCESSING")
    df_raw = loader.load_csv(config.RAW_DATA_PATH)
    df_processed = processor.preprocess_pipeline(df_raw)
    
    loader.save_to_sqlite(df_processed, config.DB_PATH, table_name="customers_cleaned")
    
  
    logger.info("MODEL TRAINING")
    df_clean = loader.load_from_sqlite(config.DB_PATH, "SELECT * FROM customers_cleaned")
    

    model_trainer = trainer.ModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = model_trainer.split_data(df_clean, target_col='Churn')
    
    X_train_res, y_train_res = model_trainer.apply_smote(X_train, y_train)
    
    rf_params = {'n_estimators': 200, 'max_depth': 10}
    model_trainer.train_model(X_train_res, y_train_res, model_type='rf', params=rf_params)
    
    model_trainer.save_model(model_dir=config.MODEL_DIR, filename='rf_smote_model.joblib')

    logger.info("MODEL EVALUATION")
    trained_model = model_trainer.model 
    y_pred = trained_model.predict(X_test)
    evaluator.evaluate_model(y_test, y_pred)
    
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")

if __name__ == "__main__":
    main()