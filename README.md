# Telco Customer Churn Prediction 

## Project Overview
This project builds an end-to-end Machine Learning pipeline to predict customer churn in the telecommunications industry. The pipeline seamlessly handles data ingestion from an SQLite database, functional data preprocessing, class imbalance resolution using SMOTE, and model training/evaluation using Random Forest. 

It is designed with production-like principles: centralized configuration, robust logging, functional data transformations, and clear separation of concerns.

## Project Structure
```text
churn_project/
├── data/
│   ├── raw/             # Raw dataset (telco_churn.csv)
│   └── processed/       # SQLite database containing cleaned data (telco.db)
├── models/              # Serialized trained models (.joblib)
├── notebooks/           # Jupyter notebooks for Exploratory Data Analysis (EDA)
├── src/                 # Source code for the ML pipeline
│   ├── config.py        # Centralized path and environment configurations
│   ├── data/            # Data loading and SQLite interactions (loader.py)
│   ├── features/        # Data cleaning and feature engineering (processor.py)
│   ├── models/          # Model training and SMOTE application (trainer.py)
│   └── evaluation/      # Metrics and performance evaluation (evaluator.py)
├── main.py              # Main orchestrator script to run the full pipeline
├── requirements.txt     # Frozen project dependencies
└── README.md            # Project documentation
```

##  How to Run

**1. Clone the repository and set up a virtual environment:**
```bash
# Create and activate a virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the Machine Learning Pipeline:**
Execute the orchestrator script to run data processing, model training, and evaluation sequentially.
```bash
python main.py
```

##  Key Results & Business Impact

* **Imbalance Handling:** Applied SMOTE exclusively on the training set to prevent data leakage, addressing the natural class imbalance of the dataset.
* **Model Performance:** The Random Forest model achieved a **70% Recall** for the minority class (Churn), successfully identifying 261 out of 374 churning customers in the test set. The overall F1-score for the churn class is 0.61.
* **Business Impact:** The pipeline is strategically optimized to minimize False Negatives. In a real-world business context, it is more cost-effective to occasionally send retention promotions to loyal customers (False Positives) than to completely miss at-risk customers who are about to leave (False Negatives).

##  Tech Stack
* **Language:** Python 3.13
* **Data Processing:** Pandas, Numpy
* **Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Storage:** SQLite
* **Tracking:** Python Native Logging