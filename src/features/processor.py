import pandas as pd
import logging 

logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize DataProcessor with the input data.
        Use.copy() to avoid unintentionally modifying the original data.
        """
        self.df= df.copy()

    def handle_missing_values(self) -> pd.DataFrame:
        """
        Handle data type issues and fill missing values in colums
        """
        logging.info("PRocessing data types and missing values...")

        # Conversion "TotalCharges" to numeric
        self.df['TotalCharges'] = pd.to_numeric(
            self.df['TotalCharges'],
            errors='coerce'
        )

        # Fill missing values with 0
        self.df['TotalCharges'].fillna(0,inplace=True)

        logging.info("Missing values has been handled successfully.")
        return self.df

    def drop_unnecessary_columns(self, columns_to_drop:list) -> pd.DataFrame:
        """
        Delete columns unnecessary for the prediction
        """
        for column in columns_to_drop: 
            self.df = self.df.drop(columns = [column])
        return self.df

    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical variables into numerical format. 
        """
        logging.info("Encoding categorical variables...")

        self.df['Churn']= self.df['Churn'].map({'Yes':1, 'No':0})
        self.df = pd.get_dummies(self.df, drop_first=True, dtype=int)

        return self.df

    def process_all(self) -> pd.DataFrame:
        """
        Run the entire data cleaning pipeline sequentially
        """        

        logging.info("---STARTING DATA PIPELINE---")
        self.handle_missing_values()
        self.drop_unnecessary_columns(columns_to_drop=['customerID'])
        self.encode_categorical_features()
        logging.info("---DATA PIPELINE COMPLETED---")

        return self.df
