import pandas as pd
import sqlite3
import logging 

# Configurer le logging 
logging.basicConfig(level= logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s' )

class DataLoader:
    def __init__(self,csv_path: str, db_path: str):
        """
        Initialiser DataLoader avec les liens vers fichier CSV et fichier SQLite (.db)
        """
        self.csv_path = csv_path
        self.db_path = db_path

    def load_csv_to_sqlite(self, table_name: str= "customers") ->None:
        """
        Doc file csv va luu vao co so du lieu
        """
        logging.info(f"Du lieu dang duoc doc tu {self.csv_path}...")
        df = pd.read_csv(self.csv_path)

        logging.info(f"Dang ket noi toi database {self.db_path}...")
        conn=sqlite3.connect(self.db_path)

        logging.info(f"Du lieu dang duoc luu vao bang {table_name}...")
        df.to_sql(name= table_name, con= conn, if_exists= 'replace', index= False)
        # if_exists= 'replace' : ghi de neu neu bang da ton tai
        # index= False: khong dien index mac dinh cua pandas vao DB
        
        #Fermer la connection:
        conn.close()
        logging.info("Da luu du lieu vaof SQLite thanh cong!")
    
    def get_data_via_sql(self, query: str) -> pd.Dataframe: # du lieu tra ve co kieu la dataframe laf mot kieu du lieu dang bang cua pandas
        """
        Keo du lieu tu SQLite bang cac query de truy van roi tra ve Pandas Dataframe.
        """
        logging.info(f"Dang thuc hien truy van: {query}")
        conn= sqlite3.connect(self.db_path)

        #Ham read_sql_query tu dong chay lenh SQL va chuyen ket qua thanh DataFrame 
        df = pd.read_sql_query(query, conn)

        conn.close()
        logging.info(f"Truy van thanh cong! ")
        return df

