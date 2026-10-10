import os
import logging
from src.data.loader import DataLoader

# Configurer logging pour main.py
logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Dinh nghia cac file path:
    # Dung os dam bao duong dan chay duoc ca tren windows/mac/linux
    csv_file = os.path.join("data","raw","telco_churn_raw.csv")
    db_file = os.path.join("data","processed","telco.db")

    # Buoc phu: kiem tra xem file csv da ton tai chua truoc khi chay:
    if not os.path.exists(csv_file):
        logging.error(f"Khong tim thay file {csv_file}. Vui long kiem tra lai thu muc data/raw/")
        return 

    # Khoi tao loader
    loader = DataLoader(csv_path=csv_file, db_path= db_file)

    # Chay ham de dua du lieu tu CSV vao SQLite
    logging.info("---BAT DAU CHUYEN DU LIEU VAO DATABASE---")
    loader.load_csv_to_sqlite()

    # Kiem tra bang mot vai truy van SQL
    logging.info("---BAT DAU TRUY VAN SQL---")
    # Query: lay ra trung binh tien cuoc hang thang nhom theo gioi tinh (gender) va tinh trang roi bo (Churn)
    sql_query=""" 
        SELECT 
            gender,
            Churn,
            COUNT(customerID) as Total_Customers,
            AVG(MonthlyCharges) as Avg_Monthly_charges
        FROM customers
        GROUP BY gender, Churn 
    """

    # Keo du lieu truy van thanh Dataframe
    df_result = loader.get_data_via_sql(sql_query)

    # In ra man hinh de kiem tra
    print("\n Ket qua truy van:")
    print(df_result)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
