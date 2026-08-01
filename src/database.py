from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine
import os

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}" #here pymysql tells "Use the PyMySQL driver to communicate with MySQL.""
)

engine = create_engine(DATABASE_URL)
print("Database connection created successfully!")


#function to put our csv files into taables of MySQL
import pandas as pd
def upload_csv_to_mysql(csv_path, table_name):
    """
    Reads a CSV file and uploads it to an existing MySQL table.
    """

    df = pd.read_csv(csv_path)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False
    )

    print(f"'{table_name}' uploaded successfully!")

    