import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

user = input("Digite o nome do usuário: ").strip()

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine_db = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

df = pd.read_sql(f'SELECT * FROM "{user}_palavras_desvio"',con=engine_db)

print(df.head)

df.to_csv(f"../data/posts/{user}_(1047)/words_data/{user}_palavras_desvio.csv", index=False)