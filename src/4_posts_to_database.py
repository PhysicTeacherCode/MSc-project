import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

core_user = input("Digite o nome do usuário principal (core_user): ").strip()

folder_path = Path(f"../data/posts")

csv_files = list(folder_path.glob(f"{core_user}_*/*.csv"))

df_posts = pd.DataFrame()

print("Criando banco de dados (pandas) de posts...")
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df_posts = pd.concat([df_posts, df], ignore_index=True)

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine_db = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

print("Salvando posts no banco de dados (PostgresSQL)...")
df_posts.to_sql(
    name=f"{core_user}_posts",
    con=engine_db,
    if_exists="replace",
    index=False
)
