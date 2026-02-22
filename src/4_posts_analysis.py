import pandas as pd
from pathlib import Path

core_user = input("Digite o nome do usuário principal (core_user): ").strip()

folder_path = Path("../data/posts")

csv_files = list(folder_path.glob(f"{core_user}_*/*.csv"))

df_posts = pd.DataFrame()

print("Criando banco de dados (pandas) de posts...")
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df_posts = pd.concat([df_posts, df], ignore_index=True)

output_dir = Path(f"../data/posts/{core_user}_merged")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{core_user}_posts.csv"

print("Salvando posts em CSV (pandas)...")
df_posts.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")
