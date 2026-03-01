import pandas as pd
from pathlib import Path
import shutil

"""
Análise de palavras por desvio padrão de tempo entre ocorrências.

Ordem de ações:
1. Localiza todas as pastas core_user em MSc-project/data/posts
2. Permite ao usuário escolher qual core_user processar
3. Lê todos os CSVs da subpasta raw_data e concatena em memória (substitui o _merged.csv)
4. Gera dois arquivos de saída:
   - _palavras_desvio.csv: uma linha por palavra, com n_ocorrencias e desvio_padrao
     calculados a partir de todos os posts de todos os handles
   - _palavras.csv: uma linha por handle, com todas as palavras utilizadas por ele
5. Remove a pasta raw_data do usuário selecionado
"""


def list_core_user_folders(base_dir):
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("core_user_")]
    return sorted(folders, key=lambda p: p.name)


base_posts_dir = Path("MSc-project") / "data" / "posts"
core_user_folders = list_core_user_folders(base_posts_dir)

if not core_user_folders:
    raise SystemExit("Nenhuma pasta core_user encontrada em MSc-project/data/posts")

print("Escolha a pasta do core user:")
for idx, folder in enumerate(core_user_folders, start=1):
    print(f"{idx}. {folder.name}")

while True:
    choice = input(f"\nDigite o numero da pasta desejada (1-{len(core_user_folders)}): ").strip()
    if choice.isdigit():
        num = int(choice)
        if 1 <= num <= len(core_user_folders):
            core_folder = core_user_folders[num - 1]
            break
    print("Escolha invalida. Tente novamente.")

core_user = core_folder.name
print(f"\nProcessando {core_user}...")

# Lê e concatena todos os CSVs da raw_data em memória (sem salvar _merged.csv)
raw_data_dir = core_folder / "raw_data"
csv_files = list(raw_data_dir.glob("*.csv"))

if not csv_files:
    raise SystemExit(f"Nenhum CSV encontrado em {raw_data_dir}")

dfs = [pd.read_csv(f) for f in csv_files]
source_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

for col in ["user", "post", "date"]:
    if col not in source_df.columns:
        raise SystemExit(f"Coluna '{col}' nao encontrada nos CSVs de {core_user}")

source_df["date"] = pd.to_datetime(source_df["date"], format="ISO8601", utc=True)
reference_date = pd.to_datetime("2023-01-01", utc=True)
source_df["idade"] = (source_df["date"] - reference_date).dt.days

source_df["post_limpo"] = (
    source_df["post"]
    .fillna("")
    .str.lower()
    .str.replace(r"[^a-zà-öø-ÿ]", " ", regex=True)
    .str.split()
)

# Explode mantendo handle e idade
df_palavras = source_df[["user", "post_limpo", "idade"]].explode("post_limpo")
df_palavras = df_palavras[df_palavras["post_limpo"].str.strip() != ""]
df_palavras = df_palavras.rename(columns={"post_limpo": "palavra"})
df_palavras["palavra"] = df_palavras["palavra"].str.strip()

if df_palavras.empty:
    raise SystemExit(f"Nenhuma palavra encontrada em {core_user}")

# --- CSV 1: _palavras_desvio.csv ---
# Desvio padrão calculado sobre todos os posts de todos os handles
min_idade = df_palavras.groupby("palavra")["idade"].transform("min")
df_palavras["dias_entre_ocorrencias"] = df_palavras["idade"] - min_idade

resultado_desvio = df_palavras.groupby("palavra").agg(
    n_ocorrencias=("palavra", "count"),
    desvio_padrao=("dias_entre_ocorrencias", "std")
).reset_index()

resultado_desvio = resultado_desvio[resultado_desvio["n_ocorrencias"] > 2]
resultado_desvio = resultado_desvio.sort_values("desvio_padrao", ascending=False)

output_dir = base_posts_dir / core_user
output_dir.mkdir(parents=True, exist_ok=True)

output_path_desvio = output_dir / f"{core_user}_palavras_desvio.csv"
resultado_desvio.to_csv(output_path_desvio, index=False)
print(f"Arquivo de desvio salvo em: {output_path_desvio}")

# --- CSV 2: _palavras.csv ---
# Uma linha por handle, com todas as palavras utilizadas por ele (separadas por "_")
palavras_por_handle = (
    df_palavras.groupby("user")["palavra"]
    .apply(lambda palavras: "_".join(map(str, palavras.unique().tolist())))
    .reset_index()
    .rename(columns={"palavra": "palavras"})
)

output_path_palavras = output_dir / f"{core_user}_palavras.csv"
palavras_por_handle.to_csv(output_path_palavras, index=False)
print(f"Arquivo de palavras por handle salvo em: {output_path_palavras}")

# Remove raw_data
print(f"\nRemovendo pasta raw_data de {core_user}...")
shutil.rmtree(raw_data_dir)
print("Pasta raw_data removida com sucesso.")