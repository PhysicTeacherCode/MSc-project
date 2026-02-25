import pandas as pd
from pathlib import Path

"""
Análise de palavras por desvio padrão de tempo entre ocorrências.

Ordem de ações:
1. Localiza todos os arquivos CSV mesclados de posts
2. Para cada arquivo, processa os posts extraindo palavras
3. Calcula o desvio padrão do tempo entre ocorrências de cada palavra
4. Salva resultado em CSV com palavras, quantidade de ocorrências e desvio padrão
"""

base_posts_dir = Path("..") / "data" / "posts"
merged_files = list(base_posts_dir.glob("core_user*/*_posts_merged.csv"))

if not merged_files:
    raise SystemExit("Nenhum arquivo *_posts_merged.csv encontrado em ../data/posts")

print("Arquivos encontrados:")
for idx, file in enumerate(merged_files, start=1):
    print(f"{idx}. {file}")

while True:
    choice = input(f"\nEscolha um arquivo (1-{len(merged_files)}): ").strip()
    if choice.isdigit():
        num = int(choice)
        if 1 <= num <= len(merged_files):
            merged_csv = merged_files[num - 1]
            break
    print("Escolha inválida. Tente novamente.")

for merged_csv in [merged_csv]:
    core_user = merged_csv.stem.replace("_posts_merged", "")
    print(f"Processando {core_user}...")

    source_df = pd.read_csv(merged_csv)

    if "post" not in source_df.columns or "date" not in source_df.columns:
        print(f"Colunas 'post' ou 'date' nao encontradas em {merged_csv}")
        continue

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

    df_palavras = source_df[["post_limpo", "idade"]].explode("post_limpo")
    df_palavras = df_palavras[df_palavras["post_limpo"].str.strip() != ""]
    df_palavras = df_palavras.rename(columns={"post_limpo": "palavras"})
    df_palavras["palavras"] = df_palavras["palavras"].str.strip()

    if df_palavras.empty:
        print(f"Nenhuma palavra encontrada em {merged_csv}")
        continue

    min_idade = df_palavras.groupby("palavras")["idade"].transform("min")
    df_palavras["dias_entre_ocorrencias"] = df_palavras["idade"] - min_idade

    resultado = df_palavras.groupby("palavras").agg(
        n_ocorrencias=("palavras", "count"),
        desvio_padrao=("dias_entre_ocorrencias", "std")
    ).reset_index()

    resultado = resultado[resultado["n_ocorrencias"] > 2]
    resultado = resultado.sort_values("desvio_padrao", ascending=False)

    output_dir = base_posts_dir / f"{core_user}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{core_user}_palavras_desvio.csv"
    resultado.to_csv(output_path, index=False)
    print(f"Arquivo salvo em: {output_path}")

    handle = core_user.split("_")[0] + "." + core_user.split("_")[1] + ".social"
    palavras_str = "_".join(resultado["palavras"].tolist())

    df_handle = pd.DataFrame({
        "handle": [handle],
        "palavras": [palavras_str]
    })

    output_path_handle = output_dir / f"{core_user}_palavras.csv"
    df_handle.to_csv(output_path_handle, index=False)
    print(f"Arquivo de palavras salvo em: {output_path_handle}")
