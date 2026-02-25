import pandas as pd
from pathlib import Path
import shutil


def list_core_user_folders(base_dir):
    """
    Lista todas as pastas core_user existentes no diretório base.

    Ordem de ações:
    1. Verifica se o diretório base existe
    2. Filtra apenas pastas que começam com "core_user_"
    3. Retorna lista ordenada das pastas encontradas
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("core_user_")]
    return sorted(folders, key=lambda p: p.name)


def choose_from_list(items, prompt):
    """
    Permite ao usuário escolher um item de uma lista numerada.

    Ordem de ações:
    1. Exibe lista numerada de itens
    2. Solicita escolha do usuário
    3. Valida a entrada e retorna o item escolhido
    """
    if not items:
        return None

    for idx, item in enumerate(items, start=1):
        print(f"{idx}. {item}")

    while True:
        choice = input(prompt).strip()
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(items):
                return items[num - 1]
        print("Escolha invalida. Tente novamente.")


base_posts_dir = Path("MSc-project/data/posts")
core_user_folders = list_core_user_folders(base_posts_dir)
if not core_user_folders:
    raise SystemExit("Nenhuma pasta core_user encontrada em MSc-project/data/posts")

print("Escolha a pasta do core user:")
selected_core_folder = choose_from_list(core_user_folders, "Digite o numero da pasta desejada: ")
if not selected_core_folder:
    raise SystemExit("Nenhuma pasta selecionada.")

raw_data_dir = Path(selected_core_folder) / "raw_data"
csv_files = list(raw_data_dir.glob("*.csv"))

core_user = Path(selected_core_folder).name

df_posts = pd.DataFrame()

print("Criando banco de dados (pandas) de posts...")
dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
df_posts = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

output_dir = Path(f"MSc-project/data/posts/{core_user}/")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{core_user}_posts_merged.csv"

print("Salvando posts em CSV (pandas)...")
df_posts.to_csv(output_path, index=False)
print(f"Arquivo salvo em: {output_path}")

print("Removendo pasta raw_data...")
shutil.rmtree(raw_data_dir)
print("Pasta raw_data removida com sucesso.")

