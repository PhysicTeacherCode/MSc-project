import pandas as pd
import asyncio
import aiohttp
import networkx as nx
import os
import re
from pathlib import Path

def clean_text(text):
    """
    Limpa e normaliza o texto do post.

    [Ordem de acoes]:
    1. Remove quebras de linha e espacos extras.
    2. Faz trim e escapa aspas.
    3. Retorna o texto limpo.
    """
    text = re.sub(r'[\n\r\t\v\f]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\n\r\t\v\f]+', ' ', text)
    text = text.strip()
    text = text.replace('"', '""')
    return text


def load_followers_list(gexf_path):
    """
    Carrega usuarios de um arquivo GEXF.

    [Ordem de acoes]:
    1. Verifica se o arquivo existe.
    2. Le o GEXF e extrai os nos.
    3. Retorna a lista de usuarios.
    """
    gexf_file = Path(gexf_path)

    if not gexf_file.exists():
        print(f"Arquivo não encontrado: {gexf_path}")
        return []

    try:
        graph = nx.read_gexf(gexf_file)
        users = list(graph.nodes())
        return users
    except Exception as e:
        print(f"Erro ao ler {gexf_file}: {e}")
        return []

async def get_posts(session, actor, semaphore):
    """
    Coleta posts de um usuario com controle de concorrencia e retry.

    [Ordem de acoes]:
    1. Pagina o feed e trata erros de rede e rate limit.
    2. Extrai textos e datas dos posts.
    3. Retorna textos, datas e informacoes de erro.
    """
    url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
    cursor = None
    texts = []
    dates = []
    tries = 0
    max_tries = 10
    error_msg = None

    async with semaphore:
        while True:
            params = {
                "actor": actor,
                "limit": 100,
                "filter": "posts_no_replies",
            }
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 429:
                        tries += 1
                        if tries > max_tries:
                            error_msg = "Rate limit excedido"
                            break
                        wait_time = min(2 ** tries, 60)
                        await asyncio.sleep(wait_time)
                        continue

                    if resp.status in [400, 404]:
                        error_msg = f"Usuário não encontrado (HTTP {resp.status})"
                        break

                    if resp.status >= 500:
                        tries += 1
                        if tries > max_tries:
                            error_msg = f"Erro do servidor (HTTP {resp.status})"
                            break
                        await asyncio.sleep(min(2 ** tries, 60))
                        continue

                    resp.raise_for_status()
                    data = await resp.json()

            except asyncio.TimeoutError:
                tries += 1
                if tries > max_tries:
                    error_msg = "Timeout excedido"
                    break
                await asyncio.sleep(min(2 ** tries, 60))
                continue
            except aiohttp.ClientError as e:
                tries += 1
                if tries > max_tries:
                    error_msg = f"Erro de conexão: {str(e)[:50]}"
                    break
                await asyncio.sleep(min(2 ** tries, 60))
                continue
            except Exception as e:
                error_msg = f"Erro inesperado: {str(e)[:50]}"
                break

            feed = data.get("feed", [])
            if not feed:
                break

            for item in feed:
                record = item.get("post", {}).get("record", {})
                text = record.get("text", "")
                if text:
                    texts.append(text)
                    dates.append(record.get("createdAt", ""))

            cursor = data.get("cursor")
            if not cursor or len(texts) >= 5000:
                break

            await asyncio.sleep(0.1)

    return actor, (texts, dates, tries, error_msg)

def list_core_user_folders(base_dir):
    """
    Lista pastas core_user_N existentes dentro do diretorio base.

    [Ordem de acoes]:
    1. Verifica se o diretorio base existe.
    2. Filtra subpastas iniciadas por core_user_.
    3. Retorna a lista ordenada.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("core_user_")]
    return sorted(folders, key=lambda p: p.name)


def choose_from_list(items, prompt):
    """
    Solicita ao usuario escolher um item da lista pelo indice.

    [Ordem de acoes]:
    1. Exibe os itens numerados.
    2. Le e valida a escolha do usuario.
    3. Retorna o item selecionado.
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


def list_gexf_files(gexf_dir):
    """
    Lista arquivos GEXF dentro do diretorio informado.

    [Ordem de acoes]:
    1. Verifica se o diretorio existe.
    2. Coleta arquivos .gexf.
    3. Retorna a lista ordenada.
    """
    gexf_path = Path(gexf_dir)
    if not gexf_path.exists():
        return []
    return sorted(gexf_path.glob("*.gexf"), key=lambda p: p.name)

async def main():
    """
    Executa o fluxo principal para escolher GEXF e baixar posts.

    [Ordem de acoes]:
    1. Seleciona pasta core_user e arquivo GEXF.
    2. Carrega usuarios e coleta posts assincronamente.
    3. Salva CSVs e gera resumo com retry opcional.
    """
    print("Escolha a pasta do core user:")

    core_user_folders = list_core_user_folders(Path("..") / "data" / "graph")
    if not core_user_folders:
        print("Nenhuma pasta core_user encontrada em ../data/graph")
        return

    selected_core_folder = choose_from_list(core_user_folders, "Digite o numero da pasta desejada: ")
    if not selected_core_folder:
        print("Nenhuma pasta selecionada.")
        return

    gexf_dir = selected_core_folder / "GEXF"
    gexf_files = list_gexf_files(gexf_dir)
    if not gexf_files:
        print(f"Nenhum arquivo GEXF encontrado em: {gexf_dir}")
        return

    print("Escolha o arquivo GEXF:")
    selected_gexf = choose_from_list(gexf_files, "Digite o numero do arquivo desejado: ")
    if not selected_gexf:
        print("Nenhum arquivo selecionado.")
        return

    gexf_path = str(selected_gexf)

    followers_list = load_followers_list(gexf_path)

    if not followers_list:
        print("Nenhum usuario encontrado no arquivo GEXF")
        return

    total_users = len(followers_list)
    print(f"{'='*60}")
    print(f"Carregados {total_users} usuários do arquivo GEXF")
    print(f"Arquivo: {Path(gexf_path).name}")
    print(f"{'='*60}")

    saved_count = 0
    error_stats = {}

    gexf_path_obj = Path(gexf_path)
    community_name = gexf_path_obj.parent.parent.name
    output_dir = f"../data/posts/{community_name}_({total_users})/"

    connector = aiohttp.TCPConnector(limit=50, limit_per_host=20, ssl=False, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    semaphore = asyncio.Semaphore(30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [get_posts(session, actor, semaphore) for actor in followers_list]

        for coro in asyncio.as_completed(tasks):
            try:
                actor, (texts, dates, tries, error_msg) = await coro

                if not texts:
                    if error_msg:
                        error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                        error_stats[error_type] = error_stats.get(error_type, 0) + 1

                        if tries > 5:
                            print(f"⚠️  {actor}: {error_msg} (tentativas: {tries})")
                    continue

                texts = [clean_text(text) for text in texts]

                df = pd.DataFrame({
                    "user": [actor] * len(texts),
                    "post": texts,
                    "date": dates
                })
                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(f"{output_dir}{actor}({len(texts)}).csv", index=False, encoding='utf-8', quoting=1, quotechar='"')
                saved_count += 1

                if saved_count % 10 == 0:
                    progress = (saved_count / total_users) * 100
                    print(f"✓ Progresso: {progress:.1f}% - Salvos: {saved_count}")

            except Exception as e:
                 print(f"❌ Erro ao processar: {e}")

    print(f"\n{'='*60}")
    print(f"RESUMO FINAL:")
    print(f"Total carregado: {total_users} usuários")
    print(f"Salvos com sucesso: {saved_count} usuários")
    print(f"Falharam/sem posts: {total_users - saved_count} usuários")
    print(f"Taxa de sucesso: {(saved_count/total_users)*100:.1f}%")

    if error_stats:
        print(f"\nERROS POR TIPO:")
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count} usuários")

    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
