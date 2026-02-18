import pandas as pd
import asyncio
import aiohttp
import networkx as nx
import os
import re
from pathlib import Path

def clean_text(text):
    """Limpa e normaliza o texto do post"""
    # Remove quebras de linha, tabulações e carriage returns
    text = re.sub(r'[\n\r\t\v\f]+', ' ', text)

    # Remove espaços múltiplos
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[\n\r\t\v\f]+', ' ', text)

    # Remove espaços no início e fim
    text = text.strip()

    # Escapa aspas para CSV
    text = text.replace('"', '""')

    return text


def load_followers_list(gexf_path):
    """Carrega todos os usuários de um arquivo GEXF usando networkx"""
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

async def get_posts(session, actor):
    url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
    cursor = None
    texts = []
    dates = []
    tries = 0
    max_tries = 5

    while True:
        params = {
            "actor": actor,
            "limit": 100,
            "filter": "posts_no_replies",
        }
        if cursor:
            params["cursor"] = cursor

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 429:  # Rate limit
                    tries += 1
                    if tries > max_tries:
                        break
                    wait_time = min(2 ** tries, 30)  # Backoff exponencial: 2, 4, 8, 16, 30s
                    await asyncio.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = await resp.json()

        except asyncio.TimeoutError:
            tries += 1
            if tries > max_tries:
                break
            await asyncio.sleep(1)
            continue
        except Exception as e:
            tries += 1
            if tries > max_tries:
                break
            wait_time = min(2 ** tries, 30)
            await asyncio.sleep(wait_time)
            continue

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

    return actor, (texts, dates, tries)

async def main():
    print("digite o caminho do arquivo GEXF e pressione Enter: ")
    gexf_path = input("exemplo -> ../data/graph/*.bsky.social/GEXF/*.gexf ")

    followers_list = load_followers_list(gexf_path)

    if not followers_list:
        print("Nenhum usuário encontrado no arquivo GEXF")
        return

    total_users = len(followers_list)
    print(f"{'='*60}")
    print(f"Carregados {total_users} usuários do arquivo GEXF")
    print(f"Arquivo: {Path(gexf_path).name}")
    print(f"{'='*60}")

    saved_count = 0
    failed_count = 0

    # Extrai o nome do diretório entre 'graph/' e '/'
    gexf_path_obj = Path(gexf_path)
    community_name = gexf_path_obj.parent.parent.name
    output_dir = f"../data/posts/{community_name}_({total_users})/"

    connector = aiohttp.TCPConnector(limit=100, limit_per_host=10, ssl=False)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [get_posts(session, actor) for actor in followers_list]

        for coro in asyncio.as_completed(tasks):
            try:
                actor, (texts, dates, tries) = await coro

                if not texts:
                    failed_count += 1
                    if tries > 0:
                        print(f"⚠️  {actor}: Sem posts (tentativas: {tries})")
                    continue

                # Limpa e normaliza todos os textos
                texts = [clean_text(text) for text in texts]

                df = pd.DataFrame({
                    "user": [actor] * len(texts),
                    "post": texts,
                    "date": dates
                })
                os.makedirs(output_dir, exist_ok=True)
                df.to_csv(f"{output_dir}{actor}({len(texts)}).csv", index=False, encoding='utf-8', quoting=1, quotechar='"')
                saved_count += 1

                if saved_count % 5 == 0:
                    print(f"✓ Salvos: {saved_count}/{total_users} usuários")

            except Exception as e:
                failed_count += 1
                print(f"❌ Erro ao processar: {e}")

    print(f"{'='*60}")
    print(f"RESUMO FINAL:")
    print(f"Total carregado: {total_users} usuários")
    print(f"Salvos com sucesso: {saved_count} usuários")
    print(f"Falharam/sem posts: {failed_count} usuários")
    print(f"Taxa de sucesso: {(saved_count/total_users)*100:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
