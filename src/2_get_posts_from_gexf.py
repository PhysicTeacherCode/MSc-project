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

async def get_posts(session, actor, semaphore):
    url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
    cursor = None
    texts = []
    dates = []
    tries = 0
    max_tries = 10  # Aumentado de 5 para 10
    error_msg = None

    async with semaphore:  # Limita requisições simultâneas
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
                    if resp.status == 429:  # Rate limit
                        tries += 1
                        if tries > max_tries:
                            error_msg = "Rate limit excedido"
                            break
                        wait_time = min(2 ** tries, 60)  # Aumentado para 60s
                        await asyncio.sleep(wait_time)
                        continue

                    # Usuário não encontrado ou privado - não é erro de rede
                    if resp.status in [400, 404]:
                        error_msg = f"Usuário não encontrado (HTTP {resp.status})"
                        break

                    # Outros erros HTTP
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

            # Pequeno delay entre páginas
            await asyncio.sleep(0.1)

    return actor, (texts, dates, tries, error_msg)

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
    failed_users = []
    error_stats = {}

    # Extrai o nome do diretório entre 'graph/' e '/'
    gexf_path_obj = Path(gexf_path)
    community_name = gexf_path_obj.parent.parent.name
    output_dir = f"../data/posts/{community_name}_({total_users})/"

    # Configuração otimizada
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=20, ssl=False, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    semaphore = asyncio.Semaphore(30)  # Limita a 30 requisições simultâneas

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [get_posts(session, actor, semaphore) for actor in followers_list]

        for coro in asyncio.as_completed(tasks):
            try:
                actor, (texts, dates, tries, error_msg) = await coro

                if not texts:
                    failed_users.append(actor)

                    # Estatísticas de erros
                    if error_msg:
                        error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                        error_stats[error_type] = error_stats.get(error_type, 0) + 1

                        if tries > 5:  # Apenas mostra se tentou muito
                            print(f"⚠️  {actor}: {error_msg} (tentativas: {tries})")
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

                if saved_count % 10 == 0:
                    progress = (saved_count + len(failed_users)) / total_users * 100
                    print(f"✓ Progresso: {progress:.1f}% - Salvos: {saved_count} | Falhas: {len(failed_users)}")

            except Exception as e:
                failed_users.append("unknown")
                print(f"❌ Erro ao processar: {e}")

    # Salva lista de usuários que falharam
    if failed_users:
        failed_file = f"{output_dir}_failed_users.txt"
        with open(failed_file, 'w', encoding='utf-8') as f:
            for user in failed_users:
                f.write(f"{user}\n")
        print(f"\n📝 Lista de falhas salva em: {failed_file}")

    print(f"\n{'='*60}")
    print(f"RESUMO FINAL:")
    print(f"Total carregado: {total_users} usuários")
    print(f"Salvos com sucesso: {saved_count} usuários")
    print(f"Falharam/sem posts: {len(failed_users)} usuários")
    print(f"Taxa de sucesso: {(saved_count/total_users)*100:.1f}%")

    if error_stats:
        print(f"\nERROS POR TIPO:")
        for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {error_type}: {count} usuários")

    print(f"{'='*60}")

    # Perguntar se quer tentar novamente os que falharam
    if failed_users and len(failed_users) > 0:
        print(f"\n🔄 Deseja tentar novamente os {len(failed_users)} usuários que falharam? (s/n)")
        retry = input().strip().lower()
        if retry == 's':
            print(f"\n{'='*60}")
            print(f"SEGUNDA TENTATIVA - {len(failed_users)} usuários")
            print(f"{'='*60}")

            retry_saved = 0
            retry_failed = []

            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                retry_tasks = [get_posts(session, actor, semaphore) for actor in failed_users if actor != "unknown"]

                for coro in asyncio.as_completed(retry_tasks):
                    try:
                        actor, (texts, dates, tries, error_msg) = await coro

                        if not texts:
                            retry_failed.append(actor)
                            continue

                        texts = [clean_text(text) for text in texts]
                        df = pd.DataFrame({
                            "user": [actor] * len(texts),
                            "post": texts,
                            "date": dates
                        })
                        df.to_csv(f"{output_dir}{actor}({len(texts)}).csv", index=False, encoding='utf-8', quoting=1, quotechar='"')
                        retry_saved += 1

                        if retry_saved % 5 == 0:
                            print(f"✓ Retry - Salvos: {retry_saved}")

                    except Exception as e:
                        print(f"❌ Erro no retry: {e}")

            print(f"\n{'='*60}")
            print(f"RESULTADO DO RETRY:")
            print(f"Salvos no retry: {retry_saved} usuários")
            print(f"Total salvos: {saved_count + retry_saved} usuários")
            print(f"Taxa final: {((saved_count + retry_saved)/total_users)*100:.1f}%")
            print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
