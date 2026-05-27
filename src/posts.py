import asyncio
import networkx as nx
import os
import aiohttp
import sys
import re


BSKY_SERVICE = "public.api.bsky.app"
REQUEST_TIMEOUT_S = 45
MAX_PAGE_RETRIES = 5


async def fetch_user_posts(session, did, semaphore, max_posts_per_user=500):
    """
    Coleta o feed de um usuário e extrai palavras únicas por post.
    Retorna (did, word_counts, word_timestamps, posts_processed, status).
    word_counts: {palavra: int} — quantas vezes a palavra apareceu nos posts.
    word_timestamps: {palavra: [createdAt]} — quando a palavra apareceu.
    """
    cursor = None
    posts_processed = 0
    word_counts = {}  # {palavra: contagem}
    word_timestamps = {}  # {palavra: [createdAt]}
    pattern = re.compile(r'\b\w{2,}\b', re.UNICODE)
    status = {"ok": True, "error": None}

    while posts_processed < max_posts_per_user:
        params = {
            "actor": did,
            "limit": 100,
            "filter": "posts_no_replies"
        }
        if cursor:
            params["cursor"] = cursor

        data = None
        for attempt in range(1, MAX_PAGE_RETRIES + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
                async with semaphore:
                    async with session.get(
                        f"https://{BSKY_SERVICE}/xrpc/app.bsky.feed.getAuthorFeed",
                        params=params,
                        timeout=timeout
                    ) as resp:
                        if resp.status == 429:
                            wait_s = min(60 * attempt, 180)
                            print(f"  > [Rate Limit] {did}: aguardando {wait_s}s...{' ' * 30}", end="\r")
                            await asyncio.sleep(wait_s)
                            continue
                        if 500 <= resp.status < 600:
                            wait_s = min(2 ** attempt, 60)
                            print(f"  > [Retry] {did}: HTTP {resp.status}, tentativa {attempt}/{MAX_PAGE_RETRIES} em {wait_s}s...{' ' * 30}", end="\r")
                            await asyncio.sleep(wait_s)
                            continue
                        if resp.status != 200:
                            status = {"ok": False, "error": f"HTTP {resp.status}"}
                            print(f"  > [Erro] {did}: HTTP {resp.status}")
                            return did, word_counts, word_timestamps, posts_processed, status

                        data = await resp.json()
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait_s = min(2 ** attempt, 60)
                if attempt < MAX_PAGE_RETRIES:
                    print(f"  > [Retry] {did}: {type(e).__name__} tentativa {attempt}/{MAX_PAGE_RETRIES}; aguardando {wait_s}s...{' ' * 30}", end="\r")
                    await asyncio.sleep(wait_s)
                else:
                    status = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    print(f"  > [Erro] {did}: coleta incompleta após {MAX_PAGE_RETRIES} tentativas ({status['error']})")
                    return did, word_counts, word_timestamps, posts_processed, status
            except Exception as e:
                status = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                print(f"  > [Erro] {did}: {status['error']}")
                return did, word_counts, word_timestamps, posts_processed, status

        if data is None:
            status = {"ok": False, "error": "sem resposta válida após retries"}
            print(f"  > [Erro] {did}: {status['error']}")
            return did, word_counts, word_timestamps, posts_processed, status

        try:
            feed = data.get("feed", [])
            if not feed:
                break
            
            for item in feed:
                post = item.get("post", {})
                record = post.get("record", {})
                text = record.get("text", "")
                created_at = record.get("createdAt") or post.get("indexedAt")
                
                if text:
                    words = set(pattern.findall(text.lower()))
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
                        if created_at:
                            word_timestamps.setdefault(word, []).append(created_at)
                    
                    posts_processed += 1
                
                if posts_processed >= max_posts_per_user:
                    break
            
            cursor = data.get("cursor")
            if not cursor or posts_processed >= max_posts_per_user:
                break
            
            if posts_processed > 0 and posts_processed % 500 == 0:
                print(f"  > [Contexto] {did}: {posts_processed} posts processados...{' ' * 30}")

        except Exception as e:
            status = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            print(f"  > [Erro] {did}: {status['error']}")
            return did, word_counts, word_timestamps, posts_processed, status

    return did, word_counts, word_timestamps, posts_processed, status


async def collect_community_posts_df(gexf_path, semaphore_limit, max_posts_per_user=5000, min_posts=0):
    """
    Lê o GEXF, dispara a coleta concorrente e agrega as palavras por usuário e globalmente.
    
    Retorna:
        global_word_counts: {word: int} — total de ocorrências em todos os posts.
        global_word_timestamps: {word: [createdAt]} — timestamps de uso da palavra.
        user_word_sets:     {did: set(words)} — palavras usadas por cada usuário.
        all_users:          [did] — lista de usuários (pós-filtro se min_posts > 0).
    """
    if not os.path.exists(gexf_path):
        print(f"[Erro] Arquivo não encontrado: {gexf_path}")
        return {}, {}, {}, []

    G = nx.read_gexf(gexf_path)
    all_users = [sys.intern(u) for u in G.nodes()]
    
    print(f"\n[Coleta] Iniciando coleta de {len(all_users)} usuários (Máx {max_posts_per_user} posts/user)...")
    if min_posts > 0:
        print(f"  [Filtro] Usuários com < {min_posts} posts serão removidos após coleta.")
    
    global_word_counts = {}    # {word_str: int}
    global_word_timestamps = {}  # {word_str: [createdAt]}
    user_word_sets = {}        # {did_interned: {word_str_interned}}
    user_post_counts = {}      # {did_interned: int}
    user_fetch_status = {}     # {did_interned: {ok: bool, error: str|None}}
    user_word_count_maps = {}  # {did_interned: {word_str_interned: int}}
    user_word_timestamp_maps = {}  # {did_interned: {word_str_interned: [createdAt]}}
    
    connector = aiohttp.TCPConnector(ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(semaphore_limit)
        
        async def fetch_and_process(u):
            return await fetch_user_posts(session, u, semaphore, max_posts_per_user)

        tasks = set()
        user_iter = iter(all_users)
        
        for _ in range(semaphore_limit + 20):
            try:
                u = next(user_iter)
                tasks.add(asyncio.create_task(fetch_and_process(u)))
            except StopIteration:
                break
        
        count = 0
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                did, word_counts, word_timestamps, post_count, fetch_status = await t
                did = sys.intern(did)
                user_post_counts[did] = post_count
                user_fetch_status[did] = fetch_status
                
                interned_user_words = set()
                interned_word_counts = {}
                interned_word_timestamps = {}
                for w, cnt in word_counts.items():
                    w_interned = sys.intern(w)
                    interned_user_words.add(w_interned)
                    interned_word_counts[w_interned] = cnt
                    if w in word_timestamps:
                        interned_word_timestamps[w_interned] = word_timestamps[w]
                
                user_word_sets[did] = interned_user_words
                user_word_count_maps[did] = interned_word_counts
                user_word_timestamp_maps[did] = interned_word_timestamps
                
                count += 1
                if count % 10 == 0 or count == len(all_users):
                    print(f"  > Progresso Total: {count}/{len(all_users)} processados...{' ' * 20}", end="\r")
                
                try:
                    u = next(user_iter)
                    tasks.add(asyncio.create_task(fetch_and_process(u)))
                except StopIteration:
                    continue

    # Garante que usuários que não postaram nada estejam no mapa
    for u in all_users:
        if u not in user_word_sets:
            user_word_sets[u] = set()
            user_post_counts[u] = 0
            user_fetch_status[u] = {"ok": False, "error": "coleta não executada"}

    # ── Filtro de Atividade Mínima ──────────────────────────────────────────
    if min_posts > 0:
        inactive_users = [u for u in all_users if user_post_counts.get(u, 0) < min_posts]
        
        if inactive_users:
            print(f"\n[Filtro] Removendo {len(inactive_users)} usuários com < {min_posts} posts...")
            failed_inactive = [
                u for u in inactive_users
                if not user_fetch_status.get(u, {}).get("ok", True)
            ]
            if failed_inactive:
                print(f"[Aviso] {len(failed_inactive)} removido(s) tiveram coleta incompleta; podem não ser realmente inativos:")
                for u in failed_inactive[:10]:
                    err = user_fetch_status.get(u, {}).get("error")
                    print(f"  - {u}: {user_post_counts.get(u, 0)} posts coletados; erro={err}")
                if len(failed_inactive) > 10:
                    print(f"  ... e mais {len(failed_inactive) - 10}")
            
            for u in inactive_users:
                if u in user_word_sets:
                    del user_word_sets[u]
                user_word_count_maps.pop(u, None)
                user_word_timestamp_maps.pop(u, None)
            
            inactive_set = set(inactive_users)
            all_users = [u for u in all_users if u not in inactive_set]
            
            print(f"[Filtro] {len(all_users)} usuários ativos mantidos (>= {min_posts} posts).")
    
    # Agrega somente os usuarios mantidos depois do filtro de atividade.
    for did, wmap in user_word_count_maps.items():
        if did not in user_word_sets:
            continue
        for w, cnt in wmap.items():
            global_word_counts[w] = global_word_counts.get(w, 0) + cnt
        for w, timestamps in user_word_timestamp_maps.get(did, {}).items():
            if timestamps:
                global_word_timestamps.setdefault(w, []).extend(timestamps)

    user_word_count_maps.clear()
    user_word_timestamp_maps.clear()

    print(f"\n[Resumo] Usuários: {len(all_users)} | Palavras Únicas: {len(global_word_counts)}")
    return global_word_counts, global_word_timestamps, user_word_sets, all_users


def interactive_select_gexf(gexf_base_dir):
    if not os.path.exists(gexf_base_dir): return None
    files = [f for f in os.listdir(gexf_base_dir) if f.endswith('.gexf')]
    if not files: return None
    
    print("\nARQUIVOS GEXF DISPONÍVEIS:")
    for i, f in enumerate(files):
        print(f"[{i}] {f}")
    
    try:
        idx = int(input("\nEscolha o índice do arquivo: "))
        if 0 <= idx < len(files):
            return os.path.join(gexf_base_dir, files[idx])
    except:
        pass
    return None

def interactive_select_csv(base_dir, keyword_filter="matriz_estados"):
    """
    Lista arquivos CSV recursivamente a partir do diretório base que correspondam ao filtro.
    Permite ao usuário selecionar um via terminal.
    """
    if not os.path.exists(base_dir): return None
    
    csv_files = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.csv') and (keyword_filter in f):
                csv_files.append(os.path.join(root, f))
                
    if not csv_files:
        print(f"[Aviso] Nenhum arquivo CSV contendo '{keyword_filter}' encontrado em {base_dir}")
        return None
    
    print(f"\nARQUIVOS CSV ({keyword_filter}) DISPONÍVEIS:")
    for i, f in enumerate(csv_files):
        rel_path = os.path.relpath(f, base_dir)
        print(f"[{i}] {rel_path}")
    
    try:
        idx = int(input("\nEscolha o índice: "))
        if 0 <= idx < len(csv_files):
            return csv_files[idx]
    except:
        pass
    return None
