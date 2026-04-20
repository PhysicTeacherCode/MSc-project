import asyncio
import pandas as pd
import networkx as nx
import os
import aiohttp
import sys
import re
from datetime import datetime, timezone
from array import array

BSKY_SERVICE = "public.api.bsky.app"

def parse_datetime(dt_str):
    # Trata carimbos de data/hora com precisão de sub-microssegundos (mais de 6 dígitos) 
    # que o fromisoformat padrão do Python não aceita.
    if "." in dt_str:
        base, rest = dt_str.split(".")
        # Procura o separador de fuso horário (+ ou - ou Z)
        tz_match = re.search(r'[Z+-]', rest)
        if tz_match:
            tz_idx = tz_match.start()
            micros = rest[:tz_idx]
            tz = rest[tz_idx:]
            # Trunca para no máximo 6 dígitos de microssegundos
            dt_str = f"{base}.{micros[:6]}{tz}"
            
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

# Data de lançamento do Bluesky (Referência fixa para reprodutibilidade)
# Lançamento do beta iOS: 17 de fevereiro de 2023
BLUESKY_START_DATE = datetime(2023, 2, 17, tzinfo=timezone.utc)

async def fetch_user_posts(session, did, semaphore, max_posts_per_user=500):
    """
    Coleta o feed de um usuário e extrai palavras e timestamps.
    Utiliza BLUESKY_START_DATE como referência fixa (T=0).
    """
    cursor = None
    posts_processed = 0
    word_map = {} # {palavra: [lista_de_idades]}
    pattern = re.compile(r'\b\w{2,}\b', re.UNICODE)

    while posts_processed < max_posts_per_user:
        try:
            params = {
                "actor": did,
                "limit": 100,
                "filter": "posts_no_replies"
            }
            if cursor:
                params["cursor"] = cursor
            
            async with semaphore:
                async with session.get(f"https://{BSKY_SERVICE}/xrpc/app.bsky.feed.getAuthorFeed", params=params) as resp:
                    if resp.status == 429:
                        print(f"  > [Rate Limit] {did} ATINGIDO!{' ' * 30}", end="\r")
                        await asyncio.sleep(60)
                        continue
                    if resp.status != 200:
                        break
                    
                    data = await resp.json()
            
            feed = data.get("feed", [])
            if not feed:
                break
            
            for item in feed:
                post = item.get("post", {})
                record = post.get("record", {})
                text = record.get("text", "")
                created_at = record.get("createdAt")
                
                if text and created_at:
                    timestamp = parse_datetime(created_at)
                    # age_days agora é "dias desde o lançamento do Bluesky" (T=0 em 17/02/2023)
                    age_days = (timestamp - BLUESKY_START_DATE).total_seconds() / 86400
                    
                    # Extração de palavras e armazenamento dos tempos de ocorrência
                    words = set(pattern.findall(text.lower()))
                    for word in words:
                        if word not in word_map:
                            word_map[word] = []
                        word_map[word].append(age_days)
                    
                    posts_processed += 1
                
                if posts_processed >= max_posts_per_user:
                    break
            
            cursor = data.get("cursor")
            if not cursor or posts_processed >= max_posts_per_user:
                break
            
            # Feedback progressivo a cada 500 posts
            if posts_processed > 0 and posts_processed % 500 == 0:
                print(f"  > [Contexto] {did}: {posts_processed} posts processados...{' ' * 30}")

        except Exception as e:
            print(f"  > [Erro] {did}: {e}")
            break

    return did, word_map


async def collect_community_posts_df(gexf_path, semaphore_limit, max_posts_per_user=3000):
    """
    Lê o GEXF, dispara a coleta concorrente e agrega as palavras por usuário e globalmente.
    Otimizado para memória usando sys.intern e array.array.
    """
    if not os.path.exists(gexf_path):
        print(f"[Erro] Arquivo não encontrado: {gexf_path}")
        return {}, {}, []

    G = nx.read_gexf(gexf_path)
    all_users = [sys.intern(u) for u in G.nodes()]
    
    print(f"\n[Coleta] Iniciando coleta de {len(all_users)} usuários (Máx {max_posts_per_user} posts/user)...")
    
    global_word_times = {} # {word_str: array.array('d')}
    user_word_sets = {}    # {did_interned: {word_str_interned}}
    
    connector = aiohttp.TCPConnector(ssl=False, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(semaphore_limit)
        
        async def fetch_and_process(u):
            return await fetch_user_posts(session, u, semaphore, max_posts_per_user)

        tasks = set()
        user_iter = iter(all_users)
        
        # Enche o pipeline inicial (aguarda no máximo o limite de concorrência real + buffer)
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
                did, word_map = await t
                did = sys.intern(did)
                
                # 1. Registrar atividade de palavras do usuário (para Ising)
                # Já internamos as palavras aqui para economizar memória global
                interned_user_words = set()
                for w, ages in word_map.items():
                    w_interned = sys.intern(w)
                    interned_user_words.add(w_interned)
                    
                    # 2. Agregar no mapa global (usando array.array que é mais pacto que list)
                    if w_interned not in global_word_times:
                        global_word_times[w_interned] = array('d')
                    global_word_times[w_interned].fromlist(ages)
                
                user_word_sets[did] = interned_user_words
                count += 1
                if count % 10 == 0 or count == len(all_users):
                    print(f"  > Progresso Total: {count}/{len(all_users)} processados...{' ' * 20}", end="\r")
                
                # Adiciona nova task se houver usuários restantes
                try:
                    u = next(user_iter)
                    tasks.add(asyncio.create_task(fetch_and_process(u)))
                except StopIteration:
                    continue

    # Garante que usuários que não postaram nada estejam no mapa
    for u in all_users:
        if u not in user_word_sets:
            user_word_sets[u] = set()

    print(f"\n[Resumo] Usuários: {len(all_users)} | Palavras Únicas: {len(global_word_times)}")
    return global_word_times, user_word_sets, all_users

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
