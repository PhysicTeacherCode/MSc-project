import aiohttp
import asyncio

BSKY_SERVICE = "public.api.bsky.app"

# ─────────────────────────────────────────────────────────────────────────────
# FILTRO DE CELEBRIDADES
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_follower_counts(session, handles: list, semaphore) -> dict:
    """
    Consulta perfis em batch de 25 handles via app.bsky.actor.getProfiles.
    Todos os lotes são disparados CONCORRENTEMENTE para máxima velocidade.
    """
    url = f"https://{BSKY_SERVICE}/xrpc/app.bsky.actor.getProfiles"

    async def fetch_batch(batch: list) -> dict:
        params = [("actors[]", h) for h in batch]
        try:
            async with semaphore:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(30)
                        return {}
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
                    return {
                        p.get("handle", ""): p.get("followersCount", 0)
                        for p in data.get("profiles", [])
                    }
        except Exception:
            return {}

    # Divide em lotes de 25 e dispara todos ao mesmo tempo
    batches = [handles[i:i + 25] for i in range(0, len(handles), 25)]
    results = await asyncio.gather(*[fetch_batch(b) for b in batches])

    # Merge de todos os dicionários retornados
    counts = {}
    for r in results:
        counts.update(r)
    return counts


async def filter_celebrities(session, handles: list, semaphore, max_followers: int) -> tuple[list, int]:
    """
    Filtra os handles que ultrapassam o limite de seguidores.
    Retorna (lista_filtrada, quantidade_removida).
    """
    if max_followers <= 0:
        return handles, 0  # Sem filtro

    counts = await fetch_follower_counts(session, handles, semaphore)

    filtered = []
    removed = 0
    for h in handles:
        fc = counts.get(h, 0)
        if fc <= max_followers:
            filtered.append(h)
        else:
            removed += 1
            print(f"  [Filtro] '{h}' removido ({fc:,} seguidores > limite {max_followers:,})")

    return filtered, removed


# ─────────────────────────────────────────────────────────────────────────────
# COLETA DE FOLLOWERS
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_followers(session, actor, semaphore, max_retries=5, limit_total=3000):
    """
    Busca de forma paginada todos os seguidores de um dado 'actor'.
    Implementa um backoff exponencial simples caso receba HTTP 429.
    Possui um limite global máximo de seguidores a serem extraídos (limit_total).
    """
    url = f"https://{BSKY_SERVICE}/xrpc/app.bsky.graph.getFollowers"
    followers = []
    cursor = None
    
    while len(followers) < limit_total:
        params = {"actor": actor, "limit": 100}
        if cursor:
            params["cursor"] = cursor
            
        retries = 0
        backoff = 1.0
        
        while retries < max_retries:
            async with semaphore:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        print(f"[Rate Limit] HTTP 429 para {actor}, aguardando {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff *= 2
                        retries += 1
                        continue
                    
                    if resp.status == 400:
                        try:
                            error_data = await resp.json()
                            err_msg = error_data.get("message", "Handle Inválido")
                        except:
                            err_msg = "Handle Inválido"
                        print(f"\n[Aviso] Falha ao coletar '{actor}': {err_msg}")
                        return followers
                        
                    if resp.status != 200:
                        return followers
                    
                    data = await resp.json()
                    for f in data.get("followers", []):
                        followers.append(f["handle"])
                        if len(followers) >= limit_total:
                            break
                        
                    cursor = data.get("cursor")
                    break
        
        if retries == max_retries or not cursor:
            break
            
    return followers


# ─────────────────────────────────────────────────────────────────────────────
# COLETA DE REDE (BFS 1ª e 2ª Ordem)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 3ª PASSAGEM: ENRIQUECIMENTO DE ARESTAS (Cross-connections)
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_following(session, actor: str, semaphore, max_pages: int = 5) -> list:
    """
    Busca a lista de usuários que 'actor' segue (follows), paginada.
    Limita a `max_pages` páginas (até 500 handles) para eficiência.
    """
    url = f"https://{BSKY_SERVICE}/xrpc/app.bsky.graph.getFollows"
    following = []
    cursor = None

    for _ in range(max_pages):
        params = {"actor": actor, "limit": 100}
        if cursor:
            params["cursor"] = cursor

        try:
            async with semaphore:
                async with session.get(url, params=params) as resp:
                    if resp.status == 429:
                        await asyncio.sleep(30)
                        continue
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    for f in data.get("follows", []):
                        following.append(f.get("handle", ""))
                    cursor = data.get("cursor")
                    if not cursor:
                        break
        except Exception:
            break

    return following


async def enrich_edges_with_cross_connections(
    session,
    second_order_nodes: list,
    known_nodes: set,
    semaphore,
    chunk_size: int = 500
) -> list:
    """
    Para cada nó de 2ª ordem, verifica quem ele segue que já está no grafo.
    Processa em lotes (chunks) para mostrar progresso e evitar sobrecarga de memória.
    """
    total = len(second_order_nodes)
    new_edges = []
    processed = 0

    print(f"\n[3ª Passagem] Cross-connections: {total} usuários em lotes de {chunk_size}...")

    for start in range(0, total, chunk_size):
        chunk = second_order_nodes[start:start + chunk_size]

        # Dispara o lote em paralelo
        results = await asyncio.gather(
            *[fetch_following(session, user, semaphore) for user in chunk],
            return_exceptions=True
        )

        for user, following_list in zip(chunk, results):
            if isinstance(following_list, Exception):
                continue
            for followed in following_list:
                if followed in known_nodes and followed != user:
                    new_edges.append((user, followed))

        processed += len(chunk)
        print(f"  [Cross] {processed}/{total} verificados | {len(new_edges)} novas arestas até agora...", end="\r")

    print(f"\n[3ª Passagem] Concluída. {len(new_edges)} novas arestas encontradas.")
    return new_edges


async def collect_network(core_user, safe_limit, max_followers: int = 5000):
    """
    Executa a coleta em largura (BFS) com 3 passagens:

    1ª: Seguidores do core_user (1ª ordem) + filtro de celebridades.
    2ª: Seguidores de cada usuário 1ª ordem (2ª ordem) + filtro de celebridades.
    3ª: Para cada usuário 2ª ordem, verifica quem ele segue que já está no grafo
        → gera arestas cruzadas, densificando o grafo.
    """
    semaphore = asyncio.Semaphore(safe_limit)
    edges = []

    async with aiohttp.ClientSession() as session:

        # ── 1ª Passagem: Followers de 1ª ordem ─────────────────────────────
        print(f"\n[Coleta] Buscando seguidores de 1ª ordem para '{core_user}'...")
        first_order_raw = await fetch_followers(session, core_user, semaphore)

        print(f"[Filtro] Verificando {len(first_order_raw)} de 1ª ordem (limite: {max_followers:,})...")
        first_order, removed_1 = await filter_celebrities(session, first_order_raw, semaphore, max_followers)
        print(f"[Filtro] 1ª ordem: {len(first_order)} mantidos, {removed_1} removidos.")

        for follower in first_order:
            edges.append((follower, core_user))

        # ── 2ª Passagem: Followers de 2ª ordem ─────────────────────────────
        print(f"\n[Coleta] Iniciando busca de 2ª ordem para {len(first_order)} usuários...")
        tasks = [asyncio.create_task(fetch_followers(session, f, semaphore)) for f in first_order]

        total_tasks = len(tasks)
        second_order_results = []
        completed = 0
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            second_order_results.append((first_order[i], result))
            completed += 1
            print(f"  [2ª] {completed}/{total_tasks} processados...{' '*20}", end="\r")

        # Filtra celebridades da 2ª ordem em batch
        all_second_handles = list({h for _, followers in second_order_results for h in followers})
        print(f"\n[Filtro] Verificando {len(all_second_handles)} handles únicos de 2ª ordem...")
        counts_map = await fetch_follower_counts(session, all_second_handles, semaphore)

        second_order_kept = set()
        removed_2 = 0
        for (follower_node, followers_list) in second_order_results:
            for second_follower in followers_list:
                fc = counts_map.get(second_follower, 0)
                if fc <= max_followers:
                    edges.append((second_follower, follower_node))
                    second_order_kept.add(second_follower)
                else:
                    removed_2 += 1

        print(f"[Filtro] 2ª ordem: {len(second_order_kept)} mantidos, {removed_2} removidos.")

        # ── 3ª Passagem: Cross-connections para densificar o grafo ──────────
        known_nodes = {core_user} | set(first_order) | second_order_kept
        cross_edges = await enrich_edges_with_cross_connections(
            session,
            list(second_order_kept),
            known_nodes,
            semaphore
        )
        edges.extend(cross_edges)

    print(f"\n[Resumo] Total de arestas: {len(edges)} "
          f"(1ª+2ª: {len(edges) - len(cross_edges)} | cross: {len(cross_edges)})")

    return edges
