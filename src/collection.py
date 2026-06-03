import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone

BSKY_SERVICE = "public.api.bsky.app"
REQUEST_TIMEOUT_S = 45

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


async def fetch_profile_post_counts(session, handles: list, semaphore, wave_size: int = 100) -> dict:
    """
    Consulta postsCount em getProfiles como pre-filtro barato.

    postsCount inclui replies, entao so e usado para remover quem certamente
    nao alcanca min_posts; a aprovacao final continua usando posts_no_replies.
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
                        p.get("handle", ""): p.get("postsCount", 0)
                        for p in data.get("profiles", [])
                    }
        except Exception:
            return {}

    batches = [handles[i:i + 25] for i in range(0, len(handles), 25)]
    counts = {}
    for start in range(0, len(batches), wave_size):
        wave = batches[start:start + wave_size]
        results = await asyncio.gather(*[fetch_batch(b) for b in wave])
        for r in results:
            counts.update(r)
    return counts


async def fetch_posts_no_replies_counts(
    session,
    handles: list,
    semaphore,
    min_posts: int,
    max_posts_per_user: int = 5000,
    max_retries: int = 3,
    handle_chunk_size: int = 250,
    progress_label: str | None = None
) -> tuple[dict, dict]:
    """
    Conta posts originais coletaveis por usuario usando o mesmo criterio da
    coleta da matriz: getAuthorFeed com filter=posts_no_replies.

    Para eficiencia, para de paginar um usuario assim que ele atinge min_posts.
    """
    url = f"https://{BSKY_SERVICE}/xrpc/app.bsky.feed.getAuthorFeed"
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    target = min(min_posts, max_posts_per_user)

    async def fetch_one(handle: str):
        cursor = None
        count = 0

        while count < target:
            params = {
                "actor": handle,
                "limit": 100,
                "filter": "posts_no_replies",
            }
            if cursor:
                params["cursor"] = cursor

            data = None
            for attempt in range(1, max_retries + 1):
                try:
                    async with semaphore:
                        async with session.get(url, params=params, timeout=timeout) as resp:
                            if resp.status == 429:
                                await asyncio.sleep(min(15 * attempt, 60))
                                continue
                            if resp.status != 200:
                                return handle, count, "unavailable" if resp.status in (400, 401, 403, 404) else "error"
                            data = await resp.json()
                            break
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    if attempt < max_retries:
                        await asyncio.sleep(min(2 ** attempt, 30))
                except Exception:
                    return handle, count, "error"

            if data is None:
                return handle, count, "error"

            feed = data.get("feed", [])
            if not feed:
                break

            for item in feed:
                post = item.get("post", {})
                record = post.get("record", {})
                if record.get("text", ""):
                    count += 1
                    if count >= target:
                        break

            cursor = data.get("cursor")
            if not cursor:
                break

        return handle, count, "ok"

    counts = {}
    statuses = {}
    total = len(handles)
    for start in range(0, total, handle_chunk_size):
        chunk = handles[start:start + handle_chunk_size]
        results = await asyncio.gather(
            *[fetch_one(h) for h in chunk],
            return_exceptions=True
        )
        for result in results:
            if isinstance(result, Exception):
                continue
            handle, count, status = result
            counts[handle] = count
            statuses[handle] = status
    return counts, statuses


async def filter_inactive_users(handles: list, min_posts: int, safe_limit: int) -> tuple[list, int]:
    """
    Filtra usuarios com menos de min_posts posts originais coletaveis.
    Usa getAuthorFeed com filter=posts_no_replies, igual a coleta da matriz.
    Retorna (lista_filtrada, quantidade_removida).
    """
    if min_posts <= 0:
        return handles, 0

    semaphore = asyncio.Semaphore(safe_limit)
    async with aiohttp.ClientSession() as session:
        filtered, stats = await filter_min_posts_no_replies_users(
            session=session,
            handles=handles,
            semaphore=semaphore,
            min_posts=min_posts,
            label="usuários"
        )

    return filtered, stats["removed"]


async def filter_min_posts_no_replies_users(
    session,
    handles: list,
    semaphore,
    min_posts: int,
    label: str = "usuários"
) -> tuple[list, dict]:
    """
    Mantem usuarios com pelo menos min_posts posts originais coletaveis.
    Usa a mesma contagem do filtro publico, mas reaproveita a sessao atual.
    """
    if min_posts <= 0 or not handles:
        return handles, {
            "removed": 0,
            "below_min": 0,
            "unverified": 0,
            "profile_below_min": 0,
        }

    print(f"[Atividade] {label}: verificando mínimo de {min_posts} posts sem replies para {len(handles)} usuários.")

    profile_counts = await fetch_profile_post_counts(session, handles, semaphore)
    candidates = []
    profile_below_min = 0
    examples = []
    for handle in handles:
        profile_count = profile_counts.get(handle)
        if profile_count is not None and profile_count < min_posts:
            profile_below_min += 1
            if len(examples) < 5:
                examples.append(f"{handle}: postsCount={profile_count}")
            continue
        candidates.append(handle)

    if profile_below_min > 0:
        print(
            f"[Atividade] {label}: {profile_below_min} removidos no pre-filtro "
            f"(postsCount < {min_posts}); {len(candidates)} precisam de validação posts_no_replies."
        )

    counts, statuses = await fetch_posts_no_replies_counts(
        session,
        candidates,
        semaphore,
        min_posts=min_posts,
        progress_label=str(label)
    )

    filtered = []
    below_min = profile_below_min
    unverified = 0
    for handle in candidates:
        count = counts.get(handle, 0)
        status = statuses.get(handle, "error")
        if status != "ok":
            unverified += 1
            if len(examples) < 5:
                examples.append(f"{handle}: não verificado ({status})")
            continue
        if count >= min_posts:
            filtered.append(handle)
            continue

        below_min += 1
        if len(examples) < 5:
            examples.append(f"{handle}: {count} posts sem replies")

    stats = {
        "removed": len(handles) - len(filtered),
        "below_min": below_min,
        "unverified": unverified,
        "profile_below_min": profile_below_min,
        "examples": examples,
    }
    print(
        f"[Atividade] {label}: {len(filtered)} mantidos, {stats['removed']} removidos "
        f"(<{min_posts} posts sem replies ou não verificados), {unverified} não verificados removidos."
    )
    if examples and len(handles) <= 1000:
        print(f"  [Atividade] Exemplos removidos: {', '.join(examples)}")
    return filtered, stats


async def filter_min_posts_with_replies_users(
    session,
    handles: list,
    semaphore,
    min_posts: int,
    label: str = "usuarios"
) -> tuple[list, dict]:
    """
    Mantem usuarios com pelo menos min_posts posts no postsCount do perfil.

    Diferente do filtro posts_no_replies, postsCount inclui replies. Este filtro
    e usado apenas na coleta inicial da rede, antes de montar as comunidades.
    """
    if min_posts <= 0 or not handles:
        return handles, {
            "removed": 0,
            "below_min": 0,
            "unverified": 0,
            "examples": [],
        }

    print(f"[Atividade] {label}: verificando minimo de {min_posts} posts com replies para {len(handles)} usuarios.")
    profile_counts = await fetch_profile_post_counts(session, handles, semaphore)

    filtered = []
    below_min = 0
    unverified = 0
    examples = []
    for handle in handles:
        profile_count = profile_counts.get(handle)
        if profile_count is None:
            unverified += 1
            if len(examples) < 5:
                examples.append(f"{handle}: postsCount indisponivel")
            continue
        if profile_count < min_posts:
            below_min += 1
            if len(examples) < 5:
                examples.append(f"{handle}: postsCount={profile_count}")
            continue
        filtered.append(handle)

    stats = {
        "removed": len(handles) - len(filtered),
        "below_min": below_min,
        "unverified": unverified,
        "examples": examples,
    }
    print(
        f"[Atividade] {label}: {len(filtered)} mantidos, {stats['removed']} removidos "
        f"(<{min_posts} posts com replies ou postsCount indisponivel)."
    )
    if examples and len(handles) <= 1000:
        print(f"  [Atividade] Exemplos removidos: {', '.join(examples)}")
    return filtered, stats


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

    return filtered, removed


def _parse_bsky_datetime(value: str | None):
    """
    Converte timestamps ISO do Bluesky para datetime UTC.
    """
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


async def fetch_last_post_dates(session, handles: list, semaphore, max_retries: int = 3) -> dict:
    """
    Busca a data do post original mais recente de cada usuario.
    Retorna {handle: {"last_post_at": datetime_utc | None, "status": str}}.
    """
    url = f"https://{BSKY_SERVICE}/xrpc/app.bsky.feed.getAuthorFeed"
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)

    async def fetch_one(handle: str):
        params = {"actor": handle, "limit": 1, "filter": "posts_no_replies"}
        for attempt in range(1, max_retries + 1):
            try:
                async with semaphore:
                    async with session.get(url, params=params, timeout=timeout) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(min(15 * attempt, 60))
                            continue
                        if resp.status != 200:
                            if resp.status in (400, 401, 403, 404):
                                return handle, None, "unavailable"
                            return handle, None, "error"

                        data = await resp.json()
                        feed = data.get("feed", [])
                        if not feed:
                            return handle, None, "no_posts"

                        post = feed[0].get("post", {})
                        record = post.get("record", {})
                        created_at = record.get("createdAt") or post.get("indexedAt")
                        return handle, _parse_bsky_datetime(created_at), "ok"
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt < max_retries:
                    await asyncio.sleep(min(2 ** attempt, 30))
            except Exception:
                return handle, None, "error"

        return handle, None, "rate_limited"

    results = await asyncio.gather(
        *[fetch_one(h) for h in handles],
        return_exceptions=True
    )

    last_dates = {}
    for result in results:
        if isinstance(result, Exception):
            continue
        handle, last_post_at, status = result
        last_dates[handle] = {
            "last_post_at": last_post_at,
            "status": status,
        }
    return last_dates


async def filter_recent_post_users(
    session,
    handles: list,
    semaphore,
    max_last_post_days: int = 90
) -> tuple[list, dict]:
    """
    Mantem apenas usuarios cujo post original mais recente esta dentro de max_last_post_days.
    Usuarios sem post original observado tambem sao removidos.
    """
    if max_last_post_days <= 0 or not handles:
        return handles, {
            "removed": 0,
            "stale": 0,
            "unknown": 0,
            "unverified": 0,
            "cutoff": None,
        }

    cutoff = datetime.now(timezone.utc) - timedelta(days=max_last_post_days)
    last_dates = await fetch_last_post_dates(session, handles, semaphore)

    filtered = []
    stale = 0
    unknown = 0
    unverified = 0
    examples = []

    for handle in handles:
        info = last_dates.get(handle, {"last_post_at": None, "status": "error"})
        last_post_at = info.get("last_post_at")
        status = info.get("status")

        if status in ("error", "rate_limited"):
            unverified += 1
            if len(examples) < 5:
                examples.append(f"{handle}: não verificado ({status})")
            continue

        if last_post_at is not None and last_post_at >= cutoff:
            filtered.append(handle)
            continue

        if last_post_at is None:
            unknown += 1
            if len(examples) < 5:
                examples.append(f"{handle}: sem post original observado")
        else:
            stale += 1
            if len(examples) < 5:
                examples.append(f"{handle}: ultimo post sem replies {last_post_at.date().isoformat()}")

    stats = {
        "removed": len(handles) - len(filtered),
        "stale": stale,
        "unknown": unknown,
        "unverified": unverified,
        "cutoff": cutoff,
        "examples": examples,
    }
    return filtered, stats


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
                        except Exception:
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



async def check_relationships_parallel(session, actor: str, known_dids: list, handle_by_did: dict, semaphore) -> list:
    url = f"https://{BSKY_SERVICE}/xrpc/app.bsky.graph.getRelationships"
    found = []

    # Chunk known_dids into batches of 30
    chunks = [known_dids[i:i+30] for i in range(0, len(known_dids), 30)]

    async def fetch_chunk(chunk):
        params = [("actor", actor)] + [("others", d) for d in chunk]
        retries = 0
        while retries < 3:
            try:
                async with semaphore:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 429:
                            await asyncio.sleep(2)
                            retries += 1
                            continue
                        if resp.status == 200:
                            data = await resp.json()
                            res = []
                            for rel in data.get("relationships", []):
                                if rel.get("following") and rel.get("did") in handle_by_did:
                                    res.append(handle_by_did[rel["did"]])
                            return res
                        return []
            except Exception:
                retries += 1
                await asyncio.sleep(1)
        return []

    results = await asyncio.gather(*[fetch_chunk(c) for c in chunks], return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            found.extend(r)
    return found


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

    print(f"\n[3ª Passagem] Cross-connections: verificando {total} usuários...")

    # Optimize if API supports parallel lookups
    # getRelationships takes up to 30 handles.
    # 5 pages of getFollows = 500 handles = 5 requests.
    # If len(known_nodes) / 30 <= 5, getRelationships makes fewer/equal requests and allows parallel fetching!
    reqs_rel = len(known_nodes) / 30
    reqs_follows = 5  # max_pages

    use_parallel_relationships = reqs_rel <= reqs_follows

    known_dids = []
    handle_by_did = {}

    if use_parallel_relationships:
        known_handles = list(known_nodes)

        async def resolve_batch(batch):
            params = [("actors", h) for h in batch]
            try:
                async with semaphore:
                    async with session.get(f"https://{BSKY_SERVICE}/xrpc/app.bsky.actor.getProfiles", params=params) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            for p in data.get("profiles", []):
                                handle_by_did[p["did"]] = p["handle"]
                                known_dids.append(p["did"])
            except Exception:
                pass

        profile_batches = [known_handles[i:i+25] for i in range(0, len(known_handles), 25)]
        await asyncio.gather(*[resolve_batch(b) for b in profile_batches])

    for start in range(0, total, chunk_size):
        chunk = second_order_nodes[start:start + chunk_size]

        if use_parallel_relationships and known_dids:
            results = await asyncio.gather(
                *[check_relationships_parallel(session, user, known_dids, handle_by_did, semaphore) for user in chunk],
                return_exceptions=True
            )
        else:
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

    print(f"[3ª Passagem] {len(new_edges)} novas arestas encontradas.")
    return new_edges


def _filter_edges_to_min_degree_core(edges: list[tuple[str, str]], min_degree: int = 2) -> tuple[list[tuple[str, str]], dict]:
    """
    Remove iterativamente nos com grau menor que min_degree usando apenas a lista de arestas.
    Isso evita devolver para o grafo final usuarios que entrariam praticamente isolados.
    """
    if min_degree <= 1 or not edges:
        return edges, {
            "kept_nodes": len({node for edge in edges for node in edge}),
            "removed_nodes": 0,
            "removed_edges": 0,
        }

    adjacency: dict[str, set[str]] = {}
    for source, target in edges:
        if source == target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    removed: set[str] = set()
    queue = [node for node, neighbors in adjacency.items() if len(neighbors) < min_degree]

    while queue:
        node = queue.pop()
        if node in removed:
            continue
        removed.add(node)
        for neighbor in list(adjacency.get(node, set())):
            adjacency.get(neighbor, set()).discard(node)
            if neighbor not in removed and len(adjacency.get(neighbor, set())) < min_degree:
                queue.append(neighbor)
        adjacency[node].clear()

    filtered_edges = [
        (source, target)
        for source, target in edges
        if source != target and source not in removed and target not in removed
    ]
    kept_nodes = {node for edge in filtered_edges for node in edge}
    return filtered_edges, {
        "kept_nodes": len(kept_nodes),
        "removed_nodes": len(removed),
        "removed_edges": len(edges) - len(filtered_edges),
    }


async def collect_network(
    core_user,
    safe_limit,
    max_followers: int = 5000,
    max_last_post_days: int = 90,
    min_posts: int = 0
):
    """
    Executa a coleta em largura (BFS) com 3 passagens:

    1ª: Seguidores do core_user (1ª ordem) + filtros de celebridades, recência e posts com replies.
    2ª: Seguidores de cada usuário 1ª ordem (2ª ordem) + filtros de celebridades, recência e posts com replies.
    3ª: Para cada usuário 2ª ordem, verifica quem ele segue que já está no grafo
        → gera arestas cruzadas, densificando o grafo.
    """
    semaphore = asyncio.Semaphore(safe_limit)
    edges = []

    async with aiohttp.ClientSession() as session:
        print(f"\n[Validação] Usuário raiz: {core_user}")
        core_recent, core_recent_stats = await filter_recent_post_users(
            session,
            [core_user],
            semaphore,
            max_last_post_days=max_last_post_days
        )
        if not core_recent:
            print(
                f"[Erro] Usuário raiz '{core_user}' removido pelo filtro de recência "
                f"(último post sem replies > {max_last_post_days} dias ou não verificado)."
            )
            return []

        if min_posts > 0:
            core_active, _ = await filter_min_posts_with_replies_users(
                session,
                [core_user],
                semaphore,
                min_posts=min_posts,
                label="usuário raiz"
            )
            if not core_active:
                print(
                    f"[Erro] Usuário raiz '{core_user}' removido pelo filtro de atividade "
                    f"(<{min_posts} posts com replies ou postsCount indisponível)."
                )
                return []

        # ── 1ª Passagem: Followers de 1ª ordem ─────────────────────────────
        print("\n[1ª ordem] Coletando seguidores...")
        first_order_raw = await fetch_followers(session, core_user, semaphore)

        first_order, removed_1 = await filter_celebrities(session, first_order_raw, semaphore, max_followers)
        print(f"[1ª ordem] Seguidores: {len(first_order_raw)} coletados | {removed_1} removidos por seguidores.")

        first_order, recent_stats_1 = await filter_recent_post_users(
            session,
            first_order,
            semaphore,
            max_last_post_days=max_last_post_days
        )
        print(
            "[1ª ordem] Recência: "
            f"{len(first_order)} mantidos, {recent_stats_1['removed']} removidos "
            f"({recent_stats_1['stale']} antigos, {recent_stats_1['unknown']} sem post original observado, "
            f"{recent_stats_1['unverified']} não verificados removidos)."
        )
        if recent_stats_1.get("examples") and len(first_order_raw) <= 1000:
            print(f"  [Recência] Exemplos removidos: {', '.join(recent_stats_1['examples'])}")

        if min_posts > 0:
            first_order, _ = await filter_min_posts_with_replies_users(
                session,
                first_order,
                semaphore,
                min_posts=min_posts,
                label="usuários de 1ª ordem"
            )

        for follower in first_order:
            edges.append((follower, core_user))

        # ── 2ª Passagem: Followers de 2ª ordem ─────────────────────────────
        print(f"\n[2ª ordem] Coletando seguidores de {len(first_order)} usuários...")
        async def fetch_followers_for(actor):
            return actor, await fetch_followers(session, actor, semaphore)

        tasks = [asyncio.create_task(fetch_followers_for(f)) for f in first_order]

        second_order_results = []
        for task in asyncio.as_completed(tasks):
            follower_node, result = await task
            second_order_results.append((follower_node, result))

        # Filtra celebridades da 2ª ordem em batch
        all_second_handles = list({h for _, followers in second_order_results for h in followers})
        counts_map = await fetch_follower_counts(session, all_second_handles, semaphore)
        second_handles_by_followers = [
            h for h in all_second_handles
            if counts_map.get(h, 0) <= max_followers
        ]
        removed_second_unique_followers = len(all_second_handles) - len(second_handles_by_followers)
        print(
            f"[2ª ordem] Handles únicos: {len(all_second_handles)} coletados | "
            f"{removed_second_unique_followers} removidos por seguidores."
        )
        recent_second_handles, recent_stats_2 = await filter_recent_post_users(
            session,
            second_handles_by_followers,
            semaphore,
            max_last_post_days=max_last_post_days
        )
        recent_second_set = set(recent_second_handles)
        print(
            "[2ª ordem] Recência: "
            f"{len(recent_second_set)} handles únicos mantidos, {recent_stats_2['removed']} removidos "
            f"({recent_stats_2['stale']} antigos, {recent_stats_2['unknown']} sem post original observado, "
            f"{recent_stats_2['unverified']} não verificados removidos)."
        )
        if recent_stats_2.get("examples") and len(second_handles_by_followers) <= 1000:
            print(f"  [Recência] Exemplos removidos: {', '.join(recent_stats_2['examples'])}")

        if min_posts > 0:
            recent_second_handles, activity_stats_2 = await filter_min_posts_with_replies_users(
                session,
                recent_second_handles,
                semaphore,
                min_posts=min_posts,
                label="handles únicos de 2ª ordem"
            )
            recent_second_set = set(recent_second_handles)

        second_order_links = {}
        for (follower_node, followers_list) in second_order_results:
            for second_follower in followers_list:
                if counts_map.get(second_follower, 0) > max_followers:
                    continue
                if second_follower not in recent_second_set:
                    continue
                second_order_links.setdefault(second_follower, set()).add(follower_node)

        second_order_with_min_connections = {
            handle
            for handle, linked_first_order in second_order_links.items()
            if len(linked_first_order) >= 2
        }
        removed_low_connection_unique = len(second_order_links) - len(second_order_with_min_connections)
        if removed_low_connection_unique > 0:
            print(
                f"[2ª ordem] Conectividade: {len(second_order_with_min_connections)} mantidos, "
                f"{removed_low_connection_unique} removidos por terem <2 conexões com a 1ª ordem."
            )

        second_order_kept = set()
        removed_2_followers = 0
        removed_2_ineligible = 0
        removed_2_low_connections = 0
        for (follower_node, followers_list) in second_order_results:
            for second_follower in followers_list:
                fc = counts_map.get(second_follower, 0)
                if fc > max_followers:
                    removed_2_followers += 1
                    continue
                if second_follower not in recent_second_set:
                    removed_2_ineligible += 1
                    continue
                if second_follower not in second_order_with_min_connections:
                    removed_2_low_connections += 1
                    continue
                if fc <= max_followers:
                    edges.append((second_follower, follower_node))
                    second_order_kept.add(second_follower)

        print(
            f"[2ª ordem] Arestas filtradas: {len(second_order_kept)} usuários mantidos, "
            f"{removed_2_followers} removidos por seguidores, "
            f"{removed_2_ineligible} removidos por recência/atividade insuficiente, "
            f"{removed_2_low_connections} removidos por <2 conexões."
        )

        # ── 3ª Passagem: Cross-connections para densificar o grafo ──────────
        known_nodes = {core_user} | set(first_order) | second_order_kept
        cross_edges = await enrich_edges_with_cross_connections(
            session,
            list(second_order_kept),
            known_nodes,
            semaphore
        )
        edges.extend(cross_edges)

    pre_degree_edges = len(edges)
    edges, degree_stats = _filter_edges_to_min_degree_core(edges, min_degree=2)
    if degree_stats["removed_nodes"] > 0:
        print(
            f"[Grau mínimo] Removidos {degree_stats['removed_nodes']} usuários com grau <2 "
            f"e {degree_stats['removed_edges']} arestas."
        )

    print(
        f"\n[Resumo] Total de arestas: {len(edges)} "
        f"(antes do grau mínimo: {pre_degree_edges}; cross brutas: {len(cross_edges)})"
    )

    return edges
