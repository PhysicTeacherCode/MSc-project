import asyncio
import aiohttp
from aiohttp import ClientSession, TCPConnector
import networkx as nx
import os

async def get_user_posts_count(session: ClientSession, actor_handle: str):
    """
    Busca e conta posts originais de um usuario, excluindo replies e reposts.

    [Ordem de acoes]:
    1. Pagina o feed do autor com cursor e timeout.
    2. Filtra registros validos e soma posts originais.
    3. Retorna o handle com a contagem total.
    """
    url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
    cursor = None
    posts_count = 0

    while True:
        params = {"actor": actor_handle, "limit": 100}
        if cursor:
            params["cursor"] = cursor

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()

        except Exception:
            break

        feed = data.get("feed", [])
        for item in feed:
            record = item.get("post", {}).get("record", {})
            if record and not record.get("reply") and "$type" in record:
                if record["$type"] == "app.bsky.feed.post":
                    posts_count += 1

        cursor = data.get("cursor")
        if not cursor:
            break

    return actor_handle, posts_count


async def fetch_posts_count_batch(session: ClientSession, user_batch):
    """
    Executa a contagem de posts para um lote de usuarios em paralelo.

    [Ordem de acoes]:
    1. Monta tarefas de contagem por usuario.
    2. Aguarda todas as tarefas e retorna os resultados.
    """
    tasks = [get_user_posts_count(session, u) for u in user_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def filter_users_by_post_count(user_list, min_posts=2, batch_size=100, per_host_limit=50):
    """
    Filtra usuarios com pelo menos um numero minimo de posts originais.

    [Ordem de acoes]:
    1. Cria sessao HTTP com limites e timeout.
    2. Processa usuarios em lotes e coleta contagens.
    3. Retorna somente usuarios com contagem suficiente.
    """
    connector = TCPConnector(
        limit_per_host=per_host_limit,
        limit=200,
        ttl_dns_cache=300
    )

    timeout = aiohttp.ClientTimeout(total=30, connect=10)

    valid_users = []

    async with ClientSession(connector=connector, timeout=timeout) as session:
        for i in range(0, len(user_list), batch_size):
            batch = user_list[i:i + batch_size]
            batch_results = await fetch_posts_count_batch(session, batch)

            for result in batch_results:
                if not isinstance(result, Exception):
                    user, post_count = result
                    if post_count >= min_posts:
                        valid_users.append(user)

            if i + batch_size < len(user_list):
                await asyncio.sleep(0.3)

    return valid_users


async def get_followers(session: ClientSession, actor_handle: str, limit=1000):
    """
    Coleta seguidores de um usuario via API do Bluesky.

    [Ordem de acoes]:
    1. Pagina seguidores usando cursor.
    2. Filtra handles invalidos e acumula resultados.
    3. Retorna o handle com a lista de seguidores.
    """
    url = "https://public.api.bsky.app/xrpc/app.bsky.graph.getFollowers"
    cursor = None
    followers = []

    while True:
        params = {"actor": actor_handle, "limit": 100}
        if cursor:
            params["cursor"] = cursor

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    break
                data = await resp.json()

        except Exception:
            break

        followers.extend(
            item["handle"]
            for item in data.get("followers", [])
            if item.get("handle") != "handle.invalid"
        )

        cursor = data.get("cursor")
        if not cursor or len(followers) >= limit:
            break

    return actor_handle, followers


async def fetch_followers_batch(session: ClientSession, user_batch):
    """
    Busca seguidores de um lote de usuarios em paralelo.

    [Ordem de acoes]:
    1. Cria tarefas por usuario para buscar seguidores.
    2. Aguarda todas as tarefas e retorna os resultados.
    """
    tasks = [get_followers(session, u) for u in user_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def fetch_followers_list(user_list, batch_size=100, per_host_limit=50):
    """
    Busca seguidores de uma lista em lotes para otimizar desempenho.

    [Ordem de acoes]:
    1. Configura sessao HTTP com limites e timeout.
    2. Processa lotes, agregando resultados validos.
    3. Retorna um dicionario usuario -> seguidores.
    """
    connector = TCPConnector(
        limit_per_host=per_host_limit,
        limit=200,
        ttl_dns_cache=300
    )

    timeout = aiohttp.ClientTimeout(total=30, connect=10)

    async with ClientSession(connector=connector, timeout=timeout) as session:
        results = []

        for i in range(0, len(user_list), batch_size):
            batch = user_list[i:i + batch_size]
            batch_results = await fetch_followers_batch(session, batch)

            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)

            if i + batch_size < len(user_list):
                await asyncio.sleep(0.3)

    return {actor: flw for actor, flw in results}


async def build_order(previous_users, order_label):
    """
    Constroi um nivel de seguidores e a lista achatada.

    [Ordem de acoes]:
    1. Busca seguidores para o conjunto de entrada.
    2. Achata os seguidores em uma lista unica.
    3. Retorna o dicionario e a lista.
    """
    results_dict = await fetch_followers_list(previous_users, batch_size=100)
    flattened = list({f for flw_list in results_dict.values() for f in flw_list})
    return results_dict, flattened


def add_edges_from_dict(G, data_dict, level):
    """
    Adiciona arestas ao grafo e define o nivel dos nos.

    [Ordem de acoes]:
    1. Monta todas as arestas a partir do dicionario.
    2. Adiciona arestas ao grafo.
    3. Marca o nivel dos nos seguidores.
    """
    edges = []
    for actor, followers in data_dict.items():
        edges.extend((f, actor) for f in followers)

    G.add_edges_from(edges)

    for actor, followers in data_dict.items():
        for f in followers:
            if f in G.nodes:
                G.nodes[f]["level"] = level

def get_next_core_user_dir(base_dir):
    """
    Escolhe a proxima pasta anonima disponivel para o core user.

    [Ordem de acoes]:
    1. Garante que o diretorio base exista.
    2. Incrementa o indice ate encontrar uma pasta livre.
    3. Retorna o caminho completo e o rotulo da pasta.
    """
    os.makedirs(base_dir, exist_ok=True)
    idx = 1
    while True:
        label = f"core_user_{idx}"
        candidate = os.path.join(base_dir, label)
        if not os.path.exists(candidate):
            return candidate, label
        idx += 1


def append_core_user_list(data_dir, handle_bsky, follower_count, core_user_label):
    """
    Registra o core user em um arquivo de lista persistente.

    [Ordem de acoes]:
    1. Garante o diretorio de dados e o arquivo.
    2. Escreve o cabecalho se for a primeira execucao.
    3. Adiciona uma linha com handle, seguidores e pasta.
    """
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "core_users_list.txt")
    is_new = not os.path.exists(output_path)
    with open(output_path, "a", encoding="utf-8") as file:
        if is_new:
            file.write("handle,followers,folder\n")
        file.write(f"{handle_bsky},{follower_count},{core_user_label}\n")


async def main():
    """
    Executa o pipeline completo: coleta seguidores, filtra usuarios e gera grafos.

    [Ordem de acoes]:
    1. Coleta seguidores de 1a e 2a ordem e filtra por posts.
    2. Cria pastas anonimas e registra o core user.
    3. Monta o grafo, aplica filtros e salva GEXF.
    """
    handle_bsky = input("Digite o handle do usuário: ")

    print("Coletando seguidores de 1ª ordem...")
    first_dict, first_list = await build_order([handle_bsky], "dos Core Users")

    print("Coletando seguidores de 2ª ordem...")
    second_dict, second_list = await build_order(first_list, "da 1ª ordem")

    all_users = set(first_list + second_list + [handle_bsky])

    print(f"Total de usuários: {len(all_users)}")

    base_graph_dir = os.path.join("MSc-project", "data", "graph")
    core_user_dir, core_user_label = get_next_core_user_dir(base_graph_dir)
    gexf_dir = os.path.join(core_user_dir, "GEXF")
    png_dir = os.path.join(core_user_dir, "PNG")

    os.makedirs(gexf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    followers_list = first_dict.get(handle_bsky, [])
    followers_count = len(set(followers_list))
    append_core_user_list(os.path.join("MSc-project", "data"), handle_bsky, followers_count, core_user_label)

    print("Criando grafo...")
    G = nx.DiGraph()

    G.add_node(handle_bsky, level=0)

    add_edges_from_dict(G, first_dict, level=1)
    add_edges_from_dict(G, second_dict, level=2)

    G = G.to_undirected()

    G.remove_edges_from(nx.selfloop_edges(G))


    G = nx.k_core(G, 2)

    nx.write_gexf(G, os.path.join(gexf_dir, f"{core_user_label}_-_nós_{G.number_of_nodes()}(comunidade_inteira).gexf"))

    print("Criando comunidades com algoritmo Clauset-Newman-Moore (greedy modularity maximization)...")

    communities = list(nx.algorithms.community.greedy_modularity_communities(G))

    print("Salvando subcomunidades...")
    for idx, comm in enumerate(communities):
        subgraph = G.subgraph(comm).copy()
        subgraph = nx.k_core(subgraph, 2)
        if subgraph.number_of_nodes() != 0:
            nx.write_gexf(subgraph, os.path.join(gexf_dir, f"comunidade_{idx + 1}_-_nós_{subgraph.number_of_nodes()}.gexf"))


asyncio.run(main())