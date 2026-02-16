import asyncio
import aiohttp
from aiohttp import ClientSession, TCPConnector
import networkx as nx
import os

async def get_followers(session: ClientSession, actor_handle: str, limit=1000):
    """
    Busca todos os seguidores de um usuário através da API do Bluesky
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
    Busca seguidores de um batch de usuários usando a mesma sessão
    """
    tasks = [get_followers(session, u) for u in user_batch]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def fetch_followers_list(user_list, batch_size=100, per_host_limit=50):
    """
    Busca seguidores de uma lista de usuários em batches para otimizar performance
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
    Constrói um nível de seguidores (1ª, 2ª ou 3ª ordem)
    """
    results_dict = await fetch_followers_list(previous_users, batch_size=100)
    flattened = list({f for flw_list in results_dict.values() for f in flw_list})
    return results_dict, flattened


def add_edges_from_dict(G, data_dict, level):
    """
    Adiciona arestas e atributos de nível ao grafo de forma otimizada
    """
    edges = []
    for actor, followers in data_dict.items():
        edges.extend((f, actor) for f in followers)

    G.add_edges_from(edges)

    for actor, followers in data_dict.items():
        for f in followers:
            if f in G.nodes:
                G.nodes[f]["level"] = level

async def main():
    """
    Pipeline principal: busca seguidores, constrói grafo e detecta comunidades
    """
    handle_bsky = input("Digite o handle do usuário: ")

    print("Coletando seguidores de 1ª ordem...")
    first_dict, first_list = await build_order([handle_bsky], "dos Core Users")

    print("Coletando seguidores de 2ª ordem...")
    second_dict, second_list = await build_order(first_list, "da 1ª ordem")

    os.makedirs(f"../data/graph/{handle_bsky}", exist_ok=True)

    print("Criando grafo...")
    G = nx.DiGraph()

    G.add_node(handle_bsky, level=0)

    add_edges_from_dict(G, first_dict, level=1)
    add_edges_from_dict(G, second_dict, level=2)

    G = G.to_undirected()

    G.remove_edges_from(nx.selfloop_edges(G))

    G = nx.k_core(G,2)

    nx.write_gexf(G, f"../data/graph/{handle_bsky}/{handle_bsky}_-_nós_{G.number_of_nodes()}(comunidade_inteira).gexf")

    print("Criando comunidades com algoritmo louvain...")

    communities = list(nx.algorithms.community.louvain_communities(G, seed=42))

    print("Salvando subcomunidades...")
    for idx, comm in enumerate(communities):
        subgraph = G.subgraph(comm).copy()
        subgraph = nx.k_core(subgraph,2)
        if subgraph.number_of_nodes() != 0:
            nx.write_gexf(subgraph, f"../data/graph/{handle_bsky}/comunidade_{idx + 1}_-_nós_{subgraph.number_of_nodes()}.gexf")


asyncio.run(main())