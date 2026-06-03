import os
import sys

# Suprime os prints de "Note: to be able to use all crisp methods..." do cdlib
_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    from cdlib import algorithms
finally:
    sys.stdout.close()
    sys.stdout = _stdout

import networkx as nx


def _avg_degree(G):
    if G.number_of_nodes() == 0:
        return 0.0
    return sum(dict(G.degree()).values()) / G.number_of_nodes()


def kcore_scenarios_for_nodes(G, nodes, k_values=None):
    """
    Calcula como uma comunidade ficaria para cada k-core local.
    k=0 representa a comunidade original, sem k-core local.
    """
    sub_G = G.subgraph(nodes).copy()
    if sub_G.number_of_nodes() == 0:
        return {0: {"nodes": 0, "edges": 0, "avg_degree": 0.0}}

    try:
        core_numbers = nx.core_number(sub_G)
        max_core = max(core_numbers.values(), default=0)
    except nx.NetworkXError:
        max_core = 0

    if k_values is None:
        k_values = range(0, max_core + 1)

    scenarios = {}
    max_degree = max(dict(sub_G.degree()).values(), default=0)
    for k in sorted({int(k) for k in k_values if int(k) >= 0}):
        if k == 0:
            core_G = sub_G
        elif k > max_degree:
            core_G = nx.Graph()
        else:
            try:
                core_G = nx.k_core(sub_G, k=k)
            except nx.NetworkXError:
                core_G = nx.Graph()

        scenarios[k] = {
            "nodes": core_G.number_of_nodes(),
            "edges": core_G.number_of_edges(),
            "avg_degree": _avg_degree(core_G),
        }

    return scenarios


def detect_communities_multi_resolution(G, resolutions=[1.0, 1.5, 2.0, 2.5, 3.0]):
    """
    Executa o algoritmo Leiden (via cdlib C++ backend) para diferentes resoluções.
    O k-core local não é aplicado aqui; apenas são calculados cenários para
    que o usuário escolha o k-core na exportação de cada subcomunidade.
    """
    print(f"\n[Comunidades] Leiden em {len(resolutions)} resoluções: {', '.join(map(str, resolutions))}.")
    
    results = {}
    
    for res in resolutions:
        try:
            # 1. Detecção Inicial
            coms = algorithms.rb_pots(G, weights=None, resolution_parameter=res)
            initial_mod = coms.newman_girvan_modularity().score

            communities = [list(nodes) for nodes in coms.communities if len(nodes) > 1]
            if not communities:
                results[res] = {
                    "partition": {},
                    "modularity": initial_mod,
                    "initial_mod": initial_mod,
                    "num_communities": 0,
                    "sizes": {},
                    "avg_degrees": {},
                    "edge_counts": {},
                    "kcore_scenarios": {},
                    "lowest_avg_degree": None,
                }
                continue

            partition = {}
            community_sizes = {}
            avg_degrees = {}
            edge_counts = {}
            kcore_scenarios = {}
            for cid, nodes in enumerate(communities):
                sub_G = G.subgraph(nodes)
                community_sizes[cid] = len(nodes)
                edge_counts[cid] = sub_G.number_of_edges()
                avg_degrees[cid] = _avg_degree(sub_G)
                kcore_scenarios[cid] = kcore_scenarios_for_nodes(G, nodes)
                for node in nodes:
                    partition[node] = cid

            lowest_cid, lowest_avg = min(avg_degrees.items(), key=lambda item: item[1])
            
            results[res] = {
                "partition": partition,
                "modularity": initial_mod,
                "initial_mod": initial_mod,
                "num_communities": len(communities),
                "sizes": community_sizes,
                "avg_degrees": avg_degrees,
                "edge_counts": edge_counts,
                "kcore_scenarios": kcore_scenarios,
                "lowest_avg_degree": {
                    "cid": lowest_cid,
                    "size": community_sizes[lowest_cid],
                    "avg_degree": lowest_avg,
                },
            }
            
        except Exception as e:
            print(f"  [Erro] Falha na resolução {res}: {e}")
            
    return results


def apply_partition(G, partition):
    """
    Adiciona o atributo 'Community ID' no grafo baseado na partição escolhida pelo usuário.
    Nós fora da partição escolhida ficam com ID -1.
    """
    # Primeiro, limpa IDs antigos
    nx.set_node_attributes(G, {node: -1 for node in G.nodes()}, name="Community ID")
    nx.set_node_attributes(G, partition, name="Community ID")

def extract_subcommunity_graph(G, community_id):
    """
    Extrai do grafo original 'G' um subgrafo contendo apenas os nós da comunidade.
    """
    nodes_in_comm = [
        node for node, data in G.nodes(data=True) 
        if data.get("Community ID") == community_id and data.get("Community ID") != -1
    ]
    return G.subgraph(nodes_in_comm).copy()


def extract_subcommunity_graph_with_kcore(G, community_id, k_core=0):
    """
    Extrai a comunidade e aplica k-core local apenas quando k_core > 0.
    """
    sub_G = extract_subcommunity_graph(G, community_id)
    if k_core <= 0 or sub_G.number_of_nodes() == 0:
        return sub_G

    max_degree = max(dict(sub_G.degree()).values(), default=0)
    if k_core > max_degree:
        return nx.Graph()

    try:
        return nx.k_core(sub_G, k=k_core).copy()
    except nx.NetworkXError:
        return nx.Graph()


def deduplicate_handles_in_graph(G) -> list:
    """
    Detecta e remove nós com handles duplicados em TODO o grafo.
    Deve ser chamada ANTES da detecção de comunidades.
    
    Agrupa nós por handle normalizado (case-insensitive + strip).
    Quando há duplicata, mantém o nó com MAIOR grau.
    Remove os demais do grafo G in-place.
    
    Retorna:
        Lista de handles removidos (para log).
    """
    handle_groups = {}
    for node in list(G.nodes()):
        key = node.strip().lower()
        handle_groups.setdefault(key, []).append(node)
    
    removed = []
    for normalized, group in handle_groups.items():
        if len(group) <= 1:
            continue
        
        group_sorted = sorted(group, key=lambda n: G.degree(n), reverse=True)
        keeper = group_sorted[0]
        duplicates = group_sorted[1:]
        
        for dup in duplicates:
            G.remove_node(dup)
            removed.append(dup)
    
    return removed


def find_core_user_community(partition: dict, core_user: str) -> int:
    """
    Identifica em qual comunidade o core_user foi alocado.
    Retorna o ID da comunidade ou -1 se o core_user não está na partição.
    """
    return partition.get(core_user, -1)


def deduplicate_handles_in_community(G, partition: dict, community_id: int) -> list:
    """
    Detecta e remove nós com handles duplicados dentro de uma comunidade.
    
    Na API do Bluesky, um mesmo handle pode aparecer mais de uma vez quando
    coletado por caminhos BFS distintos (1ª e 2ª ordem). Embora o grafo trate
    handles como nós, variações de case (maiúsc./minúsc.) podem gerar duplicatas
    semânticas (ex: 'User.bsky.social' e 'user.bsky.social').
    
    Estratégia de remoção:
        - Agrupa nós por handle normalizado (lowercase).
        - Quando há duplicata, mantém o nó com MAIOR grau (mais conexões),
          pois é o mais informativo para a rede.
        - Remove os demais do grafo G in-place.
    
    Retorna:
        Lista de handles removidos (para log).
    """
    # Filtra nós pertencentes à comunidade alvo
    nodes_in_comm = [
        node for node, data in G.nodes(data=True)
        if data.get("Community ID") == community_id
    ]
    
    # Agrupa por handle normalizado (case-insensitive)
    handle_groups = {}
    for node in nodes_in_comm:
        key = node.strip().lower()
        handle_groups.setdefault(key, []).append(node)
    
    removed = []
    for normalized, group in handle_groups.items():
        if len(group) <= 1:
            continue
        
        # Ordena por grau descendente: mantém o mais conectado
        group_sorted = sorted(group, key=lambda n: G.degree(n), reverse=True)
        keeper = group_sorted[0]
        duplicates = group_sorted[1:]
        
        for dup in duplicates:
            G.remove_node(dup)
            # Remove da partição também
            if dup in partition:
                del partition[dup]
            removed.append(dup)
    
    return removed
