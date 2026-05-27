import os
import sys
from collections import Counter

# Suprime os prints de "Note: to be able to use all crisp methods..." do cdlib
_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
try:
    from cdlib import algorithms
finally:
    sys.stdout.close()
    sys.stdout = _stdout

import networkx as nx


from cdlib.classes.node_clustering import NodeClustering

def detect_communities_multi_resolution(G, resolutions=[1.0, 1.5, 2.0, 2.5, 3.0]):
    """
    Executa o algoritmo Leiden (via cdlib C++ backend) para diferentes resoluções.
    Aplica também o filtro k-core local em cada subcomunidade conforme solicitado.
    """
    print(f"\n[Comunidades] Executando detecção Leiden (C++) e Refinamento K-Core...")
    
    results = {}
    
    for res in resolutions:
        print(f"  -> Processando resolução: {res}")
        try:
            # 1. Detecção Inicial
            coms = algorithms.rb_pots(G, weights=None, resolution_parameter=res)
            initial_mod = coms.newman_girvan_modularity().score
            
            # 2. Refinamento K-Core Local (k=2)
            # Removemos nós que não possuem pelo menos 2 conexões DENTRO de sua própria comunidade
            filtered_communities = []
            for community_nodes in coms.communities:
                if len(community_nodes) < 3: continue # k=2 requer pelo menos 3 nós
                
                sub_G = G.subgraph(community_nodes)
                k_core_sub = nx.k_core(sub_G, k=2)
                
                # Regra: remover subcomunidades com 1 ou zero nós após o filtro
                if k_core_sub.number_of_nodes() > 1:
                    filtered_communities.append(list(k_core_sub.nodes()))
            
            # 3. Recálculo da Modularidade sobre o grafo REFINADO
            if not filtered_communities:
                results[res] = {"partition": {}, "modularity": 0, "initial_mod": initial_mod, "num_communities": 0, "sizes": {}}
                continue

            # Para que a modularidade pós-filtro faça sentido, calculamos sobre o subgrafo dos sobreviventes
            surviving_nodes = [node for comm in filtered_communities for node in comm]
            G_refined = G.subgraph(surviving_nodes)
            
            refined_coms = NodeClustering(communities=filtered_communities, graph=G_refined, method_name="leiden_refined")
            final_mod = refined_coms.newman_girvan_modularity().score
            
            # Formata partição para o dicionário (nó -> id)
            partition = {}
            community_sizes = {}
            for cid, nodes in enumerate(filtered_communities):
                community_sizes[cid] = len(nodes)
                for node in nodes:
                    partition[node] = cid
            
            results[res] = {
                "partition": partition,
                "modularity": final_mod,
                "initial_mod": initial_mod,
                "num_communities": len(filtered_communities),
                "sizes": community_sizes
            }
            
        except Exception as e:
            print(f"  [Erro] Falha na resolução {res}: {e}")
            
    return results


def apply_partition(G, partition):
    """
    Adiciona o atributo 'Community ID' no grafo baseado na partição escolhida pelo usuário.
    Nós removidos pelo filtro k-core ficarão sem o atributo ou com ID -1.
    """
    # Primeiro, limpa IDs antigos
    nx.set_node_attributes(G, {node: -1 for node in G.nodes()}, name="Community ID")
    nx.set_node_attributes(G, partition, name="Community ID")

def extract_subcommunity_graph(G, community_id):
    """
    Extrai do grafo original 'G' um subgrafo contendo apenas os nós da comunidade.
    Como já aplicamos k-core durante a detecção, o subgrafo extraído já estará filtrado.
    """
    nodes_in_comm = [
        node for node, data in G.nodes(data=True) 
        if data.get("Community ID") == community_id and data.get("Community ID") != -1
    ]
    return G.subgraph(nodes_in_comm).copy()


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
        
        print(f"  [Dedup] Handle '{normalized}': mantendo '{keeper}' (grau={G.degree(keeper)}), "
              f"removendo {len(duplicates)} duplicata(s): {duplicates}")
        
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
        
        print(f"  [Dedup] Handle '{normalized}': mantendo '{keeper}' (grau={G.degree(keeper)}), "
              f"removendo {len(duplicates)} duplicata(s): {duplicates}")
        
        for dup in duplicates:
            G.remove_node(dup)
            # Remove da partição também
            if dup in partition:
                del partition[dup]
            removed.append(dup)
    
    return removed
