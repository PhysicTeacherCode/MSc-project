import networkx as nx

def build_graph(edges):
    """
    Constrói o grafo não-direcionado a partir de uma lista de arestas brutas.
    Nós: Usuários. Arestas: Relacionamento de seguidor.
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def get_network_metrics(G):
    """
    Retorna estatísticas globais e úteis sobre o grafo.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)
    
    if num_nodes == 0:
        avg_degree = 0
    else:
        avg_degree = sum(dict(G.degree()).values()) / num_nodes
        
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "avg_degree": avg_degree
    }

def get_influential_nodes(G, top_n=5):
    """
    Retorna os 'n' nós com maior grau (centralidade local).
    """
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    return [node for node, degree in sorted_nodes[:top_n]]
