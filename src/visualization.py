import matplotlib.pyplot as plt
import networkx as nx
import os

def generate_network_visualization(G, output_dir="data/processed/png", filename="visualizacao.png"):
    """
    Gera uma visualização estática no estilo Hall & Bialek (2019):
    - Fundo branco
    - Nós vermelhos, circulares e uniformes
    - Arestas pretas, finas e semi-transparentes
    - Layout spring (ForceAtlas2-like) com núcleo denso e periferia estendida
    - Se < 1000 nós: mostra todos; se >= 1000: mostra os top 1000 hubs
    """
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        print(f"[Visualização] Grafo vazio, pulando geração de {filename}.")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)

    # 1. Filtragem (redes massivas → top 1000 hubs)
    if total_nodes < 1000:
        print(f"[Visualização] {filename}: Grafo completo ({total_nodes} nós).")
        subgraph = G.copy()
    else:
        print(f"[Visualização] {filename}: Extraindo top 1000 hubs de {total_nodes} nós...")
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:1000]
        subgraph = G.subgraph(top_nodes).copy()

    if subgraph.number_of_nodes() == 0:
        print(f"[Visualização] {filename}: Subgrafo vazio, abortando.")
        return

    n = subgraph.number_of_nodes()

    # 2. Layout Fruchterman-Reingold (spring_layout)
    #    k pequeno → nós mais próximos (núcleo denso)
    #    iterations alto → melhor convergência para networks com hubs dominantes
    print(f"[Visualização] {filename}: Calculando layout Fruchterman-Reingold ({n} nós)...")
    pos = nx.spring_layout(
        subgraph,
        k=0.15,          # Distância ideal entre nós — menor = mais compacto
        iterations=200,  # Mais iterações = melhor acomodação dos hubs no centro
        seed=42          # Reprodutibilidade
    )

    # 3. Tamanho da figura escalável (máx 10x10 pol → 3000x3000px a 300dpi)
    fig_size = max(6, min(10, n / 5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), facecolor='white')
    ax.set_facecolor('white')

    # 4. Parâmetros visuais — estilo Hall & Bialek
    node_size = max(20, min(120, 2000 // n))   # Nós menores para grafos grandes
    edge_width = 0.5
    edge_alpha = 0.5 if n < 200 else 0.3

    # 5. Desenha arestas (pretas e finas)
    print(f"[Visualização] {filename}: Renderizando...")
    nx.draw_networkx_edges(
        subgraph, pos,
        ax=ax,
        width=edge_width,
        alpha=edge_alpha,
        edge_color='black'
    )

    # 6. Desenha nós (vermelhos, sem contorno, uniformes)
    nx.draw_networkx_nodes(
        subgraph, pos,
        ax=ax,
        node_size=node_size,
        node_color='red',
        linewidths=0,      # sem borda
        alpha=1.0
    )

    ax.axis('off')
    plt.tight_layout(pad=0)

    # 7. Exporta
    print(f"[Visualização] Salvando em {file_path}...")
    plt.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[Visualização] {filename}: Imagem salva com sucesso!")
