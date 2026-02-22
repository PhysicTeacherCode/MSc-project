import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def gexf_to_png_dark(gexf_file, output_file=None):
    """
    Converte arquivo GEXF para PNG com tema escuro (melhor contraste)
    """
    if output_file is None:
        output_file = gexf_file.replace('.gexf', '.png')

    G = nx.read_gexf(gexf_file)

    if G.number_of_nodes() == 0:
        print(f"⚠️  {gexf_file} está vazio, pulando...")
        return

    fig = plt.figure(figsize=(25, 25), facecolor='#1a1a1a')
    ax = fig.add_subplot(111, facecolor='#1a1a1a')

    k_value = 1.5 / np.sqrt(G.number_of_nodes())
    pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42)

    node_levels = nx.get_node_attributes(G, 'level')

    if node_levels:
        cores = {0: '#ff6b6b', 1: '#ffd93d', 2: '#6bcfff'}
        tamanhos = {0: 800, 1: 400, 2: 200}
        alphas = {0: 1.0, 1: 0.95, 2: 0.9}
        labels_dict = {0: 'Core', 1: '1ª Ordem', 2: '2ª Ordem'}

        for level in [0, 1, 2]:
            nodes = [n for n, l in node_levels.items() if l == level]
            if nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes,
                                       node_color=cores[level],
                                       node_size=tamanhos[level],
                                       alpha=alphas[level],
                                       edgecolors='white',
                                       linewidths=3,
                                       label=labels_dict[level],
                                       ax=ax)
    else:
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        node_sizes = [200 + (degrees.get(node, 1) / max_degree) * 600 for node in G.nodes()]

        nx.draw_networkx_nodes(G, pos,
                               node_color='#6bcfff',
                               node_size=node_sizes,
                               alpha=0.9,
                               edgecolors='white',
                               linewidths=2,
                               ax=ax)

    nx.draw_networkx_edges(G, pos,
                           alpha=0.7,
                           edge_color='#95a5a6',
                           width=2.5,
                           ax=ax)

    plt.title(f'{Path(gexf_file).stem}\n{G.number_of_nodes():,} nós | {G.number_of_edges():,} arestas',
              fontsize=26, fontweight='bold', pad=20, color='white')

    if node_levels:
        legend = plt.legend(scatterpoints=1, fontsize=20, loc='upper right',
                            framealpha=0.95, edgecolor='white', facecolor='#2c2c2c')
        for text in legend.get_texts():
            text.set_color('white')

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor='#1a1a1a')
    plt.close()

    print(f"✓ {output_file}")


def processar_todos():
    """
    Processa todos os arquivos GEXF encontrados (apenas tema escuro)
    """
    handle_bsky = input("Digite o handle do usuário: ")

    gexf_dir = Path(f'../data/graph/{handle_bsky}/GEXF')
    png_dir = Path(f'../data/graph/{handle_bsky}/PNG')

    if not gexf_dir.exists():
        print(f"❌ Diretório não encontrado: {gexf_dir}")
        return

    png_dir.mkdir(parents=True, exist_ok=True)

    gexf_files = list(gexf_dir.rglob('*[0-9].gexf'))

    if not gexf_files:
        print(f"❌ Nenhum arquivo GEXF encontrado em {gexf_dir}")
        return

    print(f"\n📊 Encontrados {len(gexf_files)} arquivos GEXF")
    print("🎨 Gerando gráficos com tema escuro (fundo preto)")
    print("=" * 60)

    for i, gexf_file in enumerate(gexf_files, 1):
        print(f"\n[{i}/{len(gexf_files)}] Processando: {gexf_file.name}")
        output_file = str(png_dir / gexf_file.stem) + '.png'
        gexf_to_png_dark(str(gexf_file), output_file)

    print("\n" + "=" * 60)
    print("✅ Processamento concluído!")


if __name__ == "__main__":
    processar_todos()