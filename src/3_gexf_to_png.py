import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def list_core_user_folders(base_dir):
    """
    Lista pastas core_user_N existentes dentro do diretorio base.

    [Ordem de acoes]:
    1. Verifica se o diretorio base existe.
    2. Filtra subpastas iniciadas por core_user_.
    3. Retorna a lista ordenada.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    folders = [p for p in base_path.iterdir() if p.is_dir() and p.name.startswith("core_user_")]
    return sorted(folders, key=lambda p: p.name)


def choose_from_list(items, prompt):
    """
    Solicita ao usuario escolher um item da lista pelo indice.

    [Ordem de acoes]:
    1. Exibe os itens numerados.
    2. Le e valida a escolha do usuario.
    3. Retorna o item selecionado.
    """
    if not items:
        return None

    for idx, item in enumerate(items, start=1):
        print(f"{idx}. {item}")

    while True:
        choice = input(prompt).strip()
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= len(items):
                return items[num - 1]
        print("Escolha invalida. Tente novamente.")


def gexf_to_png_dark(gexf_file, output_file=None):
    """
    Converte arquivo GEXF em PNG com tema escuro.

    [Ordem de acoes]:
    1. Carrega o grafo GEXF e valida conteudo.
    2. Calcula layout e desenha nos/arestas.
    3. Salva a figura em PNG.
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
    Processa todos os GEXF do core_user escolhido e gera PNGs.

    [Ordem de acoes]:
    1. Lista e seleciona a pasta core_user.
    2. Localiza arquivos GEXF e cria pasta PNG.
    3. Converte cada GEXF para PNG.
    """
    print("Escolha a pasta do core user:")

    core_user_folders = list_core_user_folders(Path("..") / "data" / "graph")
    if not core_user_folders:
        print("❌ Nenhuma pasta core_user encontrada em ../data/graph")
        return

    selected_core_folder = choose_from_list(core_user_folders, "Digite o numero da pasta desejada: ")
    if not selected_core_folder:
        print("❌ Nenhuma pasta selecionada.")
        return

    gexf_dir = Path(selected_core_folder) / "GEXF"
    png_dir = Path(selected_core_folder) / "PNG"

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