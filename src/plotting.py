import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def plot_figure_b1(df, output_dir="data/plots", filename="figure_B1.png"):
    """
    Gera o gráfico Figure B1 (Probability Density vs Standard Deviation in Time).
    Estilo: publicação científica (inspirado em Hall & Bialek, 2019).
    """
    if df.empty or 'desvio_padrao' not in df.columns:
        print("[Plot] DataFrame vazio ou sem a coluna 'desvio_padrao'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Filtro de dados
    data = df['desvio_padrao'].dropna()
    data = data[data > 0]
    
    if data.empty:
        print("[Plot] Sem dados válidos para plotar.")
        return

    # Bins logarítmicos com centros geométricos (1000 bins)
    log_bins = np.logspace(-1, 4, 500)
    counts, bin_edges = np.histogram(data, bins=log_bins, density=False)
    
    # Para obter as linhas horizontais de C=1, 2, 3 (como na sua imagem de referência), 
    # não podemos dividir pela largura do bin (Delta x), pois Delta x cresce com x
    # e faria o C=1 virar uma linha diagonal caindo de acordo com 1/x. 
    # Calculamos apenas a probabilidade (Frequência Relativa) ou a Densidade em log-space.
    probabilities = counts / counts.sum()
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    # ── Estilo Científico ────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',          # Computer Modern (padrão LaTeX)
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'axes.linewidth': 1.0,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Scatter com pontos sólidos pequenos (estilo paper)
    valid = probabilities > 0
    ax.scatter(bin_centers[valid], probabilities[valid], 
               s=8, color='black', edgecolors='none', zorder=3)
    
    # Escalas log-log
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Ticks em todas as potências de 10
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))

    # Limites estáticos conforme imagem de referência
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-6, 1e0)

    # Labels
    ax.set_xlabel('Standard Deviation in Time')
    ax.set_ylabel('Probability Density')

    # Sem título (padrão em artigos — o caption fica na legenda da figura)
    # Grid discreto apenas nas major ticks
    ax.grid(True, which='major', linewidth=0.4, alpha=0.4, color='gray')
    ax.grid(True, which='minor', linewidth=0.2, alpha=0.2, color='gray')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Restaura defaults para não contaminar outros plots
    plt.rcdefaults()
    
    print(f"[Plot] Gráfico B1 salvo em: {output_path}")
