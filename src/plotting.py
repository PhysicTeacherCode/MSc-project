import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_figure_b1(df, total_users=None, output_dir="data/plots", filename="figure_B1.png",
                   min_occurrences=None, max_time_std_days=None):
    """
    Figure B1 inspired by Hall & Bialek.

    Plots the probability density of temporal standard deviations in word usage.
    Candidate keywords should be frequent enough and localized in time.
    """
    required = {'occurrences', 'time_std_days'}
    if df.empty or not required.issubset(df.columns):
        print("[Plot] DataFrame vazio ou sem as colunas 'occurrences' e 'time_std_days'.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    data = df['time_std_days'].dropna()
    data = data[data > 0]
    if data.empty:
        print("[Plot] Sem dados validos para plotar.")
        return

    min_val = max(float(data.min()), 1e-3)
    max_val = float(data.max())
    if max_val <= min_val:
        max_val = min_val * 10.0

    log_bins = np.logspace(np.log10(min_val * 0.8), np.log10(max_val * 1.2), 80)
    counts, bin_edges = np.histogram(data, bins=log_bins, density=False)
    bin_widths = np.diff(bin_edges)
    density = counts / (counts.sum() * bin_widths)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
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
    valid = density > 0
    ax.scatter(bin_centers[valid], density[valid], s=10, color='black', edgecolors='none', zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))

    if max_time_std_days is not None:
        ax.axvline(max_time_std_days, color='tab:red', linestyle='--', linewidth=1.0)

    ax.set_xlabel('Standard deviation in time (days)')
    ax.set_ylabel('Probability density')

    ax.grid(True, which='major', linewidth=0.4, alpha=0.4, color='gray')
    ax.grid(True, which='minor', linewidth=0.2, alpha=0.2, color='gray')

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    plt.rcdefaults()
    print(f"[Plot] Figure B1 (densidade do desvio temporal) salva em: {output_path}")
