import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_figure_b1(df, total_users=None, output_dir="data/plots", filename="figure_B1.png",
                   max_time_std_days=None, n_bins=None, bins_per_decade=38):
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
    if max_time_std_days is not None:
        max_val = max(max_val, float(max_time_std_days))
    if max_val <= min_val:
        max_val = min_val * 10.0

    x_min = 10 ** np.floor(np.log10(min_val))
    x_max = 10 ** np.ceil(np.log10(max_val))
    if x_max <= x_min:
        x_max = x_min * 10.0

    if n_bins is None:
        n_decades = np.log10(x_max) - np.log10(x_min)
        n_bins = int(np.clip(np.ceil(n_decades * bins_per_decade), 45, 260))

    log_bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)
    counts, bin_edges = np.histogram(data, bins=log_bins, density=False)
    bin_widths = np.diff(bin_edges)
    density = counts / (len(data) * bin_widths)
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size': 14,
        'axes.labelsize': 24,
        'axes.titlesize': 18,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'axes.linewidth': 1.4,
        'xtick.major.width': 1.4,
        'ytick.major.width': 1.4,
        'xtick.minor.width': 0.9,
        'ytick.minor.width': 0.9,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    valid = density > 0
    ax.scatter(
        bin_centers[valid],
        density[valid],
        s=18,
        color='black',
        edgecolors='none',
        alpha=0.95,
        zorder=3,
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min, x_max)

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=20))
    ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

    if np.any(valid):
        y_min = 10 ** np.floor(np.log10(density[valid].min() * 0.8))
        y_max = 10 ** np.ceil(np.log10(density[valid].max() * 1.2))
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

    if max_time_std_days is not None:
        ax.axvline(max_time_std_days, color='#d62728', linestyle='--', linewidth=1.4, zorder=2)

    ax.set_xlabel('Standard deviation in time (days)')
    ax.set_ylabel('Probability density')

    ax.grid(True, which='major', linewidth=0.55, alpha=0.38, color='gray')
    ax.grid(True, which='minor', linewidth=0.35, alpha=0.18, color='gray')

    ax.tick_params(axis='both', which='both', top=True, right=True, direction='in')
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.98)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    plt.rcdefaults()
    print(f"[Plot] Figure B1 (densidade do desvio temporal, n={len(data)}) salva em: {output_path}")
