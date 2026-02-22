import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../data/posts/freebirdthirteen.bsky.social_(1047)/words_data/freebirdthirteen.bsky.social_palavras_desvio.csv')
print(df.head())

# Calcular densidade com histograma
counts, bin_edges = np.histogram(df['desvio_padrao'], bins=1000, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter(bin_centers, counts, s=10, color='black')
plt.ylim(1e-5, 1e0)
plt.xlim(1e-1, 1e4)
plt.axvline(x=120, color='r', linestyle='--', linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Standard Deviation in Time')
plt.ylabel('Probability Density')
plt.title('Probability Density vs Standard Deviation in Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()