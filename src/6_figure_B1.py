import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Encontrar todos os arquivos *palavras_desvio.csv
base_dir = Path("..") / "data" / "posts"
palavras_files = list(base_dir.glob("core_user*/*palavras_desvio.csv"))

if not palavras_files:
    raise SystemExit("Nenhum arquivo *palavras_desvio.csv encontrado em ../data/posts")

# Exibir opções
print("Arquivos encontrados:")
for idx, file in enumerate(palavras_files, start=1):
    print(f"{idx}. {file}")

# Permitir que o usuário escolha um arquivo
while True:
    choice = input(f"\nEscolha um arquivo (1-{len(palavras_files)}): ").strip()
    if choice.isdigit():
        num = int(choice)
        if 1 <= num <= len(palavras_files):
            selected_file = palavras_files[num - 1]
            break
    print("Escolha inválida. Tente novamente.")

print(f"\nProcessando: {selected_file}")

df = pd.read_csv(selected_file)
print(df.head())

# Calcular densidade com histograma
counts, bin_edges = np.histogram(df['desvio_padrao'], bins=1000, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter(bin_centers, counts, s=10, color='black')
plt.ylim(1e-5, 1e0)
plt.xlim(1e-1, 1e4)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Standard Deviation in Time')
plt.ylabel('Probability Density')
plt.title('Probability Density vs Standard Deviation in Time')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Salvar na mesma pasta do arquivo CSV selecionado
output_dir = selected_file.parent
output_path = output_dir / f"{selected_file.stem}_figure_B1.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nGráfico salvo em: {output_path}")

plt.show()

