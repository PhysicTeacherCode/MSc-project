# Bluesky Community Analysis

Pipeline em Python para coletar redes de seguidores no Bluesky, construir grafos,
detectar comunidades e gerar artefatos para análise estatística e modelos de
máxima entropia/Ising.

O projeto usa a API pública do Bluesky (`public.api.bsky.app`), sem autenticação,
com requisições assíncronas via `aiohttp`. A rede é modelada com `networkx` e a
partição de comunidades é feita com `cdlib`, usando Potts/RB com varredura de
resoluções e refinamento local por k-core.

## Requisitos

- Python 3.10 recomendado. O código usa anotações como `str | None`, que não
  funcionam em Python 3.9 sem ajustes.
- Dependências principais em `requirements.txt`.
- Dependências do modelo de Ising/ConIII em `requirements-ising.txt`, quando a
  opção 3 do menu for usada.

## Instalação

Para coleta, construção de grafos, detecção de comunidades e análise de posts:

```bash
python -m pip install -r requirements.txt
```

Para executar o fluxo de Ising com ConIII, use um ambiente Python 3.10 separado.
No WSL/EC2, o ambiente recomendado é:

```bash
conda create -n test -c conda-forge python=3.10 numpy scipy pandas matplotlib networkx aiohttp numba pip -y
conda activate test
python -m pip install --no-build-isolation -r requirements-ising.txt
```

O ConIII 3.0.1 é antigo e tenta usar aliases removidos do SciPy moderno, como
`scipy.exp`. O módulo `src/ising_coniii.py` aplica um patch antes de importar o
ConIII. Se for testar `coniii` diretamente fora do projeto, use o patch
`sitecustomize.py` descrito em `AWS_EC2_RUNBOOK.md`.

## Uso

Execute o menu principal:

```bash
python main.py
```

Opções do menu:

- `1` - Nova coleta e análise de comunidades a partir de um handle ou DID.
- `2` - Análise estatística de posts usando um arquivo GEXF já gerado.
- `3` - Aplicação do modelo de máxima entropia/Ising usando keywords filtradas.
- `4` - Sair.

Na coleta, o script calibra um limite seguro de concorrência, coleta seguidores
em camadas, aplica filtros de celebridades, recência e atividade, constrói o
grafo, remove autoarestas, aplica k-core e calcula comunidades em múltiplas
resoluções.

## Artefatos gerados

Os resultados são salvos em `data/`, principalmente:

- `data/processed/gexf/` - grafos globais e subcomunidades em GEXF.
- `data/processed/reports/sessao_*` - relatórios textuais.
- `data/processed/png/sessao_*` - visualizações de redes.
- `data/plots/sessao_*` - CSVs de keywords, matrizes de Ising, figuras e
  comparativos do ConIII.

Esses arquivos são produtos de execução local e podem crescer rapidamente.

## Observações

- A API pública do Bluesky pode retornar HTTP 429 em execuções muito
  concorrentes; o script tenta calibrar e respeitar um limite seguro.
- A opção 3 depende do ConIII e tende a ser a parte mais sensível do ambiente.
  Se a instalação falhar em Python recente, recrie o ambiente com Python 3.10.
- O pacote `optuna` não é usado atualmente pelo código principal e não está nas
  dependências padrão.
