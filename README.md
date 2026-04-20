# Bluesky Community Analysis

Este projeto implementa a coleta, modelagem e detecção de comunidades da rede social Bluesky, utilizando requisições assíncronas à API pública (sem autenticação), modelagem com a biblioteca `networkx` e detecção de comunidades através do algoritmo Leiden (`cdlib` + `leidenalg`).

## Requisitos

- Python 3.9+
- Bibliotecas do `requirements.txt`

## Instalação

Instale as dependências:

```bash
pip install -r requirements.txt
```

## Uso

Execute o arquivo principal para iniciar a sequência de scripts com o usuário inicial da coleta ("core_user"):

```bash
python main.py
```

O script automaticamente testará um "Rate Limit Seguro", iniciará a coleta em camadas, construirá um grafo, rodará a clusterização e fornecerá relatórios na pasta `data/`.
