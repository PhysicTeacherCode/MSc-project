# Prompt: Detecção, Análise de Comunidades e Física Estatística no Bluesky

Atue como um Engenheiro de Dados e Especialista em Redes Complexas e Física Estatística. Escreva um pipeline em Python que faz coleta, modelagem, detecção de comunidades, análise estatística de posts e aplicação do Modelo de Ising em dados do Bluesky a partir de um único usuário semente (`core_user`).

Não utilize o SDK `atproto`. Toda a coleta deve ser feita via requisições HTTP assíncronas na API pública do Bluesky.

---

## Bibliotecas Exigidas

- `aiohttp`, `asyncio`: Requisições assíncronas
- `networkx`: Manipulação de grafos, exportação GEXF
- `cdlib`, `leidenalg`: Algoritmo de detecção de comunidades Leiden (backend C++)
- `pandas`, `numpy`: Manipulação e análise de dados
- `matplotlib`: Visualizações
- `scipy.optimize`: Inferência de parâmetros do Modelo de Ising (PLMLE)

---

## 1. Calibração de Rate Limit (Antes de Qualquer Coleta)

- Disparar requisições concorrentes à `https://public.api.bsky.app` até receber HTTP 429.
- Calcular um limite seguro de concorrência e usar `asyncio.Semaphore` para regular toda a coleta.
- Exibir progresso do teste em `\r` (linha única).

---

## 2. Coleta de Rede — 3 Passagens BFS com Filtro de Celebridades

**Endpoint de followers:** `GET https://public.api.bsky.app/xrpc/app.bsky.graph.getFollowers?actor={handle}`  
**Endpoint de follows:** `GET https://public.api.bsky.app/xrpc/app.bsky.graph.getFollows?actor={handle}`  
**Endpoint de perfil (batch):** `GET https://public.api.bsky.app/xrpc/app.bsky.actor.getProfiles?actors[]={h1}&actors[]={h2}...` (máx 25 por chamada)

### Filtro de Celebridades (antes de montar arestas)
- Perguntar ao usuário o limite máximo de seguidores por usuário (padrão: 5000).
- Para cada lote de handles coletados, consultar `getProfiles` em **batch de 25 em paralelo** via `asyncio.gather` para obter `followersCount`.
- Descartar handles com `followersCount > max_followers`.

### Passagem 1 — Followers de 1ª Ordem
- Coletar seguidores do `core_user` com paginação via `cursor`.
- Limit total por requisição: `limit_total=3000`.
- Aplicar filtro de celebridades. Adicionar arestas `(follower → core_user)`.

### Passagem 2 — Followers de 2ª Ordem
- Para cada usuário de 1ª ordem, coletar seus seguidores (`asyncio.gather` com semáforo).
- Filtrar celebridades em **batch único** após agregar todos os handles únicos.
- Adicionar arestas `(2ª_ordem → 1ª_ordem)`.

### Passagem 3 — Cross-connections (Enriquecimento de Arestas)
- Para cada usuário de 2ª ordem, buscar quem ele **segue** (até 5 páginas × 100 = 500 handles), via `getFollows`.
- Disparar todos em paralelo com `asyncio.gather`.
- Para cada handle seguido que já existe no conjunto de nós do grafo, adicionar aresta cruzada `(2ª_ordem → nó_conhecido)`.
- Isso densifica o grafo capturando conexões recíprocas e internas da comunidade.

### Tratamento de Erros
- HTTP 429: backoff exponencial (`backoff *= 2`)
- HTTP 400: log do handle inválido, continuar execução
- HTTP != 200: retornar lista parcial silenciosamente

---

## 3. Modelagem do Grafo

- `nx.Graph()` não-dirigido. Nós = handles; Arestas = relações de seguimento.
- Remover self-loops: `remove_edges_from(nx.selfloop_edges(G))`
- Aplicar K-Core global com `k=2` para remover nós periféricos.

---

## 4. Detecção de Comunidades Leiden Multi-Resolução

- Resoluções fixas: `[1.0, 1.5, 2.0, 2.5, 3.0]`
- Algoritmo: `cdlib.algorithms.rb_pots(graph, resolution=res)`
- Exibir resumo por resolução: nº comunidades, modularity, maior e menor.
- Usuário escolhe resolução; solicitar IDs de subcomunidades para exportar.

---

## 5. Exportação e Relatórios

### Arquivos Gerados por Sessão (Anonimizados com `session_id = datetime.now().strftime(...)`)
- GEXF global: `rede_{session_id}.gexf`
- GEXF subcomunidade: `comunidade_{idx}_{core_user}.gexf` (numeração incremental, não sobrescreve)
- Relatório global: `relatorio_geral_rede.txt` (modo append)
- Relatório subcomunidade: por cada ID exportado

### Conteúdo do Relatório
- Data/hora, `core_user`, nós, arestas, densidade, grau médio, modularidade
- Top 5 nós por grau

---

## 6. Visualização de Rede (Estilo Hall & Bialek 2019)

- **Fundo branco**, nós **vermelhos uniformes**, arestas **pretas finas**
- Layout **Fruchterman-Reingold** (`nx.spring_layout`) com `k=0.05`, `iterations=200`, `seed=42`
  - `k` pequeno gera núcleo denso ao centro com periféricos estendidos, exatamente como na figura de referência
- Se < 1.000 nós: grafo completo; se ≥ 1.000 nós: top 1.000 hubs por grau
- Tamanho da figura: entre 6×6 e 10×10 polegadas (300 dpi)
- Saída: `data/processed/png/sessao_{session_id}/`

---

## 7. Coleta de Posts — Extração de Palavras On-the-Fly (Memory-Optimized)

**Endpoint:** `GET https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed?actor={did}&filter=posts_no_replies`

- Selecionar arquivo GEXF interativamente no terminal.
- Para cada usuário do GEXF, coletar até **500 posts** com paginação.
- **Deduplicação e Compactação Extrema**:
  - `sys.intern(did)` e `sys.intern(word)` para economizar RAM com strings repetidas.
  - Timestamps salvos em `array.array('d')` como **dias desde o lançamento do Bluesky** (17/02/2023). Isso garante que o tempo de um post seja fixo e reprodutível.
- **Extrair palavras durante a coleta** (não armazenar texto raw):
  - Regex: `\b\w{2,}\b` em minúsculas
  - Timestamp → idade (dias pós-lançamento): `(createdAt - 2023-02-17).total_seconds() / 86400`
  - Acumular `{palavra: array.array('d')}` por usuário
- **Gerenciamento de Tarefas**:
  - Evitar agendamento massivo imediato; processar usuários em lotes controlados para manter estabilidade da memória.
- Retornar:
  - `global_word_times`: `{word: array.array('d')}`
  - `user_word_sets`: `{did: {interned_words}}`
  - `all_users`: lista completa de DIDs internados
- Usar `aiohttp.TCPConnector(ttl_dns_cache=300)` para performance.
- HTTP 429: aguardar 60s com backoff; exibir aviso no terminal.

---

## 8. Análise Estatística Temporal (Figure B1)

- Para cada palavra em `global_word_times`:
  - Ordenar as idades cronologicamente
  - Calcular `np.diff` dos intervalos
  - Calcular `np.std` dos intervalos (desvio padrão)
  - Mínimo de 3 ocorrências para inclusão
- Retornar DataFrame: `[word, occurrences, desvio_padrao]`
- Gerar **Figure B1**: gráfico PDF (Densidade de Probabilidade × Desvio Padrão) em escala log-log
  - 1000 bins, limites cravados: X de 1e-1 a 1e4, Y de 1e-6 a 1e0.
  - Estilo visual de scatter com dots finos, fonte 'sans-serif' ampliada, sem grid.

---

## 9. Filtro Interativo de Keywords (Opção 2)

- Exibir filtros interativos: `desvio_padrao` mínimo/máximo e frequência mínima.
- Salvar keywords filtradas em `keywords_filtradas.csv`.
- Esta etapa é focada na descoberta do modelo (observáveis estatísticos relevantes).

---

## 10. Aplicação do Modelo de Ising em Subcomunidades (Opção 3)

### Fluxo de Aplicação Cruzada
- O sistema permite aplicar um modelo (keywords) criado em uma comunidade a qualquer outra subcomunidade do mesmo `core_user`.
- **Passo 1**: Selecionar `keywords_filtradas.csv` interativamente.
- **Passo 2**: Selecionar `GEXF` da subcomunidade alvo.
- **Passo 3**: Coleta automática de posts para os nós da comunidade alvo (Seção 7).

### Geração da Matriz de Ising (+1/-1)
- **Otimizações**:
  - `dtype=np.int8` para a matriz (reduz 8x o uso de RAM).
  - Uso de interseção de sets (`used_words.intersection(keywords_set)`) para gerar a matriz — velocidade exponencial em relação a loops duplos.
- Salvar em `matriz_ising_{comm_name}.csv`.
- Prosseguir para inferência e Figura 2.

---

## 11. Inferência do Modelo de Ising com ConIII (Seções 10 e 11)

Implementar inferência do Modelo de Ising usando exclusivamente o pacote `coniii` (verificar compatibilidade: `scipy < 1.10` e `pip install --no-build-isolation`), focando no método de alta precisão:
- **MCH (Monte Carlo Histogram)**: Também conhecido como Boltzmann Machine Learning. Utiliza amostragem Metropolis para ajustar os parâmetros $h$ e $J$. É considerado o "Gold Standard" de precisão para inferência em modelos de máxima entropia.
Validar o ajuste calculando o RMSE das médias estimadas ($rmse < 3/\sqrt{R}$) segundo Schneidman et al. (2006).

Ofertado por meio de um script `ising_coniii.py` que orquestra a execução. Os resultados são avaliados qualitativamente e os multiplicadores finais são salvos em `.npy`.

Geração da **Figura 2** (Covariância empírica e Acoplamentos J inferidos) com filtro topológico da rede GEXF onde `A_ij == 0` se traduz para `np.nan` (renderizado como branco). Validação opcional por amostragem Metropolis.

