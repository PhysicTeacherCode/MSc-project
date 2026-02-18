# Replicating *The Statistical Mechanics of Twitter Communities* Using the Bluesky Social Network
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-000?style=for-the-badge&logo=postgresql)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-4CB391?style=for-the-badge&logo=Seaborn&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-000?style=for-the-badge&logo=python&logoColor=white)

Este repositório é uma tentativa de reproduzir — e adaptar — os resultados do artigo:
> **Gavin Hall and William Bialek (2019)** — *The Statistical Mechanics of Twitter Communities*
> DOI: **10.1088/1742-5468/ab3af0**

O objetivo principal é investigar se as mesmas propriedades estatísticas observadas em comunidades do Twitter também emergem na rede social **Bluesky**.

---
## 📌 Motivação
O artigo original utiliza ferramentas de mecânica estatística para analisar interações entre usuários, descobrindo padrões como:
* Formação de comunidades
* Correlações entre usuários
* Estrutura de rede
* Distribuição de influência e conectividade
Com o crescimento do Bluesky e sua API aberta, este projeto explora se fenômenos similares emergem em uma plataforma mais descentralizada — ou se novas dinâmicas comportamentais surgem.
---
## 🎯 Objetivos do Projeto
* Coletar dados públicos de interação do Bluesky.
* Construir grafos de rede (seguidores, posts e interações).
* Aplicar métodos inspirados no artigo, tais como:
  * Matrizes de correlação entre pares de usuários
  * Modelos de Máxima Entropia para interações sociais
* Investigar como diferenças estruturais entre comunidades do Twitter e Bluesky afetam os resultados estatísticos.
---
## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **API:** Bluesky API
* **Principais bibliotecas:**
  * `pandas` — Manipulação e análise de dados
  * `matplotlib` / `seaborn` — Visualização de dados
  * `networkx` — Análise de redes e grafos
  * `requests` — Requisições HTTP
  * `asyncio` — Processamento assíncrono
* **Banco de Dados:** PostgreSQL
* **Ambiente:** Jupyter Notebooks para workflows de análise.
---
## 📊 Dados
Os dados são coletados diretamente da API pública do Bluesky. Dependendo das políticas de dados do Bluesky, *dados brutos podem não ser incluídos* e podem ser reconstruídos via scripts fornecidos.
---
## 📁 Estrutura do Repositório
```
MSc-project/
├── README.md                                          # Documentação principal
├── data/
│   ├── graph/                                        # Arquivos de rede (GEXF, PNG)
│   │   ├── aussieopinion.bsky.social/
│   │   ├── freebirdthirteen.bsky.social/
│   │   ├── garyrbs.bsky.social/
│   │   ├── jelle8591.bsky.social/
│   │   ├── lexicodex.bsky.social/
│   │   ├── msevelyn.bsky.social/
│   │   └── randall.gobirds.online/
│   └── posts/                                        # Dados de posts coletados
│       └── lexicodex.bsky.social_(55)/
└── src/
    ├── 1_core_users_followers.py                     # Coleta de usuários core e followers
    ├── 2_get_posts_from_gexf.py                      # Extração de posts via API
    ├── 3_gexf_to_png.py                              # Conversão de grafos para visualização
    ├── 4_posts_to_database.py                        # Armazenamento em banco de dados
    ├── 5_database_analysis.sql                       # Análises SQL
    ├── 6_database_to_df.py                           # Extração de dados para análise
    └── 7_figure_B1.py                                # Geração de figuras
```
---
## 📚 Conteúdo dos Scripts
A análise está dividida em sete etapas principais:
1. **Coleta de Usuários Core (`1_core_users_followers.py`)**: Extrai usuários principais e seus seguidores.
2. **Coleta de Posts (`2_get_posts_from_gexf.py`)**: Obtém posts via API Bluesky usando dados dos grafos.
3. **Visualização de Redes (`3_gexf_to_png.py`)**: Converte arquivos GEXF em imagens PNG com temas claros/escuros.
4. **Armazenamento em BD (`4_posts_to_database.py`)**: Salva posts em banco de dados PostgreSQL.
5. **Análises SQL (`5_database_analysis.sql`)**: Queries para análise estatística dos dados.
6. **Extração para DataFrame (`6_database_to_df.py`)**: Exporta dados do BD para análise em Python.
7. **Geração de Figuras (`7_figure_B1.py`)**: Cria visualizações e comparações estatísticas.
---
## 📊 Resultados Principais
*Análises em progresso. Resultados serão adicionados conforme as etapas forem concluídas.*
---
## 🚧 Status do Projeto
> **Em progresso.**
> Coleta de dados, construção de grafos e análises iniciais estão em andamento.
---
## 🤝 Contribuindo
Contribuições são bem-vindas! Sugestões, melhorias, métodos alternativos ou discussões sobre o artigo original são especialmente encorajadas.
---
## 👤 Autor
**Diego de Lima Fernandes**
- LinkedIn: [linkedin.com/in/diegulus](https://www.linkedin.com/in/diegulus/)
- GitHub: [@PhysicTeacherCode](https://github.com/PhysicTeacherCode)
- Email: diego196095@gmail.com
---
## 📄 License
*Em progresso*
---
## 📚 Referência Principal
Gavin Hall and William Bialek. (2019).
**The statistical mechanics of Twitter communities**.
*Journal of Statistical Mechanics: Theory and Experiment*.
DOI: 10.1088/1742-5468/ab3af0
---
## ✨ Nota Final
Este projeto visa conectar física estatística e ciência de dados aplicadas a redes sociais descentralizadas modernas. Se você está interessado em redes complexas, mecânica estatística ou plataformas sociais como Bluesky, este repositório pode ser útil para você.
