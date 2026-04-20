import os
from datetime import datetime
from .modeling import get_network_metrics, get_influential_nodes

def get_next_available_index(processed_dir, reports_dir):
    """
    Busca o primeiro número inteiro (começando de 1) que não esteja em uso
    nem como .gexf na pasta processed, nem como .txt na pasta de relatórios.
    """
    idx = 1
    while True:
        # Note: We now check for 'comunidade_{idx}_sessao_*.gexf' pattern safely or just the index
        # To simplify, we'll keep checking the index in the reports folder which is already session-isolated
        txt_exists = os.path.exists(os.path.join(reports_dir, f"relatorio_comunidade_{idx}.txt"))
        
        if not txt_exists:
            return idx
        idx += 1

def generate_global_report(G, num_communities, modularity_score, output_dir="data/reports", selected_indices=None, core_user="N/A"):
    """
    Gera o relatório geral da rede extraindo informações globais e 
    registra (append) no log local relatorio_geral_rede.txt.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = get_network_metrics(G)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = (
        f"\n{'='*50}\n"
        f"ANÁLISE EM: {timestamp}\n"
        f"USUÁRIO SEMENTE: {core_user}\n"
        f"{'-'*50}\n"
        f"Quantidade total de nós: {stats['num_nodes']}\n"
        f"Quantidade total de arestas: {stats['num_edges']}\n"
        f"Densidade do grafo: {stats['density']:.6f}\n"
        f"Grau médio dos nós: {stats['avg_degree']:.2f}\n"
        f"Número total de comunidades detectadas: {num_communities}\n"
        f"Score final de modularidade (Leiden): {modularity_score:.4f}\n"
    )

    if selected_indices:
        indices_str = ", ".join(map(str, sorted(selected_indices)))
        report_content += f"Subcomunidades exportadas (índices): {indices_str}\n"
    
    file_path = os.path.join(output_dir, "relatorio_geral_rede.txt")
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(report_content)
        
    print(f"[Relatório] Gerado {file_path}")

def generate_subcommunity_report(subgraph, display_id, original_cid, output_dir="data/reports"):
    """
    Gera as métricas de topologia específicas do subgrafo de uma comunidade.
    Utiliza o 'display_id' para o nome do arquivo (numeração incremental) e 
    registra o 'original_cid' para referência interna.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    stats = get_network_metrics(subgraph)
    # Buscando os 5 nós mais influentes na comunidade local
    top_5 = get_influential_nodes(subgraph, top_n=5)
    
    report_content = (
        f"=== RELATÓRIO DA SUBCOMUNIDADE {display_id} (ID Original: {original_cid}) ===\n"
        f"Quantidade de nós internos: {stats['num_nodes']}\n"
        f"Quantidade de arestas internas: {stats['num_edges']}\n"
        f"Densidade da subcomunidade: {stats['density']:.6f}\n"
        f"Grau médio interno: {stats['avg_degree']:.2f}\n\n"
        f"Top 5 nós com maior grau (influentes):\n"
    )
    
    for i, node in enumerate(top_5, 1):
        report_content += f"{i}. {node}\n"
        
    file_path = os.path.join(output_dir, f"relatorio_comunidade_{display_id}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report_content)
        
    print(f"[Relatório] Gerado {file_path}")
