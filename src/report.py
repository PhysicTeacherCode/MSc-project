import os
import re
from datetime import datetime
from .modeling import get_network_metrics, get_influential_nodes


def format_community_id(idx) -> str:
    return f"{int(idx):03d}"


def _extract_community_index(name: str) -> int | None:
    match = re.search(r"(?:^|_)comunidade_(\d+)(?:_|\.|$)", name)
    if not match:
        match = re.search(r"^relatorio_comunidade_(\d+)\.txt$", name)
    if not match:
        return None
    return int(match.group(1))


def _scan_community_indices(*roots) -> set[int]:
    indices = set()
    for root in roots:
        if not root or not os.path.exists(root):
            continue
        if os.path.isfile(root):
            idx = _extract_community_index(os.path.basename(root))
            if idx is not None:
                indices.add(idx)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for name in dirnames:
                idx = _extract_community_index(name)
                if idx is not None:
                    indices.add(idx)
            for name in filenames:
                idx = _extract_community_index(name)
                if idx is not None:
                    indices.add(idx)
    return indices


def get_next_available_index(processed_dir, reports_dir):
    """
    Retorna o próximo ID global de subcomunidade em três dígitos.

    A contagem não é reiniciada por sessão ou core user: a função olha os GEXF
    exportados, relatórios e pastas de plots existentes e continua do maior ID.
    """
    processed_root = processed_dir
    if os.path.basename(os.path.normpath(processed_dir)) == "gexf":
        processed_root = os.path.dirname(os.path.normpath(processed_dir))

    reports_root = reports_dir
    if os.path.basename(os.path.normpath(reports_dir)).startswith("sessao_"):
        reports_root = os.path.dirname(os.path.normpath(reports_dir))

    data_root = os.path.dirname(processed_root)
    plots_root = os.path.join(data_root, "plots")

    used = _scan_community_indices(processed_root, reports_root, plots_root)
    next_idx = (max(used) + 1) if used else 1
    return format_community_id(next_idx)

def generate_global_report(
    G,
    num_communities,
    modularity_score,
    output_dir="data/reports",
    selected_indices=None,
    core_user="N/A",
    core_user_id=None,
):
    """
    Gera o relatório geral da rede extraindo informações globais e 
    registra o relatório global no arquivo core_user_###.txt.
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
    
    report_name = f"core_user_{format_community_id(core_user_id)}.txt" if core_user_id else "core_user_000.txt"
    file_path = os.path.join(output_dir, report_name)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report_content)
        
    print(f"[Relatório] Gerado {file_path}")

def generate_subcommunity_report(
    subgraph,
    display_id,
    original_cid,
    output_dir="data/reports",
    k_core=None,
    k_core_score=None,
    core_user=None,
    verbose=True
):
    """
    Gera as métricas de topologia específicas do subgrafo de uma comunidade.
    Utiliza o 'display_id' para o nome do arquivo (numeração incremental) e 
    registra o 'original_cid' para referência interna.
    """
    os.makedirs(output_dir, exist_ok=True)
    display_id = format_community_id(display_id)
    
    stats = get_network_metrics(subgraph)
    # Buscando os 5 nós mais influentes na comunidade local
    top_5 = get_influential_nodes(subgraph, top_n=5)
    k_core_text = "sem k-core local" if not k_core else f"k={k_core}"
    score_text = "N/A" if k_core_score is None else f"{k_core_score:.6f}"
    
    report_content = (
        f"=== RELATÓRIO DA SUBCOMUNIDADE {display_id} (ID Original: {original_cid}) ===\n"
        f"USUÁRIO SEMENTE: {core_user or 'N/A'}\n"
        f"K-core aplicado na exportação: {k_core_text}\n"
        f"Critério k-core (grau médio/nós): {score_text}\n"
        f"Quantidade de nós internos: {stats['num_nodes']}\n"
        f"Quantidade de arestas internas: {stats['num_edges']}\n"
        f"Densidade da subcomunidade: {stats['density']:.6f}\n"
        f"Grau médio interno: {stats['avg_degree']:.2f}\n\n"
        f"Top 5 nós com maior grau (influentes):\n"
    )
    
    for i, node in enumerate(top_5, 1):
        report_content += f"{i}. {node}\n"
        
    file_path = os.path.join(output_dir, f"comunidade_{display_id}.txt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(report_content)
        
    if verbose:
        print(f"[Relatório] Gerado {file_path}")
