import asyncio
import glob
import os
import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime

from src.rate_limit import calibrate_rate_limit
from src.collection import collect_network
from src.modeling import build_graph
from src.community import (
    detect_communities_multi_resolution,
    apply_partition,
    extract_subcommunity_graph_with_kcore,
    deduplicate_handles_in_graph,
)
from src.report import (
    generate_global_report,
    generate_subcommunity_report,
    get_next_available_index,
)
from src.output_paths import (
    canonical_artifact_stem,
    get_next_core_user_id,
    plot_folder_name,
)
from src.visualization import generate_network_visualization
from src.posts import collect_community_posts_df, interactive_select_gexf, interactive_select_csv
from src.analysis import analyze_word_frequency
from src.plotting import plot_figure_b1

def format_kcore_scenarios(scenarios):
    """
    Formata os cenarios de k-core como usuarios/grau medio, compactando
    intervalos de k que produzem o mesmo resultado.
    """
    if not scenarios:
        return "k0:0u/g0.0"

    items = sorted((int(k), v) for k, v in scenarios.items())
    groups = []
    start_k = prev_k = items[0][0]
    prev_value = (
        items[0][1].get("nodes", 0),
        items[0][1].get("edges", 0),
        round(items[0][1].get("avg_degree", 0.0), 2),
    )

    for k, stats in items[1:]:
        value = (
            stats.get("nodes", 0),
            stats.get("edges", 0),
            round(stats.get("avg_degree", 0.0), 2),
        )
        if k == prev_k + 1 and value == prev_value:
            prev_k = k
            continue
        groups.append((start_k, prev_k, prev_value))
        start_k = prev_k = k
        prev_value = value
    groups.append((start_k, prev_k, prev_value))

    parts = []
    for start, end, value in groups:
        label = f"k{start}" if start == end else f"k{start}-{end}"
        nodes, _, avg_degree = value
        parts.append(f"{label}:{nodes}u/g{avg_degree:.1f}")

    if len(parts) > 8:
        parts = parts[:6] + ["..."] + parts[-1:]
    return " | ".join(parts)


def best_kcore_choice(scenarios):
    """
    Escolhe automaticamente o k-core que maximiza grau_medio / n_usuarios.
    Em empate, preserva mais usuarios; persistindo empate, escolhe o menor k.
    """
    best = None
    for k_core, stats in scenarios.items():
        n_users = stats.get("nodes", 0)
        if n_users < 2:
            continue
        avg_degree = stats.get("avg_degree", 0.0)
        score = avg_degree / n_users if n_users else 0.0
        candidate = (score, n_users, -int(k_core), int(k_core), avg_degree)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        return 0, 0.0

    score, _, _, selected_k, _ = best
    return selected_k, score


def ising_method_artifact_slug(method_name):
    """
    Nome curto e estavel para artefatos da inferencia Ising.
    """
    key = str(method_name or "").strip().lower()
    if "custom" in key:
        return "mch_custom"
    if "enumerate" in key or "exact" in key or "exato" in key:
        return "enumerate"
    if "mch" in key:
        return "mch_coniii"
    return "ising"


def move_artifact_replace(src, dst):
    """
    Move um artefato gerado no cwd para o destino final, substituindo versoes antigas.
    """
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    import shutil
    shutil.move(src, dst)
    return True


def interactive_select_ising_artifacts(plots_base):
    """
    Seleciona uma pasta de comunidade com matriz_ising_*.csv e multipliers_*.npy.
    """
    candidates = []
    for root, _, files in os.walk(plots_base):
        matrices = sorted(f for f in files if f.startswith("matriz_ising_") and f.endswith(".csv"))
        multipliers = sorted(f for f in files if f.startswith("multipliers_") and f.endswith(".npy"))
        if matrices and multipliers:
            candidates.append((root, matrices, multipliers))

    candidates.sort(key=lambda item: os.path.relpath(item[0], plots_base))
    if not candidates:
        print(f"[Aviso] Nenhuma pasta com matriz_ising_*.csv e multipliers_*.npy encontrada em {plots_base}")
        return None

    print("\nANÁLISES ISING DISPONÍVEIS:")
    for i, (root, matrices, multipliers) in enumerate(candidates):
        rel_path = os.path.relpath(root, plots_base)
        print(f"[{i}] {rel_path} | matriz={matrices[0]} | multipliers={multipliers[-1]}")

    try:
        idx = int(input("\nEscolha o índice da comunidade: ").strip())
        if not 0 <= idx < len(candidates):
            return None
    except ValueError:
        return None

    root, matrices, multipliers = candidates[idx]
    matrix_name = matrices[0]
    multiplier_name = multipliers[-1]

    if len(matrices) > 1:
        print("\nMATRIZES DISPONÍVEIS:")
        for i, name in enumerate(matrices):
            print(f"[{i}] {name}")
        try:
            matrix_idx = int(input(f"Escolha a matriz [padrão: 0]: ").strip() or 0)
            if 0 <= matrix_idx < len(matrices):
                matrix_name = matrices[matrix_idx]
        except ValueError:
            pass

    if len(multipliers) > 1:
        print("\nMULTIPLIERS DISPONÍVEIS:")
        for i, name in enumerate(multipliers):
            print(f"[{i}] {name}")
        try:
            multiplier_idx = int(input(f"Escolha os multipliers [padrão: {len(multipliers) - 1}]: ").strip() or (len(multipliers) - 1))
            if 0 <= multiplier_idx < len(multipliers):
                multiplier_name = multipliers[multiplier_idx]
        except ValueError:
            pass

    return {
        "output_dir": root,
        "matrix_path": os.path.join(root, matrix_name),
        "multipliers_path": os.path.join(root, multiplier_name),
    }


def prompt_int(label, default, minimum=1):
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return int(default)
        try:
            value = int(raw.replace(".", "").replace(",", ""))
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"  Informe um inteiro >= {minimum}.")


def prompt_float(label, default):
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return float(default)
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        print(f"  Valor inválido. Usando padrão {default}.")
        return float(default)


async def main():
    print("=" * 50)
    print("DETECÇÃO E ANÁLISE DE COMUNIDADES NO BLUESKY")
    print("=" * 50)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    safe_limit = await calibrate_rate_limit(max_test_concurrency=100)
    
    while True:
        print("\nMENU PRINCIPAL:")
        print("[1] Nova Coleta e Análise de Comunidades (core_user)")
        print("[2] Análise Estatística de Posts (GEXF Existente)")
        print("[3] Aplicação do Modelo de Máxima Entropia (Ising)")
        print("[4] Análise de Suscetibilidade")
        print("[5] Sair")
        
        opcao = input("\nEscolha uma opção: ").strip()
        
        if opcao == '5' or opcao.lower() == 'sair': break

        if opcao == '2':
            gexf_base = os.path.join(base_dir, "data", "processed", "gexf")
            gexf_path = interactive_select_gexf(gexf_base)
            
            if gexf_path:
                # 1. Coleta otimizada
                from src.analysis import analyze_word_frequency, create_ising_matrix_from_sets
                
                print("\n[Coleta] Usando os usuarios do GEXF sem novo filtro por minimo de posts.")
                global_word_counts, global_word_timestamps, user_word_sets, all_community_users = await collect_community_posts_df(
                    gexf_path, semaphore_limit=safe_limit
                )
                
                if global_word_counts:
                    # 2. Análise de frequência por usuários + localização temporal
                    user_count = len(all_community_users)
                    stats_df = analyze_word_frequency(
                        global_word_counts,
                        user_word_sets,
                        total_users=user_count,
                        global_word_timestamps=global_word_timestamps
                    )
                    
                    comm_name = canonical_artifact_stem(gexf_path)
                    plots_out = os.path.join(
                        base_dir,
                        "data",
                        "plots",
                        plot_folder_name(gexf_path, user_count),
                    )
                    plot_figure_b1(stats_df, total_users=user_count, output_dir=plots_out, filename="figure_B1.png")
                    print(f"\n[Sucesso] Gráfico Figure B1 salvo em: {plots_out}")

                    # --- Fase 1: Salva CSV COMPLETO para inspeção manual ---
                    full_csv_path = os.path.join(plots_out, f"keywords_todas_{comm_name}.csv")
                    stats_df_sorted = stats_df.sort_values(by='n_users', ascending=False)
                    stats_df_sorted['source_gexf'] = os.path.basename(gexf_path)
                    os.makedirs(plots_out, exist_ok=True)
                    stats_df_sorted.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
                    print(f"\n[CSV Completo] {len(stats_df_sorted)} keywords salvas em:\n  {full_csv_path}")
                    print(f"  Colunas: word | occurrences | n_users | user_freq | time_std_days")
                    print(f"  Total de usuários na comunidade: {user_count}")
                    print(f"\n  >>> Abra o arquivo para inspecionar os valores antes de filtrar. <<<")

                    try:
                        g_tmp = nx.read_gexf(gexf_path)
                        n_edges_aij = int(g_tmp.number_of_edges())
                        n_params_aij = user_count + n_edges_aij
                        n_params_dense = user_count + user_count * (user_count - 1) // 2
                        target_keywords_aij = int(np.ceil(5.0 * n_params_aij))
                        target_keywords_dense = int(np.ceil(5.0 * n_params_dense))
                        print("\n[Alvo Ising]")
                        print(
                            f"  Com A_ij do GEXF: parametros={n_params_aij} "
                            f"({user_count} h + {n_edges_aij} J) -> ideal R>5p: "
                            f">= {target_keywords_aij} keywords."
                        )
                        print(
                            f"  Se J fosse denso: parametros={n_params_dense} -> "
                            f">= {target_keywords_dense} keywords."
                        )
                    except Exception as e:
                        n_params_aij = None
                        target_keywords_aij = None
                        print(f"\n[Alvo Ising] Nao foi possivel estimar R/parametros pelo GEXF ({type(e).__name__}: {e}).")

                    # --- Fase 2: Filtro Interativo de Keywords ---
                    print("\nFILTRAGEM DE KEYWORDS:")
                    print(f"  Corte Inferior: exija que a keyword seja usada por pelo menos X usuários (ex: 3 ou 5% da comunidade)")
                    print(f"  Corte Superior: remova keywords usadas por >80-90% dos usuários (spins congelados)")
                    print(f"  Corte Temporal: remova keywords espalhadas demais no tempo (Figure B1 de Hall & Bialek)")
                    
                    while True:
                        try:
                            min_users = int(input(f"\nMínimo de Usuários (ex: 3): ").strip() or 2)
                            max_users_pct = float(input(f"Máximo de Usuários em % da comunidade (ex: 90): ").strip() or 100)
                            max_users = int(user_count * max_users_pct / 100)
                            min_freq = int(input("Frequência Mínima de ocorrências (ex: 5): ").strip() or 1)
                            max_time_std_input = input("Desvio padrão temporal máximo em dias [vazio = sem filtro; ex: 130]: ").strip()
                            max_time_std_days = float(max_time_std_input) if max_time_std_input else None

                            filtered_df = stats_df[
                                (stats_df['n_users'] >= min_users) & 
                                (stats_df['n_users'] <= max_users) & 
                                (stats_df['occurrences'] >= min_freq)
                            ]
                            if max_time_std_days is not None:
                                filtered_df = filtered_df[
                                    filtered_df['time_std_days'].notna() &
                                    (filtered_df['time_std_days'] <= max_time_std_days)
                                ]
                            filtered_df = filtered_df.sort_values(
                                by=['time_std_days', 'n_users'],
                                ascending=[True, False],
                                na_position='last'
                            )

                            print(f"\n=> Esse filtro resultou em {len(filtered_df)} palavras.")
                            print(f"   min_users={min_users}, max_users={max_users} ({max_users_pct:.0f}% de {user_count}), min_freq={min_freq}, max_time_std_days={max_time_std_days}")
                            if n_params_aij:
                                ratio_ising = len(filtered_df) / n_params_aij
                                print(
                                    f"   R/parametros estimado com A_ij: {ratio_ising:.2f} "
                                    f"(alvo > 5; ideal >= {target_keywords_aij} keywords)."
                                )
                            
                            if not filtered_df.empty:
                                confirm = input("Deseja prosseguir e salvar essas keywords? (s/n): ").strip().lower()
                                if confirm in ('s', 'sim', 'y', 'yes'):
                                    # Fase 3: Deleta CSV completo e salva apenas o filtrado
                                    if os.path.exists(full_csv_path):
                                        os.remove(full_csv_path)
                                        print(f"[Limpeza] CSV completo removido: {os.path.basename(full_csv_path)}")
                                    
                                    filtered_df['source_gexf'] = os.path.basename(gexf_path)
                                    csv_path = os.path.join(plots_out, f"keywords_filtradas_{comm_name}.csv")
                                    filtered_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                    plot_figure_b1(
                                        stats_df,
                                        total_users=user_count,
                                        output_dir=plots_out,
                                        filename="figure_B1_com_corte.png",
                                        max_time_std_days=max_time_std_days
                                    )
                                    
                                    # --- CACHE DE USO RAM PARA ACELERAR A OPÇÃO 3 ---
                                    import json
                                    cache_path = os.path.join(plots_out, f".cache_usersets_{comm_name}.json")
                                    with open(cache_path, 'w', encoding='utf-8') as f:
                                        serializable_user_word_sets = {k: list(v) for k, v in user_word_sets.items()}
                                        json.dump((serializable_user_word_sets, all_community_users), f)
                                        
                                    print(f"[Sucesso] {len(filtered_df)} palavras filtradas salvas em: {csv_path}")
                                    print(f"  [>] Cache dos dicionários locais de usuários salvo em background.")
                                    break
                                else:
                                    print("Vamos tentar outro filtro...")
                            else:
                                print("[Aviso] Nenhuma palavra atendeu aos critérios. Tente novamente.")
                                
                        except ValueError:
                            print("[Erro] Entrada inválida. Por favor, insira números válidos.")
                        except Exception as e:
                            print(f"[Erro] Falha ao filtrar keywords: {e}")
                            break
                            
            continue

        if opcao == '3':
            # ID interno apenas para arquivos temporarios gerados pelo backend Ising.
            ising_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            plots_base = os.path.join(base_dir, "data", "plots")
            # 1. Seleciona as keywords (modelo)
            kw_path = interactive_select_csv(plots_base, keyword_filter="keywords_filtradas")
            
            if kw_path:
                # 2. Seleciona a comunidade alvo
                gexf_base = os.path.join(base_dir, "data", "processed", "gexf")
                gexf_path = interactive_select_gexf(gexf_base)
                
                if gexf_path:
                    try:
                        from src.analysis import (
                            create_ising_matrix_from_sets,
                        )
                        from src.ising_coniii import (
                            construir_mascara_adjacencia,
                            inferir_modelo,
                            _amostrar_modelo_metropolis,
                            gerar_figura2,
                            gerar_figura3_multimodo,
                            gerar_figura4,
                        )
                        print("[Backend] Usando ising_coniii (ConIII + MCH custom)")
                        print("\nMÉTODO DE INFERÊNCIA ISING:")
                        print("  [0] Auto: Enumerate exato se N<=20; senão MCH")
                        print("  [1] MCH - Monte Carlo Histogram")
                        print("  [2] MCH-Custom - Numba/paralelo com execução fixa")
                        print("  [3] Enumerate exato (somente N<=20)")
                        metodo_input = input("Escolha o método [padrão: 0]: ").strip().lower()
                        if metodo_input in {"1", "m", "mch"}:
                            metodo_inferencia = "mch"
                        elif metodo_input in {"2", "mc", "custom", "custom_mch", "mch_custom", "mch-custom", "mch custom"}:
                            metodo_inferencia = "mch_custom"
                        elif metodo_input in {"3", "e", "exact", "exato", "enumerate"}:
                            metodo_inferencia = "exact"
                        else:
                            metodo_inferencia = "auto"

                        if metodo_inferencia == "mch":
                            print("  -> Método selecionado: MCH.")
                        elif metodo_inferencia == "mch_custom":
                            print("  -> Método selecionado: MCH-Custom.")
                        elif metodo_inferencia == "exact":
                            print("  -> Método selecionado: Enumerate exato.")
                        elif metodo_inferencia == "auto":
                            print("  -> Método selecionado: Auto.")
                        else:
                            print("  -> Método selecionado: MCH.")

                        mch_learning_profile = "adaptive_samples"
                        mch_sample_size = 100_000
                        mch_maxiter = 200
                        mch_interactive_continue = False
                        import json
                        
                        comm_name = canonical_artifact_stem(gexf_path)
                        dataset_plots_dir = os.path.dirname(kw_path)
                        plots_out = os.path.join(dataset_plots_dir, comm_name)
                        os.makedirs(plots_out, exist_ok=True)
                        cache_path = os.path.join(plots_out, f".cache_usersets_{comm_name}.json")
                        legacy_cache_path = os.path.join(dataset_plots_dir, f".cache_usersets_{comm_name}.json")
                        print(f"[Saida Ising] Artefatos desta subcomunidade serao salvos em: {plots_out}")
                        
                        if os.path.exists(cache_path) or os.path.exists(legacy_cache_path):
                            cache_read_path = cache_path if os.path.exists(cache_path) else legacy_cache_path
                            print("\n[Memória] Arquivo de coletas em Cache da Opção 2 recuperado!")
                            print(f"[Zero API] O sistema abortou o download duplo de 3.000 posts e os injetou instantaneamente do seu Disco local.")
                            with open(cache_read_path, 'r', encoding='utf-8') as f:
                                loaded_user_word_sets, all_community_users = json.load(f)
                                # Convert lists back to sets
                                user_word_sets = {k: set(v) for k, v in loaded_user_word_sets.items()}
                            if cache_read_path != cache_path:
                                with open(cache_path, "w", encoding="utf-8") as f:
                                    json.dump((loaded_user_word_sets, all_community_users), f)
                        else:
                            # 3. Coleta dados estrangeiros que não possuam memória na mesma base
                            print("\n[Coleta HTTP] Nenhuma memória viva dessa rede. Iniciando nova coleta via API...")
                            _, _, user_word_sets, all_community_users = await collect_community_posts_df(
                                gexf_path, semaphore_limit=safe_limit
                            )
                        
                        # 4. Carrega keywords e gera matriz de Ising
                        kw_df = pd.read_csv(kw_path)
                        keywords_list = kw_df['word'].tolist()
                        
                        print(f"\n[Ising] Gerando matriz para {len(all_community_users)} usuários...")
                        ising_matrix = create_ising_matrix_from_sets(
                            user_word_sets, keywords_list, all_community_users
                        )

                        if not ising_matrix.empty:
                            # Nome inclui o nome da comunidade para diferenciar
                            comm_name = canonical_artifact_stem(gexf_path)
                            n_keywords_csv = ising_matrix.shape[1]
                            n_keywords_present = int((ising_matrix.values > 0).any(axis=0).sum())
                            print(
                                f"[Keywords] Vocabulário fixo do CSV preservado: {n_keywords_csv} keywords "
                                f"({n_keywords_present} com ocorrência na comunidade selecionada; "
                                f"{n_keywords_csv - n_keywords_present} ausentes mantidas como -1)."
                            )

                            if ising_matrix.shape[0] < 3:
                                print("[Erro] A matriz Ising tem menos de 3 usuarios. Nao ha spins suficientes para inferencia.")
                                continue

                            if metodo_inferencia in {"mch", "mch_custom"} or (
                                metodo_inferencia == "auto" and ising_matrix.shape[0] > 20
                            ):
                                solver_label = "MCH-Custom" if metodo_inferencia == "mch_custom" else "MCH"
                                print(f"\nPERFIL DO SOLVER {solver_label}:")
                                print("  [1] Agressiva: passos maiores; mais rápida, maior risco de oscilação")
                                print("  [2] Média: passos intermediários")
                                print("  [3] Conservador: passos menores; mais estável, pode demorar mais")
                                print("  [4] Adaptado ao número de amostras: muda o perfil conforme sample_size")
                                print("  [5] Muito agressiva: passos ainda maiores; experimental")
                                perfil_input = input("Escolha o perfil [padrão: 4]: ").strip().lower()
                                if perfil_input in {"5", "ma", "muito agressiva", "muito agressivo", "muito_agressiva", "muito_agressivo", "very_aggressive"}:
                                    mch_learning_profile = "very_aggressive"
                                elif perfil_input in {"1", "a", "agressiva", "agressivo", "aggressive"}:
                                    mch_learning_profile = "aggressive"
                                elif perfil_input in {"2", "m", "media", "média", "medio", "médio", "medium"}:
                                    mch_learning_profile = "medium"
                                elif perfil_input in {"3", "c", "conservador", "conservadora", "conservative"}:
                                    mch_learning_profile = "conservative"
                                else:
                                    mch_learning_profile = "adaptive_samples"
                                perfil_label = {
                                    "very_aggressive": "Muito agressiva",
                                    "aggressive": "Agressiva",
                                    "medium": "Média",
                                    "conservative": "Conservador",
                                    "adaptive_samples": "Adaptado ao número de amostras",
                                }[mch_learning_profile]
                                print(f"  -> Perfil {solver_label} selecionado: {perfil_label}.")
                                while True:
                                    try:
                                        sample_input = input(f"Amostras {solver_label} por iteração [100000]: ").strip()
                                        mch_sample_size = int(sample_input.replace(".", "").replace(",", "")) if sample_input else 100_000
                                        if mch_sample_size >= 1000:
                                            break
                                    except ValueError:
                                        pass
                                    print(f"  [{solver_label}] Informe um inteiro >= 1000.")
                                while True:
                                    try:
                                        iter_input = input(f"Iterações {solver_label} nesta rodada [200]: ").strip()
                                        mch_maxiter = int(iter_input.replace(".", "").replace(",", "")) if iter_input else 200
                                        if mch_maxiter >= 1:
                                            break
                                    except ValueError:
                                        pass
                                    print(f"  [{solver_label}] Informe um inteiro >= 1.")
                                mch_interactive_continue = True

                            print(
                                f"  [Keywords] Sem refiltro por comunidade: {ising_matrix.shape[1]} keywords "
                                "do CSV seguem na matriz Ising."
                            )

                            ising_path = os.path.join(plots_out, f"matriz_ising_{comm_name}.csv")
                            ising_matrix.to_csv(ising_path, encoding='utf-8-sig')
                            print(f"[Ising] Matriz gerada: {ising_path}")
                            
                            # NÃO transpor aqui: inferir_modelo() já transpõe internamente
                            # (users×keywords → keywords×users).
                            S = ising_matrix.values.astype(np.int64)
                            S = np.where(S > 0, 1, -1).astype(np.int64)  # Garante +1/-1
                            
                            node_names = ising_matrix.index.tolist()
                            adjacency_mask = construir_mascara_adjacencia(gexf_path, node_names)
                            aij_path = os.path.join(plots_out, f"matriz_Aij_{comm_name}.csv")
                            pd.DataFrame(
                                adjacency_mask.astype(np.int8),
                                index=node_names,
                                columns=node_names
                            ).to_csv(aij_path, encoding='utf-8-sig')
                            print(f"[A_ij] Matriz de adjacencia salva: {aij_path}")
                             
                            # 5. Inferência do Modelo de Ising
                            print(f"\n[Ising] Iniciando inferência para {S.shape[0]} users × {S.shape[1]} keywords...")
                            resultados = inferir_modelo(
                                spin_matrix=S,
                                session_id=ising_run_id,
                                lam=0.001,
                                adjacency_mask=adjacency_mask,
                                metodo_inferencia=metodo_inferencia,
                                mch_sample_size=mch_sample_size,
                                mch_maxiter=mch_maxiter,
                                mch_learning_profile=mch_learning_profile,
                                mch_interactive_continue=mch_interactive_continue
                            )
                            result_values = next(iter(resultados.values()), {})
                            method_slug = ising_method_artifact_slug(
                                result_values.get("metodo", metodo_inferencia)
                                if isinstance(result_values, dict)
                                else metodo_inferencia
                            )
                            model_samples_figuras = None
                            n_amostras_figuras = 100_000
                            n_users_figuras = S.shape[0]
                            if (
                                isinstance(result_values, dict)
                                and "multipliers" in result_values
                                and "ERROR" not in result_values
                                and n_users_figuras > 20
                            ):
                                model_samples_figuras = _amostrar_modelo_metropolis(
                                    result_values["multipliers"],
                                    n_users_figuras,
                                    n_amostras_figuras,
                                    label="figuras Ising"
                                )
                            
                            # 6. Figura 2 (3 painéis)
                            print("\n[Ising] Gerando figuras e relatórios...")
                            gerar_figura2(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                gexf_path=gexf_path,
                                node_names=node_names,
                                session_id=ising_run_id,
                                model_samples=model_samples_figuras,
                                n_amostras_mc=n_amostras_figuras
                            )
                            
                            # 6.5. Figura 3 (Correlação de Tripletos), como em Hall & Bialek: todos os trios.
                            triplet_modes = ["all"]
                            gerar_figura3_multimodo(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                session_id=ising_run_id,
                                adjacency_mask=adjacency_mask,
                                triplet_modes=triplet_modes,
                                n_amostras_mc=n_amostras_figuras,
                                model_samples=model_samples_figuras
                            )
                            
                            # 6.6. Figura 4 (Distribuição de Atividade Coletiva Q)
                            gerar_figura4(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                session_id=ising_run_id,
                                model_samples=model_samples_figuras,
                                n_amostras_mc=n_amostras_figuras
                            )
                            
                            # 7. Move os artefatos gerados para a pasta da comunidade
                            fig_orig = f"figura2_ising_{ising_run_id}.png"
                            fig3_summary_orig = f"figura3_tripletos_resumo_{ising_run_id}.csv"
                            fig4_orig = f"figura4_distribuicao_Q_{ising_run_id}.png"
                            csv_orig = f"comparacao_metodos_{ising_run_id}.csv"
                            npy_orig = f"multiplicadores_ising_{ising_run_id}.npy"
                            
                            move_artifact_replace(
                                fig_orig,
                                os.path.join(plots_out, f"figura2_{method_slug}_{comm_name}.png")
                            )
                            for mode in triplet_modes:
                                fig3_orig = f"figura3_tripletos_{mode}_{ising_run_id}.png"
                                move_artifact_replace(
                                    fig3_orig,
                                    os.path.join(plots_out, f"figura3_tripletos_{mode}_{method_slug}_{comm_name}.png")
                                )
                            move_artifact_replace(
                                fig3_summary_orig,
                                os.path.join(plots_out, f"figura3_tripletos_resumo_{method_slug}_{comm_name}.csv")
                            )
                            move_artifact_replace(
                                fig4_orig,
                                os.path.join(plots_out, f"figura4_distribuicao_Q_{method_slug}_{comm_name}.png")
                            )
                            move_artifact_replace(
                                csv_orig,
                                os.path.join(plots_out, f"comparativo_{method_slug}_{comm_name}.csv")
                            )
                            move_artifact_replace(
                                npy_orig,
                                os.path.join(plots_out, f"multipliers_{method_slug}_{comm_name}.npy")
                            )
                            for checkpoint_orig in glob.glob(f"mch*_melhor_{ising_run_id}*.npy"):
                                if os.path.basename(checkpoint_orig).startswith("mch_custom_melhor_"):
                                    os.remove(checkpoint_orig)
                                    continue
                                checkpoint_name = os.path.basename(checkpoint_orig).replace(f"_{ising_run_id}", "")
                                move_artifact_replace(
                                    checkpoint_orig,
                                    os.path.join(plots_out, checkpoint_name)
                                )
                            for checkpoint_temp in glob.glob(os.path.join(plots_out, "mch_custom_melhor_*.npy")):
                                os.remove(checkpoint_temp)
                                
                            print(f"\n[Sucesso] Todos os artefatos Ising movidos para: {plots_out}")
                        else:

                            print("[Erro] Falha ao gerar matriz de Ising.")
                    except Exception as e:
                        print(f"[Erro] Falha na aplicação do modelo: {e}")
            continue

        if opcao == '4':
            plots_base = os.path.join(base_dir, "data", "plots")
            artifacts = interactive_select_ising_artifacts(plots_base)
            if not artifacts:
                continue

            try:
                from src.susceptibility import executar_analise_suscetibilidade

                matrix_df = pd.read_csv(artifacts["matrix_path"], index_col=0)
                spin_matrix = matrix_df.values.astype(np.int64)
                spin_matrix = np.where(spin_matrix > 0, 1, -1).astype(np.int64)
                multipliers = np.load(artifacts["multipliers_path"])
                artifact_stem = canonical_artifact_stem(artifacts["matrix_path"])

                print("\n[Análise de Suscetibilidade]")
                print(f"  Matriz: {artifacts['matrix_path']}")
                print(f"  Multipliers: {artifacts['multipliers_path']}")
                print(f"  Usuários/spins: {spin_matrix.shape[0]} | keywords/amostras empíricas: {spin_matrix.shape[1]}")
                print("  Figura 5 (Hall & Bialek 2019): χ por campo uniforme Δh.")
                print("    Preto: modelo pairwise | Ciano: modelo independente | Vermelho: Δh=0.")
                print("  Figura 6 (Hall & Bialek 2019): χ por número de indivíduos forçados em σ=+1.")
                print("    Barras de erro: desvio padrão entre configurações de indivíduos forçados.")

                field_min = prompt_float("Campo mínimo Δh", -1.0)
                field_max = prompt_float("Campo máximo Δh", 3.0)
                field_points = prompt_int("Quantidade de pontos em Δh", 41, minimum=3)
                samples_per_field = prompt_int("Amostras Monte Carlo por ponto da Figura 5", 100000, minimum=1000)
                default_max_forced = min(11, max(0, spin_matrix.shape[0] - 1))
                max_forced = prompt_int("Máximo de indivíduos forçados para Figura 6", default_max_forced, minimum=0)
                forced_configurations = prompt_int("Configurações por quantidade forçada", 25, minimum=1)
                samples_per_forced_configuration = prompt_int(
                    "Amostras Monte Carlo por configuração da Figura 6",
                    100000,
                    minimum=1000,
                )

                outputs = executar_analise_suscetibilidade(
                    spin_matrix,
                    multipliers,
                    output_dir=artifacts["output_dir"],
                    artifact_stem=artifact_stem,
                    field_min=field_min,
                    field_max=field_max,
                    field_points=field_points,
                    samples_per_field=samples_per_field,
                    max_forced=max_forced,
                    forced_configurations=forced_configurations,
                    samples_per_forced_configuration=samples_per_forced_configuration,
                )

                print("\n[Sucesso] Análise de suscetibilidade concluída.")
                print(f"  Figura 5: {outputs['figura5_png']}")
                print(f"  Dados Figura 5: {outputs['figura5_csv']}")
                print(f"  Figura 6: {outputs['figura6_png']}")
                print(f"  Dados Figura 6: {outputs['figura6_csv']}")
            except Exception as e:
                print(f"[Erro] Falha na análise de suscetibilidade: {type(e).__name__}: {e}")
            continue

        if opcao == '1':
            core_user = input("Digite o handle ou DID: ").strip()
            if not core_user: continue

            # --- Configuração dos filtros ---
            print("\nFILTRO DE CELEBRIDADES:")
            print("  Usuários com mais seguidores do que o limite serão excluídos da rede.")
            print("  (Sugestão: 5000 para comunidades temáticas, 10000 para mais abrangência)")
            try:
                max_followers_input = input("  Limite máximo de seguidores por usuário [padrão: 5000]: ").strip()
                max_followers = int(max_followers_input) if max_followers_input else 5000
            except ValueError:
                max_followers = 5000
            print(f"  → Celebridades com >{max_followers:,} seguidores serão removidas.\n")
            
            print("FILTRO DE ATIVIDADE MÍNIMA:")
            print("  Usuários com poucos posts com replies geram ruído nas comunidades e no modelo de Ising.")
            try:
                min_posts_input = input("  Mínimo de posts com replies por usuário [padrão: 0 = sem filtro]: ").strip()
                min_posts = int(min_posts_input) if min_posts_input else 0
            except ValueError:
                min_posts = 0
            if min_posts > 0:
                print(f"  → Usuários com <{min_posts} posts com replies serão removidos.\n")

            edges = await collect_network(
                core_user,
                safe_limit,
                max_followers=max_followers,
                min_posts=min_posts
            )
            if not edges: continue
                
            raw_G = build_graph(edges)
            raw_G.remove_edges_from(nx.selfloop_edges(raw_G))
            G = nx.k_core(raw_G, k=2)
            if G.number_of_nodes() == 0: continue
            
            # ── Pré-processamento: Dedup ────────────────────────────────────
            # 1. Deduplicação de handles (case-insensitive) em todo o grafo
            print(f"\n[Dedup] Verificando handles duplicados no grafo ({G.number_of_nodes()} nós)...")
            removed_dedup = deduplicate_handles_in_graph(G)
            if removed_dedup:
                print(f"[Dedup] {len(removed_dedup)} handle(s) duplicado(s) removido(s).")
            else:
                print(f"[Dedup] Nenhum handle duplicado encontrado. Grafo limpo.")
            if G.number_of_nodes() == 0:
                print("[Erro] Nenhum usuário sobreviveu aos filtros.")
                continue
            
            # ── Detecção de Comunidades (sobre grafo já limpo) ──────────────
            results = detect_communities_multi_resolution(G, [1.0, 1.5, 2.0, 2.5, 3.0])
            print("\nRESUMO LEIDEN (C++):")
            for res, data in results.items():
                if data["num_communities"] > 0:
                    sizes = data['sizes'].values()
                    max_s = max(sizes) if sizes else 0
                    min_s = min(sizes) if sizes else 0
                    lowest = data.get("lowest_avg_degree") or {}
                    print(
                        f"Res [{res}]: {data['num_communities']} coms | "
                        f"Maior: {max_s} | Menor: {min_s} | "
                        f"Menor grau medio: C{lowest.get('cid')} "
                        f"({lowest.get('size')} usuarios, g={lowest.get('avg_degree', 0.0):.2f})"
                    )
                else:
                    print(f"Res [{res}]: Nenhuma comunidade detectada.")
           
            choice = input("\nResolução (ex: 1.0) ou 'cancelar': ").strip()
            if choice.lower() == 'cancelar' or choice not in [str(r) for r in results.keys()]: continue
            chosen_res = float(choice)
                    
            # Pastas Globais Anônimas
            processed_dir = os.path.join(base_dir, "data", "processed")
            gexf_dir = os.path.join(processed_dir, "gexf")
            reports_dir = os.path.join(processed_dir, "reports")
            png_dir = os.path.join(processed_dir, "png")
            core_user_id = get_next_core_user_id(processed_dir)
            core_user_stem = f"core_user_{core_user_id}"
            
            os.makedirs(gexf_dir, exist_ok=True)
            os.makedirs(reports_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            chosen_data = results[chosen_res]
            apply_partition(G, chosen_data["partition"])

            kcore_rows = []
            selected_kcores = {}
            for cid, scenarios in chosen_data["kcore_scenarios"].items():
                selected_k, selected_score = best_kcore_choice(scenarios)
                selected_kcores[cid] = {"k_core": selected_k, "score": selected_score}

                for k_core, stats in sorted(scenarios.items()):
                    n_users = stats.get("nodes", 0)
                    avg_degree = stats.get("avg_degree", 0.0)
                    score = avg_degree / n_users if n_users else 0.0
                    kcore_rows.append({
                        "resolution": chosen_res,
                        "community_id": cid,
                        "k_core": k_core,
                        "n_users": n_users,
                        "n_edges": stats.get("edges", 0),
                        "avg_degree": avg_degree,
                        "avg_degree_per_user": score,
                        "selected": k_core == selected_k,
                    })
            kcore_csv_path = os.path.join(reports_dir, f"{core_user_stem}.csv")
            pd.DataFrame(kcore_rows).to_csv(kcore_csv_path, index=False, encoding="utf-8-sig")
            print(f"\n[K-core] Cenários completos salvos em: {kcore_csv_path}")
            
            print("\nCOMUNIDADES:")
            print("  Formato k-core: kX:n_usuarios/grau_medio. k* = escolhido por maior grau_medio/n_usuarios.")
            for cid, size in sorted(chosen_data["sizes"].items(), key=lambda x: x[1], reverse=True):
                avg_degree = chosen_data["avg_degrees"].get(cid, 0.0)
                kcore_text = format_kcore_scenarios(chosen_data["kcore_scenarios"].get(cid, {}))
                selected = selected_kcores.get(cid, {"k_core": 0, "score": 0.0})
                print(
                    f"Comunidade {cid}: {size} usuarios | grau medio={avg_degree:.2f} | "
                    f"k*={selected['k_core']} score={selected['score']:.4f} | {kcore_text}"
                )
            
            exported_indices = []
            export_failures = 0
            for cid in sorted(chosen_data["sizes"], key=lambda c: chosen_data["sizes"][c], reverse=True):
                selected = selected_kcores.get(cid, {"k_core": 0, "score": 0.0})
                selected_k = selected["k_core"]
                disp_id = get_next_available_index(gexf_dir, reports_dir)
                sub_G = extract_subcommunity_graph_with_kcore(G, cid, k_core=selected_k)
                if sub_G.number_of_nodes() < 2:
                    export_failures += 1
                    continue

                exported_indices.append(disp_id)
                nx.write_gexf(sub_G, os.path.join(gexf_dir, f"comunidade_{disp_id}.gexf"))
                generate_subcommunity_report(
                    sub_G,
                    disp_id,
                    cid,
                    output_dir=reports_dir,
                    k_core=selected_k,
                    k_core_score=selected["score"],
                    core_user=core_user,
                    verbose=False,
                )
                generate_network_visualization(
                    sub_G,
                    output_dir=png_dir,
                    filename=f"comunidade_{disp_id}.png",
                    verbose=False,
                )

            print(
                f"\n[Export] {len(exported_indices)} subcomunidades salvas automaticamente "
                f"com k-core escolhido por grau_medio/n_usuarios."
            )
            if export_failures:
                print(f"[Export] {export_failures} subcomunidade(s) ignorada(s) por ficarem com <2 usuarios.")
                
            generate_global_report(
                G,
                chosen_data["num_communities"],
                chosen_data["modularity"],
                output_dir=reports_dir,
                selected_indices=exported_indices,
                core_user=core_user,
                core_user_id=core_user_id,
            )
            
            nx.write_gexf(G, os.path.join(gexf_dir, f"{core_user_stem}.gexf"))
            generate_network_visualization(G, output_dir=png_dir, filename=f"{core_user_stem}.png")
            print(f"\n[Sucesso] Arquivos anônimos em {processed_dir}")
            print(f"Consulte o relatório em {reports_dir} para identificar o usuário.")
            
    print("\nEncerrando.")

if __name__ == "__main__":
    asyncio.run(main())
