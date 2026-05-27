import asyncio
import os
import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime

from src.rate_limit import calibrate_rate_limit
from src.collection import collect_network
from src.modeling import build_graph
from src.community import detect_communities_multi_resolution, apply_partition, extract_subcommunity_graph, deduplicate_handles_in_graph
from src.report import generate_global_report, generate_subcommunity_report, get_next_available_index
from src.visualization import generate_network_visualization
from src.posts import collect_community_posts_df, interactive_select_gexf, interactive_select_csv
from src.analysis import analyze_word_frequency
from src.plotting import plot_figure_b1

async def main():
    print("=" * 50)
    print("DETECÇÃO E ANÁLISE DE COMUNIDADES NO BLUESKY")
    print("=" * 50)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Session ID para anonimização de pastas/arquivos
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_limit = await calibrate_rate_limit(max_test_concurrency=100)
    
    while True:
        print("\nMENU PRINCIPAL:")
        print("[1] Nova Coleta e Análise de Comunidades (core_user)")
        print("[2] Análise Estatística de Posts (GEXF Existente)")
        print("[3] Aplicação do Modelo de Máxima Entropia (Ising)")
        print("[4] Sair")
        
        opcao = input("\nEscolha uma opção: ").strip()
        
        if opcao == '4' or opcao.lower() == 'sair': break

        if opcao == '2':
            gexf_base = os.path.join(base_dir, "data", "processed", "gexf")
            gexf_path = interactive_select_gexf(gexf_base)
            
            if gexf_path:
                # 1. Coleta otimizada
                from src.analysis import analyze_word_frequency, create_ising_matrix_from_sets
                
                print("\nFILTRO DE ATIVIDADE:")
                print("  Usuários com poucos posts geram ruído no modelo de Ising.")
                try:
                    min_posts_input = input("  Mínimo de posts por usuário [padrão: 0 = sem filtro]: ").strip()
                    min_posts = int(min_posts_input) if min_posts_input else 0
                except ValueError:
                    min_posts = 0
                
                global_word_counts, global_word_timestamps, user_word_sets, all_community_users = await collect_community_posts_df(
                    gexf_path, semaphore_limit=safe_limit, min_posts=min_posts
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
                    
                    plots_out = os.path.join(base_dir, "data", "plots", f"sessao_{session_id}", str(user_count))
                    plot_figure_b1(stats_df, total_users=user_count, output_dir=plots_out, filename="figure_B1.png")
                    print(f"\n[Sucesso] Gráfico Figure B1 salvo em: {plots_out}")

                    # --- Fase 1: Salva CSV COMPLETO para inspeção manual ---
                    comm_name = os.path.splitext(os.path.basename(gexf_path))[0]
                    full_csv_path = os.path.join(plots_out, f"keywords_todas_{comm_name}.csv")
                    stats_df_sorted = stats_df.sort_values(by='n_users', ascending=False)
                    stats_df_sorted['source_gexf'] = os.path.basename(gexf_path)
                    os.makedirs(plots_out, exist_ok=True)
                    stats_df_sorted.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
                    print(f"\n[CSV Completo] {len(stats_df_sorted)} keywords salvas em:\n  {full_csv_path}")
                    print(f"  Colunas: word | occurrences | n_users | user_freq | time_std_days")
                    print(f"  Total de usuários na comunidade: {user_count}")
                    print(f"\n  >>> Abra o arquivo para inspecionar os valores antes de filtrar. <<<")

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
                                        min_occurrences=min_freq,
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
                            filter_ising_keywords_by_popularity,
                            filter_ising_users_by_activity,
                        )
                        from src.ising_coniii import (
                            construir_mascara_adjacencia,
                            inferir_modelo,
                            gerar_figura2,
                            gerar_figura3_multimodo,
                            gerar_figura4,
                        )
                        print("[Backend] Usando ising_coniii (ConIII)")
                        import shutil
                        import json
                        
                        plots_out = os.path.dirname(kw_path)
                        comm_name = os.path.splitext(os.path.basename(gexf_path))[0]
                        cache_path = os.path.join(plots_out, f".cache_usersets_{comm_name}.json")
                        
                        if os.path.exists(cache_path):
                            print("\n[Memória] Arquivo de coletas em Cache da Opção 2 recuperado!")
                            print(f"[Zero API] O sistema abortou o download duplo de 3.000 posts e os injetou instantaneamente do seu Disco local.")
                            with open(cache_path, 'r', encoding='utf-8') as f:
                                loaded_user_word_sets, all_community_users = json.load(f)
                                # Convert lists back to sets
                                user_word_sets = {k: set(v) for k, v in loaded_user_word_sets.items()}
                        else:
                            # 3. Coleta dados estrangeiros que não possuam memória na mesma base
                            print("\n[Coleta HTTP] Nenhuma memória viva dessa rede. Iniciando nova coleta via API...")
                            print("\nFILTRO DE ATIVIDADE:")
                            try:
                                min_posts_input = input("  Mínimo de posts por usuário [padrão: 0 = sem filtro]: ").strip()
                                min_posts = int(min_posts_input) if min_posts_input else 0
                            except ValueError:
                                min_posts = 0
                            _, _, user_word_sets, all_community_users = await collect_community_posts_df(
                                gexf_path, semaphore_limit=safe_limit, min_posts=min_posts
                            )
                        
                        # 4. Carrega keywords e gera matriz de Ising
                        kw_df = pd.read_csv(kw_path)
                        keywords_list = kw_df['word'].tolist()
                        
                        print(f"\n[Ising] Gerando matriz para {len(all_community_users)} usuários...")
                        ising_matrix = create_ising_matrix_from_sets(
                            user_word_sets, keywords_list, all_community_users
                        )

                        if not ising_matrix.empty:
                            plots_out = os.path.dirname(kw_path)
                            # Nome inclui o nome da comunidade para diferenciar
                            comm_name = os.path.splitext(os.path.basename(gexf_path))[0]

                            print("\nFILTRO DE ATIVIDADE NA MATRIZ ISING:")
                            print("  Remove usuarios que aparecem em poucas ou em quase todas as keywords filtradas.")
                            try:
                                min_activity_input = input("  Atividade minima por usuario em % de keywords [padrao: 10]: ").strip()
                                max_activity_input = input("  Atividade maxima por usuario em % de keywords [padrao: 90]: ").strip()
                                min_activity_pct = float(min_activity_input) if min_activity_input else 10.0
                                max_activity_pct = float(max_activity_input) if max_activity_input else 90.0
                                if min_activity_pct < 0 or max_activity_pct > 100 or min_activity_pct > max_activity_pct:
                                    raise ValueError
                            except ValueError:
                                print("  [Aviso] Limites invalidos. Usando padrao: 10% a 90%.")
                                min_activity_pct = 10.0
                                max_activity_pct = 90.0

                            n_users_before = ising_matrix.shape[0]
                            ising_matrix, removed_activity_users = filter_ising_users_by_activity(
                                ising_matrix,
                                min_activity=min_activity_pct / 100.0,
                                max_activity=max_activity_pct / 100.0
                            )
                            n_removed = len(removed_activity_users)
                            print(
                                f"  [Filtro] Mantidos {ising_matrix.shape[0]}/{n_users_before} usuarios "
                                f"({min_activity_pct:.1f}% <= atividade <= {max_activity_pct:.1f}%)."
                            )
                            if n_removed > 0:
                                removed_path = os.path.join(
                                    plots_out,
                                    f"usuarios_filtrados_atividade_{comm_name}_{session_id}.csv"
                                )
                                removed_activity_users.to_csv(removed_path, index=False, encoding='utf-8-sig')
                                print(f"  [Filtro] {n_removed} usuario(s) removido(s). Relatorio: {removed_path}")

                            if ising_matrix.shape[0] < 3:
                                print("[Erro] Menos de 3 usuarios sobraram apos o filtro de atividade. Ajuste os limites.")
                                continue

                            n_keywords_before = ising_matrix.shape[1]
                            ising_matrix, removed_popularity_keywords = filter_ising_keywords_by_popularity(
                                ising_matrix,
                                min_users=3,
                                max_user_frac=0.90
                            )
                            print(
                                f"  [Keywords] Mantidas {ising_matrix.shape[1]}/{n_keywords_before} keywords "
                                "apos refiltro na matriz final (3 usuarios <= n_users <= 90%)."
                            )
                            if len(removed_popularity_keywords) > 0:
                                removed_kw_path = os.path.join(
                                    plots_out,
                                    f"keywords_refiltradas_pos_usuarios_{comm_name}_{session_id}.csv"
                                )
                                removed_popularity_keywords.to_csv(
                                    removed_kw_path,
                                    index=False,
                                    encoding='utf-8-sig'
                                )
                                print(f"  [Keywords] Relatorio de keywords removidas: {removed_kw_path}")

                            if ising_matrix.shape[1] < 1:
                                print("[Erro] Nenhuma keyword sobrou apos o refiltro pos-usuarios. Ajuste os filtros.")
                                continue

                            ising_path = os.path.join(plots_out, f"matriz_ising_{comm_name}.csv")
                            ising_matrix.to_csv(ising_path, encoding='utf-8-sig')
                            print(f"[Ising] Matriz gerada: {ising_path}")
                            
                            # NÃO transpor aqui: inferir_modelo() já transpõe internamente
                            # (users×keywords → keywords×users).
                            S = ising_matrix.values.astype(np.int64)
                            S = np.where(S > 0, 1, -1).astype(np.int64)  # Garante +1/-1
                            
                            node_names = ising_matrix.index.tolist()
                            adjacency_mask = construir_mascara_adjacencia(gexf_path, node_names)
                            aij_path = os.path.join(plots_out, f"matriz_Aij_{comm_name}_{session_id}.csv")
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
                                session_id=session_id,
                                lam=0.001,
                                adjacency_mask=adjacency_mask
                            )
                            
                            # 6. Figura 2 (Painel Duplo)
                            print("\n[Ising] Gerando figuras e relatórios...")
                            gerar_figura2(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                gexf_path=gexf_path,
                                node_names=node_names,
                                session_id=session_id
                            )
                            
                            # 6.5. Figura 3 (Correlação de Tripletos) em múltiplos filtros topológicos
                            triplet_modes = ["all", "coupled", "connected", "triangles"]
                            gerar_figura3_multimodo(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                session_id=session_id,
                                adjacency_mask=adjacency_mask,
                                triplet_modes=triplet_modes
                            )
                            
                            # 6.6. Figura 4 (Distribuição de Atividade Coletiva Q)
                            gerar_figura4(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                session_id=session_id
                            )
                            
                            # 7. Move os artefatos gerados para a pasta da comunidade
                            fig_orig = f"figura2_ising_{session_id}.png"
                            fig3_summary_orig = f"figura3_tripletos_resumo_{session_id}.csv"
                            fig4_orig = f"figura4_distribuicao_Q_{session_id}.png"
                            csv_orig = f"comparacao_metodos_{session_id}.csv"
                            npy_orig = f"multiplicadores_ising_{session_id}.npy"
                            
                            if os.path.exists(fig_orig):
                                shutil.move(fig_orig, os.path.join(plots_out, f"figura2_coniii_{comm_name}_{session_id}.png"))
                            for mode in triplet_modes:
                                fig3_orig = f"figura3_tripletos_{mode}_{session_id}.png"
                                if os.path.exists(fig3_orig):
                                    shutil.move(
                                        fig3_orig,
                                        os.path.join(plots_out, f"figura3_tripletos_{mode}_{comm_name}_{session_id}.png")
                                    )
                            if os.path.exists(fig3_summary_orig):
                                shutil.move(
                                    fig3_summary_orig,
                                    os.path.join(plots_out, f"figura3_tripletos_resumo_{comm_name}_{session_id}.csv")
                                )
                            if os.path.exists(fig4_orig):
                                shutil.move(fig4_orig, os.path.join(plots_out, f"figura4_distribuicao_Q_{comm_name}_{session_id}.png"))
                            if os.path.exists(csv_orig):
                                shutil.move(csv_orig, os.path.join(plots_out, f"comparativo_coniii_{comm_name}_{session_id}.csv"))
                            if os.path.exists(npy_orig):
                                shutil.move(npy_orig, os.path.join(plots_out, f"multipliers_{comm_name}_{session_id}.npy"))
                                
                            print(f"\n[Sucesso] Todos os artefatos Ising movidos para: {plots_out}")
                        else:

                            print("[Erro] Falha ao gerar matriz de Ising.")
                    except Exception as e:
                        print(f"[Erro] Falha na aplicação do modelo: {e}")
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
            print("  Usuários com poucos posts geram ruído nas comunidades e no modelo de Ising.")
            try:
                min_posts_input = input("  Mínimo de posts por usuário [padrão: 0 = sem filtro]: ").strip()
                min_posts = int(min_posts_input) if min_posts_input else 0
            except ValueError:
                min_posts = 0
            if min_posts > 0:
                print(f"  → Usuários com <{min_posts} posts serão removidos.\n")

            edges = await collect_network(core_user, safe_limit, max_followers=max_followers)
            if not edges: continue
                
            raw_G = build_graph(edges)
            raw_G.remove_edges_from(nx.selfloop_edges(raw_G))
            G = nx.k_core(raw_G, k=2)
            if G.number_of_nodes() == 0: continue
            
            # ── Pré-processamento: Dedup + Filtro de Atividade ──────────────
            # 1. Deduplicação de handles (case-insensitive) em todo o grafo
            print(f"\n[Dedup] Verificando handles duplicados no grafo ({G.number_of_nodes()} nós)...")
            removed_dedup = deduplicate_handles_in_graph(G)
            if removed_dedup:
                print(f"[Dedup] {len(removed_dedup)} handle(s) duplicado(s) removido(s).")
            else:
                print(f"[Dedup] Nenhum handle duplicado encontrado. Grafo limpo.")
            
            # 2. Filtro de atividade mínima (consulta postsCount via API)
            if min_posts > 0:
                from src.collection import filter_inactive_users
                all_handles = list(G.nodes())
                print(f"\n[Atividade] Verificando postsCount de {len(all_handles)} usuários (mínimo: {min_posts})...")
                active_handles, removed_count = await filter_inactive_users(all_handles, min_posts, safe_limit)
                
                if removed_count > 0:
                    inactive_set = set(all_handles) - set(active_handles)
                    G.remove_nodes_from(inactive_set)
                    print(f"[Atividade] {removed_count} usuário(s) removido(s) (<{min_posts} posts).")
                    print(f"[Atividade] {G.number_of_nodes()} usuários ativos mantidos.")
                else:
                    print(f"[Atividade] Todos os usuários atendem ao mínimo de {min_posts} posts.")
                
                if G.number_of_nodes() == 0:
                    print("[Erro] Nenhum usuário sobreviveu aos filtros.")
                    continue
            
            # ── Detecção de Comunidades (sobre grafo já limpo) ──────────────
            results = detect_communities_multi_resolution(G, [1.0, 1.5, 2.0, 2.5, 3.0])
            print("\nRESUMO LEIDEN (C++) - REFINADO COM K-CORE (k=2):")
            for res, data in results.items():
                if data['initial_mod'] > 0:
                    sizes = data['sizes'].values()
                    max_s = max(sizes) if sizes else 0
                    min_s = min(sizes) if sizes else 0
                    print(f"Res [{res}]: {data['num_communities']} coms | Mod: {data['initial_mod']:.4f} -> {data['modularity']:.4f} | Maior: {max_s} | Menor: {min_s}")
                else:
                    print(f"Res [{res}]: Nenhuma comunidade detectada.")
           
            choice = input("\nResolução (ex: 1.0) ou 'cancelar': ").strip()
            if choice.lower() == 'cancelar' or choice not in [str(r) for r in results.keys()]: continue
            chosen_res = float(choice)
                    
            # Pastas Globais Anônimas
            processed_dir = os.path.join(base_dir, "data", "processed")
            gexf_dir = os.path.join(processed_dir, "gexf")
            reports_dir = os.path.join(processed_dir, "reports", f"sessao_{session_id}")
            png_dir = os.path.join(processed_dir, "png", f"sessao_{session_id}")
            
            os.makedirs(gexf_dir, exist_ok=True)
            os.makedirs(reports_dir, exist_ok=True)
            os.makedirs(png_dir, exist_ok=True)

            chosen_data = results[chosen_res]
            apply_partition(G, chosen_data["partition"])
            
            print("\nCOMUNIDADES:")
            for cid, size in sorted(chosen_data["sizes"].items(), key=lambda x: x[1], reverse=True):
                print(f"Comunidade {cid}: {size} usuários")
            
            selection = input("\nIDs para exportar (ex: 0, 1): ").strip()
            exported_indices = []
            if selection:
                try:
                    for cid in [int(x.strip()) for x in selection.split(",")]:
                        if cid not in chosen_data["sizes"]: continue
                        disp_id = get_next_available_index(gexf_dir, reports_dir)
                        exported_indices.append(disp_id)
                        sub_G = extract_subcommunity_graph(G, cid)
                        
                        # Nome padrão: comunidade_{id}_{core_user}.gexf
                        nx.write_gexf(sub_G, os.path.join(gexf_dir, f"comunidade_{disp_id}_{core_user}.gexf"))
                        generate_subcommunity_report(sub_G, disp_id, cid, output_dir=reports_dir)
                        generate_network_visualization(sub_G, output_dir=png_dir, filename=f"comunidade_{disp_id}.png")
                except: pass
                
            generate_global_report(G, chosen_data["num_communities"], chosen_data["modularity"], output_dir=reports_dir, selected_indices=exported_indices, core_user=core_user)
            
            # Global GEXF: rede_{session_id}.gexf (Anônimo)
            nx.write_gexf(G, os.path.join(gexf_dir, f"rede_{session_id}.gexf"))
            generate_network_visualization(G, output_dir=png_dir, filename="rede_global.png")
            print(f"\n[Sucesso] Arquivos anônimos em {processed_dir}")
            print(f"Consulte o relatório em {reports_dir} para identificar o usuário.")
            
    print("\nEncerrando.")

if __name__ == "__main__":
    asyncio.run(main())
