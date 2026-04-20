import asyncio
import os
import numpy as np
import networkx as nx
import pandas as pd
from datetime import datetime

from src.rate_limit import calibrate_rate_limit
from src.collection import collect_network
from src.modeling import build_graph
from src.community import detect_communities_multi_resolution, apply_partition, extract_subcommunity_graph
from src.report import generate_global_report, generate_subcommunity_report, get_next_available_index
from src.visualization import generate_network_visualization
from src.posts import collect_community_posts_df, interactive_select_gexf, interactive_select_csv
from src.analysis import analyze_word_intervals_dict
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
                from src.analysis import analyze_word_intervals_dict, create_ising_matrix_from_sets
                
                global_word_times, user_word_sets, all_community_users = await collect_community_posts_df(
                    gexf_path, semaphore_limit=safe_limit
                )
                
                if global_word_times:
                    # 2. Análise de intervalos para Figure B1
                    stats_df = analyze_word_intervals_dict(global_word_times)
                    
                    user_count = len(all_community_users)
                    plots_out = os.path.join(base_dir, "data", "plots", f"sessao_{session_id}", str(user_count))
                    plot_figure_b1(stats_df, output_dir=plots_out, filename="figure_B1.png")
                    print(f"\n[Sucesso] Gráfico Figure B1 salvo em: {plots_out}")

                    # --- Novo: Filtro Interativo de Keywords ---
                    print("\nFILTRAGEM DE KEYWORDS:")
                    comm_name = os.path.splitext(os.path.basename(gexf_path))[0]
                    
                    while True:
                        try:
                            min_std = float(input("\nDesvio Padrão Mínimo (ex: 0.1): ").strip() or 0)
                            max_std = float(input("Desvio Padrão Máximo (ex: 10000): ").strip() or 10000)
                            min_freq = int(input("Frequência Mínima (ex: 5): ").strip() or 1)

                            filtered_df = stats_df[
                                (stats_df['desvio_padrao'] >= min_std) & 
                                (stats_df['desvio_padrao'] <= max_std) & 
                                (stats_df['occurrences'] >= min_freq)
                            ].sort_values(by='desvio_padrao')

                            print(f"\n=> Esse filtro resultou em {len(filtered_df)} palavras.")
                            
                            if not filtered_df.empty:
                                confirm = input("Deseja prosseguir e salvar essas keywords? (s/n): ").strip().lower()
                                if confirm in ('s', 'sim', 'y', 'yes'):
                                    filtered_df['source_gexf'] = os.path.basename(gexf_path)
                                    csv_path = os.path.join(plots_out, f"keywords_filtradas_{comm_name}.csv")
                                    filtered_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                                    
                                    # --- NOVO CACHE DE USO RAM PARA ACELERAR A OPÇÃO 3 ---
                                    import pickle
                                    cache_path = os.path.join(plots_out, f".cache_usersets_{comm_name}.pkl")
                                    with open(cache_path, 'wb') as f:
                                        pickle.dump((user_word_sets, all_community_users), f)
                                        
                                    print(f"[Sucesso] {len(filtered_df)} palavras salvas em: {csv_path}")
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
                        from src.analysis import create_ising_matrix_from_sets
                        from src.ising_coniii import inferir_todos, gerar_figura2
                        import shutil
                        import pickle
                        
                        plots_out = os.path.dirname(kw_path)
                        comm_name = os.path.splitext(os.path.basename(gexf_path))[0]
                        cache_path = os.path.join(plots_out, f".cache_usersets_{comm_name}.pkl")
                        
                        if os.path.exists(cache_path):
                            print("\n[Memória] Arquivo de coletas em Cache da Opção 2 recuperado!")
                            print(f"[Zero API] O sistema abortou o download duplo de 3.000 posts e os injetou instantaneamente do seu Disco local.")
                            with open(cache_path, 'rb') as f:
                                user_word_sets, all_community_users = pickle.load(f)
                        else:
                            # 3. Coleta dados estrangeiros que não possuam memória na mesma base
                            print("\n[Coleta HTTP] Nenhuma memória viva dessa rede. Iniciando nova coleta via API...")
                            _, user_word_sets, all_community_users = await collect_community_posts_df(
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
                            plots_out = os.path.dirname(kw_path)
                            # Nome inclui o nome da comunidade para diferenciar
                            comm_name = os.path.splitext(os.path.basename(gexf_path))[0]
                            ising_path = os.path.join(plots_out, f"matriz_ising_{comm_name}.csv")
                            ising_matrix.to_csv(ising_path, encoding='utf-8-sig')
                            print(f"[Ising] Matriz gerada: {ising_path}")
                            
                            # Transpondo a matriz para que os Usuários sejam os Spins (colunas)
                            # e as Keywords sejam as amostras (linhas), conforme paper original.
                            S = ising_matrix.values.T.astype(np.int64)
                            S = np.where(S > 0, 1, -1).astype(np.int64) # Garante +1/-1
                            
                            node_names = ising_matrix.index.tolist()
                            
                            # 5. Inferência com ConIII (MCH)
                            print(f"\n[Ising-ConIII] Iniciando inferência para {S.shape[0]} amostras (keywords) e {S.shape[1]} spins (usuários)...")
                            resultados = inferir_todos(
                                spin_matrix=S,
                                session_id=session_id,
                                lam=0.01
                            )
                            
                            # 6. Figura 2 (Painel Duplo)
                            print("\n[Ising-ConIII] Gerando figuras e relatórios...")
                            gerar_figura2(
                                spin_matrix=S,
                                resultados_inferencia=resultados,
                                gexf_path=gexf_path,
                                node_names=node_names,
                                session_id=session_id
                            )
                            
                            # 7. Move os artefatos gerados para a pasta da comunidade
                            fig_orig = f"figura2_ising_{session_id}.png"
                            csv_orig = f"comparacao_metodos_{session_id}.csv"
                            npy_orig = f"multiplicadores_ising_{session_id}.npy"
                            
                            if os.path.exists(fig_orig):
                                shutil.move(fig_orig, os.path.join(plots_out, f"figura2_coniii_{comm_name}_{session_id}.png"))
                            if os.path.exists(csv_orig):
                                shutil.move(csv_orig, os.path.join(plots_out, f"comparativo_coniii_{comm_name}_{session_id}.csv"))
                            if os.path.exists(npy_orig):
                                shutil.move(npy_orig, os.path.join(plots_out, f"multipliers_{comm_name}_{session_id}.npy"))
                                
                            print(f"\n[Sucesso] Todos os artefatos Ising-ConIII movidos para: {plots_out}")
                        else:

                            print("[Erro] Falha ao gerar matriz de Ising.")
                    except Exception as e:
                        print(f"[Erro] Falha na aplicação do modelo: {e}")
            continue

        if opcao == '1':
            core_user = input("Digite o handle ou DID: ").strip()
            if not core_user: continue

            # --- Configuração do filtro de celebridades ---
            print("\nFILTRO DE CELEBRIDADES:")
            print("  Usuários com mais seguidores do que o limite serão excluídos da rede.")
            print("  (Sugestão: 5000 para comunidades temáticas, 10000 para mais abrangência)")
            try:
                max_followers_input = input("  Limite máximo de seguidores por usuário [padrão: 5000]: ").strip()
                max_followers = int(max_followers_input) if max_followers_input else 5000
            except ValueError:
                max_followers = 5000
            print(f"  → Celebridades com >{max_followers:,} seguidores serão removidas.\n")

            edges = await collect_network(core_user, safe_limit, max_followers=max_followers)
            if not edges: continue
                
            raw_G = build_graph(edges)
            raw_G.remove_edges_from(nx.selfloop_edges(raw_G))
            G = nx.k_core(raw_G, k=2)
            if G.number_of_nodes() == 0: continue
            
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
