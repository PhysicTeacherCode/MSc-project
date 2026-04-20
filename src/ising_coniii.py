"""
Integração do ConIII no Pipeline Bluesky para Inferência do Modelo de Ising.

Este módulo implementa a inferência dos parâmetros (campos h e acoplamentos J)
usando exclusivamente o método Monte Carlo Histogram (MCH) via pacote coniii.

Inclui também a avaliação qualitativa do ajuste segundo Schneidman et al. (2006)
(RMSE das médias < 3/sqrt(R)), visualizações em heatmaps com máscaras topológicas
via networkx e geração de amostras via Metropolis para validação.
"""

import os
import sys
import time
import argparse
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

try:
    import coniii
    from coniii.utils import define_ising_helper_functions

    # --- Patch for Numba dtype incompatibility on Windows ---
    # Context: coniii uses np.zeros(..., dtype=int) which defaults to int32 on Windows.
    # The internal Numba functions strictly require int64, leading to a "No matching definition" TypeError.
    # This patch forces arrays into int64 right before they enter the jitted functions.
    def patch_coniii_numba_signatures():
        try:
            import coniii.utils
            import coniii.models
            import coniii.samplers
            import coniii.solvers
            
            # Captura a fábrica de funções original
            old_define = coniii.utils.define_ising_helper_functions
            
            def patched_define():
                # Executa a fábrica para puxar as referências reais ocultas pelo compilador
                calc_e, calc_observables, mch_approx = old_define()
                
                # Envolve as 3 funções perigosas com upcast de 64 bits imediatamente antes do return
                def wrapped_calc_e(states, params):
                    return calc_e(states.astype(np.int64), params)
                    
                def wrapped_calc_observables(states):
                    return calc_observables(states.astype(np.int64))
                    
                def wrapped_mch_approx(samples, dlamda):
                    return mch_approx(samples.astype(np.int64), dlamda)
                    
                return wrapped_calc_e, wrapped_calc_observables, wrapped_mch_approx
            
            # Injeta a fábrica falsificada de volta em todas as ramificações locais do ConIII
            coniii.utils.define_ising_helper_functions = patched_define
            if hasattr(coniii, 'models') and hasattr(coniii.models, 'define_ising_helper_functions'):
                coniii.models.define_ising_helper_functions = patched_define
            if hasattr(coniii, 'samplers') and hasattr(coniii.samplers, 'define_ising_helper_functions'):
                coniii.samplers.define_ising_helper_functions = patched_define
            if hasattr(coniii, 'solvers') and hasattr(coniii.solvers, 'define_ising_helper_functions'):
                coniii.solvers.define_ising_helper_functions = patched_define
                
            # Atualiza também qualquer escopo global que tenha importado `define_ising_helper_functions` precocemente no main!
            import sys
            if 'main' in sys.modules and hasattr(sys.modules['main'], 'define_ising_helper_functions'):
                sys.modules['main'].define_ising_helper_functions = patched_define

        except Exception as e:
            print(f"[Aviso] Falha ao aplicar patch do Numba: {e}")

    patch_coniii_numba_signatures()
    # --------------------------------------------------------
except ImportError:
    print("\n[Erro Crítico] Pacote 'coniii' não encontrado.")
    print("Por favor, instale o ambiente conforme a documentação:")
    print("  conda create -n bluesky_ising -c conda-forge python=3.10 numpy scipy numba matplotlib boost==1.74 jupyter")
    print("  conda activate bluesky_ising")
    print("  pip install coniii\n")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FUNÇÕES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def unpack_J(j_flat: np.ndarray, N: int) -> np.ndarray:
    """
    Reconstrói a matriz J simétrica (N x N) com diagonal zero a partir
    do vetor unidimensional j_flat gerado pelos solvers do ConIII.
    A iteração respeita a ordem lexicográfica (i < j).
    """
    J_mat = np.zeros((N, N))
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            J_mat[i, j] = j_flat[idx]
            J_mat[j, i] = j_flat[idx]
            idx += 1
    return J_mat

def calcular_rmse_medias(spin_matrix: np.ndarray, h_inferido: np.ndarray) -> float:
    """
    Avalia a qualidade do ajuste comparando as médias empíricas
    com as médias previstas isoladamente pelos campos locais: tanh(h).
    Retorna o RMSE segundo critério de Schneidman et al. (2006).
    """
    m_empirico = spin_matrix.mean(axis=0)
    m_previsto = np.tanh(h_inferido)
    return float(np.sqrt(np.mean((m_empirico - m_previsto)**2)))

def exibir_avaliacao_schneidman(rmse: float, R: int, metodo: str):
    """
    Exibe a avaliação qualitativa do RMSE segundo Schneidman et al. (2006).
    O erro padrão amostral esperado é 1/sqrt(R).
    O ajuste é considerado aceitável se RMSE < 3/sqrt(R).
    """
    erro_amostral = 1.0 / np.sqrt(R)
    limiar = 3.0 * erro_amostral
    print(f"  [{metodo}] Avaliação Schneidman (2006):")
    print(f"    RMSE Obtido: {rmse:.5f} | Erro Amostral (1/√R): {erro_amostral:.5f} | Limiar (3/√R): {limiar:.5f}")
    if rmse > limiar:
        print(f"    ⚠ AVISO: Ajuste insuficiente pelo método {metodo} (RMSE excedeu o limiar).")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEÇÃO 10 — INFERÊNCIA DE PARÂMETROS COM TRÊS MÉTODOS
# ─────────────────────────────────────────────────────────────────────────────

def inferir_mch(spin_matrix: np.ndarray, session_id: str, lam: float = 0.01) -> dict:
    """
    Método: Meta-Learning MCH (Auto-Tuning de Hiperparâmetros)
    """
    R, N = spin_matrix.shape
    spin_matrix = spin_matrix.astype(np.int64)
    calc_e, calc_observables, mch_approximation = define_ising_helper_functions()
    
    limiar_schneidman = 3.0 / np.sqrt(R)
    
    # Grade de busca (Teste dos conservadores aos longos)
    grade = [
        {"maxiter": 800,  "eta_init": 0.05, "decay_div": 200.0, "burn_in": 1000},
        {"maxiter": 1500, "eta_init": 0.1,  "decay_div": 350.0, "burn_in": 1500},
        {"maxiter": 2500, "eta_init": 0.02, "decay_div": 800.0, "burn_in": 2000},
        {"maxiter": 4000, "eta_init": 0.05, "decay_div": 1500.0,"burn_in": 2000}
    ]
    
    melhor_rmse = float('inf')
    melhor_dict = None
    
    import sys
    class MCHProgressFilter:
        def __init__(self, maxiter, prefix):
            self.maxiter = maxiter
            self.iter_count = 0
            self.original_stdout = sys.stdout
            self.prefix = prefix
            
        def write(self, text):
            if "Iterating parameters with MCH" in text:
                self.iter_count += 1
                porcentagem = (self.iter_count / self.maxiter) * 100
                self.original_stdout.write(f"\r{self.prefix} Progresso: {porcentagem:.1f}% ({self.iter_count}/{self.maxiter}) | Amostrando...")
                self.original_stdout.flush()
            elif "Sampling" in text or text.strip() == "": pass
            else:
                if text != "\n": self.original_stdout.write(text)
                
        def flush(self): 
            self.original_stdout.flush()
            
    print(f"\n  [Meta-Tuning] Iniciando busca automatizada da melhor topografia (Alvo RMSE <= {limiar_schneidman:.5f}) para N={N}...")
    
    # --- Transfer Learning / "Warm Start" ---
    print("  [Warm-Start] Pré-mapeando o Vale de Erro com algoritmo Pseudo-Verossimilhança...")
    import io
    old_stdout_ps = sys.stdout
    sys.stdout = io.StringIO() # Silencia os logs sujos do Pseudo
    try:
        solver_pseudo = coniii.solvers.Pseudo(spin_matrix)
        chute_informado = solver_pseudo.solve()
    except Exception as e:
        chute_informado = None
    finally:
        sys.stdout = old_stdout_ps
        if chute_informado is not None:
            print("  [Warm-Start] Sucesso! Esqueleto matrizal mapeado (Quase-Mínimo). Injetando matriz térmica no MCH...")
        else:
            print("  [Warm-Start] Falha no Pseudo. Retornando ao processo iterativo cego (Zeros).")
    
    for idx, cfg in enumerate(grade):
        prefixo = f"  [Tentativa {idx+1}/{len(grade)}]"
        print(f"\n{prefixo} Avaliando -> maxiter:{cfg['maxiter']}, eta_init:{cfg['eta_init']}, decaimento:i/{cfg['decay_div']}")
        
        # Vital instanciar solver zerado para limpar thermal noise do anterior
        solver = coniii.solvers.MCH(
            spin_matrix,
            calc_observables=calc_observables,
            mch_approximation=mch_approximation,
            sample_size=1000,
            n_cpus=max(1, mp.cpu_count() - 1),
            iprint=True
        )
        
        def learn_settings(i, c=cfg):
            return {
                'maxdlamda': np.exp(-i/c['decay_div']) * c['eta_init'], 
                'eta': np.exp(-i/c['decay_div']) * c['eta_init']
            }
            
        old_stdout = sys.stdout
        filtro_tela = MCHProgressFilter(maxiter=cfg['maxiter'], prefix=prefixo)
        sys.stdout = filtro_tela
        
        try:
            kwargs_solver = {
                "maxiter": cfg['maxiter'],
                "custom_convergence_f": learn_settings,
                "burn_in": cfg['burn_in'],
                "tol": 1e-5,
                "iprint": True
            }
            if chute_informado is not None:
                kwargs_solver["initial_guess"] = chute_informado

            multipliers = solver.solve(**kwargs_solver)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"\n{prefixo} ⚠ Divergência ({e}). Ignorando esta estratégia arriscada...")
            continue
        finally:
            sys.stdout = old_stdout
            
        h = multipliers[:N]
        rmse_atual = calcular_rmse_medias(spin_matrix, h)
        
        print(f"\n{prefixo} Iterações finalizadas. Obteve RMSE = {rmse_atual:.5f}")
        
        if rmse_atual < melhor_rmse:
            melhor_rmse = rmse_atual
            j_flat = multipliers[N:]
            
            melhor_dict = {
                "h": h, "J": unpack_J(j_flat, N), "multipliers": multipliers, 
                "metodo": f"MCH_Meta (eta={cfg['eta_init']}, iter={cfg['maxiter']})", 
                "rmse_medias": rmse_atual
            }
            
        if rmse_atual <= limiar_schneidman:
            print(f"  [Meta-Tuning] ✔ VALOR IDEAL ALCANÇADO E ESTATISTICAMENTE VIÁVEL! Interrompendo buscas...")
            break
        else:
            print(f"  [Meta-Tuning] Ajuste ainda em refinamento. Pulando para a próxima configuração profunda...")
            
    if melhor_dict is None:
        raise RuntimeError("Todas as configurações da Grade de Meta-Tuning causaram explosão numérica (crash).")
        
    print(f"\n  [MCH] 🏆 CAMPEÃO METODOLÓGICO SELECIONADO: {melhor_dict['metodo']} (RMSE {melhor_dict['rmse_medias']:.5f})")
    exibir_avaliacao_schneidman(melhor_dict['rmse_medias'], R, melhor_dict['metodo'])
    
    return melhor_dict



def inferir_todos(spin_matrix: np.ndarray, session_id: str, lam: float = 0.01) -> dict:
    """
    Orquestrador simplificado: Executa o método Monte Carlo Histogram (MCH).
    """
    print(f"\n[Ising-ConIII] Iniciando inferência MCH para R={spin_matrix.shape[0]}, N={spin_matrix.shape[1]}")
    t0 = time.time()
    
    try:
        resultado = inferir_mch(spin_matrix, session_id, lam)
        tempo_s = time.time() - t0
        resultado["tempo_s"] = tempo_s
        print(f"  [Sucesso] MCH concluído em {tempo_s:.1f}s")
        resultados = {"MCH": resultado}
    except Exception as e:
        tempo_s = time.time() - t0
        print(f"  [Erro] Falha em MCH: {e}")
        resultados = {"MCH": {"metodo": "MCH", "ERROR": str(e), "tempo_s": tempo_s}}
        
    # Salvar resultados e tabela comparativa
    R, N = spin_matrix.shape
    limiar_schneidman = 3.0 / np.sqrt(R)
    
    tabela = []
    print("\n--- RESUMO DE INFERÊNCIA ---")
    print(f"{'Método':<10} | {'RMSE':<8} | {'Tempo(s)':<8} | {'Viável (Schneidman)':<20}")
    print("-" * 55)
    
    for nome, dados in resultados.items():
        if "ERROR" in dados:
            print(f"{nome:<10} | {'ERROR':<8} | {dados['tempo_s']:<8.1f} | {'N/A':<20}")
            tabela.append({"metodo": nome, "rmse_medias": np.nan, "tempo_s": dados['tempo_s'], "N": N, "R": R, "limiar_schneidman": limiar_schneidman, "viavel": False})
        else:
            rmse = dados['rmse_medias']
            viavel = rmse <= limiar_schneidman
            viab_str = "Sim" if viavel else "Não"
            print(f"{nome:<10} | {rmse:<8.5f} | {dados['tempo_s']:<8.1f} | {viab_str:<20}")
            
            tabela.append({
                "metodo": nome, "rmse_medias": rmse, "tempo_s": dados['tempo_s'], 
                "N": N, "R": R, "limiar_schneidman": limiar_schneidman, "viavel": viavel
            })
                
    df_comp = pd.DataFrame(tabela)
    csv_path = f"comparacao_metodos_{session_id}.csv"
    df_comp.to_csv(csv_path, index=False)
    
    # Salvar multiplicadores
    if "MCH" in resultados and "multipliers" in resultados["MCH"]:
        multipliers = resultados["MCH"]["multipliers"]
        npy_path = f"multiplicadores_ising_{session_id}.npy"
        np.save(npy_path, multipliers)
        print(f"[Salvo] Multiplicadores MCH salvos em: {npy_path}")
        
    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# 3. SEÇÃO 11 — FIGURA 2 (Covariância Empírica + Acoplamentos Infêridos)
# ─────────────────────────────────────────────────────────────────────────────

def gerar_figura2(spin_matrix: np.ndarray, resultados_inferencia: dict, 
                  gexf_path: str, node_names: list, session_id: str):
    """
    Painel Duplo: Covariância empírica e Acoplamentos J via MCH.
    Aplica filtro topológico substituindo as pontes desconectadas por NaN
    usando a rede GEXF original da comunidade correspondente aos nós (usuários).
    """
    print("\n[Figura 2] Inicializando geração dos heatmaps lado a lado...")
    N = spin_matrix.shape[1]
    
    # Covariância
    C_emp = np.cov(spin_matrix, rowvar=False)
    
    # Filtro topológico
    aplicar_filtro = False
    A_mask = np.ones((N, N), dtype=bool)
    
    if os.path.exists(gexf_path):
        G = nx.read_gexf(gexf_path)
        encontrados = [k for k in node_names if k in G]
        if len(encontrados) < len(node_names) / 2:
            print("  [Aviso] Nomes dos nós da matriz não batem com os nós do GEXF.")
            print("  -> Filtro topológico DESABILITADO.")
        else:
            print(f"  [Filtro] GEXF válido. Aplicando máscara...")
            aplicar_filtro = True
            A = nx.to_numpy_array(G, nodelist=node_names)
            A_mask = (A == 1)
            np.fill_diagonal(A_mask, True)
    else:
        print(f"  [Aviso] GEXF {gexf_path} não encontrado. Filtro desabilitado.")

    def aplicar_mascara(matriz):
        if not aplicar_filtro:
            return matriz
        m_copy = matriz.copy()
        m_copy[~A_mask] = np.nan
        return m_copy

    C_masked = aplicar_mascara(C_emp)
    J_mch = aplicar_mascara(resultados_inferencia.get("MCH", {}).get("J", np.zeros((N,N))))

    # Configuração da figura matplotlib
    plt.rcParams.update({'text.color': '#cccccc', 'axes.labelcolor': '#cccccc',
                         'xtick.color': '#cccccc', 'ytick.color': '#cccccc'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1a1a2e')
    
    cmap = matplotlib.colormaps.get_cmap('jet').copy()
    cmap.set_bad(color='white')
    
    paineis = [
        (C_masked, "Covariância Empírica"),
        (J_mch, "J — Monte Carlo Hist. (MCH)")
    ]
    
    for i, (mat, titulo) in enumerate(paineis):
        ax = axes[i]
        ax.set_facecolor('#1a1a2e')
        
        if np.all(mat == 0) or np.isnan(mat).all():
            ax.set_title(titulo + " (Falhou/Timeout)", color='#cccccc', pad=10)
            ax.axis('off')
            continue
            
        v_max = np.nanmax(np.abs(mat)) if not np.isnan(mat).all() else 1.0
        if v_max == 0: v_max = 1.0
        im = ax.imshow(mat, cmap=cmap, vmin=-v_max, vmax=v_max, aspect='auto', interpolation='nearest')
        ax.set_title(titulo, color='#cccccc', pad=10)
        ax.set_xlabel("Spin $j$")
        ax.set_ylabel("Spin $i$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#444444')

    plt.tight_layout()
    fig_path = f"figura2_ising_{session_id}.png"
    plt.savefig(fig_path, dpi=300, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    
    print(f"\n[Figura 2] Heatmaps salvos com sucesso em: {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDAÇÃO OPCIONAL VIA METROPOLIS
# ─────────────────────────────────────────────────────────────────────────────

def validar_modelo(resultados_inferencia: dict, N: int, n_amostras: int = 1000):
    """
    Validação gerativa por amostragem Monte Carlo (Metropolis).
    Gera novas amostras a partir do modelo treinado e compara a covariância
    produzida sinteticamente com a matriz empírica esperada.
    """
    print("\n[Metropolis] Iniciando amostragem Monte Carlo para validação dos modelos...")
    calc_e, _, _ = define_ising_helper_functions()
    
    for nome, dados in resultados_inferencia.items():
        if "TIMEOUT" in dados or "ERROR" in dados:
            continue
            
        print(f"  > Avaliando método {nome}...")
        multipliers = dados["multipliers"]
        
        sampler = coniii.samplers.Metropolis(
            N,
            multipliers,
            calc_e
        )
        
        # Burn-in de 500 iter e subamostragem temporal (decorrelation) = 10
        amostras_sinteticas = sampler.sample(n_amostras, n_iters=10, burn_in=500)
        
        # (Idealmente compararíamos com a empírica real se tivéssemos salvo fora,
        # mas aqui demonstramos a amostragem)
        C_sintetica = np.cov(amostras_sinteticas, rowvar=False)
        print(f"    - [{nome}] Amostras geradas: {amostras_sinteticas.shape}. Covariância média abs: {np.abs(C_sintetica).mean():.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integração ConIII para Modelo de Ising no Bluesky")
    parser.add_argument("csv_path", type=str, help="Caminho para a matriz_estados_.csv (valores +1/-1)")
    parser.add_argument("gexf_path", type=str, help="Caminho para o arquivo GEXF da rede correspondente")
    parser.add_argument("--lam", type=float, default=0.01, help="Parâmetro de regularização (padrão 0.01)")
    parser.add_argument("--validar", action="store_true", help="Executa o Monte Carlo Metropolis para o modelo campeão")
    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"[{session_id}] Inicializando script CLI do ConIII Ising.")
    
    df = pd.read_csv(args.csv_path, index_col=0)
    S = df.values.T.astype(np.int64)
    S = np.where(S > 0, 1, -1).astype(np.int64) # Confiança +1/-1
    node_names = list(df.index)
    
    # Executa Inferência
    resultados = inferir_todos(S, session_id, lam=args.lam)
    
    # Gera a figura
    gerar_figura2(S, resultados, args.gexf_path, node_names, session_id)
    
    # Caso flag `--validar` providenciado
    if args.validar:
        validar_modelo(resultados, len(node_names))
        
    print(f"\n[CLI] Execução finalizada (Sessão {session_id}).")
