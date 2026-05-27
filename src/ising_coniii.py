"""
Integração do ConIII no Pipeline Bluesky para Inferência do Modelo de Ising.

Este módulo implementa a inferência dos parâmetros (campos h e acoplamentos J)
com suporte a pseudo-likelihood regularizada e à máscara topológica A_ij.

Inclui também a avaliação qualitativa do ajuste segundo Schneidman et al. (2006)
(RMSE das médias < 3/sqrt(R)), visualizações em heatmaps com máscaras topológicas
via networkx e geração de amostras via Metropolis para validação.
"""

import os
import sys
import time
import io
import argparse
import multiprocessing as mp
from datetime import datetime
from importlib import import_module

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

import scipy
import numpy as np
from scipy.optimize import minimize
# Patch agressivo para retrocompatibilidade total:
# Versões antigas da biblioteca SciPy exportavam TUDO do NumPy nativamente.
# Para evitar erros sequenciais (ex: exp, log, cosh, zeros, array), mapeamos 
# iterativamente todo o numpy para dentro das assinatures do scipy.
for attr in dir(np):
    if not attr.startswith('_') and not hasattr(scipy, attr):
        setattr(scipy, attr, getattr(np, attr))

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


def pack_J(J_mat: np.ndarray) -> np.ndarray:
    """
    Empacota uma matriz J simetrica (N x N) no formato ConIII: pares i < j.
    """
    idx_upper = np.triu_indices(J_mat.shape[0], k=1)
    return J_mat[idx_upper]


def preparar_mascara_adjacencia(adjacency_mask: np.ndarray | None, N: int) -> np.ndarray | None:
    """
    Valida e simetriza a mascara topologica A_ij usada para restringir J_ij.
    """
    if adjacency_mask is None:
        return None

    A = np.asarray(adjacency_mask, dtype=bool)
    if A.shape != (N, N):
        raise ValueError(f"A mascara A_ij tem shape {A.shape}, mas a inferencia espera {(N, N)}.")

    A = np.logical_or(A, A.T)
    np.fill_diagonal(A, False)
    return A


def construir_mascara_adjacencia(gexf_path: str, node_names: list[str]) -> np.ndarray:
    """
    Constroi A_ij a partir do GEXF, na mesma ordem de usuarios da matriz Ising.

    A entrada do Ising fica H(s) = sum_i h_i s_i + sum_{i<j} A_ij J_ij s_i s_j.
    Portanto, qualquer par sem aresta no grafo recebe J_ij = 0 durante a inferencia.
    """
    N = len(node_names)
    A = np.zeros((N, N), dtype=bool)
    if N == 0:
        return A

    G = nx.read_gexf(gexf_path)
    idx = {str(name): i for i, name in enumerate(node_names)}

    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su not in idx or sv not in idx or su == sv:
            continue
        i, j = idx[su], idx[sv]
        A[i, j] = True
        A[j, i] = True

    n_edges = int(np.triu(A, k=1).sum())
    dense_edges = N * (N - 1) // 2
    zero_degree = int(np.sum(A.sum(axis=1) == 0))

    print(
        f"[A_ij] Mascara topologica: {n_edges}/{dense_edges} pares com aresta "
        f"({N} usuarios filtrados)."
    )
    if zero_degree > 0:
        print(f"[A_ij] Aviso: {zero_degree} usuario(s) ficaram com grau zero apos os filtros.")
    if n_edges == 0 and N > 1:
        print("[A_ij] Aviso: nenhuma aresta encontrada para os usuarios filtrados; o modelo vira independente.")

    return A


def coniii_enumerate_disponivel(N: int) -> bool:
    """
    O solver Enumerate do ConIII depende de modulos simbolicos precompilados
    coniii.ising_eqn.ising_eqn_{N}_sym. Quando o modulo nao existe, o modelo
    Ising nao expoe calc_observables e o Enumerate nao consegue resolver.
    """
    try:
        import_module(f"coniii.ising_eqn.ising_eqn_{N}_sym")
        return True
    except ModuleNotFoundError:
        return False


def diagnosticar_matriz_inferencia(spin_matrix: np.ndarray, adjacency_mask: np.ndarray | None = None) -> dict:
    """
    Imprime diagnosticos que afetam a convergencia da inferencia de h e J.
    Entrada esperada: (R, N) = (keywords/amostras, usuarios/spins).
    """
    R, N = spin_matrix.shape
    A = preparar_mascara_adjacencia(adjacency_mask, N)
    dense_pairs = N * (N - 1) // 2
    n_edges = dense_pairs if A is None else int(np.triu(A, k=1).sum())
    n_params = N + n_edges
    ratio = R / n_params if n_params else np.inf
    plus_frac = float(np.mean(spin_matrix > 0))
    user_activity = np.mean(spin_matrix > 0, axis=0)
    keyword_popularity = np.sum(spin_matrix > 0, axis=1)

    low_user = int(np.sum(user_activity < 0.10))
    high_user = int(np.sum(user_activity > 0.90))
    frozen_user = int(np.sum((user_activity < 0.05) | (user_activity > 0.95)))
    frozen_keywords = int(np.sum((keyword_popularity == 0) | (keyword_popularity == N)))

    print("\n  [Diagnóstico] Condicionamento da inferência:")
    print(f"    R={R} amostras | N={N} spins | parâmetros={n_params} | R/parâmetros={ratio:.2f}")
    if A is not None:
        print(f"    Acoplamentos A_ij: {n_edges}/{dense_pairs} pares permitidos pela rede.")
    print(f"    Fração global de +1: {plus_frac:.3f}")
    print(
        "    Atividade por usuário (+1): "
        f"min={user_activity.min():.3f}, mediana={np.median(user_activity):.3f}, max={user_activity.max():.3f}"
    )
    print(
        "    Usuários extremos: "
        f"<10%={low_user}, >90%={high_user}, quase congelados(<5% ou >95%)={frozen_user}"
    )
    print(
        "    Popularidade das keywords (#usuários): "
        f"min={keyword_popularity.min()}, mediana={np.median(keyword_popularity):.1f}, max={keyword_popularity.max()}"
    )
    if ratio < 5.0:
        print("    ⚠ R/parâmetros < 5: inferência exata/densa tende a ser mal condicionada.")
    if frozen_user > 0 or frozen_keywords > 0:
        print("    ⚠ Há spins/keywords quase congelados; h e J podem divergir ou bater nos bounds.")

    return {
        "R": R,
        "N": N,
        "n_params": n_params,
        "n_edges": n_edges,
        "dense_pairs": dense_pairs,
        "usa_Aij": A is not None,
        "sample_param_ratio": ratio,
        "plus_frac": plus_frac,
        "low_activity_users": low_user,
        "high_activity_users": high_user,
        "frozen_users": frozen_user,
        "frozen_keywords": frozen_keywords,
    }


def resolver_pseudo_l2_coniii(sample: np.ndarray, initial_guess: np.ndarray,
                              lam: float, param_bound: float,
                              solver_kwargs: dict,
                              adjacency_mask: np.ndarray | None = None) -> tuple[np.ndarray, list]:
    """
    Resolve a pseudo-likelihood Ising com L2 real e suporte a A_ij.

    O retorno permanece no formato de multiplicadores do ConIII, mas a
    otimizacao e feita por subproblema logistico para permitir impor
    J_ij = 0 quando A_ij = 0.
    """
    sample = sample.astype(np.float64)
    R, n = sample.shape
    A = preparar_mascara_adjacencia(adjacency_mask, n)
    if A is None:
        A = np.ones((n, n), dtype=bool)
        np.fill_diagonal(A, False)

    if initial_guess is None:
        h0 = np.zeros(n, dtype=np.float64)
        J0 = np.zeros((n, n), dtype=np.float64)
    else:
        initial_guess = np.asarray(initial_guess, dtype=np.float64)
        h0 = initial_guess[:n]
        J0 = unpack_J(initial_guess[n:], n)
        J0[~A] = 0.0

    h = np.zeros(n, dtype=np.float64)
    J_directed = np.zeros((n, n), dtype=np.float64)
    soln = []

    for r in range(n):
        neighbors = np.where(A[r])[0]
        X = sample[:, neighbors]
        y = sample[:, r]

        theta0 = np.concatenate(([h0[r]], J0[r, neighbors]))
        theta0 = np.clip(theta0, -param_bound, param_bound)

        local_kwargs = dict(solver_kwargs or {})
        local_kwargs["bounds"] = [(-param_bound, param_bound)] * len(theta0)

        def f(theta):
            field = theta[0]
            if len(neighbors) > 0:
                field = field + X @ theta[1:]

            z = -2.0 * y * field
            nll = np.logaddexp(0.0, z).sum()
            coef = -2.0 * y * scipy.special.expit(z)

            grad = np.empty_like(theta)
            grad[0] = coef.sum()
            if len(neighbors) > 0:
                grad[1:] = X.T @ coef

            if lam > 0:
                nll += 0.5 * lam * np.dot(theta, theta)
                grad += lam * theta

            return nll, grad

        res = minimize(f, theta0, jac=True, **local_kwargs)
        soln.append(res)
        if not res.success:
            raise RuntimeError(f"Subproblema PL spin {r + 1}/{n} nao convergiu: {res.message}")

        this_multipliers = res.x
        if not np.all(np.isfinite(this_multipliers)):
            raise RuntimeError(f"Subproblema PL spin {r + 1}/{n} gerou NaN/Inf.")

        h[r] = this_multipliers[0]
        if len(neighbors) > 0:
            J_directed[r, neighbors] = this_multipliers[1:]

    Jmat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                Jmat[i, j] = 0.5 * (J_directed[i, j] + J_directed[j, i])
                Jmat[j, i] = Jmat[i, j]

    multipliers = np.concatenate((h, pack_J(Jmat)))
    multipliers = np.clip(multipliers, -param_bound, param_bound)
    return multipliers, soln

def calcular_rmse_medias(spin_matrix: np.ndarray, h_inferido: np.ndarray) -> float:
    """
    RMSE do modelo INDEPENDENTE: compara médias empíricas com tanh(h).
    Útil como baseline — se este RMSE é baixo, interações J são desnecessárias.
    """
    m_empirico = spin_matrix.mean(axis=0)
    m_previsto = np.tanh(h_inferido)
    return float(np.sqrt(np.mean((m_empirico - m_previsto)**2)))


def calcular_rmse_modelo_completo(spin_matrix: np.ndarray, multipliers: np.ndarray,
                                   n_amostras_mc: int = 50_000) -> float:
    """
    RMSE do modelo PAIRWISE COMPLETO: compara médias empíricas com as médias
    do modelo inferido.

    Para N ≤ 20: enumeração exata de todos os 2^N estados (garante RMSE real).
    Para N > 20: Monte Carlo Metropolis (campo médio TAP foi removido —
                 subestimava sistematicamente a qualidade do ajuste porque
                 ignora correlações entre spins no cálculo das médias).
    """
    R, N = spin_matrix.shape
    h = multipliers[:N]
    J_flat = multipliers[N:]
    J_mat = unpack_J(J_flat, N)

    m_empirico = spin_matrix.mean(axis=0).astype(np.float64)

    if N <= 20:
        # Enumeração exata — vetorizada para evitar loop Python lento
        # Gera todos os 2^N estados de uma vez via broadcasting
        n_states = 2 ** N
        idx_all = np.arange(n_states, dtype=np.int64)
        # bits[s, i] = bit i do estado s
        bits = ((idx_all[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float64)
        states = 2.0 * bits - 1.0                      # (2^N, N) em {-1,+1}
        # Energias vetorizadas: E(s) = -h·s - 0.5 s J s
        log_w = states @ h + 0.5 * np.einsum('si,ij,sj->s', states, J_mat, states)
        log_w -= log_w.max()                            # estabilidade numérica
        probs = np.exp(log_w)
        probs /= probs.sum()
        m_modelo = probs @ states                       # (N,)

    else:
        # Monte Carlo Metropolis — mais lento mas correto para N grande.
        # Campo médio TAP (h + J·m iterativo) ignorava flutuações e
        # produzia RMSE artificialmente alto para bons parâmetros.
        calc_e, _, _ = define_ising_helper_functions()
        sampler = coniii.samplers.Metropolis(N, multipliers, calc_e)
        burn_in = max(1000, N * 50)
        sampler.generate_sample_py(n_amostras_mc,
                                   n_iters=max(10, N),
                                   burn_in=burn_in)
        amostras = sampler.sample.astype(np.float64)    # (n_amostras_mc, N)
        m_modelo = amostras.mean(axis=0)

    return float(np.sqrt(np.mean((m_empirico - m_modelo) ** 2)))


def exibir_avaliacao_schneidman(rmse_indep: float, rmse_pairwise: float, R: int, metodo: str):
    """
    Exibe avaliação dual segundo Schneidman et al. (2006):
    - RMSE Independente: tanh(h) vs empírico (baseline)
    - RMSE Pairwise: modelo completo vs empírico (avaliação real)
    O ajuste pairwise é considerado aceitável se RMSE < 3/sqrt(R).
    """
    erro_amostral = 1.0 / np.sqrt(R)
    limiar = 3.0 * erro_amostral
    print(f"  [{metodo}] Avaliação Schneidman (2006):")
    print(f"    RMSE Indep. (tanh h): {rmse_indep:.5f}  ← baseline sem interações")
    print(f"    RMSE Pairwise (h+J):  {rmse_pairwise:.5f}  ← modelo completo")
    print(f"    Limiar (3/√R):        {limiar:.5f}  | Erro Amostral (1/√R): {erro_amostral:.5f}")
    if rmse_pairwise <= limiar:
        print(f"    ✔ Ajuste ACEITÁVEL pelo modelo pairwise.")
    else:
        print(f"    ⚠ AVISO: RMSE pairwise excedeu o limiar.")
    # Melhoria relativa
    if rmse_indep > 0:
        ganho = (1 - rmse_pairwise / rmse_indep) * 100
        print(f"    Ganho pairwise vs independente: {ganho:+.1f}%")




# ─────────────────────────────────────────────────────────────────────────────
# 2. INFERÊNCIA DE PARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────

def inferir_modelo(spin_matrix: np.ndarray, session_id: str, lam: float = 0.01,
                   adjacency_mask: np.ndarray | None = None) -> dict:
    # ── Orientação da matriz ─────────────────────────────────────────────────
    # create_ising_matrix_from_sets SEMPRE gera (users × keywords).
    # ConIII espera (R, N) = (keywords × users).
    # SEMPRE transpor — a heurística shape[0] < shape[1] falha quando
    # keywords < users (filtragem agressiva: 30 kw × 60 users).
    print(f"  [Matriz] Shape original: {spin_matrix.shape} (users × keywords)")
    spin_matrix = spin_matrix.T
    print(f"  [Matriz] Shape transposta: {spin_matrix.shape} (R=keywords × N=users)")

    R, N = spin_matrix.shape
    spin_matrix = spin_matrix.astype(np.int64)
    adjacency_mask = preparar_mascara_adjacencia(adjacency_mask, N)
    diagnostico = diagnosticar_matriz_inferencia(spin_matrix, adjacency_mask=adjacency_mask)
    n_params = diagnostico["n_params"]
    sample_param_ratio = diagnostico["sample_param_ratio"]

    limiar_schneidman = 3.0 / np.sqrt(R)

    t0 = time.time()
    metodo_str = None
    try:
        # Enumerate é exato e preferível até N=20 (2^20 = 1M estados, viável).
        # N=21 → 2^21 = 2M, N=25 → 33M → já fica lento.
        # Pseudo-Likelihood é usado para N > 20.
        enumerate_disponivel = coniii_enumerate_disponivel(N)
        usar_pseudo_por_amostragem = sample_param_ratio < 5.0
        usar_pseudo_por_topologia = adjacency_mask is not None
        if usar_pseudo_por_topologia:
            print("  [A_ij] Inferencia restrita a arestas do grafo; J_ij=0 fora de A_ij.")
        if N <= 30 and not enumerate_disponivel:
            print(f"  [Enumerate] Módulo simbólico do ConIII para N={N} não encontrado.")
            print("  [Fallback] Usando Pseudo-Likelihood do ConIII.")
        if N <= 30 and enumerate_disponivel and usar_pseudo_por_amostragem:
            print(f"  [Fallback] R/parâmetros={sample_param_ratio:.2f} < 5. Usando Pseudo-Likelihood regularizado.")
        if N <= 30 and enumerate_disponivel and usar_pseudo_por_topologia:
            print("  [Fallback] Enumerate denso nao impoe A_ij. Usando Pseudo-Likelihood topologico.")

        if N > 30 or not enumerate_disponivel or usar_pseudo_por_amostragem or usar_pseudo_por_topologia:
            metodo_str = "Pseudo-Likelihood"
            print(f"\n  [{metodo_str}] Iniciando inferência para N={N}, R={R}...")
            print(f"  [Regularização] λ = {lam} (L2 real na pseudo-likelihood)")
            
            # ── Pré-processamento: detectar spins degenerados ──
            medias = spin_matrix.mean(axis=0).astype(np.float64)
            spins_constantes = np.where(np.abs(medias) > 0.999)[0]
            if len(spins_constantes) > 0:
                print(f"  [Aviso] {len(spins_constantes)} spins quase constantes (|⟨σ⟩| > 0.999). "
                      f"Clamp aplicado para evitar saturação da sigmóide.")
            
            # ── Warm start: inicializa h_i com o modelo independente ──
            # e J_ij com a covariância empírica escalada.
            # J_init = 0 faz o gradiente inicial ser mínimo quando as correlações
            # são fracas, causando parada prematura do otimizador (1 iteração).
            # Inicializar com C_ij / (1 - <σ_i>²) dá ao L-BFGS-B um ponto de
            # partida informado e evita esse problema.
            medias_clamp = np.clip(medias, -0.98, 0.98)
            h_init = np.arctanh(medias_clamp)
            sigma_f = spin_matrix.astype(np.float64)
            C_emp = np.cov(sigma_f, rowvar=False)           # (N, N)
            var_i = np.maximum(1.0 - medias_clamp**2, 1e-6) # denominador estável
            # Escala: J_ij ≈ C_ij / sqrt(var_i * var_j)  (aprox. campo médio)
            scale = np.outer(np.sqrt(var_i), np.sqrt(var_i))
            J_init_mat = C_emp / scale
            np.fill_diagonal(J_init_mat, 0.0)
            if adjacency_mask is not None:
                J_init_mat[~adjacency_mask] = 0.0
            idx_upper = np.triu_indices(N, k=1)
            j_init = J_init_mat[idx_upper] * 0.1  # fator 0.1: conservador para evitar divergência
            initial_guess = np.concatenate([h_init, j_init])
            
            # ── Cadeia de fallback com múltiplos otimizadores ──
            # NOTA: cada sub-problema usa 1 + grau(i) parametros quando A_ij
            # esta ativo; por isso os bounds sao montados dentro do resolvedor.
            param_bound = 5.0  # limite conservador para |h_i| e |J_ij|
            
            _pl_strategies = [
                {"solver_kwargs": {"method": "L-BFGS-B", "options": {"maxiter": 500, "ftol": 1e-7, "gtol": 1e-5}}},
                {"solver_kwargs": {"method": "L-BFGS-B", "options": {"maxiter": 2000, "ftol": 1e-9, "gtol": 1e-6}}},
                {"solver_kwargs": {"method": "L-BFGS-B", "options": {"maxiter": 5000, "ftol": 1e-10, "gtol": 1e-7}}},
            ]
            
            multipliers = None
            for i, strategy in enumerate(_pl_strategies):
                try:
                    multipliers, soln = resolver_pseudo_l2_coniii(
                        spin_matrix,
                        initial_guess=initial_guess,
                        lam=lam,
                        param_bound=param_bound,
                        solver_kwargs=strategy["solver_kwargs"],
                        adjacency_mask=adjacency_mask
                    )
                    # Verificar se convergiu (parâmetros finitos e não-NaN)
                    if np.any(np.isnan(multipliers)) or np.any(np.isinf(multipliers)):
                        print(f"  [PL] Estratégia {i+1} produziu NaN/Inf, tentando próxima...")
                        multipliers = None
                        continue
                    
                    max_abs = np.max(np.abs(multipliers))
                    print(f"  [PL] Estratégia {i+1}/{len(_pl_strategies)} convergiu. "
                          f"|θ|_max = {max_abs:.3f}")
                    break
                except Exception as e:
                    print(f"  [PL] Estratégia {i+1} falhou ({type(e).__name__}: {e})")
            
            if multipliers is None:
                raise RuntimeError("Pseudo-Likelihood: todas as estratégias falharam.")

            # Hard clipping pós-otimização: impede parâmetros extremos
            # que causam overflow na exponencial do modelo de Ising.
            n_bound = int(np.sum(np.abs(multipliers) >= param_bound * 0.999))
            if n_bound > 0:
                print(f"  [PL] Aviso: {n_bound}/{len(multipliers)} parâmetro(s) encostaram no bound ±{param_bound:.1f}.")
        else:
            metodo_str = "Enumerate"
            print(f"\n  [{metodo_str}] Iniciando inferência exata para N={N}, R={R}...")
            
            # max_param_value limita |h_i| e |J_ij| para evitar overfitting
            # e melhorar a estabilidade em comunidades com poucas amostras.
            _max_param = 5.0
            print(f"  [Regularização] max_param_value = {_max_param:.0f}")
            solver = coniii.solvers.Enumerate(spin_matrix)
            
            # Cadeia de fallback robusta para o solver Enumerate:
            # O método padrão (krylov) falha com Jacobiano singular em dados reais.
            _solve_strategies = []
            if hasattr(getattr(solver, "model", None), "calc_observables"):
                _solve_strategies.extend([
                    {"use_root": True, "max_param_value": _max_param, "scipy_solver_kwargs": {"method": "krylov", "options": {"fatol": 1e-13, "xatol": 1e-13}}},
                    {"use_root": True, "max_param_value": _max_param, "scipy_solver_kwargs": {"method": "hybr"}},
                ])
            else:
                print("  [Enumerate] Solver ConIII sem calc_observables; pulando estratégias use_root=True.")
            _solve_strategies.append({"use_root": False, "max_param_value": _max_param})
            multipliers = None
            for i, strategy in enumerate(_solve_strategies):
                try:
                    multipliers = solver.solve(**strategy)
                    print(f"  [Enumerate] Convergiu com estratégia {i+1}/{len(_solve_strategies)}")
                    break
                except Exception as e:
                    print(f"  [Enumerate] Estratégia {i+1} falhou ({type(e).__name__}: {e}), tentando próxima...")
            if multipliers is None:
                raise RuntimeError("Todas as estratégias do Enumerate falharam.")
            
        h = multipliers[:N]
        j_flat = multipliers[N:]
        rmse_indep = calcular_rmse_medias(spin_matrix, h)
        rmse_pairwise = calcular_rmse_modelo_completo(spin_matrix, multipliers)
        tempo_s = time.time() - t0
        
        exibir_avaliacao_schneidman(rmse_indep, rmse_pairwise, R, metodo_str)
        
        resultado = {
            "h": h,
            "J": unpack_J(j_flat, N),
            "A_ij": adjacency_mask,
            "multipliers": multipliers,
            "metodo": metodo_str,
            "rmse_indep": rmse_indep,
            "rmse_pairwise": rmse_pairwise,
            "rmse_medias": rmse_pairwise,
            "tempo_s": tempo_s,
            "diagnostico": diagnostico
        }
        
        resultados = {metodo_str: resultado}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        tempo_s = time.time() - t0
        print(f"  [Erro] Falha: {e}")
        if metodo_str is None:
            metodo_str = "Pseudo-Likelihood" if N > 30 else "Enumerate"
        resultados = {
            metodo_str: {
                "metodo": metodo_str, 
                "ERROR": str(e), 
                "tempo_s": tempo_s
            }
        }
        
    # Salvar resultados e tabela comparativa
    tabela = []
    print("\n--- RESUMO DE INFERÊNCIA ---")
    print(f"{'Método':<20} | {'RMSE':<8} | {'Tempo(s)':<8} | {'Viável (Schneidman)':<20}")
    print("-" * 65)
    
    for nome, dados in resultados.items():
        if "ERROR" in dados:
            print(f"{nome:<20} | {'ERROR':<8} | {dados['tempo_s']:<8.1f} | {'N/A':<20}")
            tabela.append({
                "metodo": nome,
                "rmse_medias": np.nan,
                "tempo_s": dados['tempo_s'],
                "N": N,
                "R": R,
                "n_params": n_params,
                "n_edges": diagnostico.get("n_edges", np.nan),
                "dense_pairs": diagnostico.get("dense_pairs", np.nan),
                "usa_Aij": diagnostico.get("usa_Aij", False),
                "R_por_parametro": sample_param_ratio,
                "limiar_schneidman": limiar_schneidman,
                "viavel": False
            })
        else:
            rmse = dados['rmse_medias']
            viavel = rmse <= limiar_schneidman
            viab_str = "Sim" if viavel else "Não"
            print(f"{nome:<20} | {rmse:<8.5f} | {dados['tempo_s']:<8.1f} | {viab_str:<20}")
            
            tabela.append({
                "metodo": nome, "rmse_medias": rmse, "tempo_s": dados['tempo_s'], 
                "N": N,
                "R": R,
                "n_params": n_params,
                "n_edges": diagnostico.get("n_edges", np.nan),
                "dense_pairs": diagnostico.get("dense_pairs", np.nan),
                "usa_Aij": diagnostico.get("usa_Aij", False),
                "R_por_parametro": sample_param_ratio,
                "limiar_schneidman": limiar_schneidman,
                "viavel": viavel
            })
            
            # Salvar multiplicadores
            npy_path = f"multiplicadores_ising_{session_id}.npy"
            np.save(npy_path, dados['multipliers'])
            print(f"\n[Salvo] Multiplicadores salvos em: {npy_path}")
                
    df_comp = pd.DataFrame(tabela)
    csv_path = f"comparacao_metodos_{session_id}.csv"
    df_comp.to_csv(csv_path, index=False)
    
    return resultados

def gerar_figura2(spin_matrix: np.ndarray, resultados_inferencia: dict, 
                  gexf_path: str, node_names: list, session_id: str):
    """
    Painel Duplo: Covariância empírica e Acoplamentos J via MCH.
    Aplica filtro topológico substituindo as pontes desconectadas por NaN
    usando a rede GEXF original da comunidade correspondente aos nós (usuários).
    """
    print("\n[Figura 2] Inicializando geração dos heatmaps lado a lado...")
    # Mesma orientação da inferência: linhas = keywords/amostras, colunas = usuários/spins.
    spin_matrix = spin_matrix.T
    N = spin_matrix.shape[1]
    
    # Covariância (diagonal zerada: Cov(σ_i, σ_i) = variância, não é acoplamento)
    C_emp = np.cov(spin_matrix, rowvar=False)
    np.fill_diagonal(C_emp, 0.0)
    
    # Filtro topológico
    aplicar_filtro = False
    A_mask = np.ones((N, N), dtype=bool)
    
    if os.path.exists(gexf_path):
        G = nx.read_gexf(gexf_path)
        # Verificar quais node_names existem de fato no grafo
        nos_gexf = set(G.nodes())
        encontrados = [k for k in node_names if k in nos_gexf]
        ausentes = [k for k in node_names if k not in nos_gexf]
        if ausentes:
            print(f"  [Aviso] {len(ausentes)} nós da matriz ausentes no GEXF: {ausentes[:5]}{'...' if len(ausentes)>5 else ''}")
        if len(encontrados) < len(node_names) / 2:
            print("  [Aviso] Menos de 50% dos nós encontrados no GEXF.")
            print("  -> Filtro topológico DESABILITADO.")
        else:
            print(f"  [Filtro] GEXF válido ({len(encontrados)}/{len(node_names)} nós). Aplicando máscara...")
            aplicar_filtro = True
            # nodelist deve conter APENAS nós presentes em G para evitar KeyError.
            # Nós ausentes recebem linha/coluna de zeros (sem aresta), correto.
            nodelist_safe = [n if n in nos_gexf else None for n in node_names]
            # Constrói a máscara manualmente para suportar nós ausentes
            A_mask = np.zeros((N, N), dtype=bool)
            name_to_idx = {n: i for i, n in enumerate(node_names)}
            for u, v in G.edges():
                if u in name_to_idx and v in name_to_idx:
                    i_u, i_v = name_to_idx[u], name_to_idx[v]
                    A_mask[i_u, i_v] = True
                    A_mask[i_v, i_u] = True
            np.fill_diagonal(A_mask, True)
    else:
        print(f"  [Aviso] GEXF '{gexf_path}' não encontrado. Filtro topológico desabilitado.")

    def aplicar_mascara(matriz):
        if not aplicar_filtro:
            return matriz
        m_copy = matriz.copy()
        m_copy[~A_mask] = np.nan
        return m_copy

    C_masked = aplicar_mascara(C_emp)
    
    # Extrai o primeiro resultado independente do nome do método
    res_vals = list(resultados_inferencia.values())[0] if len(resultados_inferencia) > 0 else {}
    J_inf = aplicar_mascara(res_vals.get("J", np.zeros((N, N))))

    # Configuração da figura matplotlib
    plt.rcParams.update({'text.color': '#cccccc', 'axes.labelcolor': '#cccccc',
                         'xtick.color': '#cccccc', 'ytick.color': '#cccccc'})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor='#1a1a2e')
    
    cmap = matplotlib.colormaps.get_cmap('jet').copy()
    cmap.set_bad(color='white')
    
    paineis = [
        (C_masked, "Covariância Empírica"),
        (J_inf, "J — Inferido")
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
# 3.5. SEÇÃO 12 — FIGURA 3 (Correlação de Tripletos)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Pearson robusto para casos com poucos pontos ou variancia zero.
    """
    from scipy.stats import pearsonr

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan, np.nan
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    r, p = pearsonr(x, y)
    return float(r), float(p)


def _calcular_pares(spin_matrix: np.ndarray, pair_indices: list[tuple[int, int]] | None = None):
    """
    Calcula covariancias centradas C_ij = <(s_i-mi)(s_j-mj)>.
    """
    from itertools import combinations

    R, N = spin_matrix.shape
    medias = spin_matrix.mean(axis=0)
    centrado = spin_matrix - medias

    if pair_indices is None:
        pair_indices = list(combinations(range(N), 2))

    values = np.zeros(len(pair_indices), dtype=np.float64)
    for t, (i, j) in enumerate(pair_indices):
        values[t] = np.mean(centrado[:, i] * centrado[:, j])

    return pair_indices, values


def _estados_modelo_exato(multipliers: np.ndarray, N: int):
    """
    Estados e probabilidades exatas do modelo Ising para N pequeno.
    """
    h = multipliers[:N]
    J_mat = unpack_J(multipliers[N:], N)

    n_states = 2 ** N
    idx_all = np.arange(n_states, dtype=np.int64)
    bits = ((idx_all[:, None] >> np.arange(N)[None, :]) & 1).astype(np.float64)
    states = 2.0 * bits - 1.0

    log_weights = states @ h + 0.5 * np.einsum('si,ij,sj->s', states, J_mat, states)
    log_weights -= log_weights.max()
    probs = np.exp(log_weights)
    probs /= probs.sum()

    return states, probs


def _pares_modelo_exato(multipliers: np.ndarray, N: int,
                        pair_indices: list[tuple[int, int]] | None = None):
    """
    Calcula C_ij do modelo por enumeracao exata.
    """
    from itertools import combinations

    states, probs = _estados_modelo_exato(multipliers, N)
    medias_modelo = probs @ states
    centrado = states - medias_modelo

    if pair_indices is None:
        pair_indices = list(combinations(range(N), 2))

    values = np.zeros(len(pair_indices), dtype=np.float64)
    for t, (i, j) in enumerate(pair_indices):
        values[t] = np.sum(probs * centrado[:, i] * centrado[:, j])

    return pair_indices, values


def _contar_tripletos_por_topologia(N: int, adjacency_mask: np.ndarray | None) -> dict:
    """
    Conta tripletos por numero de arestas internas em A_ij.
    """
    from itertools import combinations

    counts = {
        "todos": 0,
        "sem_arestas": 0,
        "uma_aresta": 0,
        "acoplados": 0,
        "conectados": 0,
        "triangulos": 0,
    }

    if adjacency_mask is None:
        counts["todos"] = N * (N - 1) * (N - 2) // 6
        return counts

    A = preparar_mascara_adjacencia(adjacency_mask, N)
    for i, j, k in combinations(range(N), 3):
        edge_count = int(A[i, j]) + int(A[i, k]) + int(A[j, k])
        counts["todos"] += 1
        if edge_count == 0:
            counts["sem_arestas"] += 1
        if edge_count == 1:
            counts["uma_aresta"] += 1
        if edge_count >= 1:
            counts["acoplados"] += 1
        if edge_count >= 2:
            counts["conectados"] += 1
        if edge_count == 3:
            counts["triangulos"] += 1

    return counts


def _selecionar_tripletos_por_Aij(N: int, adjacency_mask: np.ndarray | None,
                                  mode: str = "connected"):
    """
    Seleciona quais tripletos entram na Figura 3.

    Modos:
      - all: todos os tripletos
      - coupled: pelo menos uma aresta em A_ij
      - connected: pelo menos duas arestas, sem usuario isolado no trio
      - triangles: tres arestas em A_ij
    """
    from itertools import combinations

    all_triplets = list(combinations(range(N), 3))
    if adjacency_mask is None:
        return all_triplets, "all"

    A = preparar_mascara_adjacencia(adjacency_mask, N)
    mode_key = (mode or "connected").strip().lower()
    aliases = {
        "conectados": "connected",
        "connected": "connected",
        "acoplados": "coupled",
        "coupled": "coupled",
        "triangulos": "triangles",
        "triangles": "triangles",
        "triangle": "triangles",
        "todos": "all",
        "all": "all",
    }
    mode_key = aliases.get(mode_key, "connected")

    if mode_key == "all":
        return all_triplets, "all"

    selected = []
    for i, j, k in all_triplets:
        edge_count = int(A[i, j]) + int(A[i, k]) + int(A[j, k])
        if mode_key == "coupled" and edge_count >= 1:
            selected.append((i, j, k))
        elif mode_key == "connected" and edge_count >= 2:
            selected.append((i, j, k))
        elif mode_key == "triangles" and edge_count == 3:
            selected.append((i, j, k))

    if len(selected) >= 2:
        return selected, mode_key

    # Fallback conservador: se nao ha trios conectados suficientes, ainda evita
    # trios totalmente sem acoplamento usando pelo menos uma aresta.
    coupled = []
    for i, j, k in all_triplets:
        edge_count = int(A[i, j]) + int(A[i, k]) + int(A[j, k])
        if edge_count >= 1:
            coupled.append((i, j, k))
    if len(coupled) >= 2:
        return coupled, "coupled"

    return all_triplets, "all"


def _calcular_tripletos(spin_matrix: np.ndarray,
                        triplet_indices: list[tuple[int, int, int]] | None = None):
    """
    Calcula correlações de terceira ordem (tripletos) centradas:
    C_ijk = ⟨(σ_i - ⟨σ_i⟩)(σ_j - ⟨σ_j⟩)(σ_k - ⟨σ_k⟩)⟩
    
    Retorna:
        triplet_indices: lista de tuplas (i, j, k)
        triplet_values: array com os valores C_ijk
    """
    R, N = spin_matrix.shape
    medias = spin_matrix.mean(axis=0)
    centrado = spin_matrix - medias  # (R, N)

    if triplet_indices is None:
        from itertools import combinations
        triplet_indices = list(combinations(range(N), 3))

    triplet_values = np.zeros(len(triplet_indices), dtype=np.float64)
    
    for t, (i, j, k) in enumerate(triplet_indices):
        triplet_values[t] = np.mean(centrado[:, i] * centrado[:, j] * centrado[:, k])
    
    return triplet_indices, triplet_values


def _tripletos_modelo_exato(multipliers: np.ndarray, N: int,
                            triplet_indices: list[tuple[int, int, int]] | None = None):
    """
    Calcula correlações de tripletos EXATAS via enumeração da distribuição
    de Boltzmann para N ≤ 9 (2^N estados tratáveis).
    """
    if triplet_indices is None:
        from itertools import combinations
        triplet_indices = list(combinations(range(N), 3))

    states, probs = _estados_modelo_exato(multipliers, N)
    
    # Médias do modelo
    medias_modelo = (probs[:, None] * states).sum(axis=0)
    centrado = states - medias_modelo
    
    triplet_values = np.zeros(len(triplet_indices), dtype=np.float64)
    
    for t, (i, j, k) in enumerate(triplet_indices):
        triplet_values[t] = np.sum(probs * centrado[:, i] * centrado[:, j] * centrado[:, k])
    
    return triplet_indices, triplet_values


def _amostrar_modelo_metropolis(multipliers: np.ndarray, N: int, n_amostras: int,
                                label: str = "modelo") -> np.ndarray:
    """
    Amostra o modelo Ising inferido via Metropolis-Hastings.
    """
    calc_e, _, _ = define_ising_helper_functions()
    sampler = coniii.samplers.Metropolis(N, multipliers, calc_e)

    print(f"  [Metropolis] Gerando {n_amostras:,} amostras para {label} (N={N})...")
    sampler.generate_sample_py(n_amostras, n_iters=max(10, N), burn_in=max(1000, N * 100))
    return sampler.sample.astype(np.float64)


def _tripletos_modelo_mc(multipliers: np.ndarray, N: int,
                         triplet_indices: list[tuple[int, int, int]] | None = None,
                         n_amostras: int = 500000,
                         return_samples: bool = False):
    """
    Calcula correlações de tripletos via Metropolis-Hastings para N > 9.
    Gera um grande número de amostras sintéticas e calcula C_ijk sobre elas.
    """
    amostras = _amostrar_modelo_metropolis(multipliers, N, n_amostras, label="tripletos")
    trip_idx, trip_values = _calcular_tripletos(amostras, triplet_indices=triplet_indices)
    if return_samples:
        return trip_idx, trip_values, amostras
    return trip_idx, trip_values


def _bootstrap_tripletos(spin_matrix: np.ndarray,
                         triplet_indices: list[tuple[int, int, int]] | None = None,
                         n_bootstrap: int = 200):
    """
    Estima a incerteza das correlações de tripletos empíricas via Bootstrap.
    Hall & Bialek reportam erros tipicamente ~10% do valor medido.
    
    Retorna:
        triplet_errors: array com desvio padrão bootstrap de cada C_ijk
    """
    R, N = spin_matrix.shape
    if triplet_indices is None:
        from itertools import combinations
        triplet_indices = list(combinations(range(N), 3))
    n_triplets = len(triplet_indices)
    
    boot_values = np.zeros((n_bootstrap, n_triplets))
    
    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        idx_boot = rng.choice(R, size=R, replace=True)
        sample_boot = spin_matrix[idx_boot]
        _, vals = _calcular_tripletos(sample_boot, triplet_indices=triplet_indices)
        boot_values[b] = vals
    
    return np.std(boot_values, axis=0)


def _diagnosticar_pares_figura3(spin_matrix: np.ndarray, multipliers: np.ndarray,
                                N: int, adjacency_mask: np.ndarray | None,
                                model_samples: np.ndarray | None = None):
    """
    Diagnostica se o modelo reproduz C_ij antes de interpretar tripletos.
    """
    pair_idx, c_emp = _calcular_pares(spin_matrix.astype(np.float64))
    if model_samples is not None:
        _, c_modelo = _calcular_pares(model_samples.astype(np.float64), pair_indices=pair_idx)
    else:
        _, c_modelo = _pares_modelo_exato(multipliers, N, pair_indices=pair_idx)

    print("\n  [Diagnostico pares] C_ij empirico vs modelo:")

    groups = [("todos", np.ones(len(pair_idx), dtype=bool))]
    if adjacency_mask is not None:
        A = preparar_mascara_adjacencia(adjacency_mask, N)
        pair_mask = np.array([A[i, j] for i, j in pair_idx], dtype=bool)
        groups.append(("A_ij=1", pair_mask))
        groups.append(("A_ij=0", ~pair_mask))

    for label, mask in groups:
        n = int(mask.sum())
        if n < 2:
            print(f"    {label:<8}: n={n} pares (insuficiente para Pearson)")
            continue
        r, p = _safe_pearson(c_emp[mask], c_modelo[mask])
        rmse = float(np.sqrt(np.mean((c_emp[mask] - c_modelo[mask]) ** 2)))
        if np.isnan(r):
            print(f"    {label:<8}: n={n} pares | RMSE={rmse:.5f} | Pearson indisponivel")
        else:
            print(f"    {label:<8}: n={n} pares | RMSE={rmse:.5f} | Pearson r={r:.4f} (p={p:.2e})")


def gerar_figura3(spin_matrix: np.ndarray, resultados_inferencia: dict,
                  session_id: str,
                  adjacency_mask: np.ndarray | None = None,
                  triplet_mode: str = "connected",
                  filename_suffix: str | None = None,
                  model_samples: np.ndarray | None = None,
                  diagnose_pairs: bool = True,
                  n_amostras_mc: int = 500000):
    """
    Figura 3: Correlação de Tripletos (Schneidman / Hall & Bialek).
    
    Scatter plot comparando C_ijk empírico vs C_ijk previsto pelo modelo
    de máxima entropia pairwise, com barras de erro bootstrap e coeficiente
    de Pearson.
    """
    # Mesma orientação da inferência: linhas = keywords/amostras, colunas = usuários/spins.
    spin_matrix = spin_matrix.T
    R, N = spin_matrix.shape
    print(f"\n[Figura 3] Correlação de Tripletos (N={N}, R={R})...")
    
    # Extrair multiplicadores do resultado
    res_vals = list(resultados_inferencia.values())[0] if resultados_inferencia else {}
    if "multipliers" not in res_vals or "ERROR" in res_vals:
        print("  [Erro] Nenhum modelo válido encontrado. Pulando Figura 3.")
        return None
    
    multipliers = res_vals["multipliers"]
    if adjacency_mask is None:
        adjacency_mask = res_vals.get("A_ij")
    adjacency_mask = preparar_mascara_adjacencia(adjacency_mask, N) if adjacency_mask is not None else None

    topologia_counts = _contar_tripletos_por_topologia(N, adjacency_mask)
    triplet_indices, triplet_mode_used = _selecionar_tripletos_por_Aij(
        N,
        adjacency_mask,
        mode=triplet_mode
    )
    print(
        "  [Tripletos A_ij] "
        f"todos={topologia_counts['todos']}, "
        f"sem_arestas={topologia_counts['sem_arestas']}, "
        f"uma_aresta={topologia_counts['uma_aresta']}, "
        f"conectados={topologia_counts['conectados']}, "
        f"triangulos={topologia_counts['triangulos']}"
    )
    print(f"  [Tripletos A_ij] Modo usado na Figura 3: {triplet_mode_used}")

    if len(triplet_indices) < 2:
        print("  [Erro] Menos de 2 tripletos selecionados. Pulando Figura 3.")
        return None
    
    # 1. Tripletos empíricos
    print("  [1/3] Calculando tripletos empíricos...")
    trip_idx, c_emp = _calcular_tripletos(
        spin_matrix.astype(np.float64),
        triplet_indices=triplet_indices
    )
    n_triplets = len(trip_idx)
    print(f"        {n_triplets} tripletos selecionados")
    
    # 2. Tripletos do modelo
    print("  [2/3] Calculando tripletos do modelo...")
    if N <= 20:
        print(f"        Modo EXATO (2^{N} = {2**N} estados)")
        _, c_modelo = _tripletos_modelo_exato(
            multipliers,
            N,
            triplet_indices=triplet_indices
        )
    else:
        print(f"        Modo MONTE CARLO (N={N} > 20)")
        if model_samples is not None:
            print(f"  [Metropolis] Reutilizando {model_samples.shape[0]:,} amostras do modelo.")
            _, c_modelo = _calcular_tripletos(
                model_samples.astype(np.float64),
                triplet_indices=triplet_indices
            )
        else:
            _, c_modelo, model_samples = _tripletos_modelo_mc(
                multipliers,
                N,
                triplet_indices=triplet_indices,
                n_amostras=n_amostras_mc,
                return_samples=True
            )

    if diagnose_pairs:
        _diagnosticar_pares_figura3(
            spin_matrix.astype(np.float64),
            multipliers,
            N,
            adjacency_mask,
            model_samples=model_samples
        )
    
    # 3. Bootstrap (incerteza)
    print("  [3/3] Estimando incerteza via Bootstrap (200 reamostras)...")
    erros = _bootstrap_tripletos(
        spin_matrix.astype(np.float64),
        triplet_indices=triplet_indices
    )
    
    # 4. Estatísticas
    r_pearson, p_value = _safe_pearson(c_emp, c_modelo)
    diff = c_modelo - c_emp
    rmse_triplet = float(np.sqrt(np.mean(diff ** 2)))
    mae_triplet = float(np.mean(np.abs(diff)))
    bias_triplet = float(np.mean(diff))
    std_emp = float(np.std(c_emp))
    std_modelo = float(np.std(c_modelo))
    mean_abs_emp = float(np.mean(np.abs(c_emp)))
    mean_abs_modelo = float(np.mean(np.abs(c_modelo)))
    rmse_bootstrap = float(np.sqrt(np.mean(erros ** 2))) if len(erros) else np.nan
    ratio_rmse_bootstrap = (
        rmse_triplet / rmse_bootstrap
        if np.isfinite(rmse_bootstrap) and rmse_bootstrap > 0
        else np.nan
    )
    valid_sigma = np.isfinite(erros) & (erros > 0)
    frac_within_1sigma = (
        float(np.mean(np.abs(diff[valid_sigma]) <= erros[valid_sigma]))
        if np.any(valid_sigma)
        else np.nan
    )

    if np.isnan(r_pearson):
        print("\n  [Resultado] Pearson indisponivel: variancia zero ou poucos tripletos.")
    else:
        print(f"\n  [Resultado] Pearson r = {r_pearson:.4f} (p = {p_value:.2e})")
    print("  [Metricas tripletos]")
    print(f"    RMSE={rmse_triplet:.5f} | MAE={mae_triplet:.5f} | bias(modelo-emp)={bias_triplet:+.5f}")
    print(f"    std(C_emp)={std_emp:.5f} | std(C_modelo)={std_modelo:.5f}")
    print(f"    mean|C_emp|={mean_abs_emp:.5f} | mean|C_modelo|={mean_abs_modelo:.5f}")
    if np.isfinite(ratio_rmse_bootstrap):
        print(
            f"    RMSE/erro_bootstrap={ratio_rmse_bootstrap:.2f} | "
            f"frac. dentro de 1σ={frac_within_1sigma:.3f}"
        )
    else:
        print("    RMSE/erro_bootstrap indisponivel.")
    print("  [Referencia] Hall & Bialek: r em [0.93, 0.95] para o conjunto analisado no artigo.")
    if adjacency_mask is not None and triplet_mode_used != "all":
        print("  [Nota] Aqui a Figura 3 usa apenas tripletos acoplados/conectados por A_ij.")
    if not np.isnan(r_pearson) and r_pearson >= 0.90:
        print(f"  ✔ Correlação de tripletos dentro do esperado!")
    else:
        print(f"  ⚠ Correlação abaixo do esperado. Possíveis causas: poucos dados ou correlações de alta ordem.")
    
    # 5. Gerar gráfico (estilo idêntico à Figura 2)
    plt.rcParams.update({'text.color': '#cccccc', 'axes.labelcolor': '#cccccc',
                         'xtick.color': '#cccccc', 'ytick.color': '#cccccc'})
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Escala automática baseada nos dados
    all_vals = np.concatenate([c_emp, c_modelo])
    v_max = max(np.abs(all_vals).max() * 1.3, 0.01)
    
    # Colormap jet (igual à Figura 2) — pontos coloridos pela magnitude empírica
    cmap = matplotlib.colormaps.get_cmap('jet').copy()
    norm = plt.Normalize(vmin=-v_max, vmax=v_max)
    
    # Linha diagonal y = x (previsão perfeita)
    ax.plot([-v_max, v_max], [-v_max, v_max], '--', color='#666666',
            linewidth=1, alpha=0.8, zorder=1)
    
    # Barras de erro
    ax.errorbar(
        c_emp, c_modelo,
        xerr=erros, yerr=erros,
        fmt='none', ecolor='#555555',
        elinewidth=0.6, capsize=1.5, alpha=0.4, zorder=2
    )
    
    # Scatter colorido pelo valor empírico
    sc = ax.scatter(
        c_emp, c_modelo,
        c=c_emp, cmap=cmap, norm=norm,
        s=30, edgecolors='#222222', linewidths=0.3,
        zorder=3
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    
    # Labels e título (mesmo estilo da Figura 2)
    r_label = "nan" if np.isnan(r_pearson) else f"{r_pearson:.3f}"
    ax.set_title(
        f"Tripletos {triplet_mode_used}  (Pearson $r$ = {r_label})",
        color='#cccccc', pad=10
    )
    ax.set_xlabel(r'$C_{ijk}^{\,\mathrm{emp}}$')
    ax.set_ylabel(r'$C_{ijk}^{\,\mathrm{pred}}$')
    
    # Limites simétricos
    ax.set_xlim(-v_max, v_max)
    ax.set_ylim(-v_max, v_max)
    ax.set_aspect('equal')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    
    plt.tight_layout()
    suffix = ""
    if filename_suffix:
        suffix_clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(filename_suffix))
        suffix = f"_{suffix_clean}"
    fig_path = f"figura3_tripletos{suffix}_{session_id}.png"
    plt.savefig(fig_path, dpi=300, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    
    print(f"\n[Figura 3] Scatter plot salvo em: {fig_path}")
    
    return {
        "pearson_r": r_pearson,
        "p_value": p_value,
        "rmse_triplet": rmse_triplet,
        "mae_triplet": mae_triplet,
        "bias_triplet": bias_triplet,
        "std_emp": std_emp,
        "std_modelo": std_modelo,
        "mean_abs_emp": mean_abs_emp,
        "mean_abs_modelo": mean_abs_modelo,
        "rmse_bootstrap": rmse_bootstrap,
        "ratio_rmse_bootstrap": ratio_rmse_bootstrap,
        "frac_within_1sigma": frac_within_1sigma,
        "n_triplets": n_triplets,
        "triplet_mode_requested": triplet_mode,
        "triplet_mode": triplet_mode_used,
        "triplet_topology_counts": topologia_counts,
        "fig_path": fig_path,
    }


def gerar_figura3_multimodo(spin_matrix: np.ndarray, resultados_inferencia: dict,
                            session_id: str,
                            adjacency_mask: np.ndarray | None = None,
                            triplet_modes: list[str] | tuple[str, ...] = ("all", "coupled", "connected", "triangles"),
                            n_amostras_mc: int = 500000):
    """
    Roda a Figura 3 em varios filtros topologicos de tripletos e salva um resumo.
    """
    # Prepara uma unica amostra Monte Carlo para todos os modos, quando necessario.
    spin_matrix_model = spin_matrix.T
    _, N = spin_matrix_model.shape
    res_vals = list(resultados_inferencia.values())[0] if resultados_inferencia else {}

    model_samples = None
    if N > 20 and "multipliers" in res_vals and "ERROR" not in res_vals:
        print(
            "\n[Figura 3] Gerando uma unica amostra Monte Carlo para comparar "
            f"modos {list(triplet_modes)}..."
        )
        model_samples = _amostrar_modelo_metropolis(
            res_vals["multipliers"],
            N,
            n_amostras_mc,
            label="tripletos multimodo"
        )

    resultados = []
    for i, mode in enumerate(triplet_modes):
        out = gerar_figura3(
            spin_matrix=spin_matrix,
            resultados_inferencia=resultados_inferencia,
            session_id=session_id,
            adjacency_mask=adjacency_mask,
            triplet_mode=mode,
            filename_suffix=mode,
            model_samples=model_samples,
            diagnose_pairs=(i == 0),
            n_amostras_mc=n_amostras_mc
        )
        if out is None:
            continue

        row = {k: v for k, v in out.items() if k != "triplet_topology_counts"}
        for key, value in out.get("triplet_topology_counts", {}).items():
            row[f"topologia_{key}"] = value
        resultados.append(row)

    if resultados:
        resumo_path = f"figura3_tripletos_resumo_{session_id}.csv"
        pd.DataFrame(resultados).to_csv(resumo_path, index=False)
        print(f"\n[Figura 3] Resumo multimodo salvo em: {resumo_path}")

    return resultados


# ─────────────────────────────────────────────────────────────────────────────
# 3.6. SEÇÃO 13 — FIGURA 4 (Distribuição de Atividade Coletiva Q)
# ─────────────────────────────────────────────────────────────────────────────

def _calcular_Q(spin_matrix: np.ndarray) -> np.ndarray:
    """
    Calcula a fração de usuários ativos Q para cada snapshot (linha).
    Q = 1/2 + (1/2N) * Σ σ_i
    
    Como σ ∈ {-1, +1}, Q varia de 0 (todos -1) a 1 (todos +1).
    """
    N = spin_matrix.shape[1]
    return 0.5 + spin_matrix.sum(axis=1) / (2.0 * N)


def _distribuicao_Q_independente(spin_matrix: np.ndarray, n_amostras: int = 100000):
    """
    Gera amostras do modelo INDEPENDENTE (J_ij = 0) e calcula Q.
    Os campos locais são: h_i = arctanh(⟨σ_i⟩_obs).
    """
    R, N = spin_matrix.shape
    medias = spin_matrix.mean(axis=0).astype(np.float64)
    # Clamp para evitar arctanh(±1) = ±inf
    medias = np.clip(medias, -0.9999, 0.9999)
    h_indep = np.arctanh(medias)
    
    # Probabilidade de σ_i = +1 no modelo independente: p_i = (1 + tanh(h_i)) / 2
    p_plus = (1.0 + np.tanh(h_indep)) / 2.0
    
    rng = np.random.default_rng(42)
    amostras = np.where(rng.random((n_amostras, N)) < p_plus, 1.0, -1.0)
    return _calcular_Q(amostras)


def _distribuicao_Q_pairwise_exato(multipliers: np.ndarray, N: int):
    """
    Para N ≤ 9: calcula a distribuição EXATA de Q a partir de P(s) = exp(-E(s))/Z.
    Retorna arrays (q_values, probabilities) para todos os 2^N estados.
    """
    h = multipliers[:N]
    J_flat = multipliers[N:]
    J_mat = unpack_J(J_flat, N)
    
    n_states = 2 ** N
    log_weights = np.zeros(n_states)
    q_values = np.zeros(n_states)
    
    for idx in range(n_states):
        bits = np.array([(idx >> k) & 1 for k in range(N)], dtype=np.float64)
        s = 2.0 * bits - 1.0
        energy = -np.dot(h, s) - 0.5 * s @ J_mat @ s
        log_weights[idx] = -energy
        q_values[idx] = 0.5 + s.sum() / (2.0 * N)
    
    log_Z = np.max(log_weights) + np.log(np.sum(np.exp(log_weights - np.max(log_weights))))
    probs = np.exp(log_weights - log_Z)
    
    return q_values, probs


def _distribuicao_Q_pairwise_mc(multipliers: np.ndarray, N: int, n_amostras: int = 100000):
    """
    Para N > 9: gera amostras via Metropolis-Hastings e calcula Q.
    """
    calc_e, _, _ = define_ising_helper_functions()
    
    sampler = coniii.samplers.Metropolis(N, multipliers, calc_e)
    
    print(f"  [Metropolis] Gerando {n_amostras:,} amostras para distribuição Q (N={N})...")
    sampler.generate_sample_py(n_amostras, n_iters=max(10, N), burn_in=max(1000, N * 100))
    amostras = sampler.sample.astype(np.float64)
    
    return _calcular_Q(amostras)


def gerar_figura4(spin_matrix: np.ndarray, resultados_inferencia: dict,
                  session_id: str):
    """
    Figura 4: Distribuição de Atividade Coletiva P(Q).
    
    Compara três curvas de densidade:
      - Empírica (Vermelho): histograma/KDE dos dados reais
      - Independente (Ciano): modelo sem interações (J=0)
      - Pairwise/Ising (Verde): modelo completo inferido
    
    Eixo Y em escala logarítmica para visualizar caudas.
    Inclui cálculo da suscetibilidade χ = N * Var(Q).
    """
    from scipy.stats import gaussian_kde
    
    # Mesma orientação da inferência: linhas = keywords/amostras, colunas = usuários/spins.
    spin_matrix = spin_matrix.T
    R, N = spin_matrix.shape
    print(f"\n[Figura 4] Distribuição de Atividade Coletiva Q (N={N}, R={R})...")
    
    # Extrair multiplicadores do resultado
    res_vals = list(resultados_inferencia.values())[0] if resultados_inferencia else {}
    if "multipliers" not in res_vals or "ERROR" in res_vals:
        print("  [Erro] Nenhum modelo válido encontrado. Pulando Figura 4.")
        return None
    
    multipliers = res_vals["multipliers"]
    
    # 1. Q empírico
    print("  [1/3] Calculando Q empírico...")
    Q_emp = _calcular_Q(spin_matrix.astype(np.float64))
    
    # 2. Q do modelo independente
    print("  [2/3] Gerando distribuição do modelo independente (J=0)...")
    Q_indep = _distribuicao_Q_independente(spin_matrix.astype(np.float64))
    
    # 3. Q do modelo pairwise
    print("  [3/3] Gerando distribuição do modelo pairwise...")
    use_exact = (N <= 20)
    
    if use_exact:
        print(f"        Modo EXATO (2^{N} = {2**N} estados)")
        q_exact, p_exact = _distribuicao_Q_pairwise_exato(multipliers, N)
    else:
        print(f"        Modo MONTE CARLO (N={N} > 9)")
        Q_pairwise_mc = _distribuicao_Q_pairwise_mc(multipliers, N)
    
    # 4. Estatísticas de suscetibilidade
    chi_emp = N * np.var(Q_emp)
    chi_indep = N * np.var(Q_indep)
    print(f"\n  [Suscetibilidade] χ_emp = {chi_emp:.4f} | χ_indep = {chi_indep:.4f}")
    print(f"  [Razão] χ_emp / χ_indep = {chi_emp / max(chi_indep, 1e-10):.2f}x")
    if chi_emp > 2 * chi_indep:
        print(f"  ✔ Comportamento COLETIVO detectado (χ_emp >> χ_indep).")
    else:
        print(f"  ⚠ Comportamento coletivo fraco.")
    
    # 5. Gráfico (estilo idêntico à Figura 2)
    plt.rcParams.update({'text.color': '#cccccc', 'axes.labelcolor': '#cccccc',
                         'xtick.color': '#cccccc', 'ytick.color': '#cccccc'})
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    
    # Bins para Q (valores discretos possíveis: k/N para k=0..N)
    q_bins = np.linspace(0, 1, min(50, N + 2))
    q_smooth = np.linspace(0, 1, 300)
    
    # --- Curva 1: Empírica (Vermelho) ---
    if len(np.unique(Q_emp)) > 2:
        kde_emp = gaussian_kde(Q_emp, bw_method=0.05)
        ax.plot(q_smooth, kde_emp(q_smooth), color='#ff4444', linewidth=2,
                label='Empírico', zorder=3)
        ax.fill_between(q_smooth, kde_emp(q_smooth), alpha=0.15, color='#ff4444', zorder=2)
    else:
        ax.hist(Q_emp, bins=q_bins, density=True, color='#ff4444',
                alpha=0.5, label='Empírico', zorder=2)
    
    # --- Curva 2: Independente (Ciano) ---
    kde_indep = gaussian_kde(Q_indep, bw_method=0.05)
    ax.plot(q_smooth, kde_indep(q_smooth), color='#00cccc', linewidth=2,
            linestyle='--', label='Independente ($J=0$)', zorder=3)
    
    # --- Curva 3: Pairwise/Ising (Verde) ---
    if use_exact:
        # Para enumeração exata: acumula probabilidades por valor Q discreto
        q_unique = np.unique(q_exact)
        p_by_q = np.array([p_exact[q_exact == q].sum() for q in q_unique])
        # Normaliza para densidade (divide pela largura do bin)
        dq = 1.0 / N if N > 1 else 0.1
        ax.bar(q_unique, p_by_q / dq, width=dq * 0.8, color='#44cc44',
               alpha=0.6, label='Pairwise (Ising)', zorder=2, edgecolor='#228822')
    else:
        kde_pair = gaussian_kde(Q_pairwise_mc, bw_method=0.05)
        ax.plot(q_smooth, kde_pair(q_smooth), color='#44cc44', linewidth=2,
                label='Pairwise (Ising)', zorder=3)
        ax.fill_between(q_smooth, kde_pair(q_smooth), alpha=0.1, color='#44cc44', zorder=1)
    
    # Escala logarítmica no eixo Y
    ax.set_yscale('log')
    ax.set_xlim(0, 1)
    
    # Ajusta ylim inferior para evitar cortar dados
    y_min = max(ax.get_ylim()[0], 1e-4)
    ax.set_ylim(bottom=y_min)
    
    # Labels e título (mesmo estilo da Figura 2)
    ax.set_title(
        f"Distribuição de Atividade Coletiva $P(Q)$",
        color='#cccccc', pad=10
    )
    ax.set_xlabel("Fração de Usuários Ativos $Q$")
    ax.set_ylabel("$P(Q)$")
    
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444444',
              labelcolor='#cccccc', loc='upper right')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')
    
    plt.tight_layout()
    fig_path = f"figura4_distribuicao_Q_{session_id}.png"
    plt.savefig(fig_path, dpi=300, facecolor='#1a1a2e', bbox_inches='tight')
    plt.close()
    
    print(f"\n[Figura 4] Gráfico salvo em: {fig_path}")
    
    return {
        "chi_emp": chi_emp,
        "chi_indep": chi_indep,
        "chi_ratio": chi_emp / max(chi_indep, 1e-10),
        "fig_path": fig_path,
    }

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
    # NÃO transpor: inferir_modelo() transpõe internamente
    S = df.values.astype(np.int64)
    S = np.where(S > 0, 1, -1).astype(np.int64)  # Confiança +1/-1
    node_names = list(df.index)
    
    # Executa Inferência
    resultados = inferir_modelo(S, session_id, lam=args.lam)
    
    # Gera a figura
    gerar_figura2(S, resultados, args.gexf_path, node_names, session_id)
    
    # Caso flag `--validar` providenciado
    if args.validar:
        validar_modelo(resultados, len(node_names))
        
    print(f"\n[CLI] Execução finalizada (Sessão {session_id}).")
