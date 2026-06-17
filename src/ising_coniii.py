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
import glob
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


def remover_arquivos_temporarios_npy(paths, label="arquivos temporários"):
    """
    Remove arquivos NPY temporários gerados durante a inferência.
    """
    removidos = 0
    for path in sorted(set(paths)):
        if not path or not os.path.exists(path):
            continue
        try:
            os.remove(path)
            removidos += 1
        except OSError as e:
            print(f"  [Limpeza] Aviso: não foi possível remover {path}: {e}")
    if removidos:
        print(f"  [Limpeza] {removidos} {label} removido(s).")
    return removidos

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


def normalizar_metodo_inferencia(metodo: str | None) -> str:
    """
    Normaliza a escolha interativa do metodo de inferencia.
    """
    metodo = (metodo or "auto").strip().lower()
    aliases = {
        "0": "auto",
        "a": "auto",
        "auto": "auto",
        "automatico": "auto",
        "automático": "auto",
        "1": "mch",
        "m": "mch",
        "mch": "mch",
        "monte carlo histogram": "mch",
        "monte-carlo-histogram": "mch",
        "2": "mch_custom",
        "mc": "mch_custom",
        "custom": "mch_custom",
        "custom_mch": "mch_custom",
        "mch_custom": "mch_custom",
        "mch-custom": "mch_custom",
        "mch custom": "mch_custom",
        "3": "exact",
        "e": "exact",
        "exact": "exact",
        "exato": "exact",
        "enumerate": "exact",
        "enumerate-exato": "exact",
    }
    if metodo not in aliases:
        raise ValueError("Metodo de inferencia invalido. Use 'auto', 'mch', 'mch_custom' ou 'exact'.")
    return aliases[metodo]


MCH_LEARNING_PROFILES = {
    "very_aggressive": {
        "label": "Muito agressivo",
        "learn": {"maxdlamda": 0.80, "maxdlamdaNorm": 3.0, "eta": 0.40, "maxLearningSteps": 30},
    },
    "aggressive": {
        "label": "Agressivo",
        "learn": {"maxdlamda": 0.50, "maxdlamdaNorm": 2.0, "eta": 0.30, "maxLearningSteps": 25},
    },
    "medium": {
        "label": "Medio",
        "learn": {"maxdlamda": 0.20, "maxdlamdaNorm": 1.0, "eta": 0.15, "maxLearningSteps": 30},
    },
    "conservative": {
        "label": "Conservador",
        "learn": {"maxdlamda": 0.08, "maxdlamdaNorm": 0.5, "eta": 0.05, "maxLearningSteps": 40},
    },
    "adaptive_samples": {
        "label": "Adaptado ao numero de amostras",
        "learn": None,
    },
}


def normalizar_perfil_mch(perfil: str | None) -> str:
    aliases = {
        "5": "very_aggressive",
        "ma": "very_aggressive",
        "muito agressivo": "very_aggressive",
        "muito agressiva": "very_aggressive",
        "muito_agressivo": "very_aggressive",
        "muito_agressiva": "very_aggressive",
        "muito-agressivo": "very_aggressive",
        "muito-agressiva": "very_aggressive",
        "very_aggressive": "very_aggressive",
        "very-aggressive": "very_aggressive",
        "1": "aggressive",
        "agressiva": "aggressive",
        "agressivo": "aggressive",
        "aggressive": "aggressive",
        "2": "medium",
        "media": "medium",
        "média": "medium",
        "medio": "medium",
        "médio": "medium",
        "medium": "medium",
        "3": "conservative",
        "conservador": "conservative",
        "conservadora": "conservative",
        "conservative": "conservative",
        "4": "adaptive_samples",
        "adaptado": "adaptive_samples",
        "adaptada": "adaptive_samples",
        "adaptive": "adaptive_samples",
        "adaptive_samples": "adaptive_samples",
        "amostras": "adaptive_samples",
        "sample": "adaptive_samples",
        "samples": "adaptive_samples",
    }
    key = str(perfil or "adaptive_samples").strip().lower()
    if not key:
        return "adaptive_samples"
    return aliases.get(key, "adaptive_samples")


def parametros_mch_por_perfil(perfil: str, sample_size_current: int) -> dict:
    perfil = normalizar_perfil_mch(perfil)
    if perfil != "adaptive_samples":
        return MCH_LEARNING_PROFILES[perfil]["learn"].copy()

    if sample_size_current < 200_000:
        return MCH_LEARNING_PROFILES["aggressive"]["learn"].copy()
    if sample_size_current < 500_000:
        return MCH_LEARNING_PROFILES["medium"]["learn"].copy()
    if sample_size_current < 1_200_000:
        return MCH_LEARNING_PROFILES["conservative"]["learn"].copy()
    return {"maxdlamda": 0.05, "maxdlamdaNorm": 0.3, "eta": 0.035, "maxLearningSteps": 35}


def construir_indices_parametros_aij(N: int, adjacency_mask: np.ndarray | None) -> np.ndarray | None:
    """
    Retorna os indices dos parametros livres no vetor ConIII:
    h_0..h_{N-1} e, quando A_ij existe, apenas J_ij das arestas.
    """
    if adjacency_mask is None:
        return None

    A = preparar_mascara_adjacencia(adjacency_mask, N)
    indices = list(range(N))
    param_idx = N
    for i in range(N):
        for j in range(i + 1, N):
            if A[i, j]:
                indices.append(param_idx)
            param_idx += 1
    return np.asarray(indices, dtype=np.int64)


def chute_inicial_ising(spin_matrix: np.ndarray,
                        adjacency_mask: np.ndarray | None = None,
                        param_bound: float = 5.0) -> np.ndarray:
    """
    Chute inicial comum para PL e MCH: campos independentes e acoplamentos
    proporcionais a covariancia empirica, respeitando A_ij quando presente.
    """
    R, N = spin_matrix.shape
    medias = spin_matrix.mean(axis=0).astype(np.float64)
    medias_clamp = np.clip(medias, -0.98, 0.98)
    h_init = np.arctanh(medias_clamp)

    sigma_f = spin_matrix.astype(np.float64)
    C_emp = np.cov(sigma_f, rowvar=False)
    var_i = np.maximum(1.0 - medias_clamp**2, 1e-6)
    scale = np.outer(np.sqrt(var_i), np.sqrt(var_i))
    J_init_mat = C_emp / scale
    np.fill_diagonal(J_init_mat, 0.0)

    A = preparar_mascara_adjacencia(adjacency_mask, N)
    if A is not None:
        J_init_mat[~A] = 0.0

    j_init = pack_J(J_init_mat) * 0.1
    initial_guess = np.concatenate([h_init, j_init])
    return np.clip(initial_guess, -param_bound, param_bound)


def chute_independente_suavizado(spin_matrix: np.ndarray,
                                  alpha: float = 0.5,
                                  param_bound: float = 5.0) -> np.ndarray:
    """
    Chute independente com pseudocount beta-binomial para evitar h_i extremos.
    """
    R, N = spin_matrix.shape
    n_plus = np.sum(spin_matrix > 0, axis=0).astype(np.float64)
    p_plus = (n_plus + float(alpha)) / (R + 2.0 * float(alpha))
    medias = np.clip(2.0 * p_plus - 1.0, -0.995, 0.995)
    h_init = np.arctanh(medias)
    return np.clip(np.concatenate([h_init, np.zeros(N * (N - 1) // 2)]), -param_bound, param_bound)


def chute_mean_field_shrinkage(spin_matrix: np.ndarray,
                               adjacency_mask: np.ndarray | None = None,
                               shrinkage: float = 0.5,
                               ridge: float = 1e-3,
                               scale: float = 0.1,
                               param_bound: float = 5.0) -> np.ndarray:
    """
    Chute naive mean-field com shrinkage na covariancia.

    E usado apenas como candidato de warm-start; o torneio descarta o candidato
    se ele piorar os erros pseudo-condicionais dos pares.
    """
    sample_f = spin_matrix.astype(np.float64)
    R, N = sample_f.shape
    medias = np.clip(sample_f.mean(axis=0), -0.995, 0.995)
    centered = sample_f - medias
    cov = (centered.T @ centered) / max(R, 1)
    diag = np.diag(np.diag(cov))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * diag
    cov_shrunk += np.eye(N) * max(float(ridge), 0.0)

    try:
        inv_cov = np.linalg.pinv(cov_shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_shrunk + np.eye(N) * 1e-2)

    J = -float(scale) * inv_cov
    np.fill_diagonal(J, 0.0)
    A = preparar_mascara_adjacencia(adjacency_mask, N)
    if A is not None:
        J[~A] = 0.0
    J = 0.5 * (J + J.T)

    h = np.arctanh(medias) - J @ medias
    multipliers = np.concatenate([h, pack_J(J)])
    return np.clip(multipliers, -param_bound, param_bound)


def chute_tap_shrinkage(spin_matrix: np.ndarray,
                        adjacency_mask: np.ndarray | None = None,
                        shrinkage: float = 0.5,
                        ridge: float = 0.01,
                        scale: float = 0.2,
                        param_bound: float = 5.0) -> np.ndarray:
    """
    Chute TAP regularizado para warm-start.

    A correcao TAP e usada apenas como candidato no torneio. Quando a solucao
    quadratica fica mal definida, o par volta para o limite naive mean-field.
    """
    sample_f = spin_matrix.astype(np.float64)
    R, N = sample_f.shape
    medias = np.clip(sample_f.mean(axis=0), -0.995, 0.995)
    centered = sample_f - medias
    cov = (centered.T @ centered) / max(R, 1)
    diag = np.diag(np.diag(cov))
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    cov_shrunk = (1.0 - shrinkage) * cov + shrinkage * diag
    cov_shrunk += np.eye(N) * max(float(ridge), 0.0)

    try:
        inv_cov = np.linalg.pinv(cov_shrunk)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_shrunk + np.eye(N) * 1e-2)

    J = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            q = float(medias[i] * medias[j])
            inv_ij = float(inv_cov[i, j])
            if abs(q) < 1e-8:
                value = -inv_ij
            else:
                discriminant = 1.0 - 8.0 * q * inv_ij
                if discriminant <= 0.0 or not np.isfinite(discriminant):
                    value = -inv_ij
                else:
                    value = (-1.0 + np.sqrt(discriminant)) / (4.0 * q)
            J[i, j] = value
            J[j, i] = value

    J *= float(scale)
    np.fill_diagonal(J, 0.0)
    A = preparar_mascara_adjacencia(adjacency_mask, N)
    if A is not None:
        J[~A] = 0.0
    J = 0.5 * (J + J.T)

    onsager = medias * ((J * J) @ (1.0 - medias * medias))
    h = np.arctanh(medias) - J @ medias + onsager
    multipliers = np.concatenate([h, pack_J(J)])
    return np.clip(multipliers, -param_bound, param_bound)


def listar_pares_livres(N: int, adjacency_mask: np.ndarray | None = None) -> list[tuple[int, int]]:
    """
    Lista os pares i<j que podem ter J_ij diferente de zero.
    """
    A = preparar_mascara_adjacencia(adjacency_mask, N)
    pares = []
    for i in range(N):
        for j in range(i + 1, N):
            if A is None or A[i, j]:
                pares.append((i, j))
    return pares


def extrair_parametros_livres(multipliers: np.ndarray, N: int,
                              pares_livres: list[tuple[int, int]]) -> np.ndarray:
    """
    Extrai h_i e apenas os J_ij livres de um vetor completo de multipliers.
    """
    multipliers = np.asarray(multipliers, dtype=np.float64)
    h = multipliers[:N]
    J = unpack_J(multipliers[N:], N)
    j = np.asarray([J[i, j] for i, j in pares_livres], dtype=np.float64)
    return np.concatenate([h, j])


def montar_multipliers_livres(theta: np.ndarray, N: int,
                              pares_livres: list[tuple[int, int]],
                              param_bound: float = 5.0) -> np.ndarray:
    """
    Monta o vetor completo de multipliers a partir dos parametros livres.
    """
    theta = np.clip(np.asarray(theta, dtype=np.float64), -param_bound, param_bound)
    h = theta[:N]
    J = np.zeros((N, N), dtype=np.float64)
    for value, (i, j) in zip(theta[N:], pares_livres):
        J[i, j] = value
        J[j, i] = value
    return np.concatenate([h, pack_J(J)])


def calcular_pseudo_nll_media(sample: np.ndarray, multipliers: np.ndarray,
                              adjacency_mask: np.ndarray | None = None,
                              lam: float = 0.0) -> float:
    """
    Pseudo negative log-likelihood media, usada apenas para comparar warm starts.
    """
    sample_f = sample.astype(np.float64)
    _, N = sample_f.shape
    multipliers = np.asarray(multipliers, dtype=np.float64)
    h = multipliers[:N]
    J = unpack_J(multipliers[N:], N)
    A = preparar_mascara_adjacencia(adjacency_mask, N)
    if A is not None:
        J[~A] = 0.0
    np.fill_diagonal(J, 0.0)

    fields = sample_f @ J + h
    z = -2.0 * sample_f * fields
    nll = np.logaddexp(0.0, z).mean()
    if lam > 0:
        nll += 0.5 * lam * float(np.mean(multipliers * multipliers))
    return float(nll)


def avaliar_warm_start_pseudo(sample: np.ndarray,
                              multipliers: np.ndarray,
                              adjacency_mask: np.ndarray | None = None,
                              param_bound: float = 5.0,
                              lam: float = 0.0) -> dict:
    """
    Avalia um warm-start por momentos pseudo-condicionais baratos.

    Para cada configuracao empirica, calcula tanh(h_i + sum_j J_ij s_j). Isso
    nao e uma expectativa exata do modelo Ising, mas e um bom proxy local para
    escolher o ponto inicial do MCH, especialmente para C_ij em A_ij=1.
    """
    sample_f = sample.astype(np.float64)
    R, N = sample_f.shape
    multipliers = np.asarray(multipliers, dtype=np.float64)
    h = multipliers[:N]
    J = unpack_J(multipliers[N:], N)
    A = preparar_mascara_adjacencia(adjacency_mask, N)
    if A is not None:
        J[~A] = 0.0
    np.fill_diagonal(J, 0.0)

    fields = sample_f @ J + h
    cond_mean = np.tanh(fields)
    mean_emp = sample_f.mean(axis=0)
    mean_model = cond_mean.mean(axis=0)
    mean_error = mean_model - mean_emp
    mean_rmse = float(np.sqrt(np.mean(mean_error * mean_error)))

    pares = listar_pares_livres(N, A)
    if pares:
        i_idx = np.asarray([i for i, _ in pares], dtype=np.int64)
        j_idx = np.asarray([j for _, j in pares], dtype=np.int64)
        pair_emp = (sample_f[:, i_idx] * sample_f[:, j_idx]).mean(axis=0)
        pair_model_i = (cond_mean[:, i_idx] * sample_f[:, j_idx]).mean(axis=0)
        pair_model_j = (sample_f[:, i_idx] * cond_mean[:, j_idx]).mean(axis=0)
        pair_model = 0.5 * (pair_model_i + pair_model_j)
        pair_error = pair_model - pair_emp
        pair_rmse = float(np.sqrt(np.mean(pair_error * pair_error)))
        pair_max_abs = float(np.max(np.abs(pair_error)))
        if np.std(pair_emp) > 1e-12 and np.std(pair_model) > 1e-12:
            pair_r = float(np.corrcoef(pair_emp, pair_model)[0, 1])
        else:
            pair_r = np.nan
    else:
        pair_rmse = 0.0
        pair_max_abs = 0.0
        pair_r = np.nan

    all_errors = np.concatenate([mean_error, [pair_max_abs]])
    max_abs_error = float(np.max(np.abs(all_errors))) if all_errors.size else 0.0
    max_abs_theta = float(np.max(np.abs(multipliers))) if multipliers.size else 0.0
    n_bound = int(np.sum(np.abs(multipliers) >= param_bound * 0.995))
    pseudo_nll = calcular_pseudo_nll_media(sample, multipliers, adjacency_mask, lam=lam)
    score = (
        pair_rmse
        + 0.30 * mean_rmse
        + 0.10 * max_abs_error
        + 0.002 * n_bound
        + 0.01 * max(0.0, max_abs_theta - param_bound * 0.8)
    )
    return {
        "score": float(score),
        "pair_rmse": pair_rmse,
        "pair_r": pair_r,
        "pair_max_abs": pair_max_abs,
        "mean_rmse": mean_rmse,
        "max_abs_error": max_abs_error,
        "pseudo_nll": float(pseudo_nll),
        "max_abs_theta": max_abs_theta,
        "n_bound": n_bound,
    }


def estimar_observaveis_solver_em_chunks(solver,
                                         use_sparse: bool = False,
                                         parameter_ix: np.ndarray | None = None,
                                         chunk_size: int = 10_000) -> np.ndarray:
    """
    Calcula observaveis medios do solver sem materializar toda a matriz de
    observaveis de uma vez. Evita MemoryError para amostras grandes.
    """
    states = np.asarray(solver.model.sample, dtype=np.int64)
    n_states = states.shape[0]
    total = None
    for start in range(0, n_states, int(chunk_size)):
        obs = solver.calc_observables(states[start:start + int(chunk_size)])
        obs = np.atleast_2d(obs)
        if use_sparse:
            obs = obs[:, parameter_ix]
        obs_sum = obs.sum(axis=0, dtype=np.float64)
        if total is None:
            total = np.zeros_like(obs_sum, dtype=np.float64)
        total += obs_sum
    return np.asarray(total / max(n_states, 1), dtype=np.float64)


def avaliar_warm_start_mc_curto(sample: np.ndarray,
                                multipliers: np.ndarray,
                                adjacency_mask: np.ndarray | None = None,
                                sample_size: int = 20_000,
                                n_iters: int | None = None,
                                burn_in: int | None = None) -> dict:
    """
    Avalia um candidato de warm-start pelo erro real dos constraints estimado
    com uma amostragem MCH curta.
    """
    sample = sample.astype(np.int64)
    R, N = sample.shape
    sample_size = max(1_000, int(sample_size))
    n_iters = int(n_iters or max(100, N * 20))
    burn_in = int(burn_in or max(1_000, N * 100))
    parameter_ix = construir_indices_parametros_aij(N, adjacency_mask)
    use_sparse = parameter_ix is not None
    solver_cls = coniii.solvers.SparseMCH if use_sparse else coniii.solvers.MCH
    solver_kwargs = {
        "sample": sample,
        "sample_size": sample_size,
        "iprint": False,
        "sampler_kw": {"iprint": False},
    }
    if use_sparse:
        solver_kwargs["parameter_ix"] = parameter_ix
    solver = solver_cls(**solver_kwargs)
    if use_sparse:
        constraints = solver.constraints[parameter_ix]
        solver._multipliers = solver.fill_in(np.asarray(multipliers, dtype=np.float64)[parameter_ix].copy())
    else:
        constraints = solver.constraints
        solver._multipliers = np.asarray(multipliers, dtype=np.float64).copy()

    solver.model.generate_sample(
        n_iters,
        burn_in,
        multipliers=solver._multipliers,
        generate_kwargs={"parallel": False},
    )
    estimate = estimar_observaveis_solver_em_chunks(
        solver,
        use_sparse=use_sparse,
        parameter_ix=parameter_ix,
        chunk_size=10_000,
    )
    error = estimate - constraints
    abs_error = np.abs(error)
    error_norm = float(np.linalg.norm(error))
    max_error = float(abs_error.max()) if abs_error.size else 0.0
    rms_error = float(np.sqrt(np.mean(error * error))) if error.size else 0.0
    score = max(max_error, error_norm / np.sqrt(max(len(error), 1)))
    return {
        "mc_score": float(score),
        "mc_error_norm": error_norm,
        "mc_error_max_abs": max_error,
        "mc_error_rms": rms_error,
        "mc_sample_size": sample_size,
    }


def torneio_warm_starts(sample: np.ndarray,
                        initial_guess: np.ndarray,
                        adjacency_mask: np.ndarray | None,
                        final_lam: float,
                        param_bound: float = 5.0,
                        enabled: bool = True) -> tuple[np.ndarray, dict]:
    """
    Gera varios candidatos de warm-start e escolhe o melhor por erro local de
    pares em A_ij=1, nao apenas pelo pseudo-NLL.
    """
    if not enabled:
        return initial_guess, {"enabled": False, "candidates": []}

    sample = sample.astype(np.int64)
    R, N = sample.shape
    candidates: list[tuple[str, np.ndarray, dict]] = []

    def add_candidate(name: str, multipliers: np.ndarray, origin: str = "deterministico"):
        multipliers = np.clip(np.asarray(multipliers, dtype=np.float64), -param_bound, param_bound)
        metrics = avaliar_warm_start_pseudo(
            sample,
            multipliers,
            adjacency_mask=adjacency_mask,
            param_bound=param_bound,
            lam=max(float(final_lam), 0.0),
        )
        metrics["name"] = name
        metrics["origin"] = origin
        candidates.append((name, multipliers, metrics))
        return metrics

    print("\n  [Warm-start] Torneio de candidatos para inicializar o solver:")
    add_candidate("independente_suave", chute_independente_suavizado(sample, alpha=0.5, param_bound=param_bound))
    add_candidate("covariancia_fraca", initial_guess)

    meanfield_specs = [
        (0.001, 0.05),
        (0.001, 0.10),
        (0.001, 0.15),
        (0.001, 0.20),
        (0.001, 0.30),
        (0.001, 0.40),
        (0.001, 0.60),
        (0.003, 0.10),
        (0.003, 0.20),
        (0.010, 0.10),
        (0.010, 0.20),
        (0.030, 0.20),
        (0.100, 0.20),
        (0.300, 0.20),
    ]
    for ridge, scale in meanfield_specs:
        name = f"meanfield_ridge_{ridge:g}_s{scale:g}"
        try:
            add_candidate(
                name,
                chute_mean_field_shrinkage(
                    sample,
                    adjacency_mask=adjacency_mask,
                    shrinkage=0.5,
                    ridge=ridge,
                    scale=scale,
                    param_bound=param_bound,
                ),
                origin="meanfield",
            )
        except Exception as e:
            print(f"    - {name} falhou ({type(e).__name__}: {e})")

    tap_specs = [
        (0.010, 0.10),
        (0.010, 0.20),
        (0.030, 0.20),
        (0.100, 0.20),
    ]
    for ridge, scale in tap_specs:
        name = f"tap_ridge_{ridge:g}_s{scale:g}"
        try:
            add_candidate(
                name,
                chute_tap_shrinkage(
                    sample,
                    adjacency_mask=adjacency_mask,
                    shrinkage=0.5,
                    ridge=ridge,
                    scale=scale,
                    param_bound=param_bound,
                ),
                origin="tap",
            )
        except Exception as e:
            print(f"    - {name} falhou ({type(e).__name__}: {e})")

    pl_lambdas = []
    for value in (0.1, 0.03, 0.01, 0.003, final_lam):
        value = max(float(value), 0.0)
        if value not in pl_lambdas:
            pl_lambdas.append(value)

    pl_kwargs = {
        "method": "L-BFGS-B",
        "options": {"maxiter": 300, "ftol": 1e-8, "gtol": 1e-6, "maxls": 50},
    }
    pl_start = chute_independente_suavizado(sample, alpha=1.0, param_bound=param_bound)
    for lam_stage in pl_lambdas:
        try:
            candidate, soln = resolver_pseudo_l2_coniii(
                sample,
                initial_guess=pl_start,
                lam=lam_stage,
                param_bound=param_bound,
                solver_kwargs=pl_kwargs,
                adjacency_mask=adjacency_mask,
            )
            add_candidate(f"PL_lambda_{lam_stage:g}", candidate, origin="pseudo_likelihood")
            pl_start = candidate
        except Exception as e:
            print(f"    - PL_lambda_{lam_stage:g} falhou ({type(e).__name__}: {e})")

    ranked = sorted(candidates, key=lambda item: item[2]["score"])
    for rank, (name, _, metrics) in enumerate(ranked[: min(10, len(ranked))], start=1):
        pair_r = metrics["pair_r"]
        pair_r_text = "nan" if not np.isfinite(pair_r) else f"{pair_r:.4f}"
        print(
            f"    [{rank}] {name:<22} score={metrics['score']:.5f} "
            f"pair_RMSE={metrics['pair_rmse']:.5f} pair_r={pair_r_text} "
            f"mean_RMSE={metrics['mean_rmse']:.5f} |theta|max={metrics['max_abs_theta']:.3f} "
            f"bounds={metrics['n_bound']}"
        )

    if not ranked:
        print("    [Aviso] Nenhum candidato valido; usando chute inicial base.")
        return initial_guess, {"enabled": True, "selected": "initial_guess", "candidates": []}

    mc_ranked = []
    mc_shortlist = list(ranked[: min(10, len(ranked))])
    shortlist_names = {item[0] for item in mc_shortlist}
    for required_origin in ("meanfield", "tap", "pseudo_likelihood"):
        if any(item[2].get("origin") == required_origin for item in mc_shortlist):
            continue
        for item in ranked:
            if item[2].get("origin") == required_origin and item[0] not in shortlist_names:
                mc_shortlist.append(item)
                shortlist_names.add(item[0])
                break

    print("    [MC curto] Avaliando erro real dos melhores candidatos...")
    for name, multipliers, metrics in mc_shortlist:
        try:
            mc_metrics = avaliar_warm_start_mc_curto(
                sample,
                multipliers,
                adjacency_mask=adjacency_mask,
                sample_size=20_000,
            )
            metrics.update(mc_metrics)
            mc_ranked.append((name, multipliers, metrics))
            print(
                f"      - {name:<22} mc_score={metrics['mc_score']:.5f} "
                f"max|erro|={metrics['mc_error_max_abs']:.5f} "
                f"||erro||={metrics['mc_error_norm']:.5f}"
            )
        except Exception as e:
            metrics["mc_error"] = f"{type(e).__name__}: {e}"
            print(f"      - {name:<22} MC curto falhou ({type(e).__name__}: {e})")

    if mc_ranked:
        selected_name, selected_multipliers, selected_metrics = min(
            mc_ranked,
            key=lambda item: item[2]["mc_score"],
        )
        print(
            f"    -> warm start escolhido por MC curto: {selected_name} "
            f"(mc_score={selected_metrics['mc_score']:.5f})."
        )
    else:
        selected_name, selected_multipliers, selected_metrics = ranked[0]
        print(f"    -> warm start escolhido por proxy: {selected_name} (score={selected_metrics['score']:.5f}).")

    return selected_multipliers, {
        "enabled": True,
        "selected": selected_name,
        "selected_metrics": selected_metrics,
        "candidates": [metrics for _, _, metrics in ranked],
    }


def auto_tune_pseudo_warm_start(sample: np.ndarray,
                                initial_guess: np.ndarray,
                                adjacency_mask: np.ndarray | None,
                                final_lam: float,
                                param_bound: float = 5.0,
                                enabled: bool = True) -> tuple[np.ndarray, dict]:
    """
    Roda uma continuacao curta de pseudo-likelihood com warm start.

    A cada nivel de regularizacao, testa alguns orcamentos do L-BFGS-B a partir
    do melhor ponto atual e carrega adiante o menor pseudo-NLL medio.
    """
    if not enabled:
        return initial_guess, {"enabled": False, "history": []}

    final_lam = max(float(final_lam), 0.0)
    lambda_candidates = [0.01, 0.003, 0.001, 0.0003, final_lam]
    lambda_path = []
    for value in lambda_candidates:
        if value + 1e-15 >= final_lam and value not in lambda_path:
            lambda_path.append(value)
    if final_lam not in lambda_path:
        lambda_path.append(final_lam)

    strategies = [
        ("curto", {"method": "L-BFGS-B", "options": {"maxiter": 150, "ftol": 1e-7, "gtol": 1e-5}}),
        ("medio", {"method": "L-BFGS-B", "options": {"maxiter": 400, "ftol": 1e-8, "gtol": 1e-6}}),
        ("forte", {"method": "L-BFGS-B", "options": {"maxiter": 800, "ftol": 1e-9, "gtol": 1e-7}}),
    ]

    current = np.asarray(initial_guess, dtype=np.float64)
    current_score = calcular_pseudo_nll_media(sample, current, adjacency_mask, lam=lambda_path[0])
    history = []

    print("\n  [PL-Tuning] Auto-tuning iterativo com warm start:")
    for round_idx, lam_stage in enumerate(lambda_path, start=1):
        print(f"    [Rodada {round_idx}/{len(lambda_path)}] lambda={lam_stage:g}")
        best_round = None
        for name, solver_kwargs in strategies:
            try:
                candidate, soln = resolver_pseudo_l2_coniii(
                    sample,
                    initial_guess=current,
                    lam=lam_stage,
                    param_bound=param_bound,
                    solver_kwargs=solver_kwargs,
                    adjacency_mask=adjacency_mask,
                )
                nll = calcular_pseudo_nll_media(sample, candidate, adjacency_mask, lam=lam_stage)
                max_abs = float(np.max(np.abs(candidate)))
                n_bound = int(np.sum(np.abs(candidate) >= param_bound * 0.995))
                score = nll + 0.02 * n_bound + 0.01 * max(0.0, max_abs - param_bound * 0.9)
                row = {
                    "round": round_idx,
                    "lambda": lam_stage,
                    "strategy": name,
                    "pseudo_nll": nll,
                    "score": score,
                    "max_abs": max_abs,
                    "n_bound": n_bound,
                    "success": True,
                    "n_subproblems": len(soln),
                }
                print(
                    f"      - {name:<5} score={score:.6f} nll={nll:.6f} "
                    f"|theta|max={max_abs:.3f} bounds={n_bound}"
                )
            except Exception as e:
                row = {
                    "round": round_idx,
                    "lambda": lam_stage,
                    "strategy": name,
                    "pseudo_nll": np.nan,
                    "score": np.inf,
                    "max_abs": np.nan,
                    "n_bound": np.nan,
                    "success": False,
                    "error": f"{type(e).__name__}: {e}",
                }
                candidate = None
                print(f"      - {name:<5} falhou ({type(e).__name__}: {e})")

            history.append(row)
            if candidate is not None and (best_round is None or row["score"] < best_round["row"]["score"]):
                best_round = {"row": row, "multipliers": candidate}

        if best_round is not None and best_round["row"]["score"] <= current_score * 1.05:
            current = best_round["multipliers"]
            current_score = best_round["row"]["score"]
            print(
                f"      -> warm start atualizado por '{best_round['row']['strategy']}' "
                f"(score={current_score:.6f})."
            )
        else:
            print("      -> sem melhora confiavel; mantendo warm start anterior.")

    return current, {
        "enabled": True,
        "history": history,
        "final_score": current_score,
        "lambda_path": lambda_path,
    }


def resolver_enumerate_exato(sample: np.ndarray,
                             initial_guess: np.ndarray | None = None,
                             adjacency_mask: np.ndarray | None = None,
                             lam: float = 0.0,
                             param_bound: float = 5.0,
                             solver_kwargs: dict | None = None,
                             verbose: bool = True) -> tuple[np.ndarray, dict]:
    """
    Inferencia MaxEnt exata por enumeracao para N <= 20.

    Otimiza logZ(theta) - theta . <f>_obs com gradiente exato. Quando A_ij e
    fornecida, somente arestas do grafo entram como parametros J livres.
    """
    sample = sample.astype(np.int64)
    R, N = sample.shape
    if N > 20:
        raise ValueError(f"Enumerate exato limitado a N <= 20; recebido N={N}.")

    pares_livres = listar_pares_livres(N, adjacency_mask)
    n_states = 2 ** N
    if verbose:
        print(
            f"\n  [Enumerate-Exact] Enumerando 2^{N}={n_states:,} estados; "
            f"{len(pares_livres)} acoplamentos livres."
        )

    state_idx = np.arange(n_states, dtype=np.uint64)
    bits = ((state_idx[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & 1).astype(np.int8)
    states = (2 * bits - 1).astype(np.int8)
    del bits

    if pares_livres:
        i_idx = np.asarray([i for i, _ in pares_livres], dtype=np.int64)
        j_idx = np.asarray([j for _, j in pares_livres], dtype=np.int64)
        pair_products = (states[:, i_idx] * states[:, j_idx]).astype(np.int8)
        pairs_obs = (sample[:, i_idx] * sample[:, j_idx]).mean(axis=0).astype(np.float64)
    else:
        pair_products = np.zeros((n_states, 0), dtype=np.int8)
        pairs_obs = np.zeros(0, dtype=np.float64)

    means_obs = sample.mean(axis=0).astype(np.float64)
    obs = np.concatenate([means_obs, pairs_obs])

    if initial_guess is None:
        initial_guess = chute_inicial_ising(sample, adjacency_mask=adjacency_mask, param_bound=param_bound)
    theta0 = extrair_parametros_livres(initial_guess, N, pares_livres)
    theta0 = np.clip(theta0, -param_bound, param_bound)

    def objective(theta):
        h = theta[:N]
        j = theta[N:]
        log_w = states @ h
        if len(j) > 0:
            log_w = log_w + pair_products @ j
        log_z = scipy.special.logsumexp(log_w)
        weights = np.exp(log_w - log_z)

        means_model = weights @ states
        if len(j) > 0:
            pairs_model = weights @ pair_products
        else:
            pairs_model = np.zeros(0, dtype=np.float64)
        model = np.concatenate([means_model, pairs_model])

        loss = float(log_z - np.dot(theta, obs))
        grad = model - obs
        if lam > 0:
            loss += 0.5 * lam * float(np.dot(theta, theta))
            grad = grad + lam * theta
        return loss, grad.astype(np.float64)

    local_kwargs = {
        "method": "L-BFGS-B",
        "jac": True,
        "bounds": [(-param_bound, param_bound)] * len(theta0),
        "options": {"maxiter": 500, "ftol": 1e-10, "gtol": 1e-6, "maxls": 50},
    }
    if solver_kwargs:
        local_kwargs.update(solver_kwargs)

    res = minimize(objective, theta0, **local_kwargs)
    if verbose and not res.success:
        print(f"  [Enumerate-Exact] Aviso: otimizador parou sem convergencia formal ({res.message}).")

    multipliers = montar_multipliers_livres(res.x, N, pares_livres, param_bound=param_bound)
    _, final_grad = objective(res.x)
    if lam > 0:
        final_grad = final_grad - lam * res.x
    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(getattr(res, "nit", 0)),
        "fun": float(res.fun),
        "moment_error_norm": float(np.linalg.norm(final_grad)),
        "moment_error_max_abs": float(np.max(np.abs(final_grad))) if final_grad.size else 0.0,
        "n_states": n_states,
        "n_free_couplings": len(pares_livres),
    }
    if verbose:
        print(
            "  [Enumerate-Exact] "
            f"nit={info['nit']} | max|erro_momento|={info['moment_error_max_abs']:.6f} "
            f"| ||erro||={info['moment_error_norm']:.6f}"
        )
    return multipliers, info


def _resolver_mch_loop_com_plateau(
        solver,
        constraints: np.ndarray,
        initial_selected: np.ndarray,
        use_sparse: bool,
        parameter_ix: np.ndarray | None,
        tol: float,
        tol_norm: float,
        n_iters: int,
        burn_in: int,
        maxiter: int,
        mch_schedule,
        plateau_rel_improvement: float,
        plateau_patience: int,
        plateau_max_error: float | None,
        sample_increment: int,
        sample_growth_factor: float,
        max_sample_size: int,
        sample_schedule: list[int] | tuple[int, ...] | None = None,
        generate_kwargs: dict | None = None):
    """
    Executa o loop MCH do ConIII com amostragem adaptativa por platô.

    O platô é avaliado comparando duas janelas consecutivas da média móvel do
    maior erro absoluto. Quando a melhora é pequena, a próxima amostra cresce,
    mas a inferência continua até convergir formalmente ou atingir maxiter.
    """
    generate_kwargs = generate_kwargs or {}
    constraints = np.asarray(constraints, dtype=np.float64)
    initial_selected = np.asarray(initial_selected, dtype=np.float64)
    moving_average_window = max(1, int(plateau_patience))
    sample_increment = max(1, int(sample_increment))
    sample_growth_factor = max(1.0, float(sample_growth_factor))
    initial_sample_size = int(solver.model.sampleSize)
    max_sample_size = max(initial_sample_size, int(max_sample_size))
    if sample_schedule is None:
        sample_schedule = [100_000, 200_000, 400_000, 800_000, 1_600_000, 2_500_000]
    sample_schedule = sorted({
        int(s) for s in sample_schedule
        if int(s) >= initial_sample_size and int(s) <= max_sample_size
    } | {initial_sample_size, max_sample_size})

    if use_sparse:
        solver._multipliers = solver.fill_in(initial_selected.copy())
    else:
        solver._multipliers = initial_selected.copy()

    def estimate_constraints() -> np.ndarray:
        return estimar_observaveis_solver_em_chunks(
            solver,
            use_sparse=use_sparse,
            parameter_ix=parameter_ix,
            chunk_size=10_000,
        )

    def error_metrics(error: np.ndarray) -> tuple[float, float, float]:
        abs_error = np.abs(error)
        error_norm = float(np.linalg.norm(error))
        max_error = float(abs_error.max())
        score = max(
            max_error / max(tol, np.finfo(float).eps),
            error_norm / max(tol_norm, np.finfo(float).eps),
        )
        return float(score), error_norm, max_error

    errors = []
    mch_start = time.time()
    last_report_time = mch_start

    solver.model.generate_sample(
        n_iters,
        burn_in,
        multipliers=solver._multipliers,
        generate_kwargs=generate_kwargs,
    )
    this_constraints = estimate_constraints()
    initial_error = this_constraints - constraints
    errors.append(initial_error)
    initial_score, best_error_norm, best_max_error = error_metrics(initial_error)
    n_error_terms = max(len(initial_error), 1)
    best_score = best_error_norm / np.sqrt(n_error_terms)
    best_iteration = 0
    best_error = initial_error.copy()
    best_multipliers = solver._multipliers.copy()
    best_sample_size = int(solver.model.sampleSize)
    best_updates = [{
        "iteration": best_iteration,
        "score": best_score,
        "combined_score": initial_score,
        "error_norm": best_error_norm,
        "max_error": best_max_error,
        "sample_size": best_sample_size,
    }]
    print(
        f"Iteracao 0/{maxiter} | "
        f"max|erro|={best_max_error:.5f} | "
        f"norm||erro||={best_score:.5f} | "
        f"best_norm||erro||={best_score:.5f} | "
        f"amostra={int(solver.model.sampleSize):,} | "
        f"|dtheta|=0 | "
        f"acc=n/a | "
        f"etapa=0.0s | "
        f"total={time.time() - mch_start:.1f}s",
        flush=True,
    )

    counter = 0
    moving_max_errors = []
    moving_average_previous = np.nan
    moving_average_recent = np.nan
    final_relative_improvement = np.nan
    current_error = initial_error.copy()
    current_score = initial_score
    stop_reason = "maxiter"
    errflag = 0
    sample_size_history = [int(solver.model.sampleSize)]
    sample_increase_events = []
    backtracking_events = []
    no_improvement_events = []
    pre_mc_rejection_events = []
    bad_chain_enabled = False
    backtracking_strategy = "shrink_from_base"
    max_attempts_per_iteration = 20
    attempt_decay = 0.5
    pre_mc_reject_margin = 1.02
    attempt_status_width = 58
    attempt_status_inline = sys.stdout.isatty()
    learn_params = mch_schedule(counter, int(solver.model.sampleSize))

    def print_attempt_status(iteration: int, attempt_idx: int) -> None:
        return None

    def clear_attempt_status() -> None:
        return None

    def learning_params_for_attempt(base_params: dict, attempt_idx: int) -> tuple[dict, float]:
        scale = attempt_decay ** max(0, attempt_idx - 1)
        local_params = dict(base_params)
        for key in ("eta", "maxdlamda", "maxdlamdaNorm"):
            if key in local_params:
                local_params[key] = max(float(local_params[key]) * scale, 1e-8)
        return local_params, scale

    def aumentar_amostra_apos_backtracking(reason: str) -> bool:
        nonlocal this_constraints, current_error, current_score
        nonlocal moving_average_previous, moving_average_recent, final_relative_improvement
        sample_size_used = int(solver.model.sampleSize)
        if sample_size_used >= max_sample_size:
            return False

        next_values = [s for s in sample_schedule if s > sample_size_used]
        next_sample_size = next_values[0] if next_values else max_sample_size
        next_sample_size = min(next_sample_size, max_sample_size)
        actual_increment = next_sample_size - sample_size_used
        if actual_increment <= 0:
            return False

        solver.model.sampleSize = next_sample_size
        sample_size_history.append(next_sample_size)
        sample_increase_events.append({
            "iteration": counter,
            "previous_sample_size": sample_size_used,
            "actual_increment": actual_increment,
            "next_sample_size": next_sample_size,
            "moving_average_previous": moving_average_previous,
            "moving_average_recent": moving_average_recent,
            "relative_improvement": final_relative_improvement,
            "reason": reason,
        })
        solver._multipliers = best_multipliers.copy()
        solver.multipliers = best_multipliers.copy()
        solver.model.generate_sample(
            n_iters,
            burn_in,
            multipliers=solver._multipliers,
            generate_kwargs=generate_kwargs,
        )
        this_constraints = estimate_constraints()
        current_error = this_constraints - constraints
        current_score, _, _ = error_metrics(current_error)
        moving_max_errors.clear()
        moving_average_previous = np.nan
        moving_average_recent = np.nan
        final_relative_improvement = np.nan
        return True

    while True:
        sample_size_used = int(solver.model.sampleSize)
        base_multipliers = solver._multipliers.copy()
        base_constraints = this_constraints.copy()
        base_error = current_error.copy()
        base_score = current_score
        base_learn_params = mch_schedule(counter, sample_size_used)
        base_sample = solver.model.sample
        accepted = None
        interrupted = False
        last_rejected_score = base_score

        for attempt_idx in range(1, max_attempts_per_iteration + 1):
            local_learn_params, attempt_scale = learning_params_for_attempt(
                base_learn_params,
                attempt_idx,
            )
            solver._multipliers = base_multipliers.copy()
            solver.multipliers = base_multipliers.copy()
            solver.model.sample = base_sample
            print_attempt_status(counter + 1, attempt_idx)
            try:
                predicted_constraints = solver.learn_parameters_mch(
                    base_constraints,
                    constraints,
                    **local_learn_params,
                )
                predicted_error = predicted_constraints - constraints
                predicted_score, predicted_error_norm, predicted_max_error = error_metrics(predicted_error)
                if predicted_score >= base_score * pre_mc_reject_margin:
                    pre_mc_rejection_events.append({
                        "iteration": counter + 1,
                        "attempt": attempt_idx,
                        "sample_size": sample_size_used,
                        "attempt_scale": attempt_scale,
                        "predicted_score": predicted_score,
                        "base_score": base_score,
                        "predicted_error_norm": predicted_error_norm,
                        "predicted_max_error": predicted_max_error,
                    })
                    last_rejected_score = predicted_score
                    continue

                solver.model.generate_sample(
                    n_iters,
                    burn_in,
                    multipliers=solver._multipliers,
                    generate_kwargs=generate_kwargs,
                )
                candidate_constraints = estimate_constraints()
            except KeyboardInterrupt:
                interrupted = True
                stop_reason = "manual_interrupt"
                errflag = 3
                clear_attempt_status()
                break

            candidate_error = candidate_constraints - constraints
            candidate_score, candidate_error_norm, candidate_max_error = error_metrics(candidate_error)
            if candidate_score < base_score:
                accepted = {
                    "attempt": attempt_idx,
                    "attempt_scale": attempt_scale,
                    "learn_params": local_learn_params,
                    "constraints": candidate_constraints,
                    "error": candidate_error,
                    "score": candidate_score,
                    "error_norm": candidate_error_norm,
                    "max_error": candidate_max_error,
                    "multipliers": solver._multipliers.copy(),
                }
                break

            backtracking_events.append({
                "iteration": counter + 1,
                "attempt": attempt_idx,
                "sample_size": sample_size_used,
                "attempt_scale": attempt_scale,
                "candidate_score": candidate_score,
                "base_score": base_score,
                "candidate_error_norm": candidate_error_norm,
                "candidate_max_error": candidate_max_error,
                "bad_chain_enabled": bad_chain_enabled,
                "backtracking_strategy": backtracking_strategy,
            })
            last_rejected_score = candidate_score

        if interrupted:
            break

        clear_attempt_status()
        if accepted is None:
            failed_iteration = counter + 1
            solver._multipliers = base_multipliers.copy()
            solver.multipliers = base_multipliers.copy()
            solver.model.sample = base_sample
            this_constraints = base_constraints
            current_error = base_error
            current_score = base_score
            no_improvement_events.append({
                "iteration": counter + 1,
                "sample_size": sample_size_used,
                "score": base_score,
                "last_rejected_score": last_rejected_score,
                "bad_chain_enabled": bad_chain_enabled,
                "backtracking_strategy": backtracking_strategy,
            })
            if aumentar_amostra_apos_backtracking("backtracking_sem_melhora"):
                now = time.time()
                print(
                    f"Iteracao {counter}/{maxiter} | "
                    f"max|erro|={best_max_error:.5f} | "
                    f"norm||erro||={best_error_norm / np.sqrt(n_error_terms):.5f} | "
                    f"best_norm||erro||={best_score:.5f} | "
                    f"amostra={int(solver.model.sampleSize):,} | "
                    f"|dtheta|=0 | "
                    f"acc=n/a | "
                    f"etapa={now - last_report_time:.1f}s | "
                    f"total={now - mch_start:.1f}s",
                    flush=True,
                )
                last_report_time = now
                learn_params = mch_schedule(counter, int(solver.model.sampleSize))
                continue
            now = time.time()
            counter += 1
            errors.append(base_error.copy())
            base_max_error = float(np.max(np.abs(base_error))) if base_error.size else 0.0
            print(
                f"Iteracao {counter}/{maxiter} | "
                f"max|erro|={base_max_error:.5f} | "
                f"norm||erro||={np.linalg.norm(base_error) / np.sqrt(n_error_terms):.5f} | "
                f"best_norm||erro||={best_score:.5f} | "
                f"amostra={sample_size_used:,} | "
                f"|dtheta|=0 | "
                f"acc=n/a | "
                f"etapa={now - last_report_time:.1f}s | "
                f"total={now - mch_start:.1f}s",
                flush=True,
            )
            last_report_time = now
            if counter >= maxiter:
                stop_reason = "maxiter"
                errflag = 0
                break
            continue

        solver._multipliers = accepted["multipliers"].copy()
        solver.multipliers = accepted["multipliers"].copy()
        this_constraints = accepted["constraints"]
        current_error = accepted["error"]
        current_score = accepted["score"]
        counter += 1

        error = current_error
        errors.append(error)
        abs_error = np.abs(error)
        score = accepted["score"]
        error_norm = accepted["error_norm"]
        max_error = accepted["max_error"]
        delta_norm = float(np.linalg.norm(accepted["multipliers"] - base_multipliers))

        norm_error = error_norm / np.sqrt(n_error_terms)
        if norm_error < best_score:
            best_score = norm_error
            best_error_norm = error_norm
            best_max_error = max_error
            best_iteration = counter
            best_error = error.copy()
            best_multipliers = solver._multipliers.copy()
            best_sample_size = sample_size_used
            best_updates.append({
                "iteration": best_iteration,
                "score": best_score,
                "combined_score": score,
                "error_norm": best_error_norm,
                "max_error": best_max_error,
                "sample_size": best_sample_size,
            })

        moving_max_errors.append(max_error)
        moving_average_previous = np.nan
        moving_average_recent = float(np.mean(moving_max_errors[-moving_average_window:]))
        final_relative_improvement = np.nan
        if len(moving_max_errors) >= 2 * moving_average_window:
            moving_average_previous = float(np.mean(
                moving_max_errors[-2 * moving_average_window:-moving_average_window]
            ))
            final_relative_improvement = (
                (moving_average_previous - moving_average_recent)
                / max(moving_average_previous, np.finfo(float).eps)
            )

        now = time.time()
        improvement_text = (
            "n/a"
            if not np.isfinite(final_relative_improvement)
            else f"{final_relative_improvement * 100:+.1f}%"
        )
        print(
            f"Iteracao {counter}/{maxiter} | "
            f"max|erro|={max_error:.5f} | "
            f"norm||erro||={error_norm / np.sqrt(n_error_terms):.5f} | "
            f"best_norm||erro||={best_score:.5f} | "
            f"amostra={sample_size_used:,} | "
            f"|dtheta|={delta_norm:.3g} | "
            f"acc=n/a | "
            f"etapa={now - last_report_time:.1f}s | "
            f"total={now - mch_start:.1f}s",
            flush=True,
        )
        last_report_time = now

        if counter >= maxiter:
            stop_reason = "maxiter"
            errflag = 0
            break

        learn_params = mch_schedule(counter, int(solver.model.sampleSize))

    last_error = errors[-1].copy()
    last_error_norm = float(np.linalg.norm(last_error))
    last_max_error = float(np.max(np.abs(last_error)))
    last_iteration = counter
    solver._multipliers = best_multipliers.copy()
    solver.multipliers = best_multipliers.copy()
    if not use_sparse and stop_reason != "manual_interrupt":
        solver.model.generate_sample(
            n_iters,
            burn_in,
            multipliers=solver.multipliers,
            generate_kwargs=generate_kwargs,
        )

    selected_multipliers = (
        solver.multipliers[parameter_ix] if use_sparse else solver.multipliers.copy()
    )
    run_info = {
        "stop_reason": stop_reason,
        "converged_formally": stop_reason == "converged",
        "manual_interrupt": stop_reason == "manual_interrupt",
        "plateau_triggered": bool(sample_increase_events),
        "plateau_rel_improvement": plateau_rel_improvement,
        "plateau_patience": plateau_patience,
        "plateau_max_error": plateau_max_error,
        "plateau_streak": 0,
        "final_relative_improvement": float(final_relative_improvement),
        "final_sample_size": int(solver.model.sampleSize),
        "moving_average_window": moving_average_window,
        "sample_increment": sample_increment,
        "sample_growth_factor": sample_growth_factor,
        "max_sample_size": max_sample_size,
        "sample_schedule": sample_schedule,
        "sample_size_history": sample_size_history,
        "sample_increases": len(sample_increase_events),
        "sample_increase_events": sample_increase_events,
        "bad_chain_enabled": bad_chain_enabled,
        "backtracking_strategy": backtracking_strategy,
        "max_attempts_per_iteration": max_attempts_per_iteration,
        "attempt_decay": attempt_decay,
        "pre_mc_reject_margin": pre_mc_reject_margin,
        "backtracking_rejections": len(backtracking_events),
        "backtracking_events": backtracking_events,
        "pre_mc_rejections": len(pre_mc_rejection_events),
        "pre_mc_rejection_events": pre_mc_rejection_events,
        "no_improvement_count": len(no_improvement_events),
        "no_improvement_events": no_improvement_events,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "best_error": best_error,
        "best_error_norm": best_error_norm,
        "best_error_max_abs": best_max_error,
        "best_sample_size": best_sample_size,
        "best_updates": best_updates,
        "last_iteration": last_iteration,
        "last_error_norm": last_error_norm,
        "last_error_max_abs": last_max_error,
        "elapsed_s": float(time.time() - mch_start),
    }
    return selected_multipliers, errflag, np.vstack(errors), run_info


def _resolver_mch_loop_fixo(
        solver,
        constraints: np.ndarray,
        initial_selected: np.ndarray,
        use_sparse: bool,
        parameter_ix: np.ndarray | None,
        tol: float,
        tol_norm: float,
        n_iters: int,
        burn_in: int,
        maxiter: int,
        mch_schedule,
        generate_kwargs: dict | None = None):
    """
    Executa MCH em modo fixo: uma atualização MCH e uma nova amostra por
    iteração, sem backtracking, sem subtentativas e sem aumento automático de
    sample_size.
    """
    generate_kwargs = generate_kwargs or {}
    constraints = np.asarray(constraints, dtype=np.float64)
    initial_selected = np.asarray(initial_selected, dtype=np.float64)

    if use_sparse:
        solver._multipliers = solver.fill_in(initial_selected.copy())
    else:
        solver._multipliers = initial_selected.copy()
    solver.multipliers = solver._multipliers.copy()

    def estimate_constraints() -> np.ndarray:
        return estimar_observaveis_solver_em_chunks(
            solver,
            use_sparse=use_sparse,
            parameter_ix=parameter_ix,
            chunk_size=10_000,
        )

    def error_metrics(error: np.ndarray) -> tuple[float, float, float]:
        abs_error = np.abs(error)
        error_norm = float(np.linalg.norm(error))
        max_error = float(abs_error.max())
        score = max(
            max_error / max(tol, np.finfo(float).eps),
            error_norm / max(tol_norm, np.finfo(float).eps),
        )
        return float(score), error_norm, max_error

    mch_start = time.time()
    last_report_time = mch_start
    sample_size_used = int(solver.model.sampleSize)

    solver.model.generate_sample(
        n_iters,
        burn_in,
        multipliers=solver._multipliers,
        generate_kwargs=generate_kwargs,
    )
    this_constraints = estimate_constraints()
    current_error = this_constraints - constraints
    errors = [current_error.copy()]
    current_score, current_error_norm, current_max_error = error_metrics(current_error)
    n_error_terms = max(len(current_error), 1)

    best_score = current_error_norm / np.sqrt(n_error_terms)
    best_error = current_error.copy()
    best_error_norm = current_error_norm
    best_max_error = current_max_error
    best_iteration = 0
    best_multipliers = solver._multipliers.copy()
    best_sample_size = sample_size_used
    best_updates = [{
        "iteration": 0,
        "score": best_score,
        "combined_score": current_score,
        "error_norm": best_error_norm,
        "max_error": best_max_error,
        "sample_size": best_sample_size,
    }]

    print(
        f"Iteracao 0/{maxiter} | "
        f"max|erro|={current_max_error:.5f} | "
        f"norm||erro||={current_error_norm / np.sqrt(n_error_terms):.5f} | "
        f"best_norm||erro||={best_score:.5f} | "
        f"amostra={sample_size_used:,} | "
        f"|dtheta|=n/a | "
        f"acc=n/a | "
        f"etapa=0.0s | "
        f"total={time.time() - mch_start:.1f}s",
        flush=True,
    )

    stop_reason = "maxiter"
    errflag = 0
    completed_iterations = 0
    converged_iteration = None

    for counter in range(1, maxiter + 1):
        learn_params = mch_schedule(counter - 1, sample_size_used)
        try:
            solver.learn_parameters_mch(this_constraints, constraints, **learn_params)
            solver.model.generate_sample(
                n_iters,
                burn_in,
                multipliers=solver._multipliers,
                generate_kwargs=generate_kwargs,
            )
            this_constraints = estimate_constraints()
        except KeyboardInterrupt:
            stop_reason = "manual_interrupt"
            errflag = 3
            break

        current_error = this_constraints - constraints
        errors.append(current_error.copy())
        current_score, current_error_norm, current_max_error = error_metrics(current_error)
        completed_iterations = counter

        norm_error = current_error_norm / np.sqrt(n_error_terms)
        if norm_error < best_score:
            best_score = norm_error
            best_error = current_error.copy()
            best_error_norm = current_error_norm
            best_max_error = current_max_error
            best_iteration = counter
            best_multipliers = solver._multipliers.copy()
            best_sample_size = sample_size_used
            best_updates.append({
                "iteration": best_iteration,
                "score": best_score,
                "combined_score": current_score,
                "error_norm": best_error_norm,
                "max_error": best_max_error,
                "sample_size": best_sample_size,
            })

        now = time.time()
        print(
            f"Iteracao {counter}/{maxiter} | "
            f"max|erro|={current_max_error:.5f} | "
            f"norm||erro||={current_error_norm / np.sqrt(n_error_terms):.5f} | "
            f"best_norm||erro||={best_score:.5f} | "
            f"amostra={sample_size_used:,} | "
            f"|dtheta|=n/a | "
            f"acc=n/a | "
            f"etapa={now - last_report_time:.1f}s | "
            f"total={now - mch_start:.1f}s",
            flush=True,
        )
        last_report_time = now

    solver._multipliers = best_multipliers.copy()
    solver.multipliers = best_multipliers.copy()

    selected_multipliers = (
        solver.multipliers[parameter_ix] if use_sparse else solver.multipliers.copy()
    )
    run_info = {
        "stop_reason": stop_reason,
        "converged_formally": converged_iteration is not None,
        "converged_iteration": converged_iteration,
        "manual_interrupt": stop_reason == "manual_interrupt",
        "fixed_iterations": True,
        "backtracking_strategy": "none",
        "plateau_triggered": False,
        "plateau_rel_improvement": np.nan,
        "plateau_patience": np.nan,
        "plateau_max_error": np.nan,
        "plateau_streak": 0,
        "final_relative_improvement": np.nan,
        "final_sample_size": sample_size_used,
        "moving_average_window": np.nan,
        "sample_increment": np.nan,
        "sample_growth_factor": np.nan,
        "max_sample_size": sample_size_used,
        "sample_schedule": [sample_size_used],
        "sample_size_history": [sample_size_used],
        "sample_increases": 0,
        "sample_increase_events": [],
        "bad_chain_enabled": False,
        "max_attempts_per_iteration": 1,
        "attempt_decay": np.nan,
        "pre_mc_reject_margin": np.nan,
        "backtracking_rejections": 0,
        "backtracking_events": [],
        "pre_mc_rejections": 0,
        "pre_mc_rejection_events": [],
        "no_improvement_count": 0,
        "no_improvement_events": [],
        "best_iteration": best_iteration,
        "best_score": best_score,
        "best_error": best_error,
        "best_error_norm": best_error_norm,
        "best_error_max_abs": best_max_error,
        "best_sample_size": best_sample_size,
        "best_updates": best_updates,
        "last_iteration": completed_iterations,
        "last_error_norm": current_error_norm,
        "last_error_max_abs": current_max_error,
        "elapsed_s": float(time.time() - mch_start),
    }
    return selected_multipliers, errflag, np.vstack(errors), run_info


def resolver_mch_coniii(sample: np.ndarray,
                        initial_guess: np.ndarray,
                        adjacency_mask: np.ndarray | None = None,
                        sample_size: int = 100_000,
                        maxiter: int = 200,
                        n_iters: int | None = None,
                        burn_in: int | None = None,
                        param_bound: float = 5.0,
                        plateau_rel_improvement: float = 0.0,
                        plateau_patience: int = 4,
                        plateau_max_error: float | None = None,
                        sample_increment: int = 100_000,
                        sample_growth_factor: float = 1.5,
                        max_sample_size: int = 2_500_000,
                        learning_profile: str = "adaptive_samples"
                        ) -> tuple[np.ndarray, int, np.ndarray, dict]:
    """
    Resolve o modelo pairwise por Monte Carlo Histogram (MCH) usando ConIII.

    Com A_ij, usa SparseMCH para ajustar apenas h_i e os J_ij permitidos pela
    rede; os demais acoplamentos ficam fixos em zero.
    """
    sample = sample.astype(np.int64)
    R, N = sample.shape
    sample_size = max(1000, int(sample_size))
    maxiter = max(1, int(maxiter))
    n_iters = int(n_iters or max(100, N * 20))
    burn_in = int(burn_in or max(1000, N * 100))

    parameter_ix = construir_indices_parametros_aij(N, adjacency_mask)
    use_sparse = parameter_ix is not None
    if use_sparse and not hasattr(coniii.solvers, "SparseMCH"):
        raise RuntimeError("Esta versão do ConIII não expõe SparseMCH; MCH com A_ij não é suportado.")
    solver_cls = coniii.solvers.SparseMCH if use_sparse else coniii.solvers.MCH

    solver_kwargs = {
        "sample": sample,
        "sample_size": sample_size,
        "iprint": False,
        "sampler_kw": {"iprint": False},
    }
    if use_sparse:
        solver_kwargs["parameter_ix"] = parameter_ix

    solver = solver_cls(**solver_kwargs)
    constraints = solver.constraints[parameter_ix] if use_sparse else solver.constraints
    initial_selected = initial_guess[parameter_ix] if use_sparse else initial_guess
    learning_profile = normalizar_perfil_mch(learning_profile)
    learning_label = MCH_LEARNING_PROFILES[learning_profile]["label"]

    def mch_schedule(iteration: int, sample_size_current: int):
        return parametros_mch_por_perfil(learning_profile, sample_size_current)

    tol_empirico = 3.0 / np.sqrt(R)
    tol_mc = 3.0 / np.sqrt(sample_size)
    tol_floor = 0.001
    tol = min(tol_empirico, min(tol_mc, tol_floor))
    tol_norm = tol * np.sqrt(len(initial_selected))
    plateau_rel_improvement = max(0.0, float(plateau_rel_improvement))
    plateau_patience = max(1, int(plateau_patience))
    plateau_max_error = (
        None
        if plateau_max_error is None
        else max(tol, float(plateau_max_error))
    )
    sample_increment = max(1, int(sample_increment))
    sample_growth_factor = max(1.0, float(sample_growth_factor))
    max_sample_size = max(sample_size, int(max_sample_size))
    sample_schedule = sorted({
        s for s in [100_000, 200_000, 400_000, 800_000, 1_600_000, 2_500_000]
        if s >= sample_size and s <= max_sample_size
    } | {sample_size, max_sample_size})
    multipliers_selected, errflag, errors, run_info = _resolver_mch_loop_fixo(
        solver=solver,
        constraints=constraints,
        initial_selected=initial_selected,
        use_sparse=use_sparse,
        parameter_ix=parameter_ix,
        tol=tol,
        tol_norm=tol_norm,
        n_iters=n_iters,
        burn_in=burn_in,
        maxiter=maxiter,
        mch_schedule=mch_schedule,
        generate_kwargs={"parallel": False},
    )
    run_info["learning_profile"] = learning_profile
    run_info["learning_profile_label"] = learning_label

    if use_sparse:
        multipliers = solver.multipliers.copy()
    else:
        multipliers = multipliers_selected.copy()

    multipliers = np.clip(multipliers, -param_bound, param_bound)
    if adjacency_mask is not None:
        J = unpack_J(multipliers[N:], N)
        J[~preparar_mascara_adjacencia(adjacency_mask, N)] = 0.0
        multipliers = np.concatenate([multipliers[:N], pack_J(J)])

    return multipliers, errflag, errors, run_info

def calcular_rmse_medias(spin_matrix: np.ndarray, h_inferido: np.ndarray) -> float:
    """
    RMSE do modelo INDEPENDENTE: compara médias empíricas com tanh(h).
    Útil como baseline — se este RMSE é baixo, interações J são desnecessárias.
    """
    m_empirico = spin_matrix.mean(axis=0)
    m_previsto = np.tanh(h_inferido)
    return float(np.sqrt(np.mean((m_empirico - m_previsto)**2)))


def calcular_rmse_modelo_completo(spin_matrix: np.ndarray, multipliers: np.ndarray,
                                   n_amostras_mc: int = 100_000) -> float:
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
                   adjacency_mask: np.ndarray | None = None,
                   metodo_inferencia: str = "auto",
                   mch_sample_size: int = 100_000,
                   mch_maxiter: int = 200,
                   mch_plateau_rel_improvement: float = 0.0,
                   mch_plateau_patience: int = 4,
                   mch_plateau_max_error: float | None = None,
                   mch_learning_profile: str = "adaptive_samples",
                   mch_interactive_continue: bool = False) -> dict:
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
    metodo_escolhido = normalizar_metodo_inferencia(metodo_inferencia)
    mch_info = {}
    temp_mch_custom_checkpoints = []
    try:
        usar_pseudo_por_topologia = adjacency_mask is not None
        if usar_pseudo_por_topologia:
            print("  [A_ij] Inferencia restrita a arestas do grafo; J_ij=0 fora de A_ij.")

        if metodo_escolhido == "auto":
            metodo_escolhido = "exact" if N <= 20 else "mch"
            print(f"  [Auto] Metodo escolhido: {'Enumerate exato' if metodo_escolhido == 'exact' else 'MCH'} para N={N}.")

        param_bound = 5.0
        initial_guess_base = chute_inicial_ising(
            spin_matrix,
            adjacency_mask=adjacency_mask,
            param_bound=param_bound
        )
        pseudo_tuning_info = {}
        exact_info = {}

        if metodo_escolhido in {"mch", "mch_custom"}:
            custom_mch = metodo_escolhido == "mch_custom"
            metodo_str = "MCH-Custom" if custom_mch else "MCH"
            log_prefix = "[MCH-Custom]" if custom_mch else "[MCH]"
            initial_guess, pseudo_tuning_info = torneio_warm_starts(
                spin_matrix,
                initial_guess=initial_guess_base,
                adjacency_mask=adjacency_mask,
                final_lam=lam,
                param_bound=param_bound,
                enabled=True,
            )

            def pedir_int_mch(rotulo: str, atual: int, minimo: int = 1) -> int:
                while True:
                    valor = input(f"  {log_prefix} {rotulo} [{atual:,}]: ").strip()
                    if not valor:
                        return int(atual)
                    try:
                        parsed = int(valor.replace(".", "").replace(",", ""))
                        if parsed >= minimo:
                            return parsed
                    except ValueError:
                        pass
                    print(f"  {log_prefix} Valor invalido para {rotulo}. Use inteiro >= {minimo}.")

            def pedir_perfil_mch(atual: str) -> str:
                atual_label = MCH_LEARNING_PROFILES[normalizar_perfil_mch(atual)]["label"]
                print(f"\n  {log_prefix} Perfil para a próxima rodada:")
                print("    [1] Agressiva")
                print("    [2] Média")
                print("    [3] Conservador")
                print("    [4] Adaptado ao número de amostras")
                print("    [5] Muito agressiva")
                valor = input(f"  Escolha o perfil [atual: {atual_label}]: ").strip().lower()
                if not valor:
                    return normalizar_perfil_mch(atual)
                return normalizar_perfil_mch(valor)

            all_mch_errors = []
            segment_infos = []
            current_initial = initial_guess.copy()
            best_global_score = np.inf
            best_global_multipliers = None
            best_global_error = None
            best_global_info = None
            current_sample_size = int(mch_sample_size)
            current_maxiter = int(mch_maxiter)
            current_profile = normalizar_perfil_mch(mch_learning_profile)
            segment_idx = 0

            while True:
                segment_idx += 1
                print(
                    f"\n  {log_prefix} Rodada {segment_idx}: "
                    f"perfil={MCH_LEARNING_PROFILES[current_profile]['label']}, "
                    f"sample_size={current_sample_size:,}, iteracoes={current_maxiter}."
                )
                if custom_mch:
                    from src.mch_custom import resolver_mch_custom

                    multipliers_segment, errflag_segment, mch_errors_segment, mch_run_info_segment = resolver_mch_custom(
                        spin_matrix,
                        initial_guess=current_initial,
                        adjacency_mask=adjacency_mask,
                        sample_size=current_sample_size,
                        maxiter=current_maxiter,
                        param_bound=param_bound,
                        learning_profile=current_profile,
                    )
                else:
                    multipliers_segment, errflag_segment, mch_errors_segment, mch_run_info_segment = resolver_mch_coniii(
                        spin_matrix,
                        initial_guess=current_initial,
                        adjacency_mask=adjacency_mask,
                        sample_size=current_sample_size,
                        maxiter=current_maxiter,
                        param_bound=param_bound,
                        plateau_rel_improvement=mch_plateau_rel_improvement,
                        plateau_patience=mch_plateau_patience,
                        plateau_max_error=mch_plateau_max_error,
                        learning_profile=current_profile,
                    )
                all_mch_errors.append(mch_errors_segment)
                segment_infos.append(mch_run_info_segment)

                segment_score = float(mch_run_info_segment.get("best_score", np.inf))
                if segment_score < best_global_score or best_global_multipliers is None:
                    best_global_score = segment_score
                    best_global_multipliers = multipliers_segment.copy()
                    best_global_error = np.asarray(
                        mch_run_info_segment.get(
                            "best_error",
                            mch_errors_segment[-1] if len(mch_errors_segment) else [],
                        ),
                        dtype=np.float64,
                    )
                    best_global_info = dict(mch_run_info_segment)

                checkpoint_prefix = "mch_custom" if custom_mch else "mch"
                checkpoint_path = f"{checkpoint_prefix}_melhor_{session_id}_rodada_{segment_idx}.npy"
                np.save(checkpoint_path, best_global_multipliers)
                if custom_mch:
                    temp_mch_custom_checkpoints.append(checkpoint_path)
                print(
                    f"  {log_prefix} Melhor resultado salvo em: {checkpoint_path} | "
                    f"melhor_score={best_global_score:.5f}"
                )

                current_initial = best_global_multipliers.copy()
                if not mch_interactive_continue:
                    break

                continuar = input(
                    f"\n  {log_prefix} Continuar com novas configurações antes das figuras? (s/n) [n]: "
                ).strip().lower()
                if continuar not in {"s", "sim", "y", "yes"}:
                    break

                current_profile = pedir_perfil_mch(current_profile)
                current_sample_size = pedir_int_mch("Nova quantidade de amostras por iteracao", current_sample_size, 1000)
                current_maxiter = pedir_int_mch("Nova quantidade de iteracoes", current_maxiter, 1)

            multipliers = best_global_multipliers
            errflag = int(best_global_info.get("errflag", errflag_segment)) if best_global_info else errflag_segment
            mch_errors = np.vstack(all_mch_errors) if all_mch_errors else np.zeros((0, 0))
            total_mch_iterations = int(sum(max(0, len(segment_errors) - 1) for segment_errors in all_mch_errors))
            mch_run_info = dict(best_global_info or {})
            mch_run_info.update({
                "method": metodo_escolhido,
                "method_label": metodo_str,
                "interactive_continue": bool(mch_interactive_continue),
                "segments": segment_idx,
                "segment_infos": segment_infos,
                "segment_sample_sizes": [int(info.get("best_sample_size", np.nan)) for info in segment_infos],
                "segment_iterations": [int(info.get("last_iteration", 0)) for info in segment_infos],
                "total_segment_iterations": total_mch_iterations,
                "best_global_score": best_global_score,
                "best_error": best_global_error,
                "best_score": best_global_score,
                "last_learning_profile": current_profile,
                "last_learning_profile_label": MCH_LEARNING_PROFILES[current_profile]["label"],
            })
            selected_error = np.asarray(
                best_global_error if best_global_error is not None else [],
                dtype=np.float64,
            )
            mch_info = {
                "errflag": errflag,
                "n_iter": total_mch_iterations,
                "final_error_norm": float(np.linalg.norm(selected_error)) if selected_error.size else np.nan,
                "final_error_max_abs": float(np.max(np.abs(selected_error))) if selected_error.size else np.nan,
                "sample_size": int(mch_run_info.get("best_sample_size", current_sample_size)),
                "maxiter": current_maxiter,
                **mch_run_info,
            }

        elif metodo_escolhido == "pseudo":
            metodo_str = "Pseudo-Likelihood"
            print(f"\n  [{metodo_str}] Iniciando inferência para N={N}, R={R}...")
            print(f"  [Regularização] λ = {lam} (L2 real na pseudo-likelihood)")
            medias = spin_matrix.mean(axis=0).astype(np.float64)
            spins_constantes = np.where(np.abs(medias) > 0.999)[0]
            if len(spins_constantes) > 0:
                print(f"  [Aviso] {len(spins_constantes)} spins quase constantes (|⟨σ⟩| > 0.999). "
                      f"Clamp aplicado para evitar saturação da sigmóide.")

            multipliers, pseudo_tuning_info = auto_tune_pseudo_warm_start(
                spin_matrix,
                initial_guess=initial_guess_base,
                adjacency_mask=adjacency_mask,
                final_lam=lam,
                param_bound=param_bound,
                enabled=True,
            )
            n_bound = int(np.sum(np.abs(multipliers) >= param_bound * 0.999))
            if n_bound > 0:
                print(f"  [PL] Aviso: {n_bound}/{len(multipliers)} parâmetro(s) encostaram no bound ±{param_bound:.1f}.")

        elif metodo_escolhido == "exact":
            metodo_str = "Enumerate-Exact"
            print(f"\n  [{metodo_str}] Iniciando inferência exata para N={N}, R={R}...")
            warm_start, pseudo_tuning_info = torneio_warm_starts(
                spin_matrix,
                initial_guess=initial_guess_base,
                adjacency_mask=adjacency_mask,
                final_lam=lam,
                param_bound=param_bound,
                enabled=True,
            )
            multipliers, exact_info = resolver_enumerate_exato(
                spin_matrix,
                initial_guess=warm_start,
                adjacency_mask=adjacency_mask,
                lam=lam,
                param_bound=param_bound,
            )

        else:
            raise ValueError(f"Metodo normalizado inesperado: {metodo_escolhido}")
            
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
            "diagnostico": diagnostico,
            "mch_info": mch_info or None,
            "pseudo_tuning_info": pseudo_tuning_info,
            "exact_info": exact_info if metodo_str == "Enumerate-Exact" else None,
        }
        
        resultados = {metodo_str: resultado}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        tempo_s = time.time() - t0
        print(f"  [Erro] Falha: {e}")
        if metodo_str is None:
            metodo_str = "MCH" if N > 20 else "Enumerate"
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
            mch_row = dados.get("mch_info") or {}
            pseudo_row = dados.get("pseudo_tuning_info") or {}
            warm_metrics = pseudo_row.get("selected_metrics") or {}
            exact_row = dados.get("exact_info") or {}
            
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
                "viavel": viavel,
                "mch_errflag": mch_row.get("errflag", np.nan),
                "mch_iteracoes": mch_row.get("n_iter", np.nan),
                "mch_final_error_norm": mch_row.get("final_error_norm", np.nan),
                "mch_final_error_max_abs": mch_row.get("final_error_max_abs", np.nan),
                "mch_sample_size": mch_row.get("sample_size", np.nan),
                "mch_maxiter": mch_row.get("maxiter", np.nan),
                "mch_stop_reason": mch_row.get("stop_reason", ""),
                "mch_solver": mch_row.get("solver", ""),
                "mch_backend": mch_row.get("backend", ""),
                "mch_learning_profile": mch_row.get("learning_profile", ""),
                "mch_learning_profile_label": mch_row.get("learning_profile_label", ""),
                "mch_converged_formally": mch_row.get("converged_formally", np.nan),
                "mch_manual_interrupt": mch_row.get("manual_interrupt", np.nan),
                "mch_plateau_triggered": mch_row.get("plateau_triggered", np.nan),
                "mch_plateau_rel_improvement": mch_row.get("plateau_rel_improvement", np.nan),
                "mch_plateau_patience": mch_row.get("plateau_patience", np.nan),
                "mch_plateau_max_error": mch_row.get("plateau_max_error", np.nan),
                "mch_plateau_streak": mch_row.get("plateau_streak", np.nan),
                "mch_final_relative_improvement": mch_row.get("final_relative_improvement", np.nan),
                "mch_final_sample_size": mch_row.get("final_sample_size", np.nan),
                "mch_moving_average_window": mch_row.get("moving_average_window", np.nan),
                "mch_sample_increment": mch_row.get("sample_increment", np.nan),
                "mch_sample_growth_factor": mch_row.get("sample_growth_factor", np.nan),
                "mch_max_sample_size": mch_row.get("max_sample_size", np.nan),
                "mch_sample_schedule": str(mch_row.get("sample_schedule", [])),
                "mch_sample_increases": mch_row.get("sample_increases", np.nan),
                "mch_sample_size_history": str(mch_row.get("sample_size_history", [])),
                "mch_bad_chain_enabled": mch_row.get("bad_chain_enabled", np.nan),
                "mch_backtracking_strategy": mch_row.get("backtracking_strategy", ""),
                "mch_max_attempts_per_iteration": mch_row.get("max_attempts_per_iteration", np.nan),
                "mch_attempt_decay": mch_row.get("attempt_decay", np.nan),
                "mch_pre_mc_reject_margin": mch_row.get("pre_mc_reject_margin", np.nan),
                "mch_pre_mc_rejections": mch_row.get("pre_mc_rejections", np.nan),
                "mch_backtracking_rejections": mch_row.get("backtracking_rejections", np.nan),
                "mch_baseline_resamples": mch_row.get("baseline_resamples", np.nan),
                "mch_pilot_fraction": mch_row.get("pilot_fraction", np.nan),
                "mch_pilot_min": mch_row.get("pilot_min", np.nan),
                "mch_pilot_max": mch_row.get("pilot_max", np.nan),
                "mch_max_pilot_repeats": mch_row.get("max_pilot_repeats", np.nan),
                "mch_pilot_accept_margin": mch_row.get("pilot_accept_margin", np.nan),
                "mch_pilot_reject_margin": mch_row.get("pilot_reject_margin", np.nan),
                "mch_pilot_total_evaluations": mch_row.get("pilot_total_evaluations", np.nan),
                "mch_pilot_extra_evaluations": mch_row.get("pilot_extra_evaluations", np.nan),
                "mch_pilot_uncertain_attempts": mch_row.get("pilot_uncertain_attempts", np.nan),
                "mch_attempt_scales": str(mch_row.get("attempt_scales", [])),
                "mch_subattempts_per_scale": mch_row.get("subattempts_per_scale", np.nan),
                "mch_scale_evaluations": mch_row.get("scale_evaluations", np.nan),
                "mch_subattempt_evaluations": mch_row.get("subattempt_evaluations", np.nan),
                "mch_learning_steps_mean": mch_row.get("mch_learning_steps_mean", np.nan),
                "mch_learning_steps_last": mch_row.get("mch_learning_steps_last", np.nan),
                "mch_prediction_distance_last": mch_row.get("mch_prediction_distance_last", np.nan),
                "mch_delta_norm_last": mch_row.get("mch_delta_norm_last", np.nan),
                "mch_delta_max_abs_last": mch_row.get("mch_delta_max_abs_last", np.nan),
                "mch_step_stop_reason_last": mch_row.get("mch_step_stop_reason_last", ""),
                "mch_no_improvement_count": mch_row.get("no_improvement_count", np.nan),
                "mch_n_chains": mch_row.get("n_chains", np.nan),
                "mch_acceptance_rate_mean": mch_row.get("acceptance_rate_mean", np.nan),
                "mch_acceptance_rate_last": mch_row.get("acceptance_rate_last", np.nan),
                "mch_best_iteration": mch_row.get("best_iteration", np.nan),
                "mch_best_score": mch_row.get("best_score", np.nan),
                "mch_best_error_norm": mch_row.get("best_error_norm", np.nan),
                "mch_best_error_max_abs": mch_row.get("best_error_max_abs", np.nan),
                "mch_best_sample_size": mch_row.get("best_sample_size", np.nan),
                "mch_last_iteration": mch_row.get("last_iteration", np.nan),
                "mch_last_error_norm": mch_row.get("last_error_norm", np.nan),
                "mch_last_error_max_abs": mch_row.get("last_error_max_abs", np.nan),
                "warm_start_selected": pseudo_row.get("selected", ""),
                "warm_start_score": warm_metrics.get("score", np.nan),
                "warm_start_pair_rmse": warm_metrics.get("pair_rmse", np.nan),
                "warm_start_pair_r": warm_metrics.get("pair_r", np.nan),
                "warm_start_mean_rmse": warm_metrics.get("mean_rmse", np.nan),
                "warm_start_mc_score": warm_metrics.get("mc_score", np.nan),
                "warm_start_mc_error_norm": warm_metrics.get("mc_error_norm", np.nan),
                "warm_start_mc_error_max_abs": warm_metrics.get("mc_error_max_abs", np.nan),
                "warm_start_n_bound": warm_metrics.get("n_bound", np.nan),
                "pseudo_tuning_enabled": pseudo_row.get("enabled", False),
                "pseudo_tuning_rounds": len(pseudo_row.get("history", [])),
                "pseudo_tuning_final_score": pseudo_row.get("final_score", np.nan),
                "exact_success": exact_row.get("success", np.nan),
                "exact_nit": exact_row.get("nit", np.nan),
                "exact_moment_error_norm": exact_row.get("moment_error_norm", np.nan),
                "exact_moment_error_max_abs": exact_row.get("moment_error_max_abs", np.nan),
                "exact_n_states": exact_row.get("n_states", np.nan),
            })
            
            # Salvar multiplicadores
            npy_path = f"multiplicadores_ising_{session_id}.npy"
            np.save(npy_path, dados['multipliers'])
            print(f"\n[Salvo] Multiplicadores salvos em: {npy_path}")
                
    df_comp = pd.DataFrame(tabela)
    csv_path = f"comparacao_metodos_{session_id}.csv"
    df_comp.to_csv(csv_path, index=False)
    temp_mch_custom_checkpoints.extend(glob.glob(f"mch_custom_melhor_{session_id}*.npy"))
    remover_arquivos_temporarios_npy(
        temp_mch_custom_checkpoints,
        label="checkpoint(s) temporário(s) MCH-Custom"
    )
    
    return resultados

def gerar_figura2(spin_matrix: np.ndarray, resultados_inferencia: dict,
                  gexf_path: str, node_names: list, session_id: str,
                  model_samples: np.ndarray | None = None,
                  n_amostras_mc: int = 100_000):
    """
    Figura 2: Covariância empírica e covariância reconstruída.
    Aplica filtro topológico substituindo pares sem A_ij por NaN. A diagonal i=j
    também é mascarada, pois não representa uma covariância entre usuários distintos.
    """
    print("\n[Figura 2] Inicializando geração dos 2 heatmaps de covariância lado a lado...")
    # Mesma orientação da inferência: linhas = keywords/amostras, colunas = usuários/spins.
    spin_matrix = spin_matrix.T
    R, N = spin_matrix.shape

    def calcular_covariancia_centrada(amostras: np.ndarray) -> np.ndarray:
        amostras = np.asarray(amostras, dtype=np.float64)
        if amostras.ndim != 2 or amostras.shape[0] == 0:
            return np.zeros((N, N), dtype=np.float64)
        centrado = amostras - amostras.mean(axis=0, keepdims=True)
        C = (centrado.T @ centrado) / float(amostras.shape[0])
        return C

    C_emp = calcular_covariancia_centrada(spin_matrix)

    def validar_amostras_modelo(amostras: np.ndarray, contexto: str) -> np.ndarray:
        amostras = np.asarray(amostras, dtype=np.float64)
        if amostras.ndim != 2 or amostras.shape[1] != N:
            raise ValueError(
                f"Amostras do modelo inválidas para {contexto}: esperado (*, {N}), "
                f"recebido {amostras.shape}."
            )
        return amostras
    
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
            # Constrói a máscara manualmente para suportar nós ausentes
            A_mask = np.zeros((N, N), dtype=bool)
            name_to_idx = {n: i for i, n in enumerate(node_names)}
            for u, v in G.edges():
                if u in name_to_idx and v in name_to_idx:
                    i_u, i_v = name_to_idx[u], name_to_idx[v]
                    A_mask[i_u, i_v] = True
                    A_mask[i_v, i_u] = True
    else:
        print(f"  [Aviso] GEXF '{gexf_path}' não encontrado. Filtro topológico desabilitado.")

    def aplicar_mascara(matriz):
        m_copy = matriz.copy()
        if aplicar_filtro:
            m_copy[~A_mask] = np.nan
        np.fill_diagonal(m_copy, np.nan)
        return m_copy

    # Extrai o primeiro resultado independente do nome do método
    res_vals = list(resultados_inferencia.values())[0] if len(resultados_inferencia) > 0 else {}
    multipliers = res_vals.get("multipliers")

    C_model = np.full((N, N), np.nan, dtype=np.float64)
    if multipliers is None or "ERROR" in res_vals:
        print("  [Aviso] Nenhum modelo válido encontrado. Painel reconstruído ficará vazio.")
    elif N <= 20:
        print(f"  [Modelo] Covariância reconstruída por enumeração exata (2^{N} estados).")
        states, probs = _estados_modelo_exato(multipliers, N)
        medias_modelo = probs @ states
        centrado = states - medias_modelo
        C_model = (centrado * probs[:, None]).T @ centrado
    else:
        if model_samples is not None:
            amostras_modelo = validar_amostras_modelo(model_samples, "Figura 2")
            print(
                "  [Modelo] Covariância reconstruída reutilizando "
                f"{amostras_modelo.shape[0]:,} amostras Monte Carlo."
            )
        else:
            n_amostras_mc = int(n_amostras_mc)
            print(
                "  [Modelo] Covariância reconstruída por Monte Carlo "
                f"({n_amostras_mc:,} amostras; N={N})."
            )
            amostras_modelo = _amostrar_modelo_metropolis(
                multipliers,
                N,
                n_amostras_mc,
                label="covariância reconstruída da Figura 2"
            )
        C_model = calcular_covariancia_centrada(amostras_modelo)

    C_emp_masked = aplicar_mascara(C_emp)
    C_model_masked = aplicar_mascara(C_model)

    cov_values = np.concatenate([
        C_emp_masked[np.isfinite(C_emp_masked)],
        C_model_masked[np.isfinite(C_model_masked)],
    ])
    cov_vmax = float(np.max(cov_values)) if cov_values.size else 0.0
    if not np.isfinite(cov_vmax) or cov_vmax <= 0:
        cov_vmax = 1.0

    # Configuração da figura matplotlib
    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    }):
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), facecolor='white')

        cmap = matplotlib.colormaps.get_cmap('jet').copy()
        cmap.set_bad(color='white')

        paineis = [
            (C_emp_masked, "Covariância Empírica"),
            (C_model_masked, "Covariância Reconstruída"),
        ]

        for i, (mat, titulo) in enumerate(paineis):
            ax = axes[i]
            ax.set_facecolor('white')

            mat_vis = np.asarray(mat, dtype=np.float64)
            if np.isnan(mat_vis).all():
                ax.set_title(titulo + " (Falhou/Timeout)", color='black', pad=10)
                ax.axis('off')
                continue

            im = ax.imshow(
                mat_vis,
                cmap=cmap,
                vmin=0.0,
                vmax=cov_vmax,
                aspect='auto',
                interpolation='nearest'
            )
            ax.set_title(titulo, color='black', pad=10)
            ax.set_xlabel("Spin $j$")
            ax.set_ylabel("Spin $i$")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            for spine in ax.spines.values():
                spine.set_edgecolor('black')

        plt.tight_layout()
        fig_path = f"figura2_ising_{session_id}.png"
        plt.savefig(fig_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)

    print(f"\n[Figura 2] Heatmaps com 2 painéis salvos com sucesso em: {fig_path}")


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
                         n_amostras: int = 100_000,
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
                                model_samples: np.ndarray | None = None,
                                compact_stdout: bool = False):
    """
    Diagnostica se o modelo reproduz C_ij antes de interpretar tripletos.
    """
    pair_idx, c_emp = _calcular_pares(spin_matrix.astype(np.float64))
    if model_samples is not None:
        _, c_modelo = _calcular_pares(model_samples.astype(np.float64), pair_indices=pair_idx)
    else:
        _, c_modelo = _pares_modelo_exato(multipliers, N, pair_indices=pair_idx)

    if not compact_stdout:
        print("\n  [Diagnostico pares] C_ij empirico vs modelo:")

    groups = [("todos", np.ones(len(pair_idx), dtype=bool))]
    if adjacency_mask is not None:
        A = preparar_mascara_adjacencia(adjacency_mask, N)
        pair_mask = np.array([A[i, j] for i, j in pair_idx], dtype=bool)
        groups.append(("A_ij=1", pair_mask))
        groups.append(("A_ij=0", ~pair_mask))

    compact_parts = []
    for label, mask in groups:
        n = int(mask.sum())
        if n < 2:
            if compact_stdout:
                compact_parts.append(f"{label}: n={n}")
            else:
                print(f"    {label:<8}: n={n} pares (insuficiente para Pearson)")
            continue
        r, p = _safe_pearson(c_emp[mask], c_modelo[mask])
        rmse = float(np.sqrt(np.mean((c_emp[mask] - c_modelo[mask]) ** 2)))
        if np.isnan(r):
            if compact_stdout:
                compact_parts.append(f"{label}: RMSE={rmse:.5f}, r=nan")
            else:
                print(f"    {label:<8}: n={n} pares | RMSE={rmse:.5f} | Pearson indisponivel")
        else:
            if compact_stdout:
                compact_parts.append(f"{label}: RMSE={rmse:.5f}, r={r:.4f}")
            else:
                print(f"    {label:<8}: n={n} pares | RMSE={rmse:.5f} | Pearson r={r:.4f} (p={p:.2e})")

    if compact_stdout and compact_parts:
        print("  [Pares] " + " | ".join(compact_parts))


def gerar_figura3(spin_matrix: np.ndarray, resultados_inferencia: dict,
                  session_id: str,
                  adjacency_mask: np.ndarray | None = None,
                  triplet_mode: str = "connected",
                  filename_suffix: str | None = None,
                  model_samples: np.ndarray | None = None,
                  diagnose_pairs: bool = True,
                  n_amostras_mc: int = 100_000,
                  compact_stdout: bool = False):
    """
    Figura 3: Correlação de Tripletos (Schneidman / Hall & Bialek).
    
    Scatter plot comparando C_ijk empírico vs C_ijk previsto pelo modelo
    de máxima entropia pairwise, com barras de erro bootstrap e coeficiente
    de Pearson.
    """
    # Mesma orientação da inferência: linhas = keywords/amostras, colunas = usuários/spins.
    spin_matrix = spin_matrix.T
    R, N = spin_matrix.shape

    def log(message: str = ""):
        if not compact_stdout:
            print(message)

    log(f"\n[Figura 3] Correlação de Tripletos (N={N}, R={R})...")
    
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
    log(
        "  [Tripletos A_ij] "
        f"todos={topologia_counts['todos']}, "
        f"sem_arestas={topologia_counts['sem_arestas']}, "
        f"uma_aresta={topologia_counts['uma_aresta']}, "
        f"conectados={topologia_counts['conectados']}, "
        f"triangulos={topologia_counts['triangulos']}"
    )
    log(f"  [Tripletos A_ij] Modo usado na Figura 3: {triplet_mode_used}")

    if len(triplet_indices) < 2:
        print("  [Erro] Menos de 2 tripletos selecionados. Pulando Figura 3.")
        return None
    
    # 1. Tripletos empíricos
    log("  [1/3] Calculando tripletos empíricos...")
    trip_idx, c_emp = _calcular_tripletos(
        spin_matrix.astype(np.float64),
        triplet_indices=triplet_indices
    )
    n_triplets = len(trip_idx)
    log(f"        {n_triplets} tripletos selecionados")
    
    # 2. Tripletos do modelo
    log("  [2/3] Calculando tripletos do modelo...")
    if N <= 20:
        log(f"        Modo EXATO (2^{N} = {2**N} estados)")
        _, c_modelo = _tripletos_modelo_exato(
            multipliers,
            N,
            triplet_indices=triplet_indices
        )
    else:
        log(f"        Modo MONTE CARLO (N={N} > 20)")
        if model_samples is not None:
            log(f"  [Metropolis] Reutilizando {model_samples.shape[0]:,} amostras do modelo.")
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
            model_samples=model_samples,
            compact_stdout=compact_stdout
        )
    
    # 3. Bootstrap (incerteza)
    log("  [3/3] Estimando incerteza via Bootstrap (200 reamostras)...")
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
        log("\n  [Resultado] Pearson indisponivel: variancia zero ou poucos tripletos.")
    else:
        log(f"\n  [Resultado] Pearson r = {r_pearson:.4f} (p = {p_value:.2e})")
    log("  [Metricas tripletos]")
    log(f"    RMSE={rmse_triplet:.5f} | MAE={mae_triplet:.5f} | bias(modelo-emp)={bias_triplet:+.5f}")
    log(f"    std(C_emp)={std_emp:.5f} | std(C_modelo)={std_modelo:.5f}")
    log(f"    mean|C_emp|={mean_abs_emp:.5f} | mean|C_modelo|={mean_abs_modelo:.5f}")
    if np.isfinite(ratio_rmse_bootstrap):
        log(
            f"    RMSE/erro_bootstrap={ratio_rmse_bootstrap:.2f} | "
            f"frac. dentro de 1σ={frac_within_1sigma:.3f}"
        )
    else:
        log("    RMSE/erro_bootstrap indisponivel.")
    log("  [Referencia] Hall & Bialek: r em [0.93, 0.95] para o conjunto analisado no artigo.")
    if adjacency_mask is not None and triplet_mode_used != "all":
        log("  [Nota] Aqui a Figura 3 usa apenas tripletos acoplados/conectados por A_ij.")
    if not np.isnan(r_pearson) and r_pearson >= 0.90:
        log(f"  ✔ Correlação de tripletos dentro do esperado!")
    else:
        log(f"  ⚠ Correlação abaixo do esperado. Possíveis causas: poucos dados ou correlações de alta ordem.")
    
    # 5. Gerar gráfico em estilo próximo ao artigo: pontos pretos e linha y=x.
    def axis_limits(values: np.ndarray) -> tuple[float, float]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return 0.0, 0.01

        data_min = float(finite.min())
        data_max = float(finite.max())
        span = max(data_max - data_min, abs(data_max) * 0.1, 0.01)
        lower = 0.0 if data_min >= 0 else data_min - 0.05 * span
        upper = data_max + 0.05 * span
        if upper <= lower:
            upper = lower + 0.01
        return lower, upper

    x_lim = axis_limits(c_emp)
    y_lim = axis_limits(c_modelo)
    identity_min = max(x_lim[0], y_lim[0])
    identity_max = min(x_lim[1], y_lim[1])

    plot_style = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.labelsize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    }
    with plt.rc_context(plot_style):
        fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor='white')
        ax.set_facecolor('white')

        ax.plot(
            [identity_min, identity_max],
            [identity_min, identity_max],
            color='red',
            linewidth=0.8,
            zorder=1,
        )
        ax.scatter(
            c_emp,
            c_modelo,
            s=5,
            color='black',
            edgecolors='none',
            alpha=0.95,
            label=(
                r'Pearson $r$ = nan'
                if np.isnan(r_pearson)
                else rf'Pearson $r$ = {r_pearson:.4f}'
            ),
            zorder=2,
        )

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_xlabel(r'$C_{ijk}$ Empirical')
        ax.set_ylabel(r'$C_{ijk}$ Predicted')
        ax.tick_params(direction='in', top=True, right=True, width=0.6, length=3)
        ax.legend(
            loc='best',
            frameon=True,
            fancybox=False,
            edgecolor='black',
            fontsize=9,
        )

        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(0.8)

        fig.tight_layout()
    suffix = ""
    if filename_suffix:
        suffix_clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(filename_suffix))
        suffix = f"_{suffix_clean}"
    fig_path = f"figura3_tripletos{suffix}_{session_id}.png"
    fig.savefig(fig_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    
    if compact_stdout:
        r_text = "nan" if np.isnan(r_pearson) else f"{r_pearson:.4f}"
        print(
            f"  [Figura 3] {triplet_mode_used}: n={n_triplets}, "
            f"r={r_text}, RMSE={rmse_triplet:.5f}, "
            f"std_emp={std_emp:.5f}, std_modelo={std_modelo:.5f}"
        )
    else:
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
                            triplet_modes: list[str] | tuple[str, ...] = ("all",),
                            n_amostras_mc: int = 100_000,
                            model_samples: np.ndarray | None = None):
    """
    Roda a Figura 3 nos filtros topologicos de tripletos solicitados e salva um resumo.
    """
    # Prepara uma unica amostra Monte Carlo para todos os modos, quando necessario.
    spin_matrix_model = spin_matrix.T
    _, N = spin_matrix_model.shape
    res_vals = list(resultados_inferencia.values())[0] if resultados_inferencia else {}
    if "multipliers" not in res_vals or "ERROR" in res_vals:
        print("\n[Figura 3] Nenhum modelo válido encontrado. Pulando Figura 3 multimodo.")
        return []

    if adjacency_mask is None and "A_ij" in res_vals:
        adjacency_mask = res_vals.get("A_ij")
    A = preparar_mascara_adjacencia(adjacency_mask, N) if adjacency_mask is not None else None
    topologia_counts = _contar_tripletos_por_topologia(N, A)

    print(
        "\n[Figura 3] Comparando modos de tripletos "
        f"{list(triplet_modes)} (N={N})."
    )
    print(
        "  [Tripletos A_ij] "
        f"todos={topologia_counts['todos']}, "
        f"sem_arestas={topologia_counts['sem_arestas']}, "
        f"uma_aresta={topologia_counts['uma_aresta']}, "
        f"conectados={topologia_counts['conectados']}, "
        f"triangulos={topologia_counts['triangulos']}"
    )

    if N > 20 and "multipliers" in res_vals and "ERROR" not in res_vals:
        if model_samples is not None:
            model_samples = np.asarray(model_samples, dtype=np.float64)
            if model_samples.ndim != 2 or model_samples.shape[1] != N:
                raise ValueError(
                    f"Amostras do modelo inválidas para Figura 3: esperado (*, {N}), "
                    f"recebido {model_samples.shape}."
                )
            print(f"  [Metropolis] Figura 3 reutilizando {model_samples.shape[0]:,} amostras do modelo.")
        else:
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
            n_amostras_mc=n_amostras_mc,
            compact_stdout=True
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
        print(f"\n[Figura 3] Resumo de tripletos salvo em: {resumo_path}")

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


def _distribuicao_Q_discreta(Q_values: np.ndarray, N: int, density: bool = True):
    """
    Converte amostras de Q em P(Q=k/N).

    Como Q e discreto para N usuarios, a Figura 4 deve ser plotada nos pontos
    k/N. Quando density=True, divide por Delta Q=1/N para aproximar a densidade
    usada visualmente no artigo, sem usar KDE.
    """
    if N <= 0:
        return np.array([]), np.array([])

    q = np.asarray(Q_values, dtype=np.float64)
    k = np.rint(q * N).astype(int)
    k = np.clip(k, 0, N)

    counts = np.bincount(k, minlength=N + 1).astype(np.float64)
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    if density:
        probs = probs * N

    q_grid = np.arange(N + 1, dtype=np.float64) / N
    return q_grid, probs


def _distribuicao_Q_exata_discreta(q_values: np.ndarray, probabilities: np.ndarray,
                                   N: int, density: bool = True):
    """
    Agrupa probabilidades exatas por Q=k/N.
    """
    if N <= 0:
        return np.array([]), np.array([])

    q = np.asarray(q_values, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    k = np.rint(q * N).astype(int)
    k = np.clip(k, 0, N)

    probs = np.zeros(N + 1, dtype=np.float64)
    np.add.at(probs, k, p)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    if density:
        probs = probs * N

    q_grid = np.arange(N + 1, dtype=np.float64) / N
    return q_grid, probs


def _plot_Q_discreto(ax, q_values, p_values, label, color, marker, linestyle="-",
                     linewidth=1.4, markersize=4.0, zorder=3):
    """
    Plota apenas pontos com P(Q)>0 para escala logaritmica.
    """
    q_values = np.asarray(q_values, dtype=np.float64)
    p_values = np.asarray(p_values, dtype=np.float64)
    mask = np.isfinite(q_values) & np.isfinite(p_values) & (p_values > 0)
    if not np.any(mask):
        return

    ax.plot(
        q_values[mask],
        p_values[mask],
        color=color,
        marker=marker,
        markersize=markersize,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
        zorder=zorder,
    )


def gerar_figura4(spin_matrix: np.ndarray, resultados_inferencia: dict,
                  session_id: str,
                  model_samples: np.ndarray | None = None,
                  n_amostras_mc: int = 100_000):
    """
    Figura 4: Distribuição de Atividade Coletiva P(Q).
    
    Compara três distribuições discretas em Q=k/N:
      - Empírica (Vermelho): dados reais
      - Independente (Ciano): modelo sem interações (J=0)
      - Pairwise/Ising (Verde): modelo completo inferido
    
    Eixo Y em escala logarítmica para visualizar caudas.
    Inclui cálculo da suscetibilidade χ = N * Var(Q).
    """
    # Mesma orientação da inferência: linhas = keywords/amostras, colunas = usuários/spins.
    spin_matrix = spin_matrix.T
    R, N = spin_matrix.shape
    print(f"\n[Figura 4] Gerando distribuição discreta de atividade coletiva Q (N={N}, R={R})...")
    
    # Extrair multiplicadores do resultado
    res_vals = list(resultados_inferencia.values())[0] if resultados_inferencia else {}
    if "multipliers" not in res_vals or "ERROR" in res_vals:
        print("  [Erro] Nenhum modelo válido encontrado. Pulando Figura 4.")
        return None
    
    multipliers = res_vals["multipliers"]
    
    # 1. Q empírico
    Q_emp = _calcular_Q(spin_matrix.astype(np.float64))
    
    # 2. Q do modelo independente
    Q_indep = _distribuicao_Q_independente(spin_matrix.astype(np.float64))
    
    # 3. Q do modelo pairwise
    use_exact = (N <= 20)
    
    if use_exact:
        print(f"  [Modelo] Pairwise exato (2^{N} = {2**N} estados).")
        q_exact, p_exact = _distribuicao_Q_pairwise_exato(multipliers, N)
    else:
        if model_samples is not None:
            model_samples = np.asarray(model_samples, dtype=np.float64)
            if model_samples.ndim != 2 or model_samples.shape[1] != N:
                raise ValueError(
                    f"Amostras do modelo inválidas para Figura 4: esperado (*, {N}), "
                    f"recebido {model_samples.shape}."
                )
            print(f"  [Modelo] Pairwise reutilizando {model_samples.shape[0]:,} amostras Monte Carlo.")
            Q_pairwise_mc = _calcular_Q(model_samples)
        else:
            print(f"  [Modelo] Pairwise por Monte Carlo (N={N} > 20).")
            Q_pairwise_mc = _distribuicao_Q_pairwise_mc(multipliers, N, n_amostras=int(n_amostras_mc))
    
    # 4. Estatísticas de suscetibilidade
    chi_emp = N * np.var(Q_emp)
    chi_indep = N * np.var(Q_indep)
    chi_ratio = chi_emp / max(chi_indep, 1e-10)
    print(f"  [Suscetibilidade] χ_emp={chi_emp:.4f} | χ_indep={chi_indep:.4f} | razão={chi_ratio:.2f}x")
    if chi_emp > 2 * chi_indep:
        print(f"  ✔ Comportamento coletivo detectado.")
    else:
        print(f"  ⚠ Comportamento coletivo fraco.")

    # 5. Distribuicoes discretas em Q=k/N. Dividimos por Delta Q=1/N
    # para obter densidade discreta comparavel a Figura 4 do artigo.
    q_emp, p_emp = _distribuicao_Q_discreta(Q_emp, N, density=True)
    q_indep, p_indep = _distribuicao_Q_discreta(Q_indep, N, density=True)
    if use_exact:
        q_pair, p_pair = _distribuicao_Q_exata_discreta(q_exact, p_exact, N, density=True)
    else:
        q_pair, p_pair = _distribuicao_Q_discreta(Q_pairwise_mc, N, density=True)
    
    # 6. Grafico em estilo mais proximo da Figura 4 do artigo.
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })
    fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor='white')
    ax.set_facecolor('white')

    _plot_Q_discreto(
        ax, q_emp, p_emp,
        label='Empirical',
        color='#d62728',
        marker='D',
        linestyle='-',
        linewidth=1.2,
        markersize=3.2,
        zorder=4,
    )
    _plot_Q_discreto(
        ax, q_pair, p_pair,
        label='Ising Reconstructed',
        color='#1b7f1b',
        marker='D',
        linestyle='-',
        linewidth=1.2,
        markersize=3.0,
        zorder=3,
    )
    _plot_Q_discreto(
        ax, q_indep, p_indep,
        label='Independent',
        color='#17becf',
        marker='D',
        linestyle='--',
        linewidth=1.0,
        markersize=2.8,
        zorder=2,
    )
    
    # Escala logarítmica no eixo Y
    ax.set_yscale('log')
    ax.set_xlim(0, 1)

    positive = np.concatenate([p_emp[p_emp > 0], p_indep[p_indep > 0], p_pair[p_pair > 0]])
    if positive.size:
        y_min = max(positive.min() * 0.5, 1e-5)
        y_max = min(max(positive.max() * 2.0, 1e-2), 1e3)
        ax.set_ylim(y_min, y_max)

    ax.set_xlabel("$Q$")
    ax.set_ylabel("$P(Q)$")
    ax.tick_params(direction='in', top=True, right=True)
    ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    
    plt.tight_layout()
    fig_path = f"figura4_distribuicao_Q_{session_id}.png"
    plt.savefig(fig_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    plt.rcdefaults()
    
    print(f"\n[Figura 4] Gráfico salvo em: {fig_path}")
    
    return {
        "chi_emp": chi_emp,
        "chi_indep": chi_indep,
        "chi_ratio": chi_ratio,
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
    parser.add_argument("--metodo", choices=["auto", "mch", "mch_custom", "exact"], default="auto",
                        help="Método de inferência: auto, mch, mch_custom ou exact (padrão: auto)")
    parser.add_argument("--mch-sample-size", type=int, default=100_000,
                        help="Número inicial de amostras MCMC por iteração do MCH (padrão: 100000)")
    parser.add_argument("--mch-maxiter", type=int, default=200,
                        help="Máximo de iterações externas do MCH (padrão: 200)")
    parser.add_argument("--mch-plateau-improvement", type=float, default=0.0,
                        help="Melhoria relativa mínima da média móvel antes de aumentar a amostra (padrão: 0.0; apenas piora)")
    parser.add_argument("--mch-plateau-patience", type=int, default=4,
                        help="Tamanho de cada janela da média móvel do MCH (padrão: 4)")
    parser.add_argument("--mch-plateau-max-error", type=float, default=None,
                        help="Limite opcional da média móvel para aumentar a amostra (padrão: sem limite)")
    parser.add_argument("--mch-learning-profile",
                        choices=["very_aggressive", "aggressive", "medium", "conservative", "adaptive_samples"],
                        default="adaptive_samples",
                        help="Perfil dos parâmetros do solver MCH (padrão: adaptive_samples)")
    parser.add_argument("--validar", action="store_true", help="Executa o Monte Carlo Metropolis para o modelo campeão")
    args = parser.parse_args()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"[{session_id}] Inicializando script CLI do ConIII Ising.")
    
    df = pd.read_csv(args.csv_path, index_col=0)
    # NÃO transpor: inferir_modelo() transpõe internamente
    S = df.values.astype(np.int64)
    S = np.where(S > 0, 1, -1).astype(np.int64)  # Confiança +1/-1
    node_names = list(df.index)
    adjacency_mask = construir_mascara_adjacencia(args.gexf_path, node_names)
    
    # Executa Inferência
    resultados = inferir_modelo(
        S,
        session_id,
        lam=args.lam,
        adjacency_mask=adjacency_mask,
        metodo_inferencia=args.metodo,
        mch_sample_size=args.mch_sample_size,
        mch_maxiter=args.mch_maxiter,
        mch_plateau_rel_improvement=args.mch_plateau_improvement,
        mch_plateau_patience=args.mch_plateau_patience,
        mch_plateau_max_error=args.mch_plateau_max_error,
        mch_learning_profile=args.mch_learning_profile,
    )
    
    # Gera a figura
    gerar_figura2(S, resultados, args.gexf_path, node_names, session_id)
    
    # Caso flag `--validar` providenciado
    if args.validar:
        validar_modelo(resultados, len(node_names))
        
    print(f"\n[CLI] Execução finalizada (Sessão {session_id}).")
