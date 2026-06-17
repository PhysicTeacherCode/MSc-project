"""
Custom Monte Carlo Histogram-like solver for pairwise Ising models.

This module is intentionally narrower than ConIII: spins are {-1, +1}, the
model is pairwise Ising, and optional A_ij support is represented as an edge
list. The expensive part is Metropolis sampling; Numba is used when available.
"""

from __future__ import annotations

import os
import time

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only when numba is absent.
    NUMBA_AVAILABLE = False
    njit = None
    prange = range


def _is_manual_interrupt_exception(exc: BaseException) -> bool:
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, KeyboardInterrupt):
            return True
        current = current.__cause__ or current.__context__

    text = str(exc)
    return (
        isinstance(exc, SystemError)
        and "CPUDispatcher" in text
        and "returned a result with an exception set" in text
    )


def _prepare_edges(n: int, adjacency_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if adjacency_mask is None:
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
    else:
        mask = np.asarray(adjacency_mask, dtype=bool).copy()
        mask = np.logical_or(mask, mask.T)
        np.fill_diagonal(mask, False)

    edge_i = []
    edge_j = []
    dense_idx = []
    idx = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if mask[i, j]:
                edge_i.append(i)
                edge_j.append(j)
                dense_idx.append(idx)
            idx += 1

    return (
        np.asarray(edge_i, dtype=np.int64),
        np.asarray(edge_j, dtype=np.int64),
        np.asarray(dense_idx, dtype=np.int64),
    )


def _edge_params_from_full(initial_guess: np.ndarray, n: int, dense_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    initial_guess = np.asarray(initial_guess, dtype=np.float64)
    h = initial_guess[:n].copy()
    j_full = initial_guess[n:].copy()
    j_edges = j_full[dense_idx].copy() if dense_idx.size else np.zeros(0, dtype=np.float64)
    return h, j_edges


def _assemble_full_multipliers(h: np.ndarray,
                               j_edges: np.ndarray,
                               dense_idx: np.ndarray,
                               n: int,
                               param_bound: float) -> np.ndarray:
    j_full = np.zeros(n * (n - 1) // 2, dtype=np.float64)
    if dense_idx.size:
        j_full[dense_idx] = j_edges
    multipliers = np.concatenate([h, j_full])
    return np.clip(multipliers, -float(param_bound), float(param_bound))


def _j_dense_from_edges(j_edges: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, n: int) -> np.ndarray:
    j_dense = np.zeros((n, n), dtype=np.float64)
    for k in range(len(j_edges)):
        i = int(edge_i[k])
        j = int(edge_j[k])
        value = float(j_edges[k])
        j_dense[i, j] = value
        j_dense[j, i] = value
    return j_dense


def _empirical_observables(sample: np.ndarray,
                           edge_i: np.ndarray,
                           edge_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_f = np.asarray(sample, dtype=np.float64)
    means = sample_f.mean(axis=0)
    if edge_i.size:
        pairs = (sample_f[:, edge_i] * sample_f[:, edge_j]).mean(axis=0)
    else:
        pairs = np.zeros(0, dtype=np.float64)
    return means.astype(np.float64), pairs.astype(np.float64)


if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _metropolis_flip(state, h, j_dense, site):
        local_field = h[site]
        n = state.shape[0]
        for j in range(n):
            local_field += j_dense[site, j] * state[j]

        delta_logp = -2.0 * state[site] * local_field
        if delta_logp >= 0.0 or np.random.random() < np.exp(delta_logp):
            state[site] = -state[site]
            return 1
        return 0


    @njit(parallel=True, cache=True)
    def _sample_observables_numba(h,
                                  j_dense,
                                  edge_i,
                                  edge_j,
                                  sample_size,
                                  n_chains,
                                  n_iters,
                                  burn_in,
                                  seed):
        n = h.shape[0]
        m = edge_i.shape[0]
        means_by_chain = np.zeros((n_chains, n), dtype=np.float64)
        pairs_by_chain = np.zeros((n_chains, m), dtype=np.float64)
        counts = np.zeros(n_chains, dtype=np.int64)
        accepted = np.zeros(n_chains, dtype=np.int64)
        proposed = np.zeros(n_chains, dtype=np.int64)

        base = sample_size // n_chains
        extra = sample_size % n_chains

        for chain in prange(n_chains):
            np.random.seed(seed + 1009 * chain)
            state = np.empty(n, dtype=np.int8)
            for i in range(n):
                state[i] = 1 if np.random.random() < 0.5 else -1

            for _ in range(burn_in):
                site = int(np.random.random() * n)
                accepted[chain] += _metropolis_flip(state, h, j_dense, site)
                proposed[chain] += 1

            n_local = base + (1 if chain < extra else 0)
            for _ in range(n_local):
                for _ in range(n_iters):
                    site = int(np.random.random() * n)
                    accepted[chain] += _metropolis_flip(state, h, j_dense, site)
                    proposed[chain] += 1

                for i in range(n):
                    means_by_chain[chain, i] += state[i]
                for e in range(m):
                    pairs_by_chain[chain, e] += state[edge_i[e]] * state[edge_j[e]]
                counts[chain] += 1

        total = 0
        for chain in range(n_chains):
            total += counts[chain]

        means = np.zeros(n, dtype=np.float64)
        pairs = np.zeros(m, dtype=np.float64)
        for chain in range(n_chains):
            for i in range(n):
                means[i] += means_by_chain[chain, i]
            for e in range(m):
                pairs[e] += pairs_by_chain[chain, e]

        if total > 0:
            for i in range(n):
                means[i] /= total
            for e in range(m):
                pairs[e] /= total

        accepted_total = 0
        proposed_total = 0
        for chain in range(n_chains):
            accepted_total += accepted[chain]
            proposed_total += proposed[chain]

        return means, pairs, accepted_total, proposed_total


    @njit(parallel=True, cache=True)
    def _sample_states_numba(h,
                             j_dense,
                             sample_size,
                             n_chains,
                             n_iters,
                             burn_in,
                             seed,
                             init_plus_probs):
        n = h.shape[0]
        samples = np.zeros((sample_size, n), dtype=np.int8)
        accepted = np.zeros(n_chains, dtype=np.int64)
        proposed = np.zeros(n_chains, dtype=np.int64)

        base = sample_size // n_chains
        extra = sample_size % n_chains

        for chain in prange(n_chains):
            np.random.seed(seed + 1009 * chain)
            state = np.empty(n, dtype=np.int8)
            for i in range(n):
                state[i] = 1 if np.random.random() < init_plus_probs[i] else -1

            for _ in range(burn_in):
                site = int(np.random.random() * n)
                accepted[chain] += _metropolis_flip(state, h, j_dense, site)
                proposed[chain] += 1

            n_local = base
            start = chain * base
            if chain < extra:
                n_local += 1
                start += chain
            else:
                start += extra
            for row in range(n_local):
                for _ in range(n_iters):
                    site = int(np.random.random() * n)
                    accepted[chain] += _metropolis_flip(state, h, j_dense, site)
                    proposed[chain] += 1
                for i in range(n):
                    samples[start + row, i] = state[i]

        accepted_total = 0
        proposed_total = 0
        for chain in range(n_chains):
            accepted_total += accepted[chain]
            proposed_total += proposed[chain]

        return samples, accepted_total, proposed_total


    @njit(parallel=True, cache=True)
    def _mch_reweighted_observables_numba(states, pair_products, delta):
        n_samples = states.shape[0]
        n = states.shape[1]
        m = pair_products.shape[1]
        n_obs = n + m
        log_weights = np.empty(n_samples, dtype=np.float64)

        for row in prange(n_samples):
            log_weight = 0.0
            for i in range(n):
                log_weight += states[row, i] * delta[i]
            for e in range(m):
                log_weight += pair_products[row, e] * delta[n + e]
            log_weights[row] = log_weight

        max_log_weight = np.max(log_weights)
        weights = np.empty(n_samples, dtype=np.float64)
        weight_sum = 0.0
        for row in range(n_samples):
            weight = np.exp(log_weights[row] - max_log_weight)
            weights[row] = weight
            weight_sum += weight

        if weight_sum <= 0.0:
            weight_sum = 1.0

        out = np.empty(n_obs, dtype=np.float64)
        for obs in prange(n_obs):
            total = 0.0
            if obs < n:
                for row in range(n_samples):
                    total += weights[row] * states[row, obs]
            else:
                e = obs - n
                for row in range(n_samples):
                    total += weights[row] * pair_products[row, e]
            out[obs] = total / weight_sum
        return out


    @njit(parallel=True, cache=True)
    def _observables_from_states_numba(states, edge_i, edge_j):
        n_samples = states.shape[0]
        n = states.shape[1]
        m = edge_i.shape[0]
        means = np.empty(n, dtype=np.float64)
        pairs = np.empty(m, dtype=np.float64)

        if n_samples <= 0:
            for i in prange(n):
                means[i] = 0.0
            for e in prange(m):
                pairs[e] = 0.0
            return means, pairs

        inv_n_samples = 1.0 / n_samples
        for i in prange(n):
            total = 0.0
            for row in range(n_samples):
                total += states[row, i]
            means[i] = total * inv_n_samples

        for e in prange(m):
            total = 0.0
            i = edge_i[e]
            j = edge_j[e]
            for row in range(n_samples):
                total += states[row, i] * states[row, j]
            pairs[e] = total * inv_n_samples

        return means, pairs


def _sample_observables_python(h: np.ndarray,
                               j_dense: np.ndarray,
                               edge_i: np.ndarray,
                               edge_j: np.ndarray,
                               sample_size: int,
                               n_chains: int,
                               n_iters: int,
                               burn_in: int,
                               seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    n = len(h)
    m = len(edge_i)
    means = np.zeros(n, dtype=np.float64)
    pairs = np.zeros(m, dtype=np.float64)
    accepted = 0
    proposed = 0

    counts = 0
    base = sample_size // n_chains
    extra = sample_size % n_chains

    for chain in range(n_chains):
        state = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)

        def flip_once() -> None:
            nonlocal accepted, proposed, state
            site = int(rng.integers(0, n))
            local_field = h[site] + float(j_dense[site] @ state)
            delta_logp = -2.0 * state[site] * local_field
            proposed += 1
            if delta_logp >= 0.0 or rng.random() < np.exp(delta_logp):
                state[site] = -state[site]
                accepted += 1

        for _ in range(burn_in):
            flip_once()

        n_local = base + (1 if chain < extra else 0)
        for _ in range(n_local):
            for _ in range(n_iters):
                flip_once()
            means += state
            if m:
                pairs += state[edge_i] * state[edge_j]
            counts += 1

    if counts:
        means /= counts
        pairs /= counts
    return means, pairs, accepted, proposed


def _sample_states_python(h: np.ndarray,
                          j_dense: np.ndarray,
                          sample_size: int,
                          n_chains: int,
                          n_iters: int,
                          burn_in: int,
                          seed: int,
                          init_plus_probs: np.ndarray) -> tuple[np.ndarray, int, int]:
    rng = np.random.default_rng(seed)
    n = len(h)
    samples = np.zeros((sample_size, n), dtype=np.int8)
    accepted = 0
    proposed = 0

    base = sample_size // n_chains
    extra = sample_size % n_chains

    for chain in range(n_chains):
        state = np.where(rng.random(n) < init_plus_probs, 1, -1).astype(np.int8)

        def flip_once() -> None:
            nonlocal accepted, proposed, state
            site = int(rng.integers(0, n))
            local_field = h[site] + float(j_dense[site] @ state)
            delta_logp = -2.0 * state[site] * local_field
            proposed += 1
            if delta_logp >= 0.0 or rng.random() < np.exp(delta_logp):
                state[site] = -state[site]
                accepted += 1

        for _ in range(burn_in):
            flip_once()

        n_local = base + (1 if chain < extra else 0)
        start = chain * base + min(chain, extra)
        for row in range(n_local):
            for _ in range(n_iters):
                flip_once()
            samples[start + row] = state

    return samples, accepted, proposed


def _observables_from_states(states: np.ndarray,
                             edge_i: np.ndarray,
                             edge_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    states = np.asarray(states, dtype=np.int8)
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)

    if NUMBA_AVAILABLE:
        return _observables_from_states_numba(states, edge_i, edge_j)

    means = states.mean(axis=0, dtype=np.float64)
    pairs = np.empty(edge_i.size, dtype=np.float64)
    for e, (i, j) in enumerate(zip(edge_i, edge_j)):
        pairs[e] = np.mean(states[:, i] * states[:, j], dtype=np.float64)
    return means.astype(np.float64), pairs


def _pair_products_from_states(states: np.ndarray,
                               edge_i: np.ndarray,
                               edge_j: np.ndarray) -> np.ndarray:
    if edge_i.size == 0:
        return np.zeros((states.shape[0], 0), dtype=np.int8)
    return (states[:, edge_i] * states[:, edge_j]).astype(np.int8, copy=False)


def _sample_states(h: np.ndarray,
                   j_edges: np.ndarray,
                   edge_i: np.ndarray,
                   edge_j: np.ndarray,
                   n: int,
                   sample_size: int,
                   n_chains: int,
                   n_iters: int,
                   burn_in: int,
                   seed: int,
                   init_plus_probs: np.ndarray) -> tuple[np.ndarray, dict]:
    j_dense = _j_dense_from_edges(j_edges, edge_i, edge_j, n)
    init_plus_probs = np.clip(np.asarray(init_plus_probs, dtype=np.float64), 0.0, 1.0)
    if NUMBA_AVAILABLE:
        try:
            states, accepted, proposed = _sample_states_numba(
                h.astype(np.float64),
                j_dense.astype(np.float64),
                int(sample_size),
                int(n_chains),
                int(n_iters),
                int(burn_in),
                int(seed),
                init_plus_probs.astype(np.float64),
            )
        except BaseException as exc:
            if _is_manual_interrupt_exception(exc):
                raise KeyboardInterrupt from exc
            raise
        backend = "numba_parallel"
    else:
        states, accepted, proposed = _sample_states_python(
            h,
            j_dense,
            int(sample_size),
            int(n_chains),
            int(n_iters),
            int(burn_in),
            int(seed),
            init_plus_probs,
        )
        backend = "python"

    return states, {
        "backend": backend,
        "accepted_flips": int(accepted),
        "proposed_flips": int(proposed),
        "acceptance_rate": float(accepted / max(proposed, 1)),
    }


def _sample_observables(h: np.ndarray,
                        j_edges: np.ndarray,
                        edge_i: np.ndarray,
                        edge_j: np.ndarray,
                        n: int,
                        sample_size: int,
                        n_chains: int,
                        n_iters: int,
                        burn_in: int,
                        seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    j_dense = _j_dense_from_edges(j_edges, edge_i, edge_j, n)
    if NUMBA_AVAILABLE:
        means, pairs, accepted, proposed = _sample_observables_numba(
            h.astype(np.float64),
            j_dense.astype(np.float64),
            edge_i.astype(np.int64),
            edge_j.astype(np.int64),
            int(sample_size),
            int(n_chains),
            int(n_iters),
            int(burn_in),
            int(seed),
        )
        backend = "numba_parallel"
    else:
        means, pairs, accepted, proposed = _sample_observables_python(
            h,
            j_dense,
            edge_i,
            edge_j,
            int(sample_size),
            int(n_chains),
            int(n_iters),
            int(burn_in),
            int(seed),
        )
        backend = "python"

    acceptance_rate = float(accepted / max(proposed, 1))
    return means, pairs, {
        "backend": backend,
        "accepted_flips": int(accepted),
        "proposed_flips": int(proposed),
        "acceptance_rate": acceptance_rate,
    }


def _profile_params(profile: str | None, sample_size: int) -> dict:
    profile = (profile or "adaptive_samples").strip().lower()
    if profile in {"very_aggressive", "muito_agressivo", "muito-agressivo"}:
        return {"maxdlamda": 0.80, "maxdlamdaNorm": 3.0, "eta": 0.40, "maxLearningSteps": 20}
    if profile in {"aggressive", "agressivo", "agressiva"}:
        return {"maxdlamda": 0.50, "maxdlamdaNorm": 2.0, "eta": 0.30, "maxLearningSteps": 20}
    if profile in {"medium", "medio", "media", "médio", "média"}:
        return {"maxdlamda": 0.20, "maxdlamdaNorm": 1.0, "eta": 0.15, "maxLearningSteps": 20}
    if profile in {"conservative", "conservador", "conservadora"}:
        return {"maxdlamda": 0.08, "maxdlamdaNorm": 0.5, "eta": 0.05, "maxLearningSteps": 20}

    if sample_size < 200_000:
        return {"maxdlamda": 0.50, "maxdlamdaNorm": 2.0, "eta": 0.30, "maxLearningSteps": 20}
    if sample_size < 500_000:
        return {"maxdlamda": 0.20, "maxdlamdaNorm": 1.0, "eta": 0.15, "maxLearningSteps": 20}
    if sample_size < 1_200_000:
        return {"maxdlamda": 0.08, "maxdlamdaNorm": 0.5, "eta": 0.05, "maxLearningSteps": 20}
    return {"maxdlamda": 0.05, "maxdlamdaNorm": 0.3, "eta": 0.035, "maxLearningSteps": 20}


def _profile_label(profile: str | None) -> str:
    profile = (profile or "adaptive_samples").strip().lower()
    labels = {
        "very_aggressive": "Muito agressivo",
        "aggressive": "Agressivo",
        "medium": "Medio",
        "conservative": "Conservador",
        "adaptive_samples": "Adaptado ao numero de amostras",
    }
    return labels.get(profile, labels["adaptive_samples"])


def _error_metrics(error: np.ndarray, tol: float, tol_norm: float) -> tuple[float, float, float]:
    abs_error = np.abs(error)
    error_norm = float(np.linalg.norm(error))
    max_error = float(np.max(abs_error)) if abs_error.size else 0.0
    score = max(
        max_error / max(tol, np.finfo(float).eps),
        error_norm / max(tol_norm, np.finfo(float).eps),
    )
    return float(score), error_norm, max_error


def _clip_delta(delta: np.ndarray, max_abs_step: float, max_norm_step: float) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).copy()
    abs_max = float(np.max(np.abs(delta))) if delta.size else 0.0
    if abs_max > max_abs_step > 0:
        delta *= max_abs_step / abs_max

    norm = float(np.linalg.norm(delta))
    if norm > max_norm_step > 0:
        delta *= max_norm_step / norm
    return delta


def _mch_reweighted_observables(states: np.ndarray,
                                pair_products: np.ndarray,
                                delta: np.ndarray) -> np.ndarray:
    if NUMBA_AVAILABLE:
        return _mch_reweighted_observables_numba(
            np.asarray(states, dtype=np.int8),
            np.asarray(pair_products, dtype=np.int8),
            np.asarray(delta, dtype=np.float64),
        )

    n = states.shape[1]
    dh = delta[:n]
    dj = delta[n:]

    log_weights = states @ dh
    if dj.size:
        log_weights = log_weights + pair_products @ dj
    log_weights = log_weights - float(np.max(log_weights))
    weights = np.exp(log_weights)
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        weights = np.full(states.shape[0], 1.0 / max(states.shape[0], 1), dtype=np.float64)
    else:
        weights = weights / weight_sum

    means = weights @ states
    if pair_products.shape[1]:
        pairs = weights @ pair_products
    else:
        pairs = np.zeros(0, dtype=np.float64)
    return np.concatenate([means, pairs]).astype(np.float64)


def _learn_parameters_mch_reweighted(current_observables: np.ndarray,
                                     target: np.ndarray,
                                     states: np.ndarray,
                                     pair_products: np.ndarray,
                                     learn_params: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    maxdlamda = float(learn_params["maxdlamda"])
    maxdlamda_norm = float(learn_params["maxdlamdaNorm"])
    eta = float(learn_params["eta"])
    max_learning_steps = int(learn_params["maxLearningSteps"])

    delta = np.zeros_like(target, dtype=np.float64)
    predicted = np.asarray(current_observables, dtype=np.float64).copy()
    distance = 1.0
    learning_steps = 0
    stop_reason = "maxLearningSteps"

    while learning_steps < max_learning_steps:
        candidate_delta = delta + -(predicted - target) * min(distance, 1.0) * eta
        candidate_predicted = _mch_reweighted_observables(
            states,
            pair_products,
            candidate_delta,
        )

        delta = candidate_delta
        predicted = candidate_predicted
        distance = float(np.linalg.norm(predicted - target))
        learning_steps += 1

        if np.linalg.norm(delta) > maxdlamda_norm or np.any(np.abs(delta) > maxdlamda):
            stop_reason = "step_limit"
            break

    return delta, predicted, {
        "learning_steps": learning_steps,
        "distance": distance,
        "delta_norm": float(np.linalg.norm(delta)),
        "delta_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "stop_reason": stop_reason,
    }


def resolver_mch_custom(sample: np.ndarray,
                        initial_guess: np.ndarray,
                        adjacency_mask: np.ndarray | None = None,
                        sample_size: int = 100_000,
                        maxiter: int = 200,
                        param_bound: float = 5.0,
                        n_iters: int | None = None,
                        burn_in: int | None = None,
                        max_sample_size: int = 2_500_000,
                        learning_profile: str = "adaptive_samples",
                        seed: int = 12345,
                        n_chains: int | None = None) -> tuple[np.ndarray, int, np.ndarray, dict]:
    sample = np.asarray(sample, dtype=np.int8)
    r, n = sample.shape
    sample_size = max(1_000, int(sample_size))
    maxiter = max(1, int(maxiter))
    n_iters = int(n_iters or max(100, n * 20))
    burn_in = int(burn_in or max(1_000, n * 100))
    param_bound = float(param_bound)
    n_chains = int(n_chains or min(max(os.cpu_count() or 1, 1), 16))
    n_chains = max(1, min(n_chains, sample_size))

    edge_i, edge_j, dense_idx = _prepare_edges(n, adjacency_mask)
    h, j_edges = _edge_params_from_full(initial_guess, n, dense_idx)
    h = np.clip(h, -param_bound, param_bound)
    j_edges = np.clip(j_edges, -param_bound, param_bound)

    target_means, target_pairs = _empirical_observables(sample, edge_i, edge_j)
    target = np.concatenate([target_means, target_pairs])
    init_plus_probs = np.clip((target_means + 1.0) * 0.5, 0.0, 1.0)

    tol_empirico = 3.0 / np.sqrt(r)
    tol_mc = 3.0 / np.sqrt(sample_size)
    tol_floor = 0.001
    tol = min(tol_empirico, min(tol_mc, tol_floor))
    tol_norm = tol * np.sqrt(len(target))
    learning_label = _profile_label(learning_profile)

    start = time.time()
    last_report_time = start
    current_sample_size = sample_size
    seed_counter = int(seed)
    acceptance_rates = []
    learning_steps_history = []
    mch_prediction_distances = []
    mch_delta_norms = []
    mch_delta_max_abs = []
    mch_step_stop_reasons = []

    try:
        states, sample_info = _sample_states(
            h,
            j_edges,
            edge_i,
            edge_j,
            n,
            current_sample_size,
            n_chains,
            n_iters,
            burn_in,
            seed_counter,
            init_plus_probs,
            )
        seed_counter += 10_000
        acceptance_rates.append(sample_info["acceptance_rate"])
        means, pairs = _observables_from_states(states, edge_i, edge_j)
        pair_products = _pair_products_from_states(states, edge_i, edge_j)
        current = np.concatenate([means, pairs])
        error = current - target
        current_score, current_error_norm, current_max_error = _error_metrics(error, tol, tol_norm)
    except BaseException as exc:
        if not _is_manual_interrupt_exception(exc):
            raise
        error = np.full(len(target), np.nan, dtype=np.float64)
        multipliers = _assemble_full_multipliers(h, j_edges, dense_idx, n, param_bound)
        run_info = {
            "solver": "custom_mch",
            "backend": "numba_parallel" if NUMBA_AVAILABLE else "python",
            "errflag": 3,
            "stop_reason": "manual_interrupt",
            "converged_formally": False,
            "converged_iteration": None,
            "manual_interrupt": True,
            "fixed_iterations": True,
            "backtracking_strategy": "none",
            "plateau_triggered": False,
            "plateau_rel_improvement": np.nan,
            "plateau_patience": np.nan,
            "plateau_max_error": np.nan,
            "plateau_streak": 0,
            "final_relative_improvement": np.nan,
            "n_iter": 0,
            "best_iteration": 0,
            "best_score": np.inf,
            "best_error": error.copy(),
            "best_error_norm": np.nan,
            "best_error_max_abs": np.nan,
            "best_sample_size": current_sample_size,
            "best_updates": [],
            "last_iteration": 0,
            "last_error_norm": np.nan,
            "last_error_max_abs": np.nan,
            "sample_size": sample_size,
            "final_sample_size": current_sample_size,
            "moving_average_window": np.nan,
            "sample_increment": np.nan,
            "sample_growth_factor": np.nan,
            "max_sample_size": current_sample_size,
            "sample_schedule": [current_sample_size],
            "sample_size_history": [current_sample_size],
            "sample_increases": 0,
            "sample_increase_events": [],
            "max_attempts_per_iteration": 1,
            "attempt_decay": np.nan,
            "attempt_scales": [1.0],
            "subattempts_per_scale": 1,
            "pre_mc_rejections": 0,
            "pre_mc_rejection_events": [],
            "backtracking_rejections": 0,
            "backtracking_events": [],
            "baseline_resamples": 0,
            "scale_evaluations": 0,
            "subattempt_evaluations": 0,
            "mch_learning_steps_mean": np.nan,
            "mch_learning_steps_last": np.nan,
            "mch_prediction_distance_last": np.nan,
            "mch_delta_norm_last": np.nan,
            "mch_delta_max_abs_last": np.nan,
            "mch_step_stop_reason_last": "",
            "no_improvement_count": 0,
            "no_improvement_events": [],
            "acceptance_rate_mean": np.nan,
            "acceptance_rate_last": np.nan,
            "learning_profile": learning_profile,
            "learning_profile_label": learning_label,
            "n_chains": n_chains,
            "n_edges": int(len(edge_i)),
            "chain_initialization": "empirical",
            "chain_init_plus_prob_min": float(np.min(init_plus_probs)),
            "chain_init_plus_prob_median": float(np.median(init_plus_probs)),
            "chain_init_plus_prob_max": float(np.max(init_plus_probs)),
            "elapsed_s": float(time.time() - start),
        }
        return multipliers, 3, np.vstack([error]), run_info

    n_error_terms = max(len(error), 1)
    best_score = current_error_norm / np.sqrt(n_error_terms)
    best_error = error.copy()
    best_error_norm = current_error_norm
    best_max_error = current_max_error
    best_h = h.copy()
    best_j_edges = j_edges.copy()
    best_iteration = 0
    best_sample_size = current_sample_size
    best_updates = [{
        "iteration": 0,
        "score": best_score,
        "combined_score": current_score,
        "error_norm": best_error_norm,
        "max_error": best_max_error,
        "sample_size": best_sample_size,
    }]
    errors = [error.copy()]

    print(
        f"Iteracao 0/{maxiter} | "
        f"max|erro|={current_max_error:.5f} | "
        f"norm||erro||={current_error_norm / np.sqrt(n_error_terms):.5f} | "
        f"best_norm||erro||={best_score:.5f} | "
        f"amostra={current_sample_size:,} | "
        f"|dtheta|=0 | "
        f"acc={sample_info['acceptance_rate']:.2f} | "
        f"etapa=0.0s | "
        f"total={time.time() - start:.1f}s",
        flush=True,
    )

    errflag = 0
    stop_reason = "maxiter"
    completed_iterations = 0
    converged_iteration = None

    for counter in range(1, maxiter + 1):
        try:
            profile_params = _profile_params(learning_profile, current_sample_size)
            delta, predicted, learn_info = _learn_parameters_mch_reweighted(
                current,
                target,
                states,
                pair_products,
                profile_params,
            )
            learning_steps_history.append(learn_info["learning_steps"])
            mch_prediction_distances.append(learn_info["distance"])
            mch_delta_norms.append(learn_info["delta_norm"])
            mch_delta_max_abs.append(learn_info["delta_max_abs"])
            mch_step_stop_reasons.append(learn_info["stop_reason"])

            h = np.clip(h + delta[:n], -param_bound, param_bound)
            j_edges = np.clip(j_edges + delta[n:], -param_bound, param_bound)
            states, sample_info = _sample_states(
                h,
                j_edges,
                edge_i,
                edge_j,
                n,
                current_sample_size,
                n_chains,
                n_iters,
                burn_in,
                seed_counter,
                init_plus_probs,
            )

            seed_counter += 10_000
            acceptance_rates.append(sample_info["acceptance_rate"])
            means, pairs = _observables_from_states(states, edge_i, edge_j)
            pair_products = _pair_products_from_states(states, edge_i, edge_j)
            current = np.concatenate([means, pairs])
            error = current - target
            current_score, current_error_norm, current_max_error = _error_metrics(error, tol, tol_norm)
            errors.append(error.copy())
            completed_iterations = counter

            norm_error = current_error_norm / np.sqrt(n_error_terms)
            if norm_error < best_score:
                best_score = norm_error
                best_error = error.copy()
                best_error_norm = current_error_norm
                best_max_error = current_max_error
                best_h = h.copy()
                best_j_edges = j_edges.copy()
                best_iteration = counter
                best_sample_size = current_sample_size
                best_updates.append({
                    "iteration": best_iteration,
                    "score": best_score,
                    "combined_score": current_score,
                    "error_norm": best_error_norm,
                    "max_error": best_max_error,
                    "sample_size": best_sample_size,
                })
        except BaseException as exc:
            if not _is_manual_interrupt_exception(exc):
                raise
            stop_reason = "manual_interrupt"
            errflag = 3
            break

        now = time.time()
        abs_error = np.abs(error)
        print(
            f"Iteracao {counter}/{maxiter} | "
            f"max|erro|={current_max_error:.5f} | "
            f"norm||erro||={current_error_norm / np.sqrt(n_error_terms):.5f} | "
            f"best_norm||erro||={best_score:.5f} | "
            f"amostra={current_sample_size:,} | "
            f"|dtheta|={learn_info['delta_norm']:.3g} | "
            f"acc={sample_info['acceptance_rate']:.2f} | "
            f"etapa={now - last_report_time:.1f}s | "
            f"total={now - start:.1f}s",
            flush=True,
        )
        last_report_time = now

    multipliers = _assemble_full_multipliers(best_h, best_j_edges, dense_idx, n, param_bound)
    run_info = {
        "solver": "custom_mch",
        "backend": "numba_parallel" if NUMBA_AVAILABLE else "python",
        "errflag": errflag,
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
        "n_iter": max(0, len(errors) - 1),
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
        "sample_size": sample_size,
        "final_sample_size": current_sample_size,
        "moving_average_window": np.nan,
        "sample_increment": np.nan,
        "sample_growth_factor": np.nan,
        "max_sample_size": current_sample_size,
        "sample_schedule": [current_sample_size],
        "sample_size_history": [current_sample_size],
        "sample_increases": 0,
        "sample_increase_events": [],
        "max_attempts_per_iteration": 1,
        "attempt_decay": np.nan,
        "attempt_scales": [1.0],
        "subattempts_per_scale": 1,
        "pre_mc_rejections": 0,
        "pre_mc_rejection_events": [],
        "backtracking_rejections": 0,
        "backtracking_events": [],
        "baseline_resamples": 0,
        "scale_evaluations": completed_iterations,
        "subattempt_evaluations": completed_iterations,
        "mch_learning_steps_mean": float(np.mean(learning_steps_history)) if learning_steps_history else np.nan,
        "mch_learning_steps_last": learning_steps_history[-1] if learning_steps_history else np.nan,
        "mch_prediction_distance_last": mch_prediction_distances[-1] if mch_prediction_distances else np.nan,
        "mch_delta_norm_last": mch_delta_norms[-1] if mch_delta_norms else np.nan,
        "mch_delta_max_abs_last": mch_delta_max_abs[-1] if mch_delta_max_abs else np.nan,
        "mch_step_stop_reason_last": mch_step_stop_reasons[-1] if mch_step_stop_reasons else "",
        "no_improvement_count": 0,
        "no_improvement_events": [],
        "acceptance_rate_mean": float(np.mean(acceptance_rates)) if acceptance_rates else np.nan,
        "acceptance_rate_last": float(acceptance_rates[-1]) if acceptance_rates else np.nan,
        "learning_profile": learning_profile,
        "learning_profile_label": learning_label,
        "n_chains": n_chains,
        "n_edges": int(len(edge_i)),
        "chain_initialization": "empirical",
        "chain_init_plus_prob_min": float(np.min(init_plus_probs)),
        "chain_init_plus_prob_median": float(np.median(init_plus_probs)),
        "chain_init_plus_prob_max": float(np.max(init_plus_probs)),
        "elapsed_s": float(time.time() - start),
    }
    return multipliers, errflag, np.vstack(errors), run_info
