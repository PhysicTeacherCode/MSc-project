"""Analise de suscetibilidade baseada nas Figuras 5 e 6 de Hall & Bialek (2019)."""

import itertools
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.ising_coniii import unpack_J


def _split_multipliers(multipliers: np.ndarray, n_users: int) -> tuple[np.ndarray, np.ndarray]:
    multipliers = np.asarray(multipliers, dtype=np.float64).ravel()
    expected = n_users + n_users * (n_users - 1) // 2
    if multipliers.size != expected:
        raise ValueError(
            f"Multipliers incompatíveis com N={n_users}: esperado {expected}, recebido {multipliers.size}."
        )
    return multipliers[:n_users].copy(), unpack_J(multipliers[n_users:], n_users)


def _all_spin_states(n_users: int) -> np.ndarray:
    n_states = 2 ** n_users
    idx_all = np.arange(n_states, dtype=np.uint64)
    bits = ((idx_all[:, None] >> np.arange(n_users, dtype=np.uint64)[None, :]) & 1)
    return (2.0 * bits.astype(np.float64) - 1.0)


def _log_weights(states: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    return states @ h + 0.5 * np.einsum("si,ij,sj->s", states, J, states)


def _energies(states: np.ndarray, h: np.ndarray, J: np.ndarray) -> np.ndarray:
    return -_log_weights(states, h, J)


def _normalized_probs_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = weights.sum()
    return weights / total if total > 0 else np.full_like(weights, 1.0 / len(weights))


def _chi_from_weighted_states(states: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    q = 0.5 + states.sum(axis=1) / (2.0 * states.shape[1])
    mean_q = float(np.sum(probs * q))
    var_q = float(np.sum(probs * (q - mean_q) ** 2))
    return states.shape[1] * var_q, mean_q


def _sample_ising_metropolis(
    h: np.ndarray,
    J: np.ndarray,
    n_samples: int,
    burn_in: int | None = None,
    thin: int | None = None,
    seed: int = 12345,
) -> np.ndarray:
    n_users = len(h)
    n_samples = max(1, int(n_samples))
    burn_in = int(burn_in or max(1000, n_users * 100))
    thin = int(thin or max(10, n_users))
    rng = np.random.default_rng(seed)

    state = rng.choice(np.array([-1.0, 1.0]), size=n_users)
    local_field = h + J @ state
    samples = np.empty((n_samples, n_users), dtype=np.float64)
    sample_idx = 0
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        i = int(rng.integers(0, n_users))
        old_spin = state[i]
        delta_e = 2.0 * old_spin * local_field[i]
        if delta_e <= 0.0 or rng.random() < np.exp(-delta_e):
            state[i] = -old_spin
            local_field += -2.0 * old_spin * J[:, i]

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[sample_idx] = state
            sample_idx += 1
            if sample_idx >= n_samples:
                break

    return samples


def _chi_pairwise_for_field(
    h: np.ndarray,
    J: np.ndarray,
    delta_h: float,
    n_samples: int,
    exact_threshold: int,
    seed: int,
) -> tuple[float, float]:
    h_field = h + float(delta_h)
    n_users = len(h)
    if n_users <= exact_threshold:
        states = _all_spin_states(n_users)
        probs = _normalized_probs_from_log_weights(_log_weights(states, h_field, J))
        return _chi_from_weighted_states(states, probs)

    samples = _sample_ising_metropolis(
        h_field,
        J,
        n_samples=n_samples,
        seed=seed,
    )
    q = 0.5 + samples.sum(axis=1) / (2.0 * n_users)
    return float(n_users * np.var(q)), float(np.mean(q))


def _chi_independent_for_field(h_indep: np.ndarray, delta_h: float) -> tuple[float, float]:
    means = np.tanh(h_indep + float(delta_h))
    mean_q = float(0.5 + means.mean() / 2.0)
    chi = float(np.sum(1.0 - means * means) / (4.0 * len(means)))
    return chi, mean_q


def calcular_curva_suscetibilidade(
    spin_matrix_users_keywords: np.ndarray,
    multipliers: np.ndarray,
    field_values: np.ndarray | None = None,
    n_samples: int = 100_000,
    exact_threshold: int = 20,
    seed: int = 12345,
) -> pd.DataFrame:
    """
    Calcula a Figura 5 de Hall & Bialek (2019): chi=N*Var(Q) contra campo uniforme Delta h.

    Q segue a equacao (7) do artigo: Q=(1/2N) sum_i (1 + sigma_i).
    A curva pairwise usa os multiplicadores inferidos com h_i -> h_i + Delta h; a curva
    independente usa apenas os campos que reproduzem as medias empiricas de cada usuario.
    A matriz de entrada segue o padrão do projeto: linhas=usuários, colunas=keywords.
    """
    spin_matrix = np.asarray(spin_matrix_users_keywords, dtype=np.float64)
    if spin_matrix.ndim != 2:
        raise ValueError("A matriz Ising deve ser 2D (usuários x keywords).")

    n_users = spin_matrix.shape[0]
    h, J = _split_multipliers(multipliers, n_users)
    observations = spin_matrix.T
    empirical_means = np.clip(observations.mean(axis=0), -0.9999, 0.9999)
    h_indep = np.arctanh(empirical_means)

    if field_values is None:
        field_values = np.linspace(-1.0, 3.0, 41)

    rows = []
    for idx, delta_h in enumerate(np.asarray(field_values, dtype=np.float64)):
        chi_pairwise, mean_q_pairwise = _chi_pairwise_for_field(
            h,
            J,
            float(delta_h),
            n_samples=n_samples,
            exact_threshold=exact_threshold,
            seed=seed + idx,
        )
        chi_indep, mean_q_indep = _chi_independent_for_field(h_indep, float(delta_h))
        rows.append({
            "delta_h": float(delta_h),
            "chi_pairwise": chi_pairwise,
            "chi_independente": chi_indep,
            "mean_q_pairwise": mean_q_pairwise,
            "mean_q_independente": mean_q_indep,
        })

    return pd.DataFrame(rows)


def _logcumsumexp(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values)
    running = -np.inf
    for i, value in enumerate(values):
        m = max(running, value)
        if np.isneginf(m):
            running = value
        else:
            running = m + np.log(np.exp(running - m) + np.exp(value - m))
        out[i] = running
    return out


def _entropy_energy_exact(h: np.ndarray, J: np.ndarray) -> tuple[pd.DataFrame, dict]:
    states = _all_spin_states(len(h))
    energies = np.sort(_energies(states, h, J))
    cumulative_entropy = np.log(np.arange(1, len(energies) + 1, dtype=np.float64))
    probs = _normalized_probs_from_log_weights(-energies)
    mean_energy = float(np.sum(probs * energies))
    df = pd.DataFrame({"E": energies, "S": cumulative_entropy})
    return df, {"method": "exact", "mean_energy": mean_energy, "n_states": int(len(energies))}


def _initial_energy_range(h: np.ndarray, J: np.ndarray, rng: np.random.Generator, n_probe: int = 5000):
    n_users = len(h)
    probes = rng.choice(np.array([-1.0, 1.0]), size=(n_probe, n_users))
    probes = np.vstack([
        probes,
        np.ones((1, n_users), dtype=np.float64),
        -np.ones((1, n_users), dtype=np.float64),
    ])
    energies = _energies(probes, h, J)
    e_min = float(np.min(energies))
    e_max = float(np.max(energies))
    margin = max((e_max - e_min) * 0.10, 1e-6)
    return e_min - margin, e_max + margin


def _entropy_energy_wang_landau(
    h: np.ndarray,
    J: np.ndarray,
    n_bins: int,
    wl_steps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n_users = len(h)
    e_min, e_max = _initial_energy_range(h, J, rng)
    bin_edges = np.linspace(e_min, e_max, int(n_bins) + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    def bin_index(energy: float) -> int | None:
        idx = int(np.searchsorted(bin_edges, energy, side="right") - 1)
        if idx < 0 or idx >= n_bins:
            return None
        return idx

    state = rng.choice(np.array([-1.0, 1.0]), size=n_users)
    local_field = h + J @ state
    energy = float(-state @ h - 0.5 * state @ (J @ state))
    current_bin = bin_index(energy)
    while current_bin is None:
        state = rng.choice(np.array([-1.0, 1.0]), size=n_users)
        local_field = h + J @ state
        energy = float(-state @ h - 0.5 * state @ (J @ state))
        current_bin = bin_index(energy)

    log_density = np.zeros(n_bins, dtype=np.float64)
    histogram = np.zeros(n_bins, dtype=np.int64)
    log_f = 1.0
    flatness = 0.55
    check_interval = max(1000, 20 * n_bins)
    completed_steps = 0

    for step in range(1, int(wl_steps) + 1):
        i = int(rng.integers(0, n_users))
        old_spin = state[i]
        delta_e = 2.0 * old_spin * local_field[i]
        proposal_energy = energy + delta_e
        proposal_bin = bin_index(proposal_energy)

        if proposal_bin is not None:
            accept_log_prob = log_density[current_bin] - log_density[proposal_bin]
            if accept_log_prob >= 0.0 or np.log(rng.random()) < accept_log_prob:
                state[i] = -old_spin
                local_field += -2.0 * old_spin * J[:, i]
                energy = proposal_energy
                current_bin = proposal_bin

        log_density[current_bin] += log_f
        histogram[current_bin] += 1
        completed_steps = step

        if step % check_interval == 0:
            visited = histogram > 0
            if visited.all() and histogram.min() >= flatness * histogram.mean():
                histogram[:] = 0
                log_f *= 0.5
                if log_f < 1e-3:
                    break

    log_density -= _logcumsumexp(log_density)[-1] - n_users * np.log(2.0)
    cumulative_entropy = _logcumsumexp(log_density)
    log_probs = log_density - bin_centers
    probs = np.exp(log_probs - np.max(log_probs))
    probs /= probs.sum()
    mean_energy = float(np.sum(probs * bin_centers))

    df = pd.DataFrame({"E": bin_centers, "S": cumulative_entropy})
    return df, {
        "method": "wang_landau",
        "mean_energy": mean_energy,
        "wl_steps": int(completed_steps),
        "n_bins": int(n_bins),
        "energy_min": float(e_min),
        "energy_max": float(e_max),
    }


def estimar_entropia_energia(
    multipliers: np.ndarray,
    n_users: int,
    n_bins: int = 80,
    wl_steps: int = 250_000,
    exact_threshold: int = 20,
    seed: int = 12345,
) -> tuple[pd.DataFrame, dict]:
    """
    Estima a curva de entropia cumulativa de Hall & Bialek (2019): S(E)=ln sum_sigma Theta(E - E_sigma).

    No PDF hall2019 usado pelo projeto, esta curva aparece como Figura 7. Ela fica
    mantida como utilitario, mas a opcao "Analise de Suscetibilidade" gera as Figuras
    5 e 6 do PDF: campo uniforme e individuos forcados.
    Para N<=exact_threshold usa enumeração; para N maior usa Wang-Landau binned.
    """
    h, J = _split_multipliers(multipliers, n_users)
    if n_users <= exact_threshold:
        return _entropy_energy_exact(h, J)
    return _entropy_energy_wang_landau(h, J, n_bins=n_bins, wl_steps=wl_steps, seed=seed)


def _conditional_parameters_for_forced(
    h: np.ndarray,
    J: np.ndarray,
    forced_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    n_users = len(h)
    if not forced_indices:
        return h.copy(), J.copy()

    forced = np.asarray(forced_indices, dtype=np.int64)
    remaining_mask = np.ones(n_users, dtype=bool)
    remaining_mask[forced] = False
    remaining = np.flatnonzero(remaining_mask)
    if remaining.size == 0:
        raise ValueError("E necessario manter pelo menos um usuario livre para calcular chi.")

    h_remaining = h[remaining].copy() + J[np.ix_(remaining, forced)].sum(axis=1)
    J_remaining = J[np.ix_(remaining, remaining)].copy()
    return h_remaining, J_remaining


def _forced_configurations(
    n_users: int,
    n_forced: int,
    n_configurations: int,
    rng: np.random.Generator,
) -> list[tuple[int, ...]]:
    if n_forced == 0:
        return [()]

    total_configurations = math.comb(n_users, n_forced)
    n_to_use = min(int(n_configurations), total_configurations)
    if total_configurations <= n_to_use:
        return list(itertools.combinations(range(n_users), n_forced))

    configs: set[tuple[int, ...]] = set()
    while len(configs) < n_to_use:
        configs.add(tuple(sorted(int(i) for i in rng.choice(n_users, size=n_forced, replace=False))))
    return sorted(configs)


def calcular_suscetibilidade_individuos_forcados(
    multipliers: np.ndarray,
    n_users: int,
    max_forced: int | None = None,
    n_configurations: int = 25,
    n_samples: int = 100_000,
    exact_threshold: int = 20,
    seed: int = 12345,
) -> pd.DataFrame:
    """
    Calcula a Figura 6 de Hall & Bialek (2019): chi contra numero de individuos forcados.

    Para cada conjunto de usuarios fixados em sigma=+1, os usuarios restantes seguem o
    mesmo modelo pairwise com campos condicionais h_i -> h_i + sum_alpha J_i,alpha.
    As barras de erro sao o desvio padrao de chi entre configuracoes de usuarios forcados.
    """
    n_users = int(n_users)
    if n_users <= 0:
        raise ValueError("n_users deve ser positivo.")

    h, J = _split_multipliers(multipliers, n_users)
    max_allowed = max(0, n_users - 1)
    if max_forced is None:
        max_forced = min(11, max_allowed)
    max_forced = min(max(0, int(max_forced)), max_allowed)
    n_configurations = max(1, int(n_configurations))
    rng = np.random.default_rng(seed)

    rows = []
    for n_forced in range(max_forced + 1):
        configs = _forced_configurations(n_users, n_forced, n_configurations, rng)
        chi_values = []
        remaining_users = n_users - n_forced
        for cfg_idx, forced_indices in enumerate(configs):
            h_cond, J_cond = _conditional_parameters_for_forced(h, J, forced_indices)
            chi, _mean_q = _chi_pairwise_for_field(
                h_cond,
                J_cond,
                delta_h=0.0,
                n_samples=n_samples,
                exact_threshold=exact_threshold,
                seed=seed + n_forced * 100_000 + cfg_idx,
            )
            chi_values.append(chi)

        chi_array = np.asarray(chi_values, dtype=np.float64)
        rows.append({
            "n_forced": int(n_forced),
            "chi_mean": float(np.mean(chi_array)),
            "chi_std": float(np.std(chi_array, ddof=1)) if chi_array.size > 1 else 0.0,
            "chi_min": float(np.min(chi_array)),
            "chi_max": float(np.max(chi_array)),
            "n_configurations": int(len(configs)),
            "remaining_users": int(remaining_users),
        })

    return pd.DataFrame(rows)


def plotar_figura5_suscetibilidade(df: pd.DataFrame, output_path: str, title: str | None = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
        ax.plot(df["delta_h"], df["chi_pairwise"], color="black", linewidth=1.4, marker="o", markersize=2.2, label="Pairwise Model")
        ax.plot(
            df["delta_h"],
            df["chi_independente"],
            color="#17becf",
            linewidth=1.4,
            marker="o",
            markersize=2.0,
            label="Independent Model",
        )
        ax.axvline(0.0, color="#ff0000", linewidth=1.0)
        ax.set_xlabel(r"$\Delta h$")
        ax.set_ylabel(r"$\chi$")
        if title:
            ax.set_title(title)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="black")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plotar_figura6_individuos_forcados(df: pd.DataFrame, output_path: str, title: str | None = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
        x = df["n_forced"].to_numpy(dtype=np.int64)
        y = df["chi_mean"].to_numpy(dtype=np.float64)
        yerr = df["chi_std"].to_numpy(dtype=np.float64)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color="black",
            marker="o",
            markersize=2.4,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=2.0,
        )
        ax.set_xlabel("# Forced Individuals")
        ax.set_ylabel(r"$\chi$")
        ax.set_ylim(bottom=0.0)
        if len(x) > 0:
            ax.set_xlim(left=-0.25, right=float(x.max()) + 0.75)
        if title:
            ax.set_title(title)
        ax.tick_params(direction="in", top=True, right=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def plotar_entropia_energia(df: pd.DataFrame, info: dict, n_users: int, output_path: str, title: str | None = None):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mean_energy = float(info.get("mean_energy", np.nan))
    e = df["E"].to_numpy(dtype=np.float64)
    s = df["S"].to_numpy(dtype=np.float64)
    finite = np.isfinite(e) & np.isfinite(s)
    e = e[finite]
    s = s[finite]
    if e.size == 0:
        raise ValueError("Sem pontos válidos para plotar S(E).")

    window = max((e.max() - e.min()) * 0.08, 1e-9)
    near = np.abs(e - mean_energy) <= window
    if near.sum() < 2:
        near = np.ones_like(e, dtype=bool)
    intercept = float(np.median(s[near] - e[near]))
    line_s = e + intercept

    with plt.rc_context({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
    }):
        fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
        ax.plot(e / n_users, s / n_users, color="black", linewidth=1.3, label=r"$S(E)$")
        ax.plot(e / n_users, line_s / n_users, color="#1f77b4", linewidth=1.0, label="Slope 1")
        if np.isfinite(mean_energy):
            ax.axvline(mean_energy / n_users, color="#d62728", linewidth=1.0, label="Actual energy")
        ax.set_xlabel(r"$E/N$")
        ax.set_ylabel(r"$S(E)/N$")
        if title:
            ax.set_title(title)
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(fontsize=9, frameon=True, fancybox=False, edgecolor="black")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def executar_analise_suscetibilidade(
    spin_matrix_users_keywords: np.ndarray,
    multipliers: np.ndarray,
    output_dir: str,
    artifact_stem: str,
    field_min: float = -1.0,
    field_max: float = 3.0,
    field_points: int = 41,
    samples_per_field: int = 100_000,
    max_forced: int | None = None,
    forced_configurations: int = 25,
    samples_per_forced_configuration: int = 100_000,
    seed: int = 12345,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    n_users = int(np.asarray(spin_matrix_users_keywords).shape[0])
    fields = np.linspace(float(field_min), float(field_max), int(field_points))

    df_fig5 = calcular_curva_suscetibilidade(
        spin_matrix_users_keywords,
        multipliers,
        field_values=fields,
        n_samples=int(samples_per_field),
        seed=seed,
    )
    fig5_csv = os.path.join(output_dir, f"figura5_suscetibilidade_{artifact_stem}.csv")
    fig5_png = os.path.join(output_dir, f"figura5_suscetibilidade_{artifact_stem}.png")
    df_fig5.to_csv(fig5_csv, index=False, encoding="utf-8-sig")
    plotar_figura5_suscetibilidade(df_fig5, fig5_png)

    df_fig6 = calcular_suscetibilidade_individuos_forcados(
        multipliers,
        n_users=n_users,
        max_forced=max_forced,
        n_configurations=int(forced_configurations),
        n_samples=int(samples_per_forced_configuration),
        seed=seed + 1_000_000,
    )
    fig6_csv = os.path.join(output_dir, f"figura6_individuos_forcados_{artifact_stem}.csv")
    fig6_png = os.path.join(output_dir, f"figura6_individuos_forcados_{artifact_stem}.png")
    df_fig6.to_csv(fig6_csv, index=False, encoding="utf-8-sig")
    plotar_figura6_individuos_forcados(df_fig6, fig6_png)

    metadata = {
        "source_article": "Hall & Bialek 2019 JSTAT 093406",
        "figura5_definition": "chi=N*Var(Q) versus uniform field Delta h; Q=(1/2N) sum_i (1+sigma_i)",
        "figura6_definition": "forced individuals",
        "figura5_png": fig5_png,
        "figura5_csv": fig5_csv,
        "figura6_png": fig6_png,
        "figura6_csv": fig6_csv,
        "figura6_max_forced": int(df_fig6["n_forced"].max()) if not df_fig6.empty else None,
        "figura6_forced_configurations": int(forced_configurations),
    }
    metadata_path = os.path.join(output_dir, f"analise_suscetibilidade_{artifact_stem}.csv")
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False, encoding="utf-8-sig")
    metadata["metadata_csv"] = metadata_path
    return metadata
