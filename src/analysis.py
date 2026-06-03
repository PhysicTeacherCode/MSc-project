import pandas as pd
import numpy as np

def _temporal_std_days(timestamps):
    """
    Desvio padrao temporal dos usos de uma palavra, em dias.
    """
    if not timestamps or len(timestamps) < 2:
        return np.nan

    parsed = pd.to_datetime(pd.Series(timestamps), utc=True, errors='coerce')
    parsed = parsed.dropna()
    if len(parsed) < 2:
        return np.nan

    seconds = parsed.map(lambda value: value.timestamp()).to_numpy(dtype=np.float64)
    return float(np.std(seconds, ddof=0) / 86400.0)


def analyze_word_frequency(global_word_counts, user_word_sets, total_users=None, global_word_timestamps=None):
    """
    Analisa a frequência de cada palavra pela perspectiva dos USUÁRIOS.
    
    Métricas por palavra:
        - occurrences: total de ocorrências em todos os posts.
        - n_users:     quantos usuários distintos usaram a palavra.
        - user_freq:   n_users / total de usuários na comunidade (0.0 a 1.0).
    
    Filtragem downstream:
        - Corte Inferior (min_users): exigir 3-5 usuários (ou 5% da comunidade)
          garante que o otimizador tenha "sinal" suficiente para calcular J.
        - Corte Superior (max_users): remover keywords usadas por >80-90% dos
          usuários evita spins congelados em +1 (variância → 0, h_i → ∞).
    """
    if not user_word_sets:
        print("[Análise] Mapa de palavras vazio.")
        return pd.DataFrame()

    if total_users is None:
        total_users = len(user_word_sets)
    if global_word_timestamps is None:
        global_word_timestamps = {}
    
    print(f"[Análise] Processando {len(global_word_counts)} palavras únicas para estatísticas de frequência...")
    
    # Contagem de usuários por palavra (inversão do dicionário)
    word_user_count = {}
    for user, words in user_word_sets.items():
        for w in words:
            word_user_count[w] = word_user_count.get(w, 0) + 1
    
    stats = []
    for word, count in global_word_counts.items():
        n_users = word_user_count.get(word, 0)
        if n_users < 2:  # pelo menos 2 usuários para ser relevante
            continue
        
        entry = {
            "word": word,
            "occurrences": count,
            "n_users": n_users,
            "user_freq": round(n_users / total_users, 4) if total_users > 0 else 0.0,
            "time_std_days": _temporal_std_days(global_word_timestamps.get(word, []))
        }
        stats.append(entry)
            
    result_df = pd.DataFrame(stats)
    print(f"[Análise] Concluída. {len(result_df)} palavras com >= 2 usuários.")
    return result_df

def create_ising_matrix_from_sets(user_word_sets, keywords, all_users):
    """
    Gera uma matriz de spins (+1/-1) para o modelo de Ising.
    Otimizado para usar interseção de sets (muito mais rápido que loop duplo).
    """
    if not keywords or not all_users:
        return pd.DataFrame()

    print(f"[Ising] Gerando matriz de estados (+1/-1) para {len(keywords)} keywords e {len(all_users)} usuários...")
    
    # Filtra apenas strings e remove NaNs/None
    keywords_clean = [str(k) for k in keywords if pd.notnull(k)]
    # Normaliza keywords para lowercase (garante consistência com user_word_sets que já está em lowercase)
    keywords_lower = [k.lower() for k in keywords_clean]
    # Remove duplicatas que possam surgir após o lowercase
    keywords_lower = list(dict.fromkeys(keywords_lower))
    
    # 1. Inicializar matriz com -1 (padrão Ising para inativo)
    # dtype=int8 economiza MUITA memória em relação ao padrão float64
    matrix = pd.DataFrame(np.int8(-1), index=all_users, columns=keywords_lower)
    
    # Set de keywords para interseção rápida
    keywords_set = set(keywords_lower)
    
    # 2. Mapear a atividade baseada nos conjuntos pré-coletados
    for did in all_users:
        if did in user_word_sets:
            used_words = user_word_sets[did]
            # Interseção rápida (quais keywords filtradas este usuário usou?)
            active_kws = used_words.intersection(keywords_set)
            for kw in active_kws:
                matrix.at[did, kw] = 1
                
    return matrix


def filter_ising_users_by_activity(ising_matrix, min_activity=0.10, max_activity=0.90):
    """
    Remove usuarios quase inativos ou quase sempre ativos na matriz de Ising.

    A atividade aqui e a fracao de keywords filtradas em que o usuario aparece
    com spin +1. Esses usuarios extremos deixam campos h_i muito grandes e
    podem piorar o condicionamento da inferencia de h e J.
    """
    if ising_matrix.empty:
        empty_report = pd.DataFrame(
            columns=["user", "activity_frac", "active_keywords", "total_keywords", "reason"]
        )
        return ising_matrix, empty_report

    if min_activity < 0 or max_activity > 1 or min_activity > max_activity:
        raise ValueError("Os limites de atividade devem obedecer 0 <= min <= max <= 1.")

    active = (ising_matrix.values > 0)
    activity_frac = active.mean(axis=1)
    active_keywords = active.sum(axis=1)

    keep_mask = (activity_frac >= min_activity) & (activity_frac <= max_activity)
    reasons = np.where(
        activity_frac < min_activity,
        "pouco_ativo",
        np.where(activity_frac > max_activity, "muito_ativo", "mantido")
    )

    report = pd.DataFrame({
        "user": ising_matrix.index.tolist(),
        "activity_frac": activity_frac,
        "active_keywords": active_keywords,
        "total_keywords": ising_matrix.shape[1],
        "reason": reasons,
    })

    removed = report.loc[~keep_mask].copy()
    filtered = ising_matrix.loc[keep_mask].copy()
    return filtered, removed


def filter_ising_keywords_by_popularity(ising_matrix, min_users=3, max_user_frac=0.90):
    """
    Refiltra keywords depois do filtro de usuarios.

    Quando usuarios sao removidos, uma keyword que antes passava no filtro pode
    ficar ativa em apenas 1 usuario, o que deixa os tripletos muito ruidosos.
    """
    if ising_matrix.empty:
        empty_report = pd.DataFrame(
            columns=["keyword", "n_users", "user_frac", "reason"]
        )
        return ising_matrix, empty_report

    n_users_total = ising_matrix.shape[0]
    min_users = max(1, int(min_users))
    max_users = max(min_users, int(np.floor(n_users_total * max_user_frac)))
    max_users = min(max_users, n_users_total)

    active_counts = (ising_matrix.values > 0).sum(axis=0)
    user_frac = active_counts / n_users_total if n_users_total else np.zeros_like(active_counts)
    keep_mask = (active_counts >= min_users) & (active_counts <= max_users)

    reasons = np.where(
        active_counts < min_users,
        "poucos_usuarios",
        np.where(active_counts > max_users, "usuarios_demais", "mantida")
    )

    report = pd.DataFrame({
        "keyword": ising_matrix.columns.tolist(),
        "n_users": active_counts,
        "user_frac": user_frac,
        "reason": reasons,
    })

    removed = report.loc[~keep_mask].copy()
    filtered = ising_matrix.loc[:, keep_mask].copy()
    return filtered, removed


def filter_ising_keywords_by_jaccard(ising_matrix, threshold=0.80, keyword_stats=None):
    """
    Remove keywords redundantes com padrao de usuarios ativos muito parecido.

    Jaccard(A, B) = usuarios_ativos_em_ambas / usuarios_ativos_em_pelo_menos_uma.
    Se uma keyword tiver Jaccard >= threshold com alguma keyword ja mantida, ela
    e removida. A ordem favorece keywords mais frequentes e mais localizadas no
    tempo quando keyword_stats esta disponivel.
    """
    if ising_matrix.empty:
        empty_report = pd.DataFrame(
            columns=[
                "keyword", "similar_to", "jaccard", "intersection_users",
                "union_users", "n_users", "n_users_similar"
            ]
        )
        return ising_matrix, empty_report

    threshold = float(threshold)
    if threshold <= 0 or threshold > 1:
        raise ValueError("O limiar de Jaccard deve obedecer 0 < limiar <= 1.")

    active = (ising_matrix.values > 0)
    keywords = np.asarray(ising_matrix.columns.tolist(), dtype=object)
    active_counts = active.sum(axis=0).astype(np.int64)

    rank_df = pd.DataFrame({
        "keyword": keywords,
        "active_users": active_counts,
        "original_order": np.arange(len(keywords)),
    })

    if keyword_stats is not None and not keyword_stats.empty and "word" in keyword_stats.columns:
        stats = keyword_stats.copy()
        stats["keyword"] = stats["word"].astype(str).str.lower()
        keep_cols = ["keyword"]
        for col in ("occurrences", "n_users", "time_std_days"):
            if col in stats.columns:
                keep_cols.append(col)
        stats = stats[keep_cols].drop_duplicates("keyword", keep="first")
        rank_df = rank_df.merge(stats, on="keyword", how="left")

    if "occurrences" not in rank_df.columns:
        rank_df["occurrences"] = rank_df["active_users"]
    if "n_users" not in rank_df.columns:
        rank_df["n_users"] = rank_df["active_users"]
    if "time_std_days" not in rank_df.columns:
        rank_df["time_std_days"] = np.nan

    rank_df["occurrences"] = rank_df["occurrences"].fillna(rank_df["active_users"])
    rank_df["n_users"] = rank_df["n_users"].fillna(rank_df["active_users"])
    rank_df["time_std_rank"] = rank_df["time_std_days"].fillna(np.inf)

    rank_df = rank_df.sort_values(
        by=["n_users", "occurrences", "time_std_rank", "original_order"],
        ascending=[False, False, True, True],
    )
    ordered_indices = rank_df["original_order"].to_numpy(dtype=np.int64)

    kept_indices = []
    removed_rows = []

    for idx in ordered_indices:
        users_kw = active[:, idx]
        n_users_kw = int(active_counts[idx])
        redundant = False

        for kept_idx in kept_indices:
            users_kept = active[:, kept_idx]
            intersection = int(np.logical_and(users_kw, users_kept).sum())
            union = int(np.logical_or(users_kw, users_kept).sum())
            if union == 0:
                continue

            jaccard = intersection / union
            if jaccard >= threshold:
                removed_rows.append({
                    "keyword": keywords[idx],
                    "similar_to": keywords[kept_idx],
                    "jaccard": jaccard,
                    "intersection_users": intersection,
                    "union_users": union,
                    "n_users": n_users_kw,
                    "n_users_similar": int(active_counts[kept_idx]),
                })
                redundant = True
                break

        if not redundant:
            kept_indices.append(idx)

    kept_columns = [keywords[i] for i in sorted(kept_indices)]
    filtered = ising_matrix.loc[:, kept_columns].copy()
    removed = pd.DataFrame(removed_rows)
    return filtered, removed
