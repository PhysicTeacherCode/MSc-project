import pandas as pd
import re
import numpy as np

def analyze_word_intervals_dict(global_word_times):
    """
    Calcula o desvio padrão do tempo entre ocorrências de cada palavra.
    Recebe um dicionário {palavra: [lista_de_idades]}.
    """
    if not global_word_times:
        print("[Análise] Mapa de palavras vazio.")
        return pd.DataFrame()

    print(f"[Análise] Processando {len(global_word_times)} palavras únicas para estatísticas...")
    
    stats = []
    for word, times in global_word_times.items():
        if len(times) < 3:
            continue
            
        sorted_times = sorted(times)
        intervals = np.diff(sorted_times)
        
        if len(intervals) > 0:
            std_val = np.std(intervals)
            stats.append({
                "word": word,
                "occurrences": len(times),
                "desvio_padrao": std_val
            })
            
    result_df = pd.DataFrame(stats)
    print(f"[Análise] Concluída. {len(result_df)} palavras atingiram o critério estatístico.")
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
