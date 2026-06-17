import os
import re


def format_three_digit_id(value) -> str:
    return f"{int(value):03d}"


def _safe_stem(path_or_name: str) -> str:
    stem = os.path.splitext(os.path.basename(path_or_name or ""))[0]
    return re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "sem_id"


def _extract_indexed_stem(path_or_name: str, prefix: str) -> str | None:
    match = re.search(rf"(?:^|_){re.escape(prefix)}_(\d+)(?:_|\.|\(|$)", os.path.basename(path_or_name or ""))
    if not match:
        return None
    return f"{prefix}_{format_three_digit_id(match.group(1))}"


def _legacy_rede_id_from_siblings(path_or_name: str) -> str | None:
    parent = os.path.dirname(os.path.abspath(path_or_name or ""))
    basename = os.path.basename(path_or_name or "")
    if not parent or not os.path.isdir(parent):
        return None

    legacy_rede_files = sorted(
        name for name in os.listdir(parent)
        if re.match(r"^rede(?:_|geral|global).*", name)
    )
    if basename not in legacy_rede_files:
        return None

    return format_three_digit_id(legacy_rede_files.index(basename) + 1)


def canonical_artifact_stem(path_or_name: str, core_user_id=None) -> str:
    community_stem = _extract_indexed_stem(path_or_name, "comunidade")
    if community_stem:
        return community_stem

    core_stem = _extract_indexed_stem(path_or_name, "core_user")
    if core_stem:
        return core_stem

    basename = os.path.basename(path_or_name or "")
    if re.match(r"^rede(?:_|geral|global).*", basename):
        resolved_id = core_user_id if core_user_id is not None else _legacy_rede_id_from_siblings(path_or_name)
        if resolved_id is None:
            raise ValueError(f"Nao foi possivel inferir o numero do core user para {basename}")
        return f"core_user_{format_three_digit_id(resolved_id)}"

    return _safe_stem(path_or_name)


def plot_folder_name(gexf_path: str, node_count, core_user_id=None) -> str:
    return f"{canonical_artifact_stem(gexf_path, core_user_id=core_user_id)}({int(node_count)})"


def get_next_core_user_id(processed_root: str) -> str:
    used = set()
    for dirpath, _, filenames in os.walk(processed_root):
        for name in filenames:
            match = re.search(r"(?:^|_)core_user_(\d+)(?:_|\.|\(|$)", name)
            if match:
                used.add(int(match.group(1)))

    return format_three_digit_id(max(used) + 1 if used else 1)
