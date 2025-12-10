import json
from typing import List, Optional, Tuple

import torch


def _to_tensor_2d(rows: List[List[float]], feat_dim: Optional[int] = None) -> torch.Tensor:
    """
    rows: list of feature rows; each row is list[float].
    Returns a (M, F) float32 tensor. If rows is empty → (0, F).
    """
    if not rows:
        # empty set (e.g., no joins for this query)
        if feat_dim is None:
            return torch.zeros((0, 0), dtype=torch.float32)
        else:
            return torch.zeros((0, feat_dim), dtype=torch.float32)

    F_detected = len(rows[0])
    if feat_dim is not None and F_detected != feat_dim:
        raise ValueError(f"Inconsistent feature dim: expected {feat_dim}, got {F_detected}")
    return torch.tensor(rows, dtype=torch.float32)


def _pad_and_mask_set(batch_sets: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    batch_sets: list of (Mi, F) tensors, Mi can be 0.
    Returns:
      X: (B, Mmax, F)
      mask: (B, Mmax) (True where there is a real row)
    """
    B = len(batch_sets)
    if B == 0:
        return (
            torch.zeros((0, 0, 0), dtype=torch.float32),
            torch.zeros((0, 0), dtype=torch.bool),
        )

    # 1) Find feature dim F from first non-empty tensor
    F = 0
    for t in batch_sets:
        if t.numel() > 0:
            F = t.size(1)
            break

    # Case: all sets are empty → return (B, 0, 0)
    if F == 0:
        return (
            torch.zeros((B, 0, 0), dtype=torch.float32),
            torch.zeros((B, 0), dtype=torch.bool),
        )

    # 2) Max rows across sets
    Mmax = max(t.size(0) for t in batch_sets)

    X = torch.zeros((B, Mmax, F), dtype=torch.float32)
    m = torch.zeros((B, Mmax), dtype=torch.bool)

    for i, t in enumerate(batch_sets):
        Mi = t.size(0)
        if Mi == 0:
            continue
        # note: we don't slice with :F anymore; t already has shape (Mi, F)
        X[i, :Mi, :] = t
        m[i, :Mi] = True

    return X, m

def _sanitize(x: torch.Tensor) -> torch.Tensor:
    mask = torch.isnan(x) | torch.isinf(x)
    if mask.any():
        x[mask] = 0.0
    return x

def infer_feature_dims_from_dataset(samples: List[List[List[float]]]) -> Tuple[int, int, int]:
    """
    samples: list of [table_rows, join_rows, pred_rows]
    Returns: (Ft, Fj, Fp)
    """
    ft = fj = fp = 0
    for tables, joins, preds in samples:
        if tables and ft == 0:
            ft = len(tables[0])
        if joins and fj == 0:
            fj = len(joins[0])
        if preds and fp == 0:
            fp = len(preds[0])
        if ft and fj and fp:
            break
    return ft, fj, fp


def make_mscn_batch(
    batch_samples,
    batch_labels,
    ft: Optional[int] = None,
    fj: Optional[int] = None,
    fp: Optional[int] = None,
):
    # infer feature dims from first non-empty entries
    if ft is None or fj is None or fp is None:
        ft_b = ft or 0
        fj_b = fj or 0
        fp_b = fp or 0
        for tables, joins, preds in batch_samples:
            if tables and ft_b == 0:
                ft_b = len(tables[0])
            if joins and fj_b == 0:
                fj_b = len(joins[0])
            if preds and fp_b == 0:
                fp_b = len(preds[0])
            if ft_b and fj_b and fp_b:
                break
        ft = ft_b
        fj = fj_b
        fp = fp_b

    ft = ft or 0
    fj = fj or 0
    fp = fp or 0

    tables_list, joins_list, preds_list = [], [], []
    for tables_rows, joins_rows, preds_rows in batch_samples:
        T = _sanitize(_to_tensor_2d(tables_rows, feat_dim=ft))
        J = _sanitize(_to_tensor_2d(joins_rows,  feat_dim=fj))
        P = _sanitize(_to_tensor_2d(preds_rows,  feat_dim=fp))
        tables_list.append(T)
        joins_list.append(J)
        preds_list.append(P)

    tables_X, tables_m = _pad_and_mask_set(tables_list)
    joins_X,  joins_m  = _pad_and_mask_set(joins_list)
    preds_X,  preds_m  = _pad_and_mask_set(preds_list)

    if batch_labels is not None:
        y = torch.tensor(batch_labels, dtype=torch.float32)
    else:
        y = None

    return tables_X, tables_m, joins_X, joins_m, preds_X, preds_m, y


def _tuple_key_dict_to_str(d: dict, sep: str = "||") -> dict:
    """
    Convert dict with tuple keys to dict with string keys,
    e.g. ('movie', 'cast') -> 'movie||cast'.
    Leaves non-tuple keys intact.
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, tuple):
            k_str = sep.join(str(part) for part in k)
        else:
            k_str = k
        out[k_str] = v
    return out


def save_stats(stats, file_path):
    stats_json = dict(stats)
    if "edge_to_id" in stats_json:
        stats_json["edge_to_id"] = _tuple_key_dict_to_str(stats_json["edge_to_id"])

    with open(file_path, "w") as f:
        json.dump(stats_json, f, indent=2)

def _str_key_dict_to_tuple(d: dict, sep: str = "||") -> dict:
    """
    Inverse of _tuple_key_dict_to_str:
    'movie||cast' -> ('movie', 'cast') if sep is found.
    Leaves non-string keys as-is.
    """
    out = {}
    for k, v in d.items():
        if isinstance(k, str) and sep in k:
            parts = tuple(k.split(sep))
            out[parts] = v
        else:
            out[k] = v
    return out


def load_stats(file_path):
    with open(file_path) as f:
        stats_json = json.load(f)

    if "edge_to_id" in stats_json:
        stats_json["edge_to_id"] = _str_key_dict_to_tuple(stats_json["edge_to_id"])

    return stats_json




