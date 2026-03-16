import math
from typing import Dict, Any, Tuple, List, Optional

Edge = Tuple[str, str]


class FeatureExtractor:
    def __init__(self, plan: Dict, table_stats: Dict[str, Dict[str, Any]],
                 scan_norms: Optional[Dict[str, Dict[str, float]]] = None):
        self.plan = plan
        self.table_stats = table_stats
        self.scan_norms = scan_norms or {}

    def extract_features_for_tables(self) -> Tuple[List[str], List[List[float]], Dict[str, int]]:
        """
           Returns:
             onehots  : List[List[float]]  -> shape (m, K), one row per table used in this plan
             tables   : List[str]          -> the table names in the same order as onehots rows
             tab2idx  : Dict[str, int]     -> global mapping used to build onehots (stable across queries)
           """

        TABLE_SCALAR_KEYS = [
            "log_rows",
            "rows_frac",
            "approx_size_mb",
            "cols_count",
            "avg_col_width",
            "max_col_width",
            "mean_null_frac",
            "max_null_frac",
            "mean_ndv_ratio",
            "max_ndv_ratio",
            "frac_high_ndv",
            "mean_abs_corr",
            "frac_numeric",
            "frac_text",
        ]

        def _estimate_ndv(n_distinct, table_rows):
            """
            Postgres semantics: n_distinct < 0 means -1 * (ndv / table_rows).
            Return an estimated NDV count (>=0).
            """
            if n_distinct is None:
                return 0.0
            nd = n_distinct
            rows = max(table_rows, 0.0)
            if nd >= 0:
                return nd
            # negative ratio * rows
            return max(-nd * rows, 0.0)

        def compute_table_scalars(table_rec: Dict[str, Any], total_rows_in_schema: float) -> Dict[str, float]:
            # Be robust to missing keys / different stat formats
            cols = table_rec.get("columns") or []
            rows = table_rec.get("tableSize", None)
            if rows is None:
                rows = table_rec.get("reltuples", None)

            try:
                rows = float(rows) if rows is not None else 0.0
            except Exception:
                rows = 0.0

            rows_total = max(float(total_rows_in_schema or 0.0), 1.0)

            def _get_col_val(c: Dict[str, Any], *keys, default=None):
                for k in keys:
                    if k in c and c.get(k) is not None:
                        return c.get(k)
                return default

            # widths
            widths = []
            for c in cols:
                w = _get_col_val(c, "avgWidth", "avg_width")
                if w is None:
                    continue
                try:
                    widths.append(float(w))
                except Exception:
                    pass
            avg_col_width = sum(widths) / len(widths) if widths else 0.0
            max_col_width = max(widths) if widths else 0.0

            # null fractions
            nulls = []
            for c in cols:
                nf = _get_col_val(c, "nullFrac", "null_frac")
                if nf is None:
                    continue
                try:
                    nulls.append(float(nf))
                except Exception:
                    pass
            mean_null_frac = sum(nulls) / len(nulls) if nulls else 0.0
            max_null_frac = max(nulls) if nulls else 0.0

            # NDV ratios
            ndv_ratios = []
            high_ndv_cnt = 0
            for c in cols:
                nd = _get_col_val(c, "nDistinct", "n_distinct")
                ndv = _estimate_ndv(nd, rows)
                denom = max(rows, 1.0)
                r = min(max(ndv / denom, 0.0), 1.0)  # clamp to [0,1]
                ndv_ratios.append(r)
                if r >= 0.5:
                    high_ndv_cnt += 1
            mean_ndv_ratio = sum(ndv_ratios) / len(ndv_ratios) if ndv_ratios else 0.0
            max_ndv_ratio = max(ndv_ratios) if ndv_ratios else 0.0
            frac_high_ndv = (high_ndv_cnt / len(ndv_ratios)) if ndv_ratios else 0.0

            # correlations (abs)
            corrs = []
            for c in cols:
                corr = _get_col_val(c, "correlation")
                if corr is None:
                    continue
                try:
                    corrs.append(abs(float(corr)))
                except Exception:
                    pass
            mean_abs_corr = sum(corrs) / len(corrs) if corrs else 0.0

            numeric_prefixes = ("int", "integer", "smallint", "bigint", "serial", "float", "double", "real", "decimal",
                                "numeric", "bool")
            text_prefixes = ("varchar", "char", "text")

            numeric_cnt = 0
            text_cnt = 0
            for c in cols:
                dt = _get_col_val(c, "data_type", "dataType")
                dt = (str(dt).lower().strip() if dt is not None else "")
                if dt.startswith(numeric_prefixes):
                    numeric_cnt += 1
                if dt.startswith(text_prefixes):
                    text_cnt += 1

            ncols = len(cols) if cols else 0
            frac_numeric = (numeric_cnt / ncols) if ncols else 0.0
            frac_text = (text_cnt / ncols) if ncols else 0.0

            # use avg_col_width * rows as rough payload size, bytes to MB
            approx_size_mb = (avg_col_width * rows) / (1024.0 * 1024.0) if rows > 0 else 0.0

            out = {
                "log_rows": math.log1p(max(rows, 0.0)),
                "rows_frac": rows / rows_total,
                "approx_size_mb": approx_size_mb,
                "cols_count": float(ncols),
                "avg_col_width": avg_col_width,
                "max_col_width": max_col_width,
                "mean_null_frac": mean_null_frac,
                "max_null_frac": max_null_frac,
                "mean_ndv_ratio": mean_ndv_ratio,
                "max_ndv_ratio": max_ndv_ratio,
                "frac_high_ndv": frac_high_ndv,
                "mean_abs_corr": mean_abs_corr,
                "frac_numeric": frac_numeric,
                "frac_text": frac_text,
            }
            return out

        def get_table_norm_stats(tr):
            buckets = {k: [] for k in TABLE_SCALAR_KEYS}
            for t in self.table_stats.values():
                scalars = compute_table_scalars(t, tr)
                for k in TABLE_SCALAR_KEYS:
                    buckets[k].append(scalars.get(k, 0.0))
            nor: Dict[str, Dict[str, float]] = {}
            for k, arr in buckets.items():
                arr_sorted = sorted(arr)
                n = max(len(arr_sorted), 1)
                mean = sum(arr_sorted) / n
                var = sum((x - mean) ** 2 for x in arr_sorted) / max(n - 1, 1)
                std = (var ** 0.5) or 1.0
                p1_idx = int(0.01 * (n - 1))
                p99_idx = int(0.99 * (n - 1))
                nor[k] = {
                    "mean": mean,
                    "std": std,
                    "p1": arr_sorted[p1_idx],
                    "p99": arr_sorted[p99_idx],
                }
            return nor

        def _norm_val(x: float, key: str, nor: Dict[str, Dict[str, float]]) -> float:
            spec = nor.get(key)
            if not spec:
                return x
            p1, p99 = spec["p1"], spec["p99"]
            mean, std = spec["mean"], spec["std"] or 1.0
            if x < p1: x = p1
            if x > p99: x = p99
            return (x - mean) / std


        def _build_table_index_map() -> Dict[str, int]:
            """
            Create a stable table->index mapping
            Sort by table names for reproducibility.
            """
            all_tables = sorted(self.table_stats.keys())
            return {t: i for i, t in enumerate(all_tables)}

        # ── Scan-type vocabulary (6 categories) ─────────────────────
        SCAN_TYPES = [
            "Seq Scan",
            "Index Scan",
            "Index Only Scan",
            "Bitmap Heap Scan",
            "Bitmap Index Scan",
            "Other",
        ]
        _scan_type_to_idx = {s: i for i, s in enumerate(SCAN_TYPES)}

        # Scalar keys that need z-score normalization (from scan_norms)
        SCAN_SCALAR_KEYS = [
            "scan_est_card_log",
            "scan_est_loops_log",
            "scan_selectivity",
            "total_scan_work_log",
            "num_local_filters",
        ]

        def _count_atomic_filters(filt: Any) -> int:
            """Count atomic filter predicates (recursing through AND/OR/NOT)."""
            if not isinstance(filt, dict):
                return 0
            op = (filt.get("operator") or "").upper()
            if "children" in filt and isinstance(filt["children"], list) and op in {"AND", "OR", "NOT"}:
                return sum(_count_atomic_filters(ch) for ch in filt["children"])
            # atomic filter
            if "table_name" in filt and "col_name" in filt:
                return 1
            return 0

        def _has_any_index_cond(filt: Any) -> bool:
            """Return True if any atomic predicate has is_index_cond=True."""
            if not isinstance(filt, dict):
                return False
            op = (filt.get("operator") or "").upper()
            if "children" in filt and isinstance(filt["children"], list) and op in {"AND", "OR", "NOT"}:
                return any(_has_any_index_cond(ch) for ch in filt["children"])
            return bool(filt.get("is_index_cond"))

        def _get_tables_and_scan_info(node: Dict[str, Any],
                                      scan_info: Dict[str, Dict[str, Any]]):
            """
            DFS over plan tree.  For each node with table_name, record scan info.
            If a table appears in multiple nodes (e.g. self-join),
            we keep the scan with the largest total work (est_card × est_loops).
            """
            pp = node.get("plan_parameters", {})
            tname = pp.get("table_name")
            if isinstance(tname, str) and tname:
                est_card = float(pp.get("est_card") or 0.0)
                est_loops = float(pp.get("est_loops") or 1.0)
                total_work = est_card * est_loops

                # Keep scan with largest total_work for this table
                prev = scan_info.get(tname)
                if prev is None or total_work > prev.get("_total_work", 0.0):
                    filt = pp.get("filter")
                    scan_info[tname] = {
                        "op_name": pp.get("op_name", ""),
                        "est_card": est_card,
                        "est_loops": est_loops,
                        "_total_work": total_work,
                        "has_index_cond": _has_any_index_cond(filt) if filt else False,
                        "num_filters": _count_atomic_filters(filt) if filt else 0,
                    }
            for child in node.get("children", []):
                _get_tables_and_scan_info(child, scan_info)

        def _norm_scan_val(x: float, key: str) -> float:
            """Z-score normalize a scan scalar using pre-computed scan_norms."""
            spec = self.scan_norms.get(key)
            if not spec:
                return x
            p1, p99 = spec.get("p1", x), spec.get("p99", x)
            mean, std = spec.get("mean", 0.0), spec.get("std", 1.0) or 1.0
            if x < p1: x = p1
            if x > p99: x = p99
            return (x - mean) / std

        tab2idx = _build_table_index_map()
        K = len(tab2idx)
        # Collect scan-level info per table via DFS
        scan_info: Dict[str, Dict[str, Any]] = {}
        _get_tables_and_scan_info(self.plan, scan_info)

        feature_vectors = []
        feature_tables = []
        total_rows = sum((t.get("tableSize")) for t in self.table_stats.values())
        norms = get_table_norm_stats(total_rows)

        # Build feature vectors for each table that appears in the plan
        for table_in_plan in sorted(scan_info.keys()):
            if table_in_plan not in tab2idx:
                continue

            # ── One-hot table ID ──
            row = [0.0] * K
            row[tab2idx[table_in_plan]] = 1.0

            # ── Static schema scalars (14 dims) ──
            rec = self.table_stats.get(table_in_plan)
            raw = compute_table_scalars(rec, total_rows)
            scalars_norm = [_norm_val(raw[k], k, norms) for k in TABLE_SCALAR_KEYS]

            # ── Scan-level features (13 dims) ──
            si = scan_info.get(table_in_plan, {})

            # Scan type one-hot (6 dims)
            op_name = si.get("op_name", "")
            scan_idx = _scan_type_to_idx.get(op_name, _scan_type_to_idx["Other"])
            scan_onehot = [0.0] * len(SCAN_TYPES)
            scan_onehot[scan_idx] = 1.0

            # Scan scalars (5 dims, z-score normalized)
            est_card = si.get("est_card", 0.0)
            est_loops = si.get("est_loops", 1.0)
            table_rows = float((rec or {}).get("tableSize", 0.0) or 0.0)
            selectivity = min(est_card / max(table_rows, 1.0), 1.0) if table_rows > 0 else 0.0

            scan_scalars_raw = {
                "scan_est_card_log": math.log1p(est_card),
                "scan_est_loops_log": math.log1p(est_loops),
                "scan_selectivity": selectivity,
                "total_scan_work_log": math.log1p(est_card * est_loops),
                "num_local_filters": float(si.get("num_filters", 0)),
            }
            scan_scalars_norm = [_norm_scan_val(scan_scalars_raw[k], k) for k in SCAN_SCALAR_KEYS]

            # Binary flag: has_index_cond (1 dim)
            has_idx_cond = 1.0 if si.get("has_index_cond", False) else 0.0

            # Assemble: one_hot_table + schema_scalars + scan_onehot + scan_scalars + has_index_cond
            feature_vectors.append(row + scalars_norm + scan_onehot + scan_scalars_norm + [has_idx_cond])
            feature_tables.append(table_in_plan)
        return feature_tables, feature_vectors, tab2idx

    def extract_features_for_joins(self, edge_to_id, algo_to_id, op_to_id, norms, md) -> list[Any]:
        J, A, O = len(edge_to_id), len(algo_to_id), len(op_to_id)
        """
        Build the Joins set for ONE plan as an n x |J| one-hot matrix.

        Inputs:
          - plan: one parsed_plan (JSON object with 'plan_parameters' and 'children', or with a 'plan' key).
          - edge_to_id: global vocabulary mapping (from unique_join_pairs).
          - unseen edges are encoded as <UNK_EDGE>

        Returns:
          - onehots: List[List[float]] of shape (n, |J|), one row per *join node* encountered.
          - edges_used: List[Edge], same order as onehots rows.
        """

        def _canonical_edge(t1: str, t2: str) -> Edge:
            """Make the edge order-invariant (unordered pair)."""
            return (t1, t2) if t1 <= t2 else (t2, t1)

        def _one_hot(idx: int, size: int) -> List[float]:
            r = [0.0] * size
            if 0 <= idx < size:
                r[idx] = 1.0
            return r

        def _norm(x: float, key: str, n: Dict[str, Dict[str, float]]) -> float:
            """Clip to [p1,p99] then z-score with train mean/std. Safe for missing keys."""
            spec = n.get(key)
            if spec is None:
                return float(x)
            p1, p99 = spec.get("p1", x), spec.get("p99", x)
            mean, std = spec.get("mean", 0.0), spec.get("std", 1.0) or 1.0
            # clip
            if x < p1: x = p1
            if x > p99: x = p99
            return (x - mean) / std

        def _is_join(node: Dict[str, Any]) -> bool:
            j = node.get("plan_parameters", {}).get("join", {})
            if not isinstance(j, dict):
                return False
            t1 = j.get("table_name1") or j.get("left_table")
            t2 = j.get("table_name2") or j.get("right_table")
            return isinstance(t1, str) and bool(t1) and isinstance(t2, str) and bool(t2)

        def _gather_join_edges_from_node(node: Dict, out, depth=0):
            """
            DFS a single plan tree; append one edge per *join node* found in node['plan_parameters'].
            Only reads from 'plan_parameters' and recurses into 'children'.
            """
            if not isinstance(node, dict):
                return

            plan_parameters = node.get("plan_parameters", {})
            join = plan_parameters.get("join")
            new_depth = depth
            children = node.get("children", [])
            if isinstance(join, dict):
                t1 = join.get("table_name1")
                t2 = join.get("table_name2")
                new_depth = depth + 1
                if isinstance(t1, str) and t1 and isinstance(t2, str) and t2:
                    edge = _canonical_edge(t1, t2)
                    edge_idx = edge_to_id.get(edge, edge_to_id["<UNK_EDGE>"])

                    alg_name = plan_parameters.get("op_name")
                    alg_idx = algo_to_id.get(alg_name, algo_to_id["<UNK_ALG>"])
                    left_op_name = children[0].get("plan_parameters", {}).get("op_name") if len(children) > 0 else None
                    right_op_name = children[1].get("plan_parameters", {}).get("op_name") if len(children) > 1 else None
                    left_op_idx = op_to_id.get(left_op_name, op_to_id["<UNK_OP>"])
                    right_op_idx = op_to_id.get(right_op_name, op_to_id["<UNK_OP>"])
                    onehot_edge = _one_hot(edge_idx, J)
                    onehot_alg = _one_hot(alg_idx, A)
                    onehot_left = _one_hot(left_op_idx, O)
                    onehot_right = _one_hot(right_op_idx, O)

                    est_card_out_log = math.log1p(plan_parameters.get("est_card", 0.0))
                    est_width_out = float(plan_parameters.get("est_width", 0.0))
                    est_loops_log = math.log1p((plan_parameters.get("est_loops", 0.0)))

                    lpp = children[0].get("plan_parameters", {}) if len(children) > 0 else {}
                    rpp = children[1].get("plan_parameters", {}) if len(children) > 1 else {}
                    left_raw = float(lpp.get("est_card") or 0.0)
                    right_raw = float(rpp.get("est_card") or 0.0)
                    left_card_log = math.log1p(left_raw)
                    right_card_log = math.log1p(right_raw)

                    md_safe = max(float(md), 1.0)
                    raw_depth_norm = float(depth) / md_safe
                    is_root_join = 1.0 if depth == 0 else 0.0
                    right_is_join = _is_join(children[1]) if len(children) > 1 else False
                    left_deep_hint = 1.0 if (len(children) > 1 and not right_is_join) else 0.0

                    denom = max(left_raw * right_raw, 1.0)
                    join_sel_log = math.log1p((plan_parameters.get("est_card", 0.0)) / denom)

                    # normalize numerics using your collected norms
                    est_card_out_log = _norm(est_card_out_log, "est_card_out_log", norms)
                    est_width_out = _norm(est_width_out, "est_width_out", norms)
                    est_loops_log = _norm(est_loops_log, "est_loops_log", norms)
                    left_card_log = _norm(left_card_log, "left_card_log", norms)
                    right_card_log = _norm(right_card_log, "right_card_log", norms)
                    join_sel_log = _norm(join_sel_log, "join_sel_log", norms)

                    ic = join.get("is_index_cond")
                    index_cond_flag = 1.0 if (isinstance(ic, (bool, int)) and bool(ic)) else 0.0

                    r = (onehot_edge
                        + onehot_alg
                        + onehot_left
                        + onehot_right
                        + [raw_depth_norm, is_root_join, left_deep_hint]
                        + [est_card_out_log, est_width_out, est_loops_log]
                        + [left_card_log, right_card_log, join_sel_log]
                        + [index_cond_flag])

                    out.append(r)

            for child in children:
                _gather_join_edges_from_node(child, out, new_depth)

        edges = []
        _gather_join_edges_from_node(self.plan, edges)
        return edges

    def extract_feature_for_predicates(
            self,
            column_to_id: Dict[str, int],  # frozen from schema
            predop_to_id: Dict[str, int],  # frozen from train (+ "<UNK_OP>")
            col_norms: Dict[str, Dict[str, float]],  # per-column stats {mean,std,p1,p99}
    ) -> List[Any]:
        """
        Returns n × Fp matrix with one row per *atomic* predicate:
          [ one_hot_colA(|C|) | one_hot_colB(|C|) | one_hot_op(|P|)
          | val_low_norm, val_high_norm | is_join_pred, is_index_cond ]
        - Flattens boolean filters with 'children' (AND/OR/NOT) into atomic predicates.
        - Keeps join predicates as before (two columns, operator from join.operator, values = 0,0).
        """

        C, P = len(column_to_id), len(predop_to_id)
        rows: List[Any] = []

        def _clip_z(x: float, spec: Dict[str, float]) -> float:
            # spec keys: mean, std, p1, p99
            p1, p99 = spec.get("p1", x), spec.get("p99", x)
            mean, std = spec.get("mean", 0.0), spec.get("std", 1.0) or 1.0
            if x < p1: x = p1
            if x > p99: x = p99
            return (x - mean) / std

        def norm_val(col_key, v: Optional[float]) -> float:
            if v is None:
                return 0.0
            spec = col_norms.get(col_key)
            return _clip_z(float(v), spec) if spec else 0.0

        def _to_float(x: Any) -> Optional[float]:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, bool):
                return float(int(x))
            if isinstance(x, str):
                s = x.strip()
                if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
                    s = s[1:-1].strip()
                s = s.replace(",", "")
                if s == "":
                    return None
                try:
                    return float(s)
                except Exception:
                    return None
            return None

        def _one_hot(idx: int, size: int) -> List[float]:
            row = [0.0] * size
            if 0 <= idx < size:
                row[idx] = 1.0
            return row

        def _gather_atomic_filters(filt: Any, out: List[Dict[str, Any]]) -> None:
            """
            Append *atomic* filter dicts to `out`.
            Accepts either:
              - atomic dict with keys {table_name, col_name, operator, literal, is_index_cond, ...}
              - boolean dict: {"operator": "AND"/"OR"/"NOT", "children": [ ... ]}
            """
            if not isinstance(filt, dict):
                return

            # Boolean node with children
            op = (filt.get("operator") or "").upper()
            if "children" in filt and isinstance(filt["children"], list) and op in {"AND", "OR", "NOT"}:
                for child in filt["children"]:
                    _gather_atomic_filters(child, out)
                return

            # Atomic node (the shape from your second example)
            # Expected keys: table_name, col_name, operator, literal, is_index_cond
            if "table_name" in filt and "col_name" in filt and "operator" in filt:
                out.append(filt)

        def encode_atomic_filter(f: Dict[str, Any]):
            # Build column key
            t = f.get("table_name")
            c = f.get("col_name")
            if not (isinstance(t, str) and isinstance(c, str) and t and c):
                return
            colA = f"{t}.{c}"
            a_idx = column_to_id.get(colA)
            if a_idx is None:
                return  # unknown column in schema

            # Operator one-hot
            op = (f.get("operator") or "").upper()
            p_idx = predop_to_id.get(op, predop_to_id.get("<UNK_OP>", 0))
            op_hot = _one_hot(p_idx, P)

            # Numeric values (two slots). Non-numeric → 0,0.
            lit = f.get("literal")
            # Handle BETWEEN/IN if your dataset sometimes places them as atomic (rare); otherwise treat as unary
            if op == "BETWEEN":
                low = high = None
                if isinstance(lit, (list, tuple)) and len(lit) >= 2:
                    low, high = _to_float(lit[0]), _to_float(lit[1])
                elif isinstance(lit, str) and "AND" in lit:
                    parts = [p.strip() for p in lit.split("AND", 1)]
                    if len(parts) == 2:
                        low, high = _to_float(parts[0]), _to_float(parts[1])
                if low is not None and high is not None and low > high:
                    low, high = high, low
                v1 = norm_val(colA, low)
                v2 = norm_val(colA, high)
            elif op == "IN":
                v1 = v2 = 0.0
                if isinstance(lit, (list, tuple)) and len(lit) > 0:
                    nums = [v for v in (_to_float(x) for x in lit) if v is not None]
                    if nums:
                        m = sum(nums) / len(nums)
                        v1 = v2 = norm_val(colA, m)
                else:
                    v = _to_float(lit)
                    v1 = v2 = norm_val(colA, v)
            else:
                v = _to_float(lit)
                v1 = v2 = norm_val(colA, v)

            is_join_pred = 0.0
            is_index_cond = 1.0 if bool(f.get("is_index_cond")) else 0.0
            row = (
                    _one_hot(a_idx, C) + [0.0] * C +  # colB empty
                    op_hot +
                    [v1, v2] +
                    [is_join_pred, is_index_cond]
            )
            rows.append(row)

        def encode_join(pp: Dict[str, Any]):
            j = pp.get("join")
            if not isinstance(j, dict):
                return
            t1 = j.get("table_name1") or j.get("left_table")
            c1 = j.get("column_name1") or j.get("col_name1") or j.get("col1")
            t2 = j.get("table_name2") or j.get("right_table")
            c2 = j.get("column_name2") or j.get("col_name2") or j.get("col2")
            if not (isinstance(t1, str) and isinstance(c1, str) and isinstance(t2, str) and isinstance(c2, str)):
                return
            colA = f"{t1}.{c1}"
            colB = f"{t2}.{c2}"
            a_idx = column_to_id.get(colA)
            b_idx = column_to_id.get(colB)
            if a_idx is None or b_idx is None:
                return

            op = (j.get("operator") or "=").upper()
            p_idx = predop_to_id.get(op, predop_to_id.get("<UNK_OP>", 0))
            op_hot = _one_hot(p_idx, P)

            is_join_pred = 1.0
            is_index_cond = 1.0 if bool(j.get("is_index_cond")) else 0.0

            row = (
                    _one_hot(a_idx, C) +
                    _one_hot(b_idx, C) +
                    op_hot +
                    [0.0, 0.0] +
                    [is_join_pred, is_index_cond]
            )
            rows.append(row)

        def dfs(node: Dict[str, Any]):
            pp = node.get("plan_parameters", {})

            filt = pp.get("filter")
            atoms: List[Dict[str, Any]] = []

            if isinstance(filt, dict):
                _gather_atomic_filters(filt, atoms)
            elif isinstance(filt, list):
                for item in filt:
                    _gather_atomic_filters(item, atoms)

            for af in atoms:
                encode_atomic_filter(af)

            encode_join(pp)

            for ch in node.get("children", []):
                dfs(ch)

        dfs(self.plan)
        return rows