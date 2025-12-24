import math
from typing import Dict, Any, Tuple, List, Optional

Edge = Tuple[str, str]


class FeatureExtractor:
    def __init__(self, plan: Dict, table_stats: Dict[str, Dict[str, Any]]):
        self.plan = plan
        self.table_stats = table_stats

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
            "log_pages",
            "pages_frac",
            "rows_per_page",
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
            return max(-nd * rows, 0.0)

        def compute_table_scalars(
            table_rec: Dict[str, Any],
            total_rows_in_schema: float,
            total_pages_in_schema: float,
        ) -> Dict[str, float]:
            cols = table_rec.get("columns") or []
            rows = table_rec.get("tableSize", None)
            if rows is None:
                rows = table_rec.get("reltuples", None)

            pages = table_rec.get("tablePages", None)
            if pages is None:
                pages = table_rec.get("relpages", None)

            try:
                rows = float(rows) if rows is not None else 0.0
            except Exception:
                rows = 0.0

            try:
                pages = float(pages) if pages is not None else 0.0
            except Exception:
                pages = 0.0

            rows_total = max(float(total_rows_in_schema or 0.0), 1.0)
            pages_total = max(float(total_pages_in_schema or 0.0), 1.0)

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

            # pages are a strong I/O proxy; rows_per_page is a density proxy
            rows_per_page = (rows / pages) if pages > 0 else 0.0

            out = {
                "log_rows": math.log1p(max(rows, 0.0)),
                "rows_frac": rows / rows_total,
                "log_pages": math.log1p(max(pages, 0.0)),
                "pages_frac": pages / pages_total,
                "rows_per_page": rows_per_page,
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

        def get_table_norm_stats(tr, tp):
            buckets = {k: [] for k in TABLE_SCALAR_KEYS}
            for t in self.table_stats.values():
                scalars = compute_table_scalars(t, tr, tp)
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

        def _get_tables_used_in_plan(node: Dict[str, Any], distinct_tables: set):
            """
                DFS over the plan tree; add table names found in node['plan_parameters']
            """
            plan_parameters = node.get("plan_parameters", {})
            t = plan_parameters.get("table_name")
            if isinstance(t, str) and t:
                distinct_tables.add(t)
            for child in node.get("children", []):
                _get_tables_used_in_plan(child, distinct_tables)

        tab2idx = _build_table_index_map()
        K = len(tab2idx)
        tables_used_in_plan = set()
        _get_tables_used_in_plan(self.plan, tables_used_in_plan)
        feature_vectors = []
        feature_tables = []
        total_rows = sum(float(t.get("tableSize") or t.get("reltuples") or 0.0) for t in self.table_stats.values())
        total_pages = sum(float(t.get("tablePages") or t.get("relpages") or 0.0) for t in self.table_stats.values())
        norms = get_table_norm_stats(total_rows, total_pages)
        for table_in_plan in sorted(tables_used_in_plan):
            if table_in_plan not in tab2idx:
                # handle case where table is not in table_stats
                continue
            row = [0.0] * K
            row[tab2idx[table_in_plan]] = 1.0
            rec = self.table_stats.get(table_in_plan)
            raw = compute_table_scalars(rec, total_rows, total_pages)
            scalars_norm = [_norm_val(raw[k], k, norms) for k in TABLE_SCALAR_KEYS]
            feature_vectors.append(row+scalars_norm)
            feature_tables.append(table_in_plan)
        return feature_tables, feature_vectors, tab2idx

    def extract_features_for_joins(
        self,
        edge_to_id,
        algo_to_id,
        op_to_id,
        norms,
        md,
        join_key_norms: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> list[Any]:
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

        # ----- column-stat lookup (for join key features) -----
        def _estimate_ndv(n_distinct, table_rows: float) -> float:
            """Postgres semantics: n_distinct < 0 means -1 * (ndv / table_rows)."""
            if n_distinct is None:
                return 0.0
            try:
                nd = float(n_distinct)
            except Exception:
                return 0.0
            rows = max(float(table_rows or 0.0), 0.0)
            if nd >= 0:
                return nd
            return max(-nd * rows, 0.0)

        col_lookup: Dict[str, Dict[str, Any]] = {}
        for tname, trec in self.table_stats.items():
            t_rows = float(trec.get("tableSize") or trec.get("reltuples") or 0.0)
            for c in (trec.get("columns") or []):
                cname = c.get("colName")
                if not cname:
                    continue
                col_lookup[f"{tname}.{cname}"] = {
                    "nullFrac": float(c.get("nullFrac") or c.get("null_frac") or 0.0),
                    "avgWidth": float(c.get("avgWidth") or c.get("avg_width") or 0.0),
                    "nDistinct": c.get("nDistinct") or c.get("n_distinct"),
                    "correlation": float(c.get("correlation") or 0.0),
                    "tableRows": t_rows,
                }

        def _key_features(table: Any, col: Any) -> Tuple[float, float, float, float]:
            """Return (null_frac, ndv_ratio, abs_corr, avg_width_log) for a join key column."""
            if not (isinstance(table, str) and isinstance(col, str) and table and col):
                return 0.0, 0.0, 0.0, 0.0
            rec = col_lookup.get(f"{table}.{col}")
            if not rec:
                return 0.0, 0.0, 0.0, 0.0
            nf = float(rec.get("nullFrac") or 0.0)
            corr = abs(float(rec.get("correlation") or 0.0))
            aw_log = math.log1p(max(float(rec.get("avgWidth") or 0.0), 0.0))
            rows = float(rec.get("tableRows") or 0.0)
            ndv = _estimate_ndv(rec.get("nDistinct"), rows)
            ndv_ratio = min(max(ndv / max(rows, 1.0), 0.0), 1.0)
            return nf, ndv_ratio, corr, aw_log

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

        def _norm_key(x: float, key: str) -> float:
            """Normalise join-key column scalars (built from DB stats)."""
            if not join_key_norms:
                return float(x)
            spec = join_key_norms.get(key)
            if not spec:
                return float(x)
            p1, p99 = spec.get("p1", x), spec.get("p99", x)
            mean, std = spec.get("mean", 0.0), spec.get("std", 1.0) or 1.0
            if x < p1:
                x = p1
            if x > p99:
                x = p99
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

                    # Join-key column statistics (very predictive when the optimizer chooses between
                    # NLJ/MJ/HJ and when index usage is possible).
                    c1 = join.get("column_name1") or join.get("col_name1") or join.get("col1")
                    c2 = join.get("column_name2") or join.get("col_name2") or join.get("col2")
                    nf1, ndv1, corr1, aw1 = _key_features(t1, c1)
                    nf2, ndv2, corr2, aw2 = _key_features(t2, c2)
                    join_key_feats = [
                        _norm_key(nf1, "key_null_frac"),
                        _norm_key(nf2, "key_null_frac"),
                        _norm_key(ndv1, "key_ndv_ratio"),
                        _norm_key(ndv2, "key_ndv_ratio"),
                        _norm_key(corr1, "key_abs_corr"),
                        _norm_key(corr2, "key_abs_corr"),
                        _norm_key(aw1, "key_avg_width_log"),
                        _norm_key(aw2, "key_avg_width_log"),
                    ]

                    ic = join.get("is_index_cond")
                    index_cond_flag = 1.0 if (isinstance(ic, (bool, int)) and bool(ic)) else 0.0

                    r = (onehot_edge
                        + onehot_alg
                        + onehot_left
                        + onehot_right
                        + [raw_depth_norm, is_root_join, left_deep_hint]
                        + [est_card_out_log, est_width_out, est_loops_log]
                        + [left_card_log, right_card_log, join_sel_log]
                        + join_key_feats
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
          | val_low_norm, val_high_norm
          | colA_stats(6)
          | sel_proxy_log, lit_is_numeric, lit_strlen_log, in_list_len_log, like_wildcard
          | under_or, under_not
          | is_join_pred, is_index_cond ]
        """

        C, P = len(column_to_id), len(predop_to_id)
        rows: List[Any] = []

        # column stat lookup
        def _estimate_ndv(n_distinct: Any, table_rows: float) -> float:
            if n_distinct is None:
                return 0.0
            try:
                nd = float(n_distinct)
            except Exception:
                return 0.0
            rows_ = max(float(table_rows or 0.0), 0.0)
            if nd >= 0:
                return nd
            return max(-nd * rows_, 0.0)

        col_lookup: Dict[str, Dict[str, Any]] = {}
        for tname, trec in self.table_stats.items():
            t_rows = float(trec.get("tableSize") or trec.get("reltuples") or 0.0)
            for c in (trec.get("columns") or []):
                cname = c.get("colName") or c.get("attname")
                if not cname:
                    continue
                col_lookup[f"{tname}.{cname}"] = {
                    "nullFrac": float(c.get("nullFrac") or c.get("null_frac") or 0.0),
                    "avgWidth": float(c.get("avgWidth") or c.get("avg_width") or 0.0),
                    "nDistinct": c.get("nDistinct") or c.get("n_distinct"),
                    "correlation": float(c.get("correlation") or 0.0),
                    "dataType": (c.get("dataType") or c.get("data_type") or ""),
                    "tableRows": t_rows,
                }

        def _dtype_flags(dt: str) -> Tuple[float, float]:
            d = (dt or "").lower()
            is_num = 1.0 if any(k in d for k in ["int", "numeric", "real", "double", "float", "decimal"]) else 0.0
            is_txt = 1.0 if any(k in d for k in ["char", "text", "varchar"]) else 0.0
            return is_num, is_txt

        def _colA_stats(col_key: str) -> List[float]:
            rec = col_lookup.get(col_key)
            if not rec:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            nf = float(rec.get("nullFrac") or 0.0)
            rows = float(rec.get("tableRows") or 0.0)
            ndv = _estimate_ndv(rec.get("nDistinct"), rows)
            ndv_ratio = min(max(ndv / max(rows, 1.0), 0.0), 1.0)
            abs_corr = abs(float(rec.get("correlation") or 0.0))
            aw_log = math.log1p(max(float(rec.get("avgWidth") or 0.0), 0.0))
            is_num, is_txt = _dtype_flags(str(rec.get("dataType") or ""))
            return [nf, ndv_ratio, abs_corr, aw_log, is_num, is_txt]

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

        def _gather_atomic_filters(
            filt: Any,
            out: List[Tuple[Dict[str, Any], float, float]],
            under_or: float = 0.0,
            under_not: float = 0.0,
        ) -> None:
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
                next_under_or = 1.0 if (under_or > 0.0 or op == "OR") else 0.0
                next_under_not = 1.0 if (under_not > 0.0 or op == "NOT") else 0.0
                for child in filt["children"]:
                    _gather_atomic_filters(child, out, next_under_or, next_under_not)
                return

            # Atomic node (the shape from your second example)
            # Expected keys: table_name, col_name, operator, literal, is_index_cond
            if "table_name" in filt and "col_name" in filt and "operator" in filt:
                out.append((filt, under_or, under_not))

        def encode_atomic_filter(f: Dict[str, Any], under_or: float, under_not: float):
            # Build column key
            t = f.get("table_name")
            c = f.get("col_name")
            if not (isinstance(t, str) and isinstance(c, str) and t and c):
                return
            colA = f"{t}.{c}"
            a_idx = column_to_id.get(colA)
            if a_idx is None:
                return

            # Operator one-hot
            op = (f.get("operator") or "").upper()
            p_idx = predop_to_id.get(op, predop_to_id.get("<UNK_OP>", 0))
            op_hot = _one_hot(p_idx, P)

            # Numeric values (two slots).
            lit = f.get("literal")
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

            # extra predicate features (schema-driven) 
            colA_stats = _colA_stats(colA)

            sel_proxy = 0.0
            rec = col_lookup.get(colA)
            if rec:
                nf = float(rec.get("nullFrac") or 0.0)
                rows_ = float(rec.get("tableRows") or 0.0)
                ndv = _estimate_ndv(rec.get("nDistinct"), rows_)
                base = (1.0 - nf) / max(ndv, 1.0)
                if op in {"=", "=="}:
                    sel_proxy = min(max(base, 0.0), 1.0)
                elif op == "IN":
                    L = len(lit) if isinstance(lit, (list, tuple)) else 1
                    sel_proxy = min(max(base * float(L), 0.0), 1.0)
            sel_proxy_log = math.log1p(sel_proxy)

            lit_is_numeric = 0.0
            if op == "BETWEEN" and isinstance(lit, (list, tuple)) and len(lit) >= 2:
                lit_is_numeric = 1.0 if (_to_float(lit[0]) is not None and _to_float(lit[1]) is not None) else 0.0
            elif op == "IN" and isinstance(lit, (list, tuple)):
                lit_is_numeric = 1.0 if any(_to_float(x) is not None for x in lit) else 0.0
            else:
                lit_is_numeric = 1.0 if _to_float(lit) is not None else 0.0

            lit_strlen_log = 0.0
            if isinstance(lit, str):
                s = lit.strip()
                if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
                    s = s[1:-1]
                lit_strlen_log = math.log1p(len(s))

            in_list_len_log = math.log1p(len(lit)) if (op == "IN" and isinstance(lit, (list, tuple))) else 0.0

            like_wildcard = 0.0
            if op in {"LIKE", "ILIKE"} and isinstance(lit, str):
                s = lit.strip()
                if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
                    s = s[1:-1]
                like_wildcard = 1.0 if ("%" in s or "_" in s) else 0.0

            is_join_pred = 0.0
            is_index_cond = 1.0 if bool(f.get("is_index_cond")) else 0.0
            row = (
                _one_hot(a_idx, C) + [0.0] * C +  # colB empty
                op_hot +
                [v1, v2] +
                colA_stats +
                [sel_proxy_log, lit_is_numeric, lit_strlen_log, in_list_len_log, like_wildcard] +
                [under_or, under_not] +
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

            colA_stats = _colA_stats(colA)
            
            sel_proxy_log = 0.0
            lit_is_numeric = 0.0
            lit_strlen_log = 0.0
            in_list_len_log = 0.0
            like_wildcard = 0.0
            under_or = 0.0
            under_not = 0.0

            row = (
                    _one_hot(a_idx, C) +
                    _one_hot(b_idx, C) +
                    op_hot +
                    [0.0, 0.0] +
                    colA_stats +
                    [sel_proxy_log, lit_is_numeric, lit_strlen_log, in_list_len_log, like_wildcard] +
                    [under_or, under_not] +
                    [is_join_pred, is_index_cond]
            )
            rows.append(row)

        def dfs(node: Dict[str, Any]):
            pp = node.get("plan_parameters", {})

            filt = pp.get("filter")
            atoms: List[Tuple[Dict[str, Any], float, float]] = []

            if isinstance(filt, dict):
                _gather_atomic_filters(filt, atoms)
            elif isinstance(filt, list):
                for item in filt:
                    _gather_atomic_filters(item, atoms)

            for af, uo, un in atoms:
                encode_atomic_filter(af, uo, un)

            encode_join(pp)

            for ch in node.get("children", []):
                dfs(ch)

        dfs(self.plan)
        return rows
