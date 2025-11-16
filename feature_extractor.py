import math
from typing import Dict, Any, Tuple, List

Edge = Tuple[str, str]


class FeatureExtractor:
    def __init__(self, plan: Dict, table_stats: Dict[str, Dict[str, Any]]):
        self.plan = plan
        self.table_stats = table_stats
        # print(table_stats)

    def extract_features_for_tables(self) -> Tuple[List[str], List[List[float]], Dict[str, int]]:
        """
           Returns:
             onehots  : List[List[float]]  -> shape (m, K), one row per table used in this plan
             tables   : List[str]          -> the table names in the same order as onehots rows
             tab2idx  : Dict[str, int]     -> global mapping used to build onehots (stable across queries)
           """

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
        for table_in_plan in sorted(tables_used_in_plan):
            if table_in_plan not in tab2idx:
                # handle case where table is not in table_stats
                continue
            row = [0.0] * K
            row[tab2idx[table_in_plan]] = 1.0
            feature_vectors.append(row)
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
                    rpp = children[0].get("plan_parameters", {}) if len(children) > 1 else {}
                    left_raw = float(lpp.get("est_card") or 0.0)
                    right_raw = float(rpp.get("est_card") or 0.0)
                    left_card_log = math.log1p(left_raw)
                    right_card_log = math.log1p(right_raw)

                    raw_depth_norm = float(depth) / float(md)
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

                    r = [
                        onehot_edge
                        + onehot_alg
                        + onehot_left
                        + onehot_right
                        + [raw_depth_norm, is_root_join, left_deep_hint]
                        + [est_card_out_log, est_width_out, est_loops_log]
                        + [left_card_log, right_card_log, join_sel_log]
                        + [index_cond_flag]
                    ]
                    out.append(r)

            for child in children:
                _gather_join_edges_from_node(child, out, new_depth)

        edges = []
        _gather_join_edges_from_node(self.plan, edges)
        return edges
