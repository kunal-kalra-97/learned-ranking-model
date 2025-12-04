import json
import math
import os
import re
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set, Optional

from feature_extractor import FeatureExtractor


def extract_feature(plan:Dict, table_stats: Dict[str, Dict[str, Any]], edge_to_id, algo_to_id, op_to_id, norms, md, col_to_id, pred_op_to_id, col_norms)->Any:
    feature_extractor = FeatureExtractor(plan, table_stats)
    table_features, feature_vectors, tab2idx = feature_extractor.extract_features_for_tables()
    join_features = feature_extractor.extract_features_for_joins(edge_to_id, algo_to_id, op_to_id, norms, md)
    pred_features = feature_extractor.extract_feature_for_predicates(col_to_id, pred_op_to_id, col_norms)

    return [feature_vectors, join_features, pred_features]


    # DEMO Implementation: return the depth of the plan
    def compute_depth(node:Dict)->int:
        if 'children' not in node or not node['children']:
            return 1
        return 1 + max(compute_depth(child) for child in node['children'])

    return [42, compute_depth(plan)]  # Replace with actual feature vector


def extract_table_column_map(column_stats: List, table_stats: List)->Dict[str, Dict[str, Any]]:
    """
        Extracts table statistics from the provided column statistics.

        Parameters:
            table_stats:
            column_stats (list): List of column statistics for each table.

        Returns:
            Dictionary of table statistics in the following format:
        {
            <tableName>: {
                "tableName": <str>,
                "tableSize": <float or None>,
                "columns": [
                    {
                    }, ...
                ]
            }, ...
        }

    """

    table_column_map:Dict[str, Dict[str, Any]] = {}

    for table_stat in table_stats:
        table_name = table_stat['relname']
        if table_name not in table_column_map:
            table_column_map[table_name] = {
                "tableSize": table_stat["reltuples"] or 0.0,
                "columns": []
            }

    for column_stat in column_stats:
        table_name = column_stat['tablename']
        table_column_map[table_name]["columns"].append({
            "colName": column_stat["attname"],
            "dataType": column_stat["data_type"],
            "avgWidth": column_stat["avg_width"],
            "nullFrac": column_stat["null_frac"],
            "nDistinct": column_stat["n_distinct"],
            "correlation": column_stat["correlation"],
        })
    return table_column_map

Edge = Tuple[str, str]

def get_edge_dictionary_and_join_stats(parsed_plans: List[Any]):
    """
        One-time pass over training plans to build a stable join-edge vocabulary.
        Returns: edge_to_id mapping (unordered table pair -> index).
    """

    def _canonical_edge(t1: str, t2: str) -> Edge:
        """Make the edge order-invariant (unordered pair)."""
        return (t1, t2) if t1 <= t2 else (t2, t1)

    def _push(stats, key, val):
        if val is None: return
        stats[key].append(float(val))

    buckets = defaultdict(list)
    norms = {}


    def _gather_join_edges_from_node(node: Dict, edges_list: List[Edge], algos_list: List[str], ops_list: List[str], depth = 0):
        """
        DFS a single plan tree; append one edge per *join node* found in node['plan_parameters'].
        """
        if not isinstance(node, dict):
            return depth

        plan_parameters = node.get("plan_parameters", {})
        join = plan_parameters.get("join")
        new_depth = depth
        if isinstance(join, dict):
            op_name = plan_parameters.get("op_name")
            children = node.get("children")
            t1 = join.get("table_name1")
            t2 = join.get("table_name2")
            if isinstance(t1, str) and t1 and isinstance(t2, str) and t2:
                edges_list.append(_canonical_edge(t1, t2))
            if isinstance(op_name, str) and op_name:
                algos_list.append(op_name)
            _push(buckets, "est_card_out_log", math.log1p(plan_parameters.get("est_card", 0.0)))
            _push(buckets, "est_width_out", plan_parameters.get("est_width", 0.0))
            _push(buckets, "est_loops_log", math.log1p(plan_parameters.get("est_loops", 0.0)))
            # children
            lpp = children[0].get("plan_parameters", {}) if len(children) > 0 else {}
            rpp = (children[1].get("plan_parameters") or {}) if len(children) > 1 else {}
            left_raw = float(lpp.get("est_card") or 0.0)
            right_raw = float(rpp.get("est_card") or 0.0)
            _push(buckets, "left_card_log", math.log1p(left_raw))
            _push(buckets, "right_card_log", math.log1p(right_raw))
            denom = max(left_raw * right_raw, 1.0)
            _push(buckets, "join_sel_log", math.log1p((plan_parameters.get("est_card", 0.0)) / denom))
            new_depth = depth + 1
            # Will recursively reach into the leaf nodes of the join tree.
            for ch in node.get("children"):
                ch_op = ch.get("plan_parameters", {}).get("op_name")
                if isinstance(ch_op, str) and ch_op:
                    ops_list.append(ch_op)

        for child in (node.get("children", [])):
            return max(new_depth, _gather_join_edges_from_node(child, edges_list, algos_list, ops_list, new_depth))

        return new_depth

    for k, arr in buckets.items():
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        mean = sum(arr_sorted) / n if n else 0.0
        var = sum((x - mean) ** 2 for x in arr_sorted) / max(n - 1, 1)
        std = var ** 0.5 or 1.0
        p1 = arr_sorted[int(0.01 * (n - 1))] if n else 0.0
        p99 = arr_sorted[int(0.99 * (n - 1))] if n else 1.0
        norms[k] = {"mean": mean, "std": std, "p1": p1, "p99": p99}

    edges_seen: Set[Edge] = set()
    algos_seen: Set[str] = set()
    ops_seen: Set[str] = set()
    md = 0
    for plan in parsed_plans:
        edges: List[Edge] = []
        algos: List[str] = []
        ops: List[str] = []
        d = _gather_join_edges_from_node(plan, edges, algos, ops, 1)
        md = max(d, md)
        edges_seen.update(edges)
        algos_seen.update(algos)
        ops_seen.update(ops)
    edge_list = sorted(edges_seen)
    algo_list = sorted(algos_seen)
    op_list  = sorted(ops_seen)
    edge_to_id = {edge: i for i, edge in enumerate(edge_list)}
    algo_to_id = {algo: i for i, algo in enumerate(algo_list)}
    op_to_id = {op: i for i, op in enumerate(op_list)}
    edge_to_id["<UNK_EDGE>"] = len(edge_to_id)
    algo_to_id["<UNK_ALG>"] = len(algo_to_id)
    op_to_id["<UNK_OP>"] = len(op_to_id)
    return edge_to_id, algo_to_id, {k: v for k, v in op_to_id.items() if k not in algo_to_id}, norms, md

def build_predicate_vocabs(parsed_plans, column_stats):
    """
    Returns:
      column_to_id : fixed mapping over schema columns (table_map) -> index in [0, |C|-1]
      predop_to_id : observed predicate operators -> index; includes '<UNK_OP>'
    """
    # Columns: derive from schema (stable)
    cols = []
    for col in column_stats:
        c = col['attname']
        tn = col['tablename']
        cols.append(f"{tn}.{c}")
    column_to_id = {c: i for i, c in enumerate(sorted(set(cols)))}

    # Operators: collect from filters and joins in training plans
    ops_seen: Set[str] = set()
    def dfs(node):
        plan_parameters = node["plan_parameters"]
        # filter operator
        filt = plan_parameters.get("filter")
        if isinstance(filt, dict):
            op = filt.get("operator")
            if isinstance(op, str) and op:
                ops_seen.add(op.upper())
                for fc in filt.get("children", []):
                    ops_seen.add(fc.get("operator").upper())
        # join operator (typically '=')
        j = plan_parameters.get("join")
        if isinstance(j, dict):
            op = j.get("operator")
            if isinstance(op, str) and op:
                ops_seen.add(op.upper())
        for ch in node.get("children", []):
            dfs(ch)

    for pl in parsed_plans:
        dfs(pl)

    op_list = sorted(ops_seen) + ["<UNK_OP>"]
    predop_to_id = {name: i for i, name in enumerate(op_list)}
    return column_to_id, predop_to_id

def build_predicate_norm_stats(parsed_plans,column_to_id):
    """
    Scan training plans and collect numeric literal statistics per column:
      stats[col_key] = {mean, std, p1, p99}
    We look only at filter.literals (JOINs typically have no literal).
    """

    def _gather_atomic_filters(filt: Any, out: List[Dict[str, Any]]) -> None:
        """
        Append *atomic* filters to out.
        Accepts:
          - boolean dict: {"operator": "AND"/"OR"/"NOT", "children":[...]}
          - atomic dict: {table_name, col_name, operator, literal, is_index_cond, ...}
        """
        if not isinstance(filt, dict):
            return
        op = (filt.get("operator") or "").upper()
        if "children" in filt and isinstance(filt["children"], list) and op in {"AND", "OR", "NOT"}:
            for ch in filt["children"]:
                _gather_atomic_filters(ch, out)
            return
        # atomic
        if "table_name" in filt and "col_name" in filt and "operator" in filt:
            out.append(filt)

    buckets: Dict[str, List[float]] = defaultdict(list)

    def add(col_key: str, v: Optional[float]):
        if v is None or col_key not in column_to_id:
            return
        buckets[col_key].append(float(v))

    _NUM_RE = re.compile(r"^[\s'\"]*([-+]?\d+(\.\d+)?)[\s'\%]*$")  # matches numbers like '123', "12.3", %99
    def _to_float_strict(x: Any) -> Optional[float]:
        """Return float if x is numeric or a quoted/percent-wrapped numeric string; else None."""
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            m = _NUM_RE.match(x)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
        return None

    def handle_atomic(f: Dict[str, Any]):
        t = f.get("table_name");
        c = f.get("col_name")
        if not (isinstance(t, str) and isinstance(c, str) and t and c):
            return
        key = f"{t}.{c}"
        op = (f.get("operator") or "").upper()
        lit = f.get("literal")

        if op == "BETWEEN":
            low = high = None
            # common encodings: list/tuple; "a AND b"; "a..b"
            if isinstance(lit, (list, tuple)) and len(lit) >= 2:
                low, high = _to_float_strict(lit[0]), _to_float_strict(lit[1])
            elif isinstance(lit, str):
                if "AND" in lit:
                    parts = [p.strip() for p in lit.split("AND", 1)]
                    if len(parts) == 2:
                        low, high = _to_float_strict(parts[0]), _to_float_strict(parts[1])
                elif ".." in lit:
                    parts = [p.strip() for p in lit.split("..", 1)]
                    if len(parts) == 2:
                        low, high = _to_float_strict(parts[0]), _to_float_strict(parts[1])
            # record if parseable
            if low is not None and high is not None:
                if low > high:
                    low, high = high, low
                add(key, low)
                add(key, high)

        elif op == "IN":
            if isinstance(lit, (list, tuple)):
                for v in lit:
                    add(key, _to_float_strict(v))
            else:
                add(key, _to_float_strict(lit))

        else:
            # unary/binary ops with single literal: =, <, <=, >, >=, <> ...
            add(key, _to_float_strict(lit))

    def dfs(node: Dict[str, Any]):
        pp = node.get("plan_parameters", {})
        filt = pp.get("filter")
        if isinstance(filt, dict):
            atoms: List[Dict[str, Any]] = []
            _gather_atomic_filters(filt, atoms)
            for af in atoms:
                handle_atomic(af)
        for ch in node.get("children", []):
            dfs(ch)

    # walk all training plans
    for pl in parsed_plans:
        dfs(pl)

    # reduce to stats
    stats: Dict[str, Dict[str, float]] = {}
    for col, arr in buckets.items():
        if not arr:
            continue
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        mean = sum(arr_sorted) / n
        var = sum((x - mean) ** 2 for x in arr_sorted) / max(n - 1, 1)
        std = math.sqrt(var)
        if std < 1e-12:  # avoid zero-division later
            std = 1.0
        p1_idx = int(0.01 * (n - 1))
        p99_idx = int(0.99 * (n - 1))
        stats[col] = {
            "mean": mean,
            "std": std,
            "p1": arr_sorted[p1_idx],
            "p99": arr_sorted[p99_idx],
        }
    return stats

def extract_features(file_path:str):
    """
    Extracts features and labels from the provided JSON file.

    Parameters:
        file_path (str): Path to the input JSON file.
        estimated_regex (re.Pattern): Regular expression to match and extract estimated values.

    Returns:
        tuple: A tuple containing a list of feature vectors and corresponding labels.
    """
    with open(file_path, 'r') as file: 
        json_data = json.load(file)
    
    plans = json_data['parsed_plans']

    column_stats = json_data['database_stats']['column_stats']
    table_stats = json_data['database_stats']['table_stats']
    table_column_map = extract_table_column_map(column_stats, table_stats)
    edge_to_id, algo_to_id, op_to_id, norms, md = get_edge_dictionary_and_join_stats(plans)
    col_to_id, preop_to_id = build_predicate_vocabs(plans, column_stats)
    pred_norm_stats = build_predicate_norm_stats(plans,col_to_id)
    feature_vectors = []
    max_table, max_join, max_pred = 0, 0, 0
    for plan in plans[:2]:
        # extract label
        label = plan.get("plan_runtime_ms")

        # if label is none (timeout during execution), we skip this entry
        if label is None:
            continue

        # extract query identifier (to map this plan to the corresponding query)
        sql = plan.pop("sql")
        # extract feature information
        features = extract_feature(plan, table_column_map, edge_to_id, algo_to_id,
                                   op_to_id, norms, md, col_to_id, preop_to_id, pred_norm_stats)

        max_table = max(max_table, len(features[0]))
        max_join = max(max_join, len(features[1]))
        max_pred = max(max_pred, len(features[2]))
        feature_vectors.append({
            'sql': sql,
            'features': features,
            'label': label
        })
    print(max_table, max_join, max_pred)

    return feature_vectors, (max_table, max_join, max_pred)

def save_data(file_path, data):
    """
    Saves data to the specified file path in JSON format.

    Parameters:
        file_path (str): Path to save the output file.
        data (list): Data to be saved.

    Returns:
        None
    """

    # make sure the output path exists
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, 'w') as file:
        json.dump(data, file)

def main():
    parser = argparse.ArgumentParser(description="Extract features from the provided JSON file.")
    parser.add_argument("--file_path", type=str, help="Path to the input workload JSON file.", required=True)
    parser.add_argument("--output_path", type=str, help="Path to the output features JSON file", required=True)
    args = parser.parse_args()

    feature_vectors, max_values = extract_features(args.file_path)

    save_data(args.output_path, feature_vectors)
    print("Feature vectors saved successfully!")

if __name__ == "__main__":
    main()