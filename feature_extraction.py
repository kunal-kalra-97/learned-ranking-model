import json
import os
import re
import argparse
from typing import Any, Dict, List, Tuple, Set

from feature_extractor import FeatureExtractor


def extract_feature(plan:Dict, table_stats: Dict[str, Dict[str, Any]], unique_join_pairs: dict[tuple[str, str], int])->Any:
    feature_extractor = FeatureExtractor(plan, table_stats)
    feature_tables, feature_vectors, tab2idx = feature_extractor.extract_features_for_tables()
    feature_joins, edges_used, edge_to_id = feature_extractor.extract_features_for_joins(unique_join_pairs)
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
            column_stats (list): List of column statistics for each table.

        Returns:
            Dictionary of table statistics in the following format:
        {
            <tableName>: {
                "tableName": <str>,
                "tableSize": <float or None>,
                "columns": [
                    {
                        "colName": <str>,
                        "dataType":
                        "avgWidth":
                        "nullFrac":
                        "nDistinct":
                        "correlation":
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
                "tableName": table_name,
                "tableSize": table_stat["reltuples"],
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
def extract_unique_join_pairs(parsed_plans: List[Any])-> dict[tuple[str, str], int]:
    """
        One-time pass over training plans to build a stable join-edge vocabulary.
        Returns: edge_to_id mapping (unordered table pair -> index).
    """

    def _canonical_edge(t1: str, t2: str) -> Edge:
        """Make the edge order-invariant (unordered pair)."""
        return (t1, t2) if t1 <= t2 else (t2, t1)

    def _gather_join_edges_from_node(node: Dict, out: List[Edge]) -> None:
        """
        DFS a single plan tree; append one edge per *join node* found in node['plan_parameters'].
        Only reads from 'plan_parameters' and recurses into 'children'.
        """
        if not isinstance(node, dict):
            return

        plan_parameters = node.get("plan_parameters", {})
        join = plan_parameters.get("join")

        if isinstance(join, dict):
            t1 = join.get("table_name1")
            t2 = join.get("table_name2")
            if isinstance(t1, str) and t1 and isinstance(t2, str) and t2:
                out.append(_canonical_edge(t1, t2))

        for child in (node.get("children") or []):
            _gather_join_edges_from_node(child, out)

    edges_seen: Set[Edge] = set()

    for plan in parsed_plans:
        edges: List[Edge] = []
        _gather_join_edges_from_node(plan, edges)
        edges_seen.update(edges)
    edge_list = sorted(edges_seen)
    edge_to_id = {edge: i for i, edge in enumerate(edge_list)}
    edge_to_id["<UNK_EDGE>"] = len(edge_to_id)
    return edge_to_id


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
    unique_join_pairs = extract_unique_join_pairs(plans)
    feature_vectors = []

    for plan in plans:
        # extract label
        label = plan.get("plan_runtime_ms")

        # if label is none (timeout during execution), we skip this entry
        if label is None:
            continue

        # extract query identifier (to map this plan to the corresponding query)
        sql = plan.pop("sql")
        print(sql)
        # extract feature information
        features = extract_feature(plan, table_column_map, unique_join_pairs)
        feature_vectors.append({
            'sql': sql,
            'features': features,
            'label': label
        })
        break

    return feature_vectors

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

    feature_vectors = extract_features(args.file_path)

    save_data(args.output_path, feature_vectors)

    print("Feature vectors saved successfully!")

if __name__ == "__main__":
    main()