from typing import Dict, Any, Tuple, List

Edge = Tuple[str, str]
class FeatureExtractor:
    def __init__(self, plan:Dict, table_stats: Dict[str, Dict[str, Any]]):
        self.plan = plan
        self.table_stats = table_stats

    def extract_features_for_tables(self)->Tuple[List[str], List[List[float]], Dict[str, int]]:
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


    def extract_features_for_joins(self, unique_join_pairs: dict[tuple[str, str], int])-> Tuple[List[List[float]], List[Edge], Dict[Edge, int]]:
        def _one_hot_encode_joins_for_plan(
                plan: Dict[str, Any],
                edge_to_id: Dict[Edge, int],
        ) -> Tuple[List[List[float]], List[Edge], Dict[Edge, int]]:
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

            edges: List[Edge] = []
            _gather_join_edges_from_node(plan, edges)

            K = len(edge_to_id)

            onehots: List[List[float]] = []
            edges_used: List[Edge] = []

            for e in edges:
                idx = edge_to_id.get(e)
                if idx is None:
                    idx = edge_to_id.get("<UNK_EDGE>")
                row = [0.0] * K
                row[idx] = 1.0
                onehots.append(row)
                edges_used.append(e)

            return onehots, edges_used, edge_to_id

        return _one_hot_encode_joins_for_plan(self.plan, unique_join_pairs)







