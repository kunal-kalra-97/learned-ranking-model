from typing import Dict, Any, Tuple, List


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
            plan_parameters = node["plan_parameters"] or {}
            for key in ("table_name", "table_name1", "table_name2", "relname"):
                t = plan_parameters.get(key)
                if isinstance(t, str) and t:
                    distinct_tables.add(t)
                    break
            for child in node["children"] or []:
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








