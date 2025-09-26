from typing import Dict, List
import numpy as np
from joblib import load


def evaluate_model(ranking_benchmark:Dict):
    """
    Evaluate the pre-trained ranking model on the provided ranking benchmark.
    """
    # Load pre-trained model from 'model/model.pkl' file
    loaded_model = load('model/model.pkl')

    # for each query in the ranking benchmark, we have multiple plan candidates
    # from this list of candidates, the model should pick the supposedly fastest plan
    picked_runtimes_ms = []
    for entry in ranking_benchmark:
        sql = entry['sql']
        plan_candidates_features_list:List[List[float]] = entry['plan_candidates_features']
        plan_candidates_runtimes:List[float] = entry['plan_candidates_runtimes']

        # validate data
        assert len(plan_candidates_features_list) == len(plan_candidates_runtimes)
        assert len(plan_candidates_features_list) > 0
        assert all([isinstance(f, list) for f in plan_candidates_features_list])
        assert all([isinstance(r, (int, float)) for r in plan_candidates_runtimes])

        # run inference
        fastest_plan_idx = loaded_model.inference(sql, plan_candidates_features_list)

        # lookup runtime of the picked plan
        picked_runtimes_ms.append(plan_candidates_runtimes[fastest_plan_idx])

    print(f'Sum of picked runtimes: {sum(picked_runtimes_ms)/1000:.2f}s')
    return sum(picked_runtimes_ms)/1000