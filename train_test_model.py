import json
from typing import Any, Dict, List
import pandas as pd
import joblib
import argparse
from evaluate_model import evaluate_model
from model.data_utils import make_mscn_batch, infer_feature_dims_from_dataset
from model.mscn_dataset import make_dataloaders
from model.ranking_model_wrapper import RankingModelWrapper
import numpy as np


def load_features_from_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        data  = json.load(f)
    return data

def extract_runtime_from_labels(data):
    labels = []
    """Load only runtimes for training"""
    for entry in data:
        if "runtime" in entry:
            labels.append(entry["runtime"])

    return labels

def organize_as_ranking_benchmark(df:pd.DataFrame)->List[Dict[str,Any]]:
    """
    Organize the per-query features and labels as a ranking benchmark.
    For this we group by the sql query. The task is then to pick the fastest plan for each query.
    """
    ranking_benchmark = []
    grouped = df.groupby('sql')
    for sql, group in grouped:
        # Pick the fastest plan (minimum runtime) for each query
        fastest_plan = group.loc[group['label'].idxmin()]
        ranking_benchmark.append({
            'sql': sql,
            'fastest_runtime': fastest_plan['label'],
            'plan_candidates_features': group['features'].tolist(),
            'plan_candidates_runtimes': group['label'].tolist()
        })
    return ranking_benchmark

def main(args):
    # Load features and labels
    data = load_features_from_file(args.train_data)
    test_data = load_features_from_file(args.test_data)
    #
    # # convert to pd dataframe
    train_df = pd.DataFrame(data)
    test_df = pd.DataFrame(test_data)
    batch_samples = train_df['features'].tolist()
    # batch_labels = train_df['label'].tolist()
    ft, fj, fp = infer_feature_dims_from_dataset(batch_samples)

    # # extract info
    # train_sql = train_df['sql'].to_numpy()
    #
    # train_features = np.array(train_df['features'].tolist())
    # train_runtimes = train_df['label'].to_numpy()

    # Initialize and train the model
    model = RankingModelWrapper(feature_dims = (ft, fj, fp))
    model.fit()
    
    # Dump the model for later be used in the evaluation platform
    joblib.dump(model, 'model/model.pkl')
    # organize test data as ranking test
    ranking_benchmark = organize_as_ranking_benchmark(test_df)
    evaluate_model(ranking_benchmark)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load features and labels from the provided JSON file.")
    parser.add_argument("--train_data", type=str, help="Path to the training features JSON file.", required=True)
    parser.add_argument("--test_data", type=str, help="Path to the testing queries JSON file", required=True)
    args = parser.parse_args()
    main(args)