import json
import os
import re
import argparse
from typing import Any, Dict


def extract_feature(plan:Dict)->Any:

    # TODO implement feature extraction logic here

    # DEMO Implementation: return the depth of the plan
    def compute_depth(node:Dict)->int:
        if 'children' not in node or not node['children']:
            return 1
        return 1 + max(compute_depth(child) for child in node['children'])

    return [42, compute_depth(plan)]  # Replace with actual feature vector


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

    feature_vectors = []

    for plan in plans:
        # extract label
        label = plan.get("plan_runtime_ms")

        # if label is none (timeout during execution), we skip this entry
        if label is None:
            continue

        # extract query identifier (to map this plan to the corresponding query)
        sql = plan.pop("sql")
        hint = plan.pop("hint")

        # extract feature information
        features = extract_feature(plan)

        feature_vectors.append({
            'sql': sql,
            'hint': hint,
            'features': features,
            'label': label
        })

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