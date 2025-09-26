# AIDM 2025 - Learned Query Optimizer Training

This repository contains materials for the lab component of the *Extended Seminar - AI for Data Management*. The goal is to equip you with practical skills to develop a learned query optimizer from scratch. Throughout the mini-course and lectures, you will explore database internals, with a focus on query optimization. This hands-on lab will reinforce those concepts by guiding you through the process of building and evaluating a learned query optimizer.

Your primary objective is to develop a model that predicts query runtimes more accurately than the naive baseline. We encourage you to improve your solution iteratively—your submissions are automatically benchmarked on a hidden test set, with results displayed on the [leaderboard](http://runner-aidm.dm.informatik.tu-darmstadt.de:60085). Top performers will receive bonus points. Details on grading are provided in the kick-off slides.

## Task Overview
You will train a learned query optimizer using the provided query workload and database statistics. The model should predict the runtime of each query (labels provided). 80% of the workload is available for training; the remaining 20% is reserved for evaluation.

To participate:
- Upload your trained model to the `model/` directory. Your model should outperform the naive baseline.
- Submit:
    - Your cost model implementation, with a README describing training and inference steps.
    - The trained model, ready for inference on the test data.

## Demo Implementation
A demo query optimizer is provided. To run it:

```
# Set up a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Extract features for the baseline model
python feature_extraction.py --file_path plans/train_plans1.json --output_path datasets/train_features.json
python feature_extraction.py --file_path plans/test_plans.json --output_path datasets/test_features.json

# Train and evaluate the query optimizer
python train_test_model.py --train_data datasets/train_features.json --test_data datasets/test_features.json
```

`feature_extraction.py` generates features for model training. 

## Feature Engineering and Model Extensions
To improve upon the baseline, consider augmenting features with operator types and estimated cardinalities. While true cardinalities are unknown before execution, you can use cardinality estimation techniques. Including operator type and estimated cardinality in your feature vector can help the model select better plans. For more details, see this [paper](https://ieeexplore.ieee.org/document/4812438).

Possible model architectures:
- Cost-based Query Optimizer: predict costs and pick the plan with the lowest runtime
- Ranking Model: select the fastest plan from a list of plans (e.g. pairwise/listwise)

Datasets - we have provided two datasets plus a (public) test dataset:
- `plans/test_plans.json`: Dataset with multiple plans for the same query
- `plans/train_plans1.json`: Dataset with multiple plans for the same query
- `plans/train_plans2.json`: Dataset with one plan per query

## Advanced Models (Bonus)
Explore creative feature engineering and model architectures to further enhance your learned query optimizer. Consider what additional information or model types could improve runtime prediction accuracy.

### Custom Feature Extraction for Test Data
If your model uses additional features, provide an updated `feature_extraction` script that generates these features for the test data. Only include parameters available prior to query execution.

## Evaluation Metric
Model performance is measured by the runtime of the picked plans

