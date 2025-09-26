from typing import List
from sklearn.linear_model import LinearRegression
import numpy as np

import pandas as pd

class RankingModelWrapper:
    def __init__(self):
        # TODO implement model initialization here

        # For demo purposes we do ranking via a cost model and use a simple linear regression model
        self.model = LinearRegression()

    def fit(self, train_sql:np.ndarray, train_features:np.ndarray, train_labels:np.ndarray):
        # TODO implement all model training code here

        assert len(train_features.shape)==2
        assert len(train_labels.shape)==1
        assert len(train_sql.shape)==1
        assert train_features.shape[0]==train_labels.shape[0]==train_sql.shape[0]

        # logscale labels
        train_labels = np.log(train_labels)
        self.model.fit(train_features, train_labels)

    def inference(self, sql:str, plan_candidates_features:List[List[float]])->int:
        # TODO implement inference code here

        # since the demo uses a cost-model approach, we predict the runtime for each plan candidate
        data = np.array(plan_candidates_features)
        predictions = self.model.predict(data)

        # return the index of the plan with the lowest predicted runtime
        return int(np.argmin(predictions))
        