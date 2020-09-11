import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class NaiveBayesClassifier(BaseEstimator):
    def __init__(self, *args, **kwargs):
        super.__init__(args, kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X, use_labels=False):
        pass

    def predict_proba(self, X, use_labels=False):
        pass

    def score(self, y_true, y_pred):
        pass
