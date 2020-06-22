import math
import scipy as sp
import numpy as np
import pandas as pd


class Transformer:
    def fit(self, X_train, y_train):
        pass

    def transform(self, X_test) -> np.ndarray:
        pass


class NaiveBayesClassifier:
    def __init__(self):
        pass

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test: pd.DataFrame):
        pass

    def score(self, X_test, y_test) -> float:
        pass


if __name__ == '__main__':
    print("Naive Bayes Classifier Docs")
    model = NaiveBayesClassifier()
    transformer = Transformer()
    data = []
    print("given the following data:", data)
    label = " "
    print("use a NaiveBayes Classification to predict label:", label)
    acc = 0.0
    print("the training accuracy is: ", acc)
    acc = 1.0
    print("the testing accuracy is: ", acc)
