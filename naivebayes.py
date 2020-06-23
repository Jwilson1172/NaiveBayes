import math
from math import exp, pi
import scipy as sp
import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    """An Implementation of the Naive Bayes classification algorithm
    """
    def __init__(self):

        return

    def mean(self, x: list) -> float:
        """Function that calculates the mean of the input array"""
        return sum(x) / float(len(x))

    def sqrt(self, x):
        "takes the square root of the input number"
        return x**(1 / 2)

    def stdev(self, a):
        """takes the standard deviation of the input array"""
        avg = self.mean(a)
        # insane one liner
        variance = sum([(x - avg)**2 for x in a]) / float(len(a) - 1)
        return self.sqrt(variance)

    def calculate_probability(self, x, mean, stdev):
        """calculates the Gaussian PDF for x"""
        exponent = exp(-((x - mean)**2 / (2 * self.stdev**2)))
        return (1 / (self.sqrt(2 * pi) * self.stdev)) * exponent

    def fit(self, X_train, y_train):
        return None

    def predict(self, X_test):
        return None

    def score(self, X_test, y_test) -> float:
        return None


if __name__ == '__main__':
    print("Naive Bayes Classifier Docs")
    model = NaiveBayesClassifier()
    data = []
    print("given the following data:", data)
    label = " "
    print("use a NaiveBayes Classification to predict label:", label)
    acc = 0.0
    print("the training accuracy is: ", acc)
    acc = 1.0
    print("the testing accuracy is: ", acc)
