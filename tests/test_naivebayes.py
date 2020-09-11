import unittest
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from naivebayes import NaiveBayesClassifier


class TestModel(unittest.TestCase):
    def __init__(self):
        self.project_model = NaiveBayesClassifier()
        self.sk_catNB = CategoricalNB()
        self.sk_contNB = GaussianNB()
