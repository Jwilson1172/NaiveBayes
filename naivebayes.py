import math
from math import exp, pi, sqrt
import scipy as sp
import numpy as np
import pandas as pd
from csv import reader
import random


class FileHandle():
    def __init__(self):
        self.dataset = None
        self.__help__ = []
        return

    def load_csv(self, filename: str) -> list:
        """A method to load a csv into a list of list's
        """
        # init empty list to store data from the file
        self.dataset = list()
        try:
            with open(filename, 'r') as file:
                csv_reader = reader(file)
                # iterate the lines in the csv and
                # append that data to the dataset
                for row in csv_reader:
                    if not row:
                        continue
                    self.dataset.append(row)
        # error catching to handle the file not being there.
        except FileExistsError as e:
            # script friendly error message
            FileExistsError(e,
                            "\n",
                            u"That File Doesn't Exist:\n",
                            filename,
                            "\n",
                            sep='')
        return self.dataset


"""
    def str_column_to_float(self, dataset: list, column):
        \"\"\"A method that takes a column and convert's it into float type
        \"\"\"
        for row in self.dataset:
            row[column] = float(row[column].strip())

    def str_column_to_int(self, dataset, column):
        \"\"\"A method that takes the column values and convert's them to int type
        \"\"\"
        class_values = [row[column] for row in self.dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup
"""


class NaiveBayesClassifier:
    """An Implementation of the Naive Bayes classification algorithm
    """
    def __init__(self):
        self.dataset = None
        self.acc = 0
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
        """calculate the gaussian probability distribution

        Args:
            x (list): dataset
            mean (float): mean of the dataset
            stdev (float): standard deviation of the dataset

        Returns:

        """
        exponent = exp(-((x - mean)**2 / (2 * stdev**2)))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent

    def score(self, actual, predicted):
        """A method that takes the predictions from the model and the labels
        from the dataset then compares them and calculates the error rate in
        the prediction
        """
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        self.acc = correct / float(len(actual)) * 100.0
        return self.acc

        #
    def summarize_dataset(self, dataset: list) -> list:
        """Calculate the mean, stdev and count for each column in a dataset
            Note that is is not the fitted values of the dataset these are for
            just getting more information about the columns.
        Args:
            dataset (list): list of data that is the dataset

        Returns:
            list: the summaries for the dataset
        """
        summaries = [(self.mean(column), self.stdev(column), len(column))
                     for column in zip(*dataset)]
        del (summaries[-1])
        return summaries

    def summarize_by_class(self, dataset: list) -> list:
        """Split dataset by class then calculate statistics for each row

        Args:
            dataset (list): dataset

        Returns:
            summaries (list): processed dataset similar to the encoded weights
            of a NN .fit() method, or the dtm that is passed to a KNN. these
            values are what are the results of "fitting" the model.
        """
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    def calculate_class_probabilities(self, summaries: list,
                                      row: list) -> dict:
        """Calculate the probabilities of predicting each class for a given row

        Args:
            summaries (list): the fitted values in the model
            row (list): a row that the class probability is calculated for

        Returns: probabilities
            dict: the probabilities of the classes occurring in the given row.
        """
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = summaries[class_value][0][2] / float(
                total_rows)
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(
                    row[i], mean, stdev)
        return probabilities

    def predict(self, summaries: list, row: list):
        """Predict the class for a given row

        Args:
            summaries (list): the fitted values that are returned by the model
            row (list): the row that is going to be used for the prediction

        Returns:
            best_label (undefined type): predicted class label of the input
            row and the fitted values.
        """
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def separate_by_class(self, dataset: list) -> dict:
        """Split the dataset by class values, returns a dictionary

        """
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    def fit_predict(self, train, test):
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return (predictions)


if __name__ == '__main__':

    class_choices = [0, 1]
    size = 10000
    dataset = []
    for i in range(size):
        col1 = random.gauss(3.0, 1.0)
        col2 = random.gauss(3.0, 1.0)
        label = random.choice(class_choices)
        dataset.append([col1, col2, label])

    # some QA on the dataset
    class_weights = pd.DataFrame(dataset)[2].value_counts(normalize=True)
    assert class_weights.values[0] <= 0.55
    assert class_weights.values[1] <= 0.55

    print("Generated a dataset with a gaussian\nclass weight distribution:",
          class_weights)
    nb = NaiveBayesClassifier()
    summaries = nb.summarize_by_class(dataset)

    probabilities = nb.calculate_class_probabilities(summaries, dataset[0])
    print(probabilities)
