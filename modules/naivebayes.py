from sklearn.base import BaseEstimator, TransformerMixin
from pandas import DataFrame, Series


class BayesTherory:
    def __init__(self):
        pass

    def prob_label_occuring(self, y, cond_y):
        # copy input to avoid changing original
        y_cp = y[:]
        # start count
        i = 0
        # save len of original y
        bottom = len(y_cp)
        # check if last element is None (breakpoint case)
        if y_cp[-1] is not None:
            y_cp.append(None)

        # count target observations
        while True:
            if y_cp[0] == cond_y:
                i += 1
            # break on None item
            elif y_cp[0] is None:
                break
            y_cp.pop(0)
        # return ratio of labels to len
        return i / bottom

    def prob_feature_given_label(self, X: DataFrame, y: Series,
                                 cond_y) -> dict:
        # function that takes an input array of features and returns the
        # marginal probability of a feature effecting the class label
        ret = {}
        y = y.copy().to_list()
        for feature in X.columns:
            d = self.__conditional_counter(X[feature].to_list(), y, cond_y)
            d = self.__calc_f_proba(d, y, cond_y)
            ret[feature] = d
        return ret

    def __conditional_counter(self, x: list, y: list, cond_y) -> dict:
        # function that counts unique values in X that meat cond_y value in y
        # for instance:
        # given
        # x,y = ([1,0,1,0,1],['y','n','y','n','y'])
        # and
        # cond_y = 'y'
        # this function will look through x where y == 'y' and count
        # the unique x values then return a dictionary with those keys and
        # and the number of observations
        # __conditional_counter(x,y,cond_y)
        # output -> {1: 3}

        d = {}
        for k, v in zip(x, y):
            if v == cond_y:
                if k in d:
                    d[k] += 1
                else:
                    d[k] = 1
        return d

    def __calc_f_proba(self, d: dict, y: list, cond_y) -> dict:
        # return a dictionary of each feature value as keys and the
        # probability of that feature being in the target class
        n = y.count(cond_y)
        return {k: v / n for k, v in zip(d.keys(), d.values())}

    def calc_post(priors, marginal_prob, label_prob):
        pass

class NaiveBayesClassifier(BaseEstimator):
    def __init__(self, *args, **kwargs):
        super.__init__(args, kwargs)

    def fit(self, X, y):
        pass

    def predict(self, X, use_labels=False):
        pass

    def predict_proba(self, X, use_labels=False):
        pass
