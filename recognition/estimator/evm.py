import libmr
from multiprocessing import Pool, cpu_count

import numpy as np
# from numba import float_, int_, bool_
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


# spec = [
#     ('tail', int_),
#     ('open_set_threshold', float_),
#     ('biased_distance', float_),
#     ('k', int_),
#     ("redundancy_rate", float_),
#     ("use_multithreading", bool_),
#     ("n_jobs", int_),
#     ("classes", dict),
#     ("dists", dict)
# ]


# @jitclass(spec)
class EVM(BaseEstimator):

    def __init__(self,
                 tail: int = 10,
                 open_set_threshold: float = 0.5,
                 biased_distance: float = 0.5,
                 k: int = 5,
                 redundancy_rate: float = 0):
        self.tail = tail
        self.biased_distance = biased_distance
        self.classes = {}
        self.dists = {}
        self.open_set_threshold = open_set_threshold
        self.k = k
        self.redundancy_rate = redundancy_rate

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes = dict()
        self.dists = dict()
        for clz in classes:
            self.classes[clz] = X[y == clz]
        self._infer()
        if self.redundancy_rate > 0:
            self._reduce()

    def fit_new_data(self, X, y):
        values = np.unique(y)
        classes = {}
        for val in values:
            classes[val] = X[y == val]
            if val in self.classes.keys():
                pass

    def _infer(self):
        self._infer_classes(list(self.classes.keys()))

    def _infer_classes(self, indices):
        temp = [self._infer_class(i) for i in indices]
        for i in range(len(temp)):
            self.dists[indices[i]] = temp[i]

    def _infer_class(self, class_index):
        in_class = self.classes[class_index]
        out_class = np.concatenate([self.classes[i] for i in self.classes.keys() if i != class_index])
        distances = cdist(in_class, out_class)
        distances.sort(axis=1)
        distances = self.biased_distance * distances[:, :self.tail]
        return np.apply_along_axis(self._fit_weibull, 1, distances)

    def _fit_weibull(self, row):
        mr = libmr.MR()
        mr.fit_low(row, self.tail)
        return mr

    def predict(self, X):
        return np.apply_along_axis(self._predict_row, 1, X)

    def _predict_row_generalized(self, row):
        max_prop, max_class = -1, -1
        for i in self.classes.keys():
            clz, dist = self.classes[i], self.dists[i]
            distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            temp = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
            props = sorted(temp, reverse=True)[:self.k]
            prop = sum(props) / self.k
            if prop > max_prop:
                max_prop = prop
                max_class = i
        # -1 if from another class
        return max_class if max_prop >= self.open_set_threshold else -1

    def _predict_row(self, row):
        max_prop, max_class = -1, -1
        for i in self.classes.keys():
            clz, dist = self.classes[i], self.dists[i]
            distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
            prop = max(props)
            if prop > max_prop:
                max_prop = prop
                max_class = i
        # -1 if from another class
        return max_class if max_prop >= self.open_set_threshold else -1

    def _predict_class_generalized(self, item):
        row, class_index = item
        clz, dist = self.classes[class_index], self.dists[class_index]
        distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
        temp = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
        props = sorted(temp, reverse=True)[:self.k]
        prop = sum(props) / self.k
        return prop, class_index

    def predict_with_prop(self, X):
        return np.apply_along_axis(self._predict_with_prob, 1, X)

    def _predict_with_prob(self, row):
        max_prop, max_class = -1, -1
        for i in self.classes.keys():
            clz, dist = self.classes[i], self.dists[i]
            distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
            prop = max(props)
            if prop > max_prop:
                max_prop = prop
                max_class = i
        # -1 if from another class
        return (max_class, max_prop) if max_prop >= self.open_set_threshold else (-1, 1 - max_prop)

    def _predict_class(self, item):
        row, class_index = item
        clz, dist = self.classes[class_index], self.dists[class_index]
        distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
        props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
        prop = max(props)
        return prop, class_index

    def _reduce(self):
        with Pool(cpu_count()) as p:
            p.map(self._reduce_class, self.classes.keys())
            p.close()
            p.join()

    def _reduce_class(self, class_index):
        clz, dist = self.classes[class_index], self.dists[class_index]
        distances = cdist(clz, clz)
        l = clz.shape[0]
        s = [0] * l
        for i in range(l):
            s[i] = []
            for j in range(l):
                if dist[i].w_score(distances[i, j]) >= self.redundancy_rate:
                    s[i].append(j)
        u = set(range(l))
        c = set()
        indices = []
        s = [set(l) for l in s]
        len_s = [len(j) for j in s]
        while u != c:
            temp = max(len_s)
            ind = len_s.index(temp)
            c |= s[ind]
            indices.append(ind)
            s.pop(ind)
            len_s.pop(ind)
        self.classes[class_index], self.dists[class_index] = self.classes[class_index][indices], \
                                                             self.dists[class_index][indices]


if __name__ == '__main__':
    X, y = load_digits(return_X_y=True)
    # X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print("number of training samples = {}, obviously choosing a small tail will yield a very bad result".format(
        X_train.shape[0]))
    estimator = EVM(tail=8, open_set_threshold=0)
    params = {"tail": range(500, 600, 10)}
    grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score), n_jobs=-1)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    predicted = best_estimator.predict(X_test)
    accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
    print("best accuracy = {}".format(accuracy))
