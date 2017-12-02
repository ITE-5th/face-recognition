import libmr
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.datasets import load_digits
from sklearn.externals import joblib
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV


class EVM(BaseEstimator):
    def __init__(self,
                 tail: int = 100,
                 open_set_threshold: float = 0.5,
                 k: int = 5,
                 redundancy_rate: float = 0,
                 n_jobs: int = cpu_count()):
        super().__init__()
        self.tail = tail
        self.classes = []
        self.dists = []
        self.open_set_threshold = open_set_threshold
        self.k = k
        self.redundancy_rate = redundancy_rate
        self.max_class = 0
        self.n_jobs = n_jobs

    def fit(self, X, y):
        max_class = y.max()
        self.max_class = max_class
        self.classes = [0] * (max_class + 1)
        self.dists = [0] * (max_class + 1)
        for i in range(max_class + 1):
            self.classes[i] = X[y == i]
        self._infer()
        if self.redundancy_rate > 0:
            self._reduce()

    def fit_new_data(self, X, y):
        max_class = y.max()
        self.classes += [0] * (max_class - self.max_class)
        self.dists += [0] * (max_class - self.max_class)
        old_max_class = self.max_class
        self.max_class = max_class
        for i in range(old_max_class + 1, max_class + 1):
            self.classes[i] = X[y == i]
        self._infer_classes(list(range(old_max_class + 1, max_class + 1)))

    def _infer(self):
        l = list(range(len(self.classes)))
        self._infer_classes(l)

    def _infer_classes(self, indices):
        f = indices[0]
        with Pool(self.n_jobs) as p:
            self.dists[f:] = p.map(self._infer_class, indices)
            p.close()
            p.join()

    def _infer_class(self, class_index):
        in_class = self.classes[class_index]
        out_class = np.concatenate([self.classes[i] for i in range(len(self.classes)) if i != class_index])
        distances = cdist(in_class, out_class)
        distances.sort(axis=1)
        distances = 0.5 * distances[:, :self.tail]
        return np.apply_along_axis(self._fit_weibull, 1, distances)

    def _fit_weibull(self, row):
        mr = libmr.MR()
        mr.fit_low(row, self.tail)
        return mr

    def predict(self, X):
        return np.apply_along_axis(self._predict_row, 1, X)

    def _predict_row_generalized(self, row):
        max_prop, max_class = 0, -1
        for i in range(len(self.dists)):
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
        max_prop = 0
        max_class = -1
        for i in range(len(self.dists)):
            clz, dist = self.classes[i], self.dists[i]
            distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
            prop = max(props)
            if prop > max_prop:
                max_prop = prop
                max_class = i
        # -1 if from another class
        return max_class if max_prop >= self.open_set_threshold else -1

    def _reduce(self):
        with Pool(self.n_jobs) as p:
            p.map(self._reduce_class, range(len(self.classes)))
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

    def save(self, model_path: str):
        with open(model_path, "w") as _:
            joblib.dump(self, model_path)

    @staticmethod
    def load(model_path: str):
        return joblib.load(model_path)


def cat_column(df, name):
    column_types = df.loc[:, name].unique().tolist()
    df.loc[:, name] = df.loc[:, name].factorize(column_types)[0]
    return df


if __name__ == '__main__':
    # df = pd.read_csv("../data/chess.csv", delimiter=",")
    # cat_cols = ["white_king_column", "white_rook_column", "black_king_column", "result"]
    # for col in cat_cols:
    #     df = cat_column(df, col)
    # X, y = df.loc[:, df.columns != "result"], df.loc[:, "result"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # iris test

    # X, y = load_iris(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # mnist Test

    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    print("number of training samples = {}, obviously choosing a small tail will yield a very bad result".format(
        X_train.shape[0]))
    estimator = EVM(open_set_threshold=0)
    params = {"tail": [700]}
    grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score))
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    predicted = best_estimator.predict(X_test)
    accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
    print("best accuracy = {}".format(accuracy))
