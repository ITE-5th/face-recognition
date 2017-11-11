import libmr

import numpy as np
from scipy.spatial.distance import cdist


class EVM:
    def __init__(self, tail_size: int = 100, threshold=0.75):
        self.tail_size = tail_size
        self.X = None
        self.classes = []
        self.dists = []
        self.threshold = threshold

    def fit(self, X, y):
        self.X = X
        max_class = y.max()
        self.classes = [0] * (max_class + 1)
        self.dists = [0] * (max_class + 1)
        for i in range(max_class + 1):
            self.classes[i] = X[y == i]
        self._infer()

    def _infer(self):
        for i in range(len(self.classes)):
            self._infer_class(i)

    def _infer_class(self, class_index):
        in_class = self.classes[class_index]
        out_class = np.concatenate([self.classes[i] for i in range(len(self.classes)) if i != class_index])
        distances = cdist(in_class, out_class)
        distances.sort(axis=1)
        distances = 0.5 * distances[:, :self.tail_size]
        self.dists[class_index] = np.apply_along_axis(self._fit_weibull, 1, distances)

    def _fit_weibull(self, row):
        mr = libmr.MR()
        mr.fit_low(row, self.tail_size)
        return mr

    def predict(self, X):
        return np.apply_along_axis(self._predict_row, 1, X)

    def _predict_row_generalized(self, row):
        result = []
        for i in range(len(self.dists)):
            clz, dist = self.classes[i], self.dists[i]
            distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            result.append(sum(dist[j].w_score(distances[j]) for j in range(dist.shape[0])) / dist.shape[0])
        m = max(result)
        # -1 if from another class
        return result.index(m) if m >= self.threshold else -1

    def _predict_row(self, row):
        max_prop = 0
        max_class = -1
        # row = row.reshape(1, -1)
        for i in range(len(self.dists)):
            clz, dist = self.classes[i], self.dists[i]
            distances = np.linalg.norm(clz - row, axis=1, keepdims=True)
            props = [dist[j].w_score(distances[j]) for j in range(dist.shape[0])]
            prop = max(props)
            if prop > max_prop:
                max_prop = prop
                max_class = i
        # -1 if from another class
        return max_class if max_prop >= self.threshold else -1