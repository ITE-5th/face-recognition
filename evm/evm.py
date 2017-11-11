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

    def predict(self, X):
        return np.apply_along_axis(self._predict_row, 1, X)

    def _predict_row_generalized(self, row):
        result = []
        for i in range(len(self.dists)):
            clz, dist = self.classes[i], self.dists[i]
            res = 0
            for j in range(dist.shape[0]):
                distance = np.linalg.norm(clz[j] - row)
                res += dist[j].w_score(distance)
            result.append(res / dist.shape[0])
        m = max(result)
        # -1 if from another class
        return result.index(m) if m >= self.threshold else -1

    def _predict_row(self, row):
        max_prop = 0
        max_class = -1
        for i in range(len(self.dists)):
            clz, dist = self.classes[i], self.dists[i]
            for j in range(dist.shape[0]):
                distance = np.linalg.norm(clz[j] - row)
                prop = dist[j].w_score(distance)
                if prop > max_prop:
                    max_prop = prop
                    max_class = i
        return max_class if max_prop >= self.threshold else -1

    def _infer(self):
        for i in range(len(self.classes)):
            self._infer_class(i)

    def _infer_class(self, class_index):
        in_class = self.classes[class_index]
        out_class = []
        for i in range(len(self.classes)):
            if i != class_index:
                for row in self.classes[i]:
                    out_class.append(row)
        out_class = np.array(out_class)
        distances = cdist(in_class, out_class)
        distances.sort(axis=1)
        distances = 0.5 * distances[:, :self.tail_size]
        self.dists[class_index] = np.apply_along_axis(self._fit_weibull, 1, distances)

    def _fit_weibull(self, row):
        mr = libmr.MR()
        mr.fit_low(row, self.tail_size)
        return mr
