import os

import cv2
import joblib
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from aligners.one_millisecond_aligner import OneMillisecondAligner
from bases.pipeline import Pipeline
from classifiers.evm import EVM
from detectors.dlib_detector import DLibDetector
from extractors.dlib_extractor import DLibExtractor
# from extractors.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager

root_path = FilePathManager.resolve("faces")
classes = os.listdir(root_path)
pipeline = Pipeline([
    DLibDetector(scale=1),
    OneMillisecondAligner(224),
    # VggExtractor()
    DLibExtractor()
])
X, y = [], []
for clz in classes:
    files = os.listdir(f"{root_path}/{clz}")
    for file in files:
        path = f"{root_path}/{clz}/{file}"
        image = cv2.imread(path)
        temp = pipeline(image)[0].reshape(-1)
        if temp.shape[0] == 0:
            print(f"wrong : {path}")
            continue
        print(f"correct : {path}")
        X.append(temp)
        y.append(clz)
X, y = np.array(X), np.array(y).reshape(-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True, random_state=42)
estimator = EVM()
params = {
    "tail": range(5, 20),
    "open_set_threshold": [0.3],
    "biased_distance": [0.5]
}
grid = GridSearchCV(estimator, params, make_scorer(accuracy_score), n_jobs=-1, cv=3)
grid.fit(X, y)
estimator = grid.best_estimator_
path = FilePathManager.resolve("trained_models")
if not os.path.exists(path):
    os.makedirs(path)
path = path + "/evm.model"
joblib.dump(estimator, path)
