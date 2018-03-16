import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost.sklearn import XGBClassifier

from file_path_manager import FilePathManager
from recognition.dataset.image_feature_extractor import ImageFeatureExtractor
from recognition.estimator.evm import EVM

if __name__ == '__main__':
    root_path = FilePathManager.resolve("data")
    just_train = True
    features = ImageFeatureExtractor.load(root_path)
    X, y = zip(*features)
    X, y = np.array([x.float().numpy() for x in X]), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)
    type = "forest"
    estimator, params = None, {}
    if type == "evm":
        estimator = EVM()
        params = {
            "tail": range(7, 15),
            "open_set_threshold": [0.2, 0.3, 0.4, 0.5],
            "biased_distance": [0.5, 0.6, 0.7]
        }
    elif type == "forest":
        estimator = RandomForestClassifier()
        params = {
            'n_estimators': [50, 100, 200, 300],
            'max_features': ['log2', 'sqrt', 0.8],
            "max_depth": list(range(3, 6))
        }
    elif type == "xgboost":
        estimator = XGBClassifier()
        params = {
            'n_estimators': [50, 100, 200, 300],
            "max_depth": list(range(3, 6)),
            "learning_rate": [0.001, 0.01, 0.1]
        }
    grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score), n_jobs=-1)
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    predicted = best_estimator.predict(X_test)
    accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
    print("best accuracy = {}".format(accuracy))
    best_estimator.fit(X, y)
    path = FilePathManager.resolve(f"recognition/models/{type}.model")
    joblib.dump(best_estimator, path)
