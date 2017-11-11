from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
from evm.evm import EVM

# iris test
# types = ["Kama", "Rosa", "Canadian"]
# df = pd.read_csv("data/temp.tsv")
# X, y = df.loc[:, df.columns != "Type"], df["Type"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.factorize(types)[0], y_test.factorize(types)[0]
# mnist Test
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
tails = [50, 100, 200, 500, 750, 1000]
best_error = 100
best_model = None
print("number of training samples = {}, obviously choosing a small tail will yield a very bad result".format(X_train.shape[0]))
for tail in tails:
    if tail > 2 * X_train.shape[0] / 3:
        continue
    evm = EVM(tail)
    evm.fit(X_train, y_train)
    result = evm.predict(X_test)
    err = ((result != y_test).sum() / X_test.shape[0]) * 100
    if err < best_error:
        best_model = evm
        best_error = err
    print("tail = {}, error = {}".format(tail, err))
print("best error = {}".format(best_error))
