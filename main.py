from multiprocessing import Pool, cpu_count

import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import pandas as pd
from evm.evm import EVM
from skimage import io
import dlib
import openface
import glob
import os

# iris test

# types = ["Kama", "Rosa", "Canadian"]
# df = pd.read_csv("data/temp.tsv")
# X, y = df.loc[:, df.columns != "Type"], df["Type"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# X_train, X_test, y_train, y_test = X_train.values, X_test.values, y_train.factorize(types)[0], y_test.factorize(types)[0]

# mnist Test

# X, y = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# tails = range(100, 1500, 100)
# best_accuracy = 0
# best_model = None
# print("number of training samples = {}, obviously choosing a small tail will yield a very bad result".format(X_train.shape[0]))
# for tail in tails:
#     if tail > 9 * X_train.shape[0] / 10:
#         break
#     evm = EVM(tail)
#     evm.fit(X_train, y_train)
#     result = evm.predict(X_test)
#     err = ((result != y_test).sum() / X_test.shape[0]) * 100
#     acc = 100 - err
#     if acc > best_accuracy:
#         best_model = evm
#         best_accuracy = acc
#     print("tail = {}, accuracy = {}%".format(tail, acc))
# print("best accuracy = {}%".format(best_accuracy))

path_to_pretrained_model = "data/shape_predictor_68_face_landmarks.dat"
path_to_cnn_model = "data/mmod_human_face_detector.dat"
# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1(path_to_cnn_model)
predictor = dlib.shape_predictor(path_to_pretrained_model)
aligner = openface.AlignDlib(path_to_pretrained_model)


def process_face(face):
    image = cv2.imread(face)
    rect = detector(image, 0)[0].rect
    aligned = aligner.align(299, image, rect,
                            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    temp = face[:10] + "2" + face[10:]
    dirname = temp[:temp.rfind("/")]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    cv2.imwrite(temp, aligned)
    print(face)


faces = glob.glob("./data/lfw/**/*.jpg")
with Pool(cpu_count()) as p:
    p.map(process_face, faces)
