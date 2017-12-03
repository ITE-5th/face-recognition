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


path_to_pretrained_model = "data/shape_predictor_68_face_landmarks.dat"
path_to_cnn_model = "data/mmod_human_face_detector.dat"
# detector = dlib.get_frontal_face_detector()
detector = dlib.cnn_face_detection_model_v1(path_to_cnn_model)
predictor = dlib.shape_predictor(path_to_pretrained_model)
aligner = openface.AlignDlib(path_to_pretrained_model)

wrong_faces = []


def process_face(face, lfw=False):
    try:
        image = cv2.imread(face)
        rect = detector(image, 1)[0].rect
        aligned = aligner.align(299, image, rect,
                                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        temp = face[:(20 if not lfw else 10)] + "2" + face[(20 if not lfw else 10):]
        dirname = temp[:temp.rfind("/")]
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(temp, aligned)
        print("correct : {}".format(face))

    except:
        print("wrong : {}".format(face))
        wrong_faces.append(face)


faces = glob.glob("./data/custom_images/**/*")
with Pool(cpu_count()) as p:
    p.map(process_face, faces)
    p.close()
    p.join()
with open("temp.txt", "w") as f:
    f.write("\n".join(wrong_faces))
