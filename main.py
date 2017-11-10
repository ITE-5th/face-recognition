import os
import glob
import cv2

from recognition.face_recognition_dataset import FaceRecognitionDataset

files = glob.glob("data/lfw/**/*.jpg")
for file in files:
    print(file)
    image = cv2.imread(file)
    rect = FaceRecognitionDataset.detector(image, 1)[0]