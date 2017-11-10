import cv2

from recognition.data_set import FaceRecognitionDataset

image = cv2.imread("data/lfw/Gerry_Adams/Gerry_Adams_0002.jpg")
rect = FaceRecognitionDataset.detector(image, 1)[0]