import cv2
import os

from aligners.one_millisecond_aligner import OneMillisecondAligner
from bases.pipeline import Pipeline
from detectors.dlib_detector import DLibDetector
from extractors.vgg_extractor import VggExtractor
from file_path_manager import FilePathManager

#
# class ClosedPredictor:
#     pipeline = Pipeline([
#         DLibDetector(scale=1),
#         OneMillisecondAligner(224),
#         VggExtractor()
#     ])
#
#     def __init__(self, model_path):
#         self.model_path = model_path
#         self.reload()
