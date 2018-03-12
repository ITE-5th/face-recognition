import glob

import os

from recognition.dataset.image_feature_extractor import ImageFeatureExtractor
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor
from file_path_manager import FilePathManager

if __name__ == '__main__':
    path = FilePathManager.load_path("data")
    os.system("rm -rf {}/custom_images".format(path))
    os.system("rm -rf {}/custom_features".format(path))
    faces = sorted(glob.glob(FilePathManager.load_path("data/custom_images/**/*")))
    p = AlignerPreprocessor(scale=1)
    p.preprocess_faces(faces)
    print("finish aligning")
    ImageFeatureExtractor.extract(FilePathManager.load_path("data"), vgg_face=True)
