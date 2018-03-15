import glob

import os

from recognition.dataset.image_feature_extractor import ImageFeatureExtractor
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor
from file_path_manager import FilePathManager

if __name__ == '__main__':
    path = FilePathManager.resolve("data")
    os.system("rm -rf {}/custom_images2".format(path))
    os.system("rm -rf {}/custom_features".format(path))
    faces = sorted(glob.glob(FilePathManager.resolve("data/custom_images/**/*")))
    p = AlignerPreprocessor(scale=1)
    p.preprocess_faces(faces)
    print("finish aligning")
    ImageFeatureExtractor.extract(FilePathManager.resolve("data"), vgg_face=True)
