import glob

from recognition.dataset.image_feature_extractor import ImageFeatureExtractor
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor
from util.file_path_manager import FilePathManager

if __name__ == '__main__':
    faces = sorted(glob.glob(FilePathManager.load_path("data/custom_images/**/*")))
    p = AlignerPreprocessor(scale=1)
    p.preprocess_faces(faces)
    ImageFeatureExtractor.extract(FilePathManager.load_path("data"), vgg_face=True)
