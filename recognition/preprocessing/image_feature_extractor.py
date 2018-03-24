import glob
import os

import cv2
import numpy as np
import torch
from dlt.util import cv2torch
from torch.autograd import Variable

from file_path_manager import FilePathManager
from recognition.extractor.extractors import vgg_extractor
from recognition.preprocessing.aligner_preprocessor import AlignerPreprocessor


class ImageFeatureExtractor:
    aligner = AlignerPreprocessor()
    extractor = vgg_extractor()

    @staticmethod
    def extract_from_images(images):
        result = []
        for image in images:
            aligned = ImageFeatureExtractor.aligner.preprocess_face_from_image(image)
            aligned = cv2torch(aligned).float().unsqueeze(0).cuda()
            aligned = ImageFeatureExtractor.extractor(Variable(aligned))
            aligned = aligned.view(-1).cpu().data.numpy()
            result.append(aligned)
        return np.array(result)

    @staticmethod
    def extract_from_dir(root_dir: str):
        extractor = vgg_extractor()
        names = sorted(os.listdir(root_dir + "/custom_images2"))
        if not os.path.exists(root_dir + "/custom_features"):
            os.makedirs(root_dir + "/custom_features")
        for i in range(len(names)):
            name = names[i]
            path = root_dir + "/custom_images2/" + name
            if not os.path.exists(root_dir + "/custom_features/" + name):
                os.makedirs(root_dir + "/custom_features/" + name)
            faces = os.listdir(path)
            for face in faces:
                p = path + "/" + face
                image = cv2.imread(p)
                image = cv2.resize(image, (224, 224))
                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 1, 2)
                image = torch.from_numpy(image.astype(np.float)).float().unsqueeze(0).cuda()
                image = extractor(Variable(image))
                image = image.view(-1).cpu()
                res = (image.data, name)
                temp = root_dir + "/custom_features/" + name + "/" + face[
                                                                       :face.rfind(
                                                                           ".")] + ".features"
                torch.save(res, temp)

    @staticmethod
    def load(root_dir: str, lfw=False):
        temp = sorted(glob.glob(root_dir + "/{}_features/**/*.features".format("lfw" if lfw else "custom")))
        return [torch.load(face) for face in temp]


if __name__ == '__main__':
    ImageFeatureExtractor.extract_from_dir(FilePathManager.resolve("data"))
