import glob
import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable

from util.file_path_manager import FilePathManager
from recognition.pretrained.extractors import inception_extractor, vgg_extractor


class ImageFeatureExtractor:
    @staticmethod
    def extract(root_dir: str, vgg_face=False, lfw=False):
        extractor = vgg_extractor() if vgg_face else inception_extractor()
        names = sorted(os.listdir(root_dir + ("/lfw2" if lfw else "/custom_images2")))
        if not os.path.exists(root_dir + ("/custom_features" if not lfw else "/lfw_features")):
            os.makedirs(root_dir + ("/custom_features" if not lfw else "/lfw_features"))
        for i in range(len(names)):
            name = names[i]
            path = root_dir + ("/custom_images2/" if not lfw else "/lfw2/") + name
            if not os.path.exists(root_dir + ("/custom_features/" if not lfw else "/lfw_features/") + name):
                os.makedirs(root_dir + ("/custom_features/" if not lfw else "/lfw_features/") + name)
            faces = os.listdir(path)
            for face in faces:
                p = path + "/" + face
                image = cv2.imread(p)
                if vgg_face:
                    image = cv2.resize(image, (224, 224))
                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 1, 2)
                image = torch.from_numpy(image.astype(np.float)).float().unsqueeze(0).cuda()
                image = extractor(Variable(image))
                image = image.view(-1).cpu()
                res = (image.data, i)
                temp = root_dir + ("/custom_features/" if not lfw else "/lfw_features/") + name + "/" + face[:face.rfind(".")] + ".features"
                print(temp)
                torch.save(res, temp)

    @staticmethod
    def load(root_dir: str, lfw=False):
        temp = sorted(glob.glob(root_dir + "/{}_features/**/*.features".format("lfw" if lfw else "custom")))
        return [torch.load(face) for face in temp]


if __name__ == '__main__':
    ImageFeatureExtractor.extract(FilePathManager.load_path("data"), vgg_face=True)
