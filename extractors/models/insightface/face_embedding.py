import os
import sys

import mxnet as mx
import numpy as np
from sklearn.preprocessing import normalize

from file_path_manager import FilePathManager

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


class FaceModel:
    def __init__(self, threshold, image_size, model, det, use_gpu=False, flip=0):
        self.det = det
        self.flip = flip
        self.threshold = threshold
        self.det_minsize = 50
        self.det_threshold = [0.4, 0.6, 0.6]
        self.det_factor = 0.9
        assert len(image_size) == 2
        self.image_size = image_size
        _vec = model.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        ctx = mx.gpu() if use_gpu else mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model
        # mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
        # detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=4, accurate_landmark=True,
        #                          threshold=[0.0, 0.0, 0.2])
        # self.detector = detector

    def get_feature(self, aligned):
        embedding = None
        for flipid in [0, 1]:
            if flipid == 1:
                if self.flip == 0:
                    break
                do_flip(aligned)
            input_blob = np.expand_dims(aligned, axis=0)
            # input_blob = aligned
            data = mx.nd.array(input_blob)
            # mx.nd.
            # models = np.array(models)
            db = mx.io.DataBatch(data=list(data))
            self.model.forward(db, is_train=False)
            _embedding = self.model.get_outputs()[0].asnumpy()
            if embedding is None:
                embedding = _embedding
            else:
                embedding += _embedding
        embedding = normalize(embedding)
        return embedding


if __name__ == '__main__':
    FaceModel(threshold=1.24, det=2, image_size=[112, 112],
              model=FilePathManager.resolve(
                  "data/model-r50-am-lfw/model,0"))
