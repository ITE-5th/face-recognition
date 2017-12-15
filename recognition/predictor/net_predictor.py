import torch

from recognition.estimator.net import Net, to_module
from recognition.predictor.predictor import Predictor


class NetPredictor(Predictor):
    def __init__(self, model_path: str, use_custom: bool = True, use_cuda: bool = True,
                 highest_number: int = 1):
        super().__init__(use_custom, use_cuda)
        self.highest_number = highest_number
        state = torch.load(model_path)
        temp = state["state_dict"]
        state_dict = to_module(temp)
        net = Net(state["num_classes"])
        if use_cuda:
            net = net.cuda()
        net.load_state_dict(state_dict)
        net.eval()
        self.net = net

    def predict_from_image(self, image):
        items = super().predict_from_image(image)
        result = []
        for (face, rect) in items:
            x = face.view(1, -1)
            y = self.net(x)
            k = self.highest_number
            vals, inds = y.data.topk(k)
            inds = inds.view(k)
            ind = inds[0]
            result.append((self.names[ind], rect))
        return result
