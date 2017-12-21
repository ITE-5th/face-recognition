from util.file_path_manager import FilePathManager


def inception_extractor(use_cuda=True):
    from torch import nn

    from recognition.pretrained.inceptionresnetv2 import inceptionresnetv2

    path_of_pretrained_model = FilePathManager.load_path("data/inceptionresnetv2.pth")
    extractor = inceptionresnetv2(path_of_pretrained_model)
    extractor = nn.Sequential(*list(extractor.children())[:-1])
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    if use_cuda:
        extractor = extractor.cuda()
    return extractor


def vgg_extractor(use_cuda=True):
    import torch
    from recognition.pretrained.VGG_FACE import VGG_FACE
    extractor = VGG_FACE
    state = torch.load(FilePathManager.load_path('data/VGG_FACE.pth'))
    extractor.load_state_dict(state)
    extractor = torch.nn.Sequential(*list(extractor.children())[:-7])
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    if use_cuda:
        extractor = extractor.cuda()
    return extractor
