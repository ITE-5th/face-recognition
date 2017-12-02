

def inception_extractor():
    from torch import nn

    from recognition.inceptionresnetv2 import inceptionresnetv2

    path_of_pretrained_model = "../data/inceptionresnetv2.pth"
    extractor = inceptionresnetv2(path_of_pretrained_model)
    extractor = nn.Sequential(*list(extractor.children())[:-1])
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    extractor = extractor.cuda()
    return extractor


def vgg_extractor():
    import torch
    from recognition.VGG_FACE import VGG_FACE

    extractor = VGG_FACE
    extractor.load_state_dict(torch.load('../data/VGG_FACE.pth'))
    extractor = torch.nn.Sequential(*list(extractor.children())[:-5])
    for param in extractor.parameters():
        param.requires_grad = False
    extractor.eval()
    extractor = extractor.cuda()
    return extractor