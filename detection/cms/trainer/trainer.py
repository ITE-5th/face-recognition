from multiprocessing import cpu_count

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from detection.cms.dataset.face_detection_dataset import FaceDetectionDataset
from detection.cms.estimator.faster_rcnn import network
from detection.cms.estimator.faster_rcnn.fast_rcnn.config import cfg
from detection.cms.estimator.temp_cms import CMSRCNN
from detection.cms.transforms.mean_subtract_transform import MeanSubtract
from detection.cms.transforms.scale_transform import Scale
from file_path_manager import FilePathManager


def save_checkpoint(state, epoch):
    torch.save(state, FilePathManager.resolve("detection/cms/models/net/checkpoint-{}.pth.tar".format(epoch)))


def val_accuracy(dataloader, net):
    epoch_loss, epoch_correct = 0, 0
    for batch, sample in enumerate(dataloader, 0):
        inputs = sample["image"]
        info = sample["scale"]

        gt_bboxes = FaceDetectionDataset.remove_padded_bboxes(sample)

        inputs = Variable(inputs.cuda())

        outputs = net(inputs)
        loss = net.module.loss + net.module.rpn.loss

        # TODO: following statements are incorrect
        # bbox + output must overlap by 50% or more
        _, first = outputs.data.max(1)
        # second = labels.data
        # correct = torch.eq(first, second).sum()
        epoch_correct += 0
        epoch_loss += loss.data[0]
    return epoch_correct, epoch_loss


def split_data(data):
    train, val = [], []
    for i in range(0, len(data), 5):
        train.extend(data[i: i + 4])
        val.append(data[i + 4])
    return train, val


if __name__ == '__main__':

    batch_size = cfg.TRAIN.IMS_PER_BATCH

    faces_image_path = "../data/WIDER_train"
    json_path = "../data/wider_face_split/wider_face_train_bbx_gt.json"

    transformer = Compose([MeanSubtract(), Scale((224, 224))])

    train_dataset = FaceDetectionDataset(json_file=json_path, root_dir=faces_image_path, transform=transformer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())

    net = CMSRCNN()
    # net.load_pretrained()
    net = nn.DataParallel(net).cuda()

    optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01)

    num_classes = 2

    epochs = 2000
    batch_loss, total_loss = 0, 0
    batches = len(train_dataset.names) / batch_size

    print("images = {}".format(len(train_dataset.names)))
    print("batches = {}".format(batches))
    print("Begin Training")
    for epoch in range(epochs):

        epoch_train_loss, epoch_train_correct, epoch_val_loss, epoch_val_correct = 0, 0, 0, 0

        for batch, sample in enumerate(train_dataloader, 0):
            inputs = sample["image"]
            info = sample["scale"]

            gt_bboxes = FaceDetectionDataset.remove_padded_bboxes(sample)
            inputs = Variable(inputs.cuda())

            net(inputs, info.numpy(), gt_bboxes)
            optimizer.zero_grad()
            loss = net.module.total_loss
            loss.backward()
            network.clip_gradient(net, 10.)
            optimizer.step()

            # TODO: following statements are incorrect
            # bbox + output must overlap by 50% or more
            # correct = torch.eq(first, second).sum()
            epoch_train_correct += 0
            epoch_train_loss += loss.data[0]

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "num_classes": num_classes
        }, epoch + 1)
        print("epoch finished")
        # print(
        #     'Epoch {} done, average train loss: {}, average train accuracy: {}%, average val loss: {}, average val '
        #     'accuracy: {}%'.format(
        #         epoch + 1,
        #         epoch_train_loss / batches,
        #         epoch_train_correct * 100 / (batches * batch_size),
        #         epoch_val_loss / batches,
        #         epoch_val_correct * 100 / (batches * batch_size)
        #     )
        # )
