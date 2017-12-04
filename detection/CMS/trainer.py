import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from CMS.dataset.face_detection_dataset import FaceDetectionDataset
from CMS.temp_cms import CMSRCNN
from CMS.transforms.mean_subtract_transform import MeanSubtract
from CMS.transforms.scale_transform import Scale
from CMS.faster_rcnn.fast_rcnn.config import cfg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


faces_image_path = "./data/WIDER_train/images/"
json_path = "./data/wider_face_split/wider_face_train_bbx_gt.json"

transformer = Compose([MeanSubtract(), Scale((224, 224))])
dataset = FaceDetectionDataset(json_file=json_path, root_dir=faces_image_path, transform=transformer)
dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.IMS_PER_BATCH, shuffle=True, num_workers=1)
net = CMSRCNN()
net = nn.DataParallel(net).cuda()
optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001)
epochs = 100
print("Begin Training")
batch_loss, total_loss = 0, 0
batches = 0

for epoch in range(epochs):
    for batch, sample in enumerate(dataloader, 0):

        inputs = sample["image"]
        bboxes = sample["bboxes"].numpy()
        info = sample["scale"]
        bboxes_count = sample["bboxes_count"].numpy()

        temp = []
        # np.empty((len(bboxes), bboxes_count, 5))

        for i in range(len(bboxes)):
            temp.append(bboxes[i, :bboxes_count[i]])

        bboxes = temp
        inputs = Variable(inputs.cuda())

        optimizer.zero_grad()
        outputs = net(inputs, info.numpy(), bboxes)
        loss = net.module.loss + net.module.rpn.loss
        loss.backward()
        optimizer.step()
        batch_loss += loss.data[0]
        batches += 1
        if batch != 0 and batch % 30 == 0:
            print("Epoch = {}, Batch = {}, Error = {}".format(epoch, batch, batch_loss))
            total_loss += batch_loss
            batch_loss = 0
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    })
    total_loss = total_loss / batches
    print("epoch finished with loss = {}".format(total_loss))
    total_loss, batches = 0, 0
