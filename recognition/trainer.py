import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from recognition.face_recognition_dataset import FaceRecognitionDataset
from recognition.net import Net


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


faces_image_path = "../data/lfw2/"
dataset = FaceRecognitionDataset(faces_image_path)
# dataset = FaceRecognitionDataset()
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=8)
l = len(dataset.names)
net = Net(l)
net = nn.DataParallel(net).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.001)
epochs = 100
print("Begin Training")
batch_loss, total_loss = 0, 0
batches = 0
for epoch in range(epochs):
    for batch, (inputs, labels) in enumerate(dataloader, 0):
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
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
        'optimizer': optimizer.state_dict(),
        "num_classes": l
    })
    total_loss = total_loss / batches
    print("epoch finished with loss = {}".format(total_loss))
    total_loss, batches = 0, 0
