import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from recognition.face_recognition_dataset import FaceRecognitionDataset
from recognition.net import Net

faces_image_path = "../data/lfw/"
dataset = FaceRecognitionDataset(faces_image_path)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)
net = Net(len(dataset.names))
net = nn.DataParallel(net).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=0.001)
epochs = 1000
running_loss = 0
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
        print("batch finished")
    print("epoch finished")
