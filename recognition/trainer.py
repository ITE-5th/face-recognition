import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from evm.evm import EVM
from recognition.face_recognition_dataset import FaceRecognitionDataset
from recognition.image_feature_extractor import ImageFeatureExtractor
from recognition.net import Net


def save_checkpoint(state, epoch):
    torch.save(state, "../models/checkpoint-{}.pth.tar".format(epoch))


root_path = "../data"
use_evm = False
if not use_evm:
    batch_size = 256
    dataset = FaceRecognitionDataset(root_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    l = len(os.listdir(root_path + "/lfw2"))
    net = Net(l, vgg_face=True)
    net = nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01)
    epochs = 1000
    batch_loss, total_loss = 0, 0
    batches = len(dataset.faces) / batch_size
    print("faces = {}".format(len(dataset.faces)))
    print("batches = {}".format(batches))
    print("Begin Training")
    for epoch in range(epochs):
        epoch_loss, epoch_correct = 0, 0
        for batch, (inputs, labels) in enumerate(dataloader, 0):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            _, first = outputs.data.max(1)
            second = labels.data
            correct = torch.eq(first, second).sum()
            epoch_correct += correct
            epoch_loss += loss.data[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "num_classes": l
        }, epoch + 1)
        print('Epoch {} done, average loss: {}, average accuracy: {}%'.format(
            epoch + 1, epoch_loss / batches, epoch_correct * 100 / (batches * batch_size)))
else:
    # TODO: review :)
    features = ImageFeatureExtractor.load("./data")
    X, y = zip(*features)
    X, y = np.array([x.numpy() for x in X]), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print("number of training samples = {}, obviously choosing a small tail will yield a very bad result".format(
        X_train.shape[0]))
    estimator = EVM(open_set_threshold=0)
    params = {"tail": [1000, 4000, 8000]}
    grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score))
    grid.fit(X_train, y_train)
    best_estimator = grid.best_estimator_
    predicted = best_estimator.predict(X_test)
    accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
    print("best accuracy = {}".format(accuracy))
