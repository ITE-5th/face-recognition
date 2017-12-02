import glob
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
from preprocessing.preprocessor import to_tensor
from recognition.face_recognition_dataset import FaceRecognitionDataset
from recognition.image_feature_extractor import ImageFeatureExtractor
from recognition.net import Net


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


faces_image_path = "../data/lfw2/"
use_evm = False
if not use_evm:
    batch_size = 1024
    dataset = FaceRecognitionDataset(faces_image_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count())
    l = len(os.listdir(faces_image_path))
    net = Net(l)
    net = nn.DataParallel(net).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01)
    epochs = 100
    batch_loss, total_loss = 0, 0
    batches = 0
    print("faces = {}".format(len(dataset.faces)))
    print("batches = {}".format(len(dataset.faces) / batch_size))
    print("Begin Training")
    for epoch in range(epochs):
        for batch, (inputs, labels) in enumerate(dataloader, 0):
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            "num_classes": l
        })
        print("epoch {} finished".format(epoch))
        total_loss, batches = 0, 0
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
