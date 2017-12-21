import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from util.file_path_manager import FilePathManager
from recognition.dataset.face_recognition_dataset import FaceRecognitionDataset
from recognition.dataset.image_feature_extractor import ImageFeatureExtractor
from recognition.estimator.evm import EVM
from recognition.estimator.net import Net


def save_checkpoint(state, epoch):
    torch.save(state, FilePathManager.load_path("models/net/checkpoint-{}.pth.tar".format(epoch)))


def val_accuracy(dataloader, net):
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
    return epoch_correct, epoch_loss


def split_data(data):
    train, val = [], []
    for i in range(0, len(data), 5):
        train.extend(data[i: i + 4])
        val.append(data[i + 4])
    return train, val


if __name__ == '__main__':

    root_path = FilePathManager.load_path("data")
    use_evm = True
    if not use_evm:
        batch_size = 512
        faces = ImageFeatureExtractor.load(root_path)
        train, val = split_data(faces)
        train_dataset = FaceRecognitionDataset(train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count() // 2)
        val_dataset = FaceRecognitionDataset(val)
        val_dataloader = DataLoader(val_dataset, batch_size=4 * batch_size, shuffle=True, num_workers=cpu_count() // 2)
        l = len(os.listdir(root_path + "/lfw2"))
        net = Net(l, vgg_face=True)
        net = nn.DataParallel(net).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=0.01)
        epochs = 2000
        batch_loss, total_loss = 0, 0
        batches = len(train_dataset.faces) / batch_size
        print("faces = {}".format(len(train_dataset.faces)))
        print("batches = {}".format(batches))
        print("Begin Training")
        for epoch in range(epochs):
            epoch_train_loss, epoch_train_correct, epoch_val_loss, epoch_val_correct = 0, 0, 0, 0
            for batch, (inputs, labels) in enumerate(train_dataloader, 0):
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                epoch_val_correct, epoch_val_loss = val_accuracy(val_dataloader, net)
                _, first = outputs.data.max(1)
                second = labels.data
                correct = torch.eq(first, second).sum()
                epoch_train_correct += correct
                epoch_train_loss += loss.data[0]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "num_classes": l
            }, epoch + 1)
            print(
                'Epoch {} done, average train loss: {}, average train accuracy: {}%, average val loss: {}, average val accuracy: {}%'.format
                    (
                    epoch + 1,
                    epoch_train_loss / batches,
                    epoch_train_correct * 100 / (batches * batch_size),
                    epoch_val_loss / batches,
                    epoch_val_correct * 100 / (batches * batch_size)
                )
            )

    else:
        features = ImageFeatureExtractor.load(root_path)
        X, y = zip(*features)
        X, y = np.array([x.float().numpy() for x in X]), np.array(y)
        temp = split_data(X)
        X_train, X_test = np.array(temp[0]), np.array(temp[1])
        temp = split_data(y)
        y_train, y_test = np.array(temp[0]), np.array(temp[1])
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)
        print("number of training samples = {}, obviously choosing a small tail will yield a very bad result".format(
            X_train.shape[0]))
        # estimator = SVC(kernel="rbf")
        # params = {}
        estimator = EVM(tail=8, open_set_threshold=0.3, biased_distance=0.7)
        params = {"tail": range(5, 10), "open_set_threshold": [0.2, 0.4, 0.6], "biased_distance": [0.5, 0.7]}
        grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score))
        grid.fit(X_train, y_train)
        best_estimator = grid.best_estimator_
        path = FilePathManager.load_path("models/evm")
        if not os.path.exists(path):
            os.makedirs(path)
        path += "/evm.model"
        best_estimator.save(path)
        predicted = best_estimator.predict(X_test)
        # supported_classes = list(best_estimator.classes.keys())
        # y_test = np.array([y if y in supported_classes else -1 for y in y_test])
        accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
        print("best accuracy = {}".format(accuracy))
