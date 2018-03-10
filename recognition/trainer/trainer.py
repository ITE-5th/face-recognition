import os
from multiprocessing import cpu_count

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from recognition.dataset.face_recognition_dataset import FaceRecognitionDataset
from recognition.dataset.image_feature_extractor import ImageFeatureExtractor
from recognition.estimator.evm import EVM
from recognition.estimator.net import Net
from util.file_path_manager import FilePathManager


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


samples_per_class = 7


def split_data(data):
    train, val = [], []
    for i in range(0, len(data), samples_per_class):
        train.extend(data[i: i + samples_per_class - 1])
        val.append(data[i + samples_per_class - 1])
    return train, val


if __name__ == '__main__':

    root_path = FilePathManager.load_path("data")
    type = "evm"
    if type == "net":
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
        just_train = True
        features = ImageFeatureExtractor.load(root_path)
        X, y = zip(*features)
        X, y = np.array([x.float().numpy() for x in X]), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)
        if just_train:
            print("number of training samples = {}".format(X.shape[0]))
            estimator = EVM(tail=13, open_set_threshold=0.74,
                            biased_distance=0.7) if type == "evm" else RandomForestClassifier(n_estimators=100)
            # params = {"tail": range(3, 13), "open_set_threshold": [0.4, 0.5], "biased_distance": [0.5, 0.7]}
            # grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score))
            # grid.fit(X, y)
            # best_estimator = grid.best_estimator_
            estimator.fit(X, y)
            best_estimator = estimator
        else:
            estimator = EVM() if type == "evm" else RandomForestClassifier(n_estimators=100)
            params = {"tail": range(3, 10), "open_set_threshold": [0.5],
                      "biased_distance": [0.5, 0.7]} if type == "evm" \
                else \
                {
                    'rf__n_estimators': [50, 100, 200],
                    'rf__max_features': ['log2', 'sqrt', 0.8]

                }
            grid = GridSearchCV(estimator, param_grid=params, scoring=make_scorer(accuracy_score))
            grid.fit(X_train, y_train)
            best_estimator = grid.best_estimator_
            predicted = best_estimator.predict(X_test)
            accuracy = (predicted == y_test).sum() * 100 / X_test.shape[0]
            print("best accuracy = {}".format(accuracy))
        path = FilePathManager.load_path(f"models/{type}")
        if not os.path.exists(path):
            os.makedirs(path)
        path += f"/{type}.model"
        best_estimator.save(path)
