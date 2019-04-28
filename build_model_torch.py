import os
import sys
import time
import math
import json
import datetime

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import assets
import settings
from preprocessing import import_image
from mailer import send_email

"""
@param  filename    filename consisting of images to be trained on
@return data        a dict of training and testing data, where the
                    training data is split into an array of batches
"""
def dataloader(filename, batch_size=16):
    with open(os.path.join(assets.TRAINING_PATH, filename), 'r') as f:
        lines = [line.strip('\n') for line in f]
    data = {
        "train": [],
        "test": []
    }
    x, y = [], []
    print("Preprocessing training data...")
    for file in lines:
        img, mask = import_image(file)
        if mask is not None:
            x.append(np.moveaxis(img, -1, 0))
            y.append(mask)
    x_train = np.array(x[:int(0.75*len(lines))])
    y_train = np.array(y[:int(0.75*len(lines))])
    x_test = np.array(x[int(0.75*len(lines)):])
    y_test = np.array(y[int(0.75*len(lines)):])
    num_batches = len(x_train) // batch_size
    x_train = torch.tensor(x_train).type(torch.FloatTensor)
    y_train = torch.tensor(y_train).type(torch.LongTensor)
    print("(Torch) x_train shape:", list(x_train.size()))
    print("(Torch) y_train shape:", list(y_train.size()))
    print("\n")
    processed = 0
    for batch_id in range(num_batches):
        print("Building batch %d / %d" % (batch_id, num_batches))
        num_points = batch_size
        data["train"].append((
            x_train[processed:processed+num_points],
            y_train[processed:processed+num_points]))
        processed += num_points
    print("Building test data...")
    for img, mask in zip(x_test, y_test):
        data["test"].append((
            torch.tensor(img).type(torch.FloatTensor),
            torch.tensor(mask).type(torch.LongTensor)))
    print("Done loading data...")
    return data

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            #384x256x3 ==> 384x256x128
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #384x256x128 ==> 192x128x128
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            #192x128x128 ==> 192x128x64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #192x128x64 ==> 96x64x64
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            #96x64x64 ==> 96x64x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #96x64x32 ==> 48x32x32
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            #48x32x32 ==> 48x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            #96x64x64 ==> 96x64x218
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, len(settings.LABELS), kernel_size=3, padding=1),
            nn.Softmax())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #48x32x64 ==> 96x64x64
        out = F.interpolate(self.layer4(out), scale_factor=2)
        #96x64x64 ==> 192x128x128
        out = F.interpolate(self.layer5(out), scale_factor=2)
        #192x128x128 ==> 384x256x10
        out = F.interpolate(self.layer6(out), scale_factor=2)
        return out

"""
@param  model       the model being trained
@param  data_file  list of filenames to be training
"""
def train(data_file):
    batch_size = 16
    num_epochs = 64
    data = dataloader(data_file, batch_size)

    print("\n")
    print("Training / Testing Data Info:")
    print("Num batches (batch_size=%d):" % batch_size, len(data["train"]))
    print("Input image shape:", data["train"][0][0].shape)
    print("Output mask shape:", data["train"][0][1].shape)
    print("Num testing datapoints:", len(data["test"]))
    print("="*50)
    print("\n")

    cnn_model = CNN()
    loss_func = F.cross_entropy
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(data["train"]):
            images = Variable(images.float())
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = cnn_model(images)
            loss = loss_func(outputs, labels, ignore_index=0)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            # if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch+1, num_epochs, i+1, len(images), loss.item()))

    epoch_time = int(time.time())
    model_directory = assets.DATA_PATH + "/ml/{0}".format(epoch_time)
    os.mkdir(model_directory)
    torch.save(cnn_model.state_dict(), model_directory)


if __name__ == "__main__":
    filename = "train.train"
    print("Start training process on file %s @ time" % filename, datetime.datetime.now())
    send_email("[CHTC Job Update]", "Training job started on file %s @ time %s" % (filename, datetime.datetime.now()))
    try:
        train("32020191045.train")
        send_email("[CHTC Job Update]", "Ending job successfully on file %s @ time %s" % (filename, datetime.datetime.now()))
    except Exception as e:
        send_email("[CHTC Job Update]", "Ended job with error %s on file %s @ time %s" % (str(e), filename, datetime.datetime.now()))
    print("End training @ time", datetime.datetime.now())
