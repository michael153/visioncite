import os
import time
import datetime
import traceback
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import assets
import settings
from preprocessing import import_image

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCH_SIZE = 32

def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('datafile', dest='data_file', help='path to train file')
    parser.add_argument('images',
                        dest='image_dir',
                        help='path to dataset image directory')
    parser.add_argument('labels',
                        dest='label_dir',
                        help='path to dataset label directory')
    parser.add_argument('-b',
                        '--batches',
                        type=int,
                        dest='batch_size',
                        default=DEFAULT_BATCH_SIZE,
                        help='number of samples to propogate (default: 64)')
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        dest='num_epochs',
                        default=DEFAULT_EPOCH_SIZE,
                        help='number of passes through dataset (default: 32)')
    parser.add_argument('--disable-cuda',
                        dest='cuda_disabled',
                        action='store_true',
                        help='Disable CUDA')

    args = parser.parse_args()

    if not args.cuda_disabled and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    train(args.data_file, args.image_dir, args.label_dir, args.batch_size,
          args.num_epochs, device)


def train(data_file, image_dir, label_dir, batch_size, num_epochs, device):
    dataset = PRImADataset(data_file, root_dir)
    data = DataLoader(dataset, batch_size)

    num_classes = len(settings.LABELS)
    model = CNN(num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_func = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for image, labels in data:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels, ignore_index=0)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model-b%s-e%s" % (batch_size, num_epochs))





class CNN(nn.Module):

    def __init__(self, num_classes):
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
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1),
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


if __name__ == "__main__":
    main()
