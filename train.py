import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import PRImADataset
from models import CNN

DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCH_SIZE = 32
NUM_CLASSES = len(PRImADataset.CLASSES)
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('x_train_dir',
                        metavar='x_train',
                        help='path to train dataset image directory')
    parser.add_argument('y_train_dir',
                        metavar='y_train',
                        help='path to train dataset label directory')
    parser.add_argument('x_valid_dir',
                        metavar='x_valid',
                        help='path to validation dataset image directory')
    parser.add_argument('y_valid_dir',
                        metavar='y_valid',
                        help='path to validation dataset label directory')
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

    train_dataset = PRImADataset(args.x_train_dir,
                                 args.y_train_dir)
    validation_dataset = PRImADataset(args.x_valid_dir,
                                      args.y_valid_dir)
    train(train_dataset, args.batch_size, args.num_epochs, device,
          validation_dataset)


def train(train_dataset,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          device=DEFAULT_DEVICE,
          validation_dataset=None):
    dataloader = DataLoader(train_dataset, batch_size)
    debug_train(train_dataset, batch_size, num_epochs, device)

    model = CNN(NUM_CLASSES)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_func = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(num_epochs):
        model.train()
        for batch_number, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images)
            loss = loss_func(predictions, labels)
            debug_batch(batch_number, epoch, loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if validation_dataset:
            validate(model, validation_dataset, device, loss_func,
                     batch_size * 2)

    save_model(model, batch_size, num_epochs)


def validate(model, dataset, device, loss_func, batch_size):
    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images)
            batch_accuracy = accuracy(predictions, labels)

            total_loss += loss_func(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    debug_validate(total_loss, average_batch_accuracy)


def accuracy(predictions, labels):
    accuracies = []
    for prediction, label in zip(predictions, labels):
        # torch.Size([1, 12, 384, 256]) => torch.Size([12, 384, 256])
        prediction = prediction.squeeze()
        # torch.Size([12, 384, 256]) => torch.Size([384, 256])
        mask = prediction.argmax(dim=0)
        difference = torch.abs(mask - label)
        # This a 384x256 matrix where each element is 1 if the prediction for
        # that pixel was correct and 0 otherwise.
        correct = difference.clamp(0, 1)

        num_elements = np.prod(correct.shape)
        num_correct = (num_elements - torch.sum(correct)).float()
        accuracies.append(num_correct / num_elements)

    return sum(accuracies) / len(accuracies)


def debug_validate(loss, accuracy):
    print("VALID\tACCURACY")
    print("%.4f\t%.4f" % (loss.item(), accuracy), end="\n\n")


def debug_train(dataset, batch_size, num_epochs, device):
    assert dataset, "Expected non-empty dataset."

    start_time = datetime.datetime.now()
    print("Starting training at time %s." % start_time, end="\n\n")

    print("BATCH_SIZE=%d" % batch_size)
    print("NUM_EPOCHS=%d" % num_epochs)
    if torch.cuda.device_count() > 1:
        print("NUM_GPUS=%d" % torch.cuda.device_count())
    print("DEVICE=%s" % device, end="\n\n")

    sample = dataset[0]
    print("INPUT_SHAPE=", sample["image"].shape, sep="")
    print("MASK_SHAPE=", sample["label"].shape, sep="")
    print("DATASET_SIZE=%d" % len(dataset), end="\n\n")


def debug_batch(batch_number, epoch, loss_value):
    if not batch_number:
        print("BATCH\tEPOCH\tLOSS")
    print("%d\t%d\t%.4f" % (batch_number, epoch, loss_value))


def save_model(model, batch_size, num_epochs):
    end_time = datetime.datetime.now()
    filename = "%s%s%s%s.b%de%d.model" % (end_time.month, end_time.day,
                                          end_time.hour, end_time.minute,
                                          batch_size, num_epochs)
    print("Saving model to %s" % filename)
    torch.save(model.module.state_dict(), filename)


if __name__ == "__main__":
    main()
