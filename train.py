"""Implements functions for traning a neural network.

usage: train.py [-h] [--xtest XTEST] [--ytest YTEST] [--batches BATCH_SIZE]
                [--epochs NUM_EPOCHS] [--disable-cuda]
                xtrain ytrain

Train model

positional arguments:
  xtrain                path to train dataset image directory
  ytrain                path to train dataset label directory

optional arguments:
  -h, --help            show this help message and exit
  --xtest XTEST         path to validation dataset image directory
  --ytest YTEST         path to validation dataset label directory
  --batches BATCH_SIZE  number of samples to propogate (default: 64)
  --epochs NUM_EPOCHS   number of passes through dataset (default: 32)
  --disable-cuda        disable CUDA support
"""
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
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ModelType = CNN
MODEL_ARGS = [len(PRImADataset.CLASSES)]
MODEL_KWARGS = {}


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('x_train_dir',
                        metavar='xtrain',
                        help='path to train dataset image directory')
    parser.add_argument('y_train_dir',
                        metavar='ytrain',
                        help='path to train dataset label directory')
    parser.add_argument('--xtest',
                        dest="x_test_dir",
                        metavar="XTEST",
                        default=None,
                        help='path to validation dataset image directory')
    parser.add_argument('--ytest',
                        dest="y_test_dir",
                        metavar="YTEST",
                        default=None,
                        help='path to validation dataset label directory')
    parser.add_argument('--batches',
                        type=int,
                        dest='batch_size',
                        default=DEFAULT_BATCH_SIZE,
                        help='number of samples to propogate (default: 64)')
    parser.add_argument('--epochs',
                        type=int,
                        dest='num_epochs',
                        default=DEFAULT_EPOCH_SIZE,
                        help='number of passes through dataset (default: 32)')
    parser.add_argument('--disable-cuda',
                        dest='cuda_disabled',
                        action='store_true',
                        help='disable CUDA support')

    args = parser.parse_args()

    if not args.cuda_disabled and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    model = ModelType(*MODEL_ARGS, **MODEL_KWARGS)
    train_dataset = PRImADataset(args.x_train_dir, args.y_train_dir)
    if args.x_test_dir and args.y_test_dir:
        validation_dataset = PRImADataset(args.x_test_dir, args.y_test_dir)
    else:
        validation_dataset = None

    train(model,
          train_dataset,
          validation_dataset=validation_dataset,
          batch_size=args.batch_size,
          num_epochs=args.num_epochs,
          device=device)
    save_model(model, batch_size=args.batch_size, num_epochs=args.num_epochs)


#pylint: disable=too-many-arguments, too-many-locals
def train(model,
          train_dataset,
          validation_dataset=None,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=F.cross_entropy,
          device=DEFAULT_DEVICE):
    """Trains the specified model.

    If a validation dataset is supplied, then this function will run validation
    tests every epoch.

    The specified model will always be trained using the Adam optimizer.
    """

    def debug_train(dataset, batch_size, num_epochs, device):
        """Prints information about the training hyperparameters."""
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

    dataloader = DataLoader(train_dataset, batch_size)
    debug_train(train_dataset, batch_size, num_epochs, device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters())

    def debug_batch(batch_number, epoch, loss):
        """Prints results from training a model on a data batch.

        Arguments:
            batch_number: A non-negative integer
            epoch: A non-negative integer
            loss: A scalar tensor
        """
        if not batch_number:
            print("BATCH\tEPOCH\tLOSS")
        print("%d\t%d\t%.4f" % (batch_number, epoch, loss.item()))

    for epoch in range(num_epochs):
        model.train()
        for batch_number, batch in enumerate(dataloader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images)
            loss = loss_func(predictions, labels)
            debug_batch(batch_number, epoch, loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if validation_dataset:
            validate(model,
                     validation_dataset,
                     device=device,
                     batch_size=batch_size * 2)

    return model


def validate(model,
             dataset,
             batch_size=DEFAULT_DEVICE * 2,
             loss_func=F.cross_entropy,
             device=DEFAULT_DEVICE):
    """Runs validation testing on a model over the specified dataset."""

    def debug_validate(loss, accuracy):
        """Prints results from validating a model over a dataset.

        Arguments:
            loss: A scalar tensor
            accuracy: A floating-point number between 0 and 1.
        """
        assert 0 <= accuracy <= 1, "accuracy must be in range [0, 1]"

        print("VALID\tACCURACY")
        print("%.4f\t%.4f" % (loss.item(), accuracy), end="\n\n")

    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        total_loss = 0
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            predictions = model(images)
            batch_accuracy = average_accuracy(predictions, labels)

            total_loss += loss_func(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    debug_validate(total_loss, average_batch_accuracy)


def average_accuracy(predictions, labels):
    """Calculates the average prediction accuracy.

    N is the number of predictions. C is the number of classes.

    Arguments:
        predictions: A tensor of shape N x C x H x W.
        labels: A tensor of shape H x W

    Returns:
        The average prediction accuracy.
    """
    accuracies = []
    for prediction, label in zip(predictions, labels):
        # torch.Size([1, 12, 384, 256]) => torch.Size([12, 384, 256])
        prediction = prediction.squeeze()
        # torch.Size([12, 384, 256]) => torch.Size([384, 256])
        mask = prediction.argmax(dim=0)
        difference = torch.abs(mask - label)  #pylint: disable=no-member
        # This a 384x256 matrix where each element is 1 if the prediction for
        # that pixel was correct and 0 otherwise.
        correct = difference.clamp(0, 1)

        num_elements = np.prod(correct.shape)
        num_correct = (num_elements - torch.sum(correct)).float()  #pylint: disable=no-member
        accuracies.append(num_correct / num_elements)

    return sum(accuracies) / len(accuracies)


def save_model(model, batch_size, num_epochs):
    """Saves a model's state dictionary to a .pt file

    An example of a file that might be produced by this function
    is 6132234.b64.e32.pt. The first part of the filename indicates the month,
    day, hour, and minute at which the model was saved. The second part
    indicates the batch size, and the third part indicates the number of epochs.

    Arguments:
        model: An nn.Module object
        batch_size: An integer representing how many batches the model was
            trained on
        num_epochs: An integer representing how many epochs the model was
            trained over
    """
    end_time = datetime.datetime.now()
    filename = "%s%s%s%s.b%de%d.pt" % (end_time.month, end_time.day,
                                       end_time.hour, end_time.minute,
                                       batch_size, num_epochs)
    print("Saving model to %s" % filename)
    torch.save(model.module.state_dict(), filename)


if __name__ == "__main__":
    main()
