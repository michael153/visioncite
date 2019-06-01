# TODO: Add external mail functionality.
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import PRImADataset
from models import CNN

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCH_SIZE = 32
NUM_CLASSES = len(PRImADataset.CLASSES)


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('data_file',
                        metavar='datafile',
                        help='path to train file')
    parser.add_argument('image_dir',
                        metavar='images',
                        help='path to dataset image directory')
    parser.add_argument('label_dir',
                        metavar='labels',
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

    dataset = PRImADataset(args.data_file, args.image_dir, args.label_dir)
    train(dataset, args.batch_size, args.num_epochs, device)


def train(dataset, batch_size, num_epochs, device):
    dataloader = DataLoader(dataset, batch_size)
    debug_train(dataset, batch_size, num_epochs, device)

    model = CNN(NUM_CLASSES)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    loss_func = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for batch_number, batch in enumerate(dataloader):
            images, labels = batch["image"], batch["label"]

            predictions = model(images)
            loss = loss_func(predictions, labels, ignore_index=0)
            debug_batch(batch_number, epoch, loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    save_model(model, batch_size, num_epochs)


def debug_train(dataset, batch_size, num_epochs, device):
    assert dataset, "Expected non-empty dataset."

    start_time = datetime.datetime.now()
    print("Starting training at time %s" % start_time, end="\n\n")

    print("BATCH_SIZE=%d" % batch_size)
    print("NUM_EPOCHS=%d" % num_epochs)
    print("DEVICE=%s" % device, end="\n\n")

    sample = dataset[0]
    print("INPUT_SHAPE=", sample["image"].shape, sep="")
    print("MASK_SHAPE=", sample["label"].shape, sep="")
    print("DATASET_SIZE=%d" % len(dataset), end="\n\n")


def debug_batch(batch_number, epoch, loss_value):
    if not batch_number and not epoch:
        print("BATCH\tEPOCH\tLOSS")
    print("%d\t%d\t%.4f" % (batch_number, epoch, loss_value))


def save_model(model, batch_size, num_epochs):
    end_time = datetime.datetime.now()
    filename = "%s%s%s%s.b%de%d.model" % (end_time.month, end_time.day,
                                          end_time.hour, end_time.minute,
                                          batch_size, num_epochs)
    print("Saving model to %s" % filename)
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    main()
