import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import PRImADataset
from models import CNN

DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCH_SIZE = 32


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('datafile', metavar='data_file', help='path to train file')
    parser.add_argument('images',
                        metavar='image_dir',
                        help='path to dataset image directory')
    parser.add_argument('labels',
                        metavar='label_dir',
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

    num_classes = len(PRImADataset.CLASSES)
    model = CNN(num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_func = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(num_epochs):
        for batch in dataloader:
            images, labels = batch["image"], batch["label"]

            predictions = model(images)
            loss = loss_func(predictions, labels, ignore_index=0)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(model.state_dict(), "model-b%s-e%s" % (batch_size, num_epochs))


if __name__ == "__main__":
    main()
