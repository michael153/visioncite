"""Implements functions for traning a neural network."""
import functools
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mail import send_email  # TODO: Remove this once CHTC stuff is figured out


#pylint: disable=too-many-arguments, too-many-locals
def train(model,
          dataset,
          batch_size=16,
          num_epochs=64,
          loss_func=torch.nn.CrossEntropyLoss(),
          optimizer_class=torch.optim.Adam,
          learning_rate=0.001):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)
    device = next(model.parameters()).device

    model.train()
    for epoch in range(num_epochs):
        for batch_number, (observations, labels) in enumerate(dataloader):
            model.zero_grad()

            observations = observations.to(device)
            labels = labels.to(device)

            predictions = model(observations)
            # torch.Size([N, 1]) => torch.Size([N])
            predictions = torch.squeeze(predictions)

            loss = loss_func(predictions, labels)

            loss.backward()
            optimizer.step()

            # TODO: Refactor this once CHTC stuff is figured out
            if not epoch and not batch_number:
                print("\tEPOCH\tBATCH\tLOSS")
            print("\t%d\t%d\t%.4f" % (epoch, batch_number, loss.item()))

            # TODO: Remove this once CHTC stuff is figured out
            send_email("[CHTC] Training Update", "Batch %d finished at %s with loss %s." % (batch_number, datetime.now(), loss.item()), "bveeramani@berkeley.edu")
            send_email("[CHTC] Training Update", "Batch %d finished at %s with loss %s." % (batch_number, datetime.now(), loss.item()), "m.wan@berkeley.edu")

        # TODO: Remove this once CHTC stuff is figured out
        send_email("[CHTC] Training Update", "Epoch %d finished at %s." % (epoch, datetime.now()), "bveeramani@berkeley.edu")
        send_email("[CHTC] Training Update", "Epoch %d finished at %s." % (epoch, datetime.now()), "m.wan@berkeley.edu")


def test(model, dataset, batch_size=32):

    def average_accuracy(predictions, labels):
        predicted_labels = torch.argmax(predictions, dim=1)
        num_correct = torch.sum(predicted_labels == labels)
        num_labels = functools.reduce(lambda a, b : a * b, labels.shape)
        return float(num_correct) / num_labels

    dataloader = DataLoader(dataset, batch_size)
    device = next(model.parameters()).device

    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        for observations, labels in dataloader:
            observations = observations.to(device)
            labels = labels.to(device)

            predictions = model(observations)

            batch_accuracy = average_accuracy(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    return average_batch_accuracy


def save(model, filename):
    if hasattr(model, "module"):
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    torch.save(state, filename)


if __name__ == "__main__":
    main()
