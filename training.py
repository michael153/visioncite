"""Implements functions for traning a neural network."""
import functools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

OPTIMAL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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

    model.train()
    model.to(OPTIMAL_DEVICE)

    for epoch in range(num_epochs):
        for batch_number, (observations, labels) in enumerate(dataloader):
            model.zero_grad()

            observations = observations.to(OPTIMAL_DEVICE)
            labels = labels.to(OPTIMAL_DEVICE)

            predictions = model(observations)
            # torch.Size([N, 1]) => torch.Size([N])
            predictions = torch.squeeze(predictions)

            loss = loss_func(predictions, labels)

            loss.backward()
            optimizer.step()

            if not epoch and not batch_number:
                print("\tEPOCH\tBATCH\tLOSS")
            print("\t%d\t%d\t%.4f" % (epoch, batch_number, loss.item()))


def test(model, dataset, batch_size=32):

    def average_accuracy(predictions, labels):
        predicted_labels = torch.argmax(predictions, dim=1)
        num_correct = torch.sum(predicted_labels == labels)
        num_labels = functools.reduce(lambda a, b : a * b, labels.shape)
        return float(num_correct) / num_labels

    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model.eval()
    model = model.to(OPTIMAL_DEVICE)
    with torch.no_grad():
        for observations, labels in dataloader:
            observations = observations.to(OPTIMAL_DEVICE)
            labels = labels.to(OPTIMAL_DEVICE)

            predictions = model(observations)

            batch_accuracy = average_accuracy(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    return average_batch_accuracy


def save(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    main()
