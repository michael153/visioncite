"""Implements functions for traning a neural network."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_SIZE = 64


#pylint: disable=too-many-arguments, too-many-locals
def train(model,
          dataset,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=torch.nn.CrossEntropyLoss()):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        for observations, labels in dataloader:
            model.zero_grad()

            predictions = model(observations)
            # torch.Size([N, 1]) => torch.Size([N])
            predictions = torch.squeeze(predictions)
            loss = loss_func(predictions, labels)

            loss.backward()
            optimizer.step()


def test(model, dataset, batch_size=DEFAULT_BATCH_SIZE * 2):

    def average_accuracy(predictions, labels):
        num_correct = 0
        for prediction, label in zip(predictions, labels):
            if prediction.argmax(dim=0) == label:
                num_correct += 1
        return num_correct / len(predictions)

    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model.eval()
    with torch.no_grad():
        for observations, labels in dataloader:
            predictions = model(observations)
            batch_accuracy = average_accuracy(predictions, labels)
            batch_accuracies.append(batch_accuracy)

    average_batch_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    return average_batch_accuracy


def save(model, filename):
    torch.save(model.state_dict(), filename)


if __name__ == "__main__":
    main()
