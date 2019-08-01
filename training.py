"""Implements functions for traning a neural network."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCH_SIZE = 64
OPTIMAL_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


#pylint: disable=too-many-arguments, too-many-locals
def train(model,
          dataset,
          batch_size=DEFAULT_BATCH_SIZE,
          num_epochs=DEFAULT_EPOCH_SIZE,
          loss_func=torch.nn.CrossEntropyLoss()):
    dataloader = DataLoader(dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.to(OPTIMAL_DEVICE)

    for epoch in range(num_epochs):
        model.train()
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
            
            print("Epoch", epoch, "Batch", batch_number, "Loss", loss.item())


def test(model, dataset, batch_size=DEFAULT_BATCH_SIZE * 2):

    def average_accuracy(predictions, labels):
        num_correct = 0
        for prediction, label in zip(predictions, labels):
            if prediction.argmax(dim=0) == label:
                num_correct += 1
        return num_correct / len(predictions)

    dataloader = DataLoader(dataset, batch_size)
    batch_accuracies = []

    model = model.to(OPTIMAL_DEVICE)

    model.eval()
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
