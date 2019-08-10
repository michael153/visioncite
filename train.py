import torch
import torch.nn.functional as F
from torch.utils.data import random_split, ConcatDataset
from torchvision  import transforms

from training import train, test, save
from datasets import VIADataset
from models import CNN

transform = transforms.Normalize([217.5426, 216.6502, 214.7937], [74.5412, 74.2549, 75.9361])

print("[ INFO ] Loading Michael's data...")
dataset1 = VIADataset("WLA-500c1", "metadata1.json", transform=transform) # Michael's data

print("[ INFO ] Loading Balaji's data...")
dataset2 = VIADataset("WLA-500c2", "metadata2.json", transform=transform) # Balaji's data

print("[ INFO ] Concatinating data...")
dataset = ConcatDataset([dataset1, dataset2])

lengths = [int(len(dataset) * 0.8), int(len(dataset) * 0.2)]

print("[ INFO ] Splitting data...")
train_dataset, test_dataset = random_split(dataset, lengths)

model = CNN(len(VIADataset.CLASSES))

print("[ INFO ] Training model...")
train(model, train_dataset)

print("[ INFO ] Testing model...")
accuracy = test(model, test_dataset)
print("\tACCURACY\n\t%.4f" % accuracy)

print("[ INFO ] Saving model...")
save(model, "model.pt")
