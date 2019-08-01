import torch
import torch.nn.functional as F
from torch.utils.data import random_split, ConcatDataset

from training import train, save
from datasets import VIADataset
from models import CNN


def transform(image, mask):
    size = 384, 512

    # Add mini-batch dimension
    image = image.unsqueeze(0)
    image = F.interpolate(image, size=size)
    # Remove mini-batch dimension
    image = image.squeeze(0)

    # Add mini-batch and channel dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = F.interpolate(mask, size=size)
    # Remove mini-batch and channel dimensions
    mask = mask.squeeze(0).squeeze(0)

    mask = mask.type(torch.LongTensor)
    return image, mask

print("Loading first dataset...")
dataset1 = VIADataset("WLA-500c1", "metadata1.json", transform=transform) # Michael's data
print("Loading second dataset...")
dataset2 = VIADataset("WLA-500c2", "metadata2.json", transform=transform) # Balaji's data
dataset = ConcatDataset([dataset1, dataset2])

model = CNN(len(VIADataset.CLASSES))
print("Training model...")
train(model, dataset)
print("Saving model...")
save(model, "model.pt")
