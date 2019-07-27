import torch
import torch.nn.functional as F
from torch.utils.data import random_split

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


dataset = VIADataset("WLA-500c", "metadata.json", transform=transform)
model = CNN(len(dataset.CLASSES))
traning_dataset, testing_dataset = random_split(dataset, (2, len(dataset) - 2))
train(model, traning_dataset, batch_size=2, num_epochs=1)
save(model, "model.pt")
