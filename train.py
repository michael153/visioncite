from torch.utils.data import ConcatDataset
from torchvision  import transforms
import torch

from training import train, save
from datasets import VIADataset
from models import SSM1

transform = transforms.Normalize([217.5426, 216.6502, 214.7937], [74.5412, 74.2549, 75.9361])
dataset1 = VIADataset("WLA-500c1", "metadata1.json", transform=transform) # Michael's data
dataset2 = VIADataset("WLA-500c2", "metadata2.json", transform=transform) # Balaji's data
dataset = ConcatDataset([dataset1, dataset2])

model = SSM1(len(VIADataset.CLASSES))
if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
model.to("cuda:0" if torch.cuda.is_available() else "cpu")

train(model, dataset, batch_size=2, num_epochs=1, optimizer_class=torch.optim.Adadelta)
save(model, "model.pt")
