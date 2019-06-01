import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            #384x256x3 ==> 384x256x128
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #384x256x128 ==> 192x128x128
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            #192x128x128 ==> 192x128x64
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #192x128x64 ==> 96x64x64
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            #96x64x64 ==> 96x64x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #96x64x32 ==> 48x32x32
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            #48x32x32 ==> 48x32x64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            #96x64x64 ==> 96x64x218
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1),
            nn.Softmax(dim=1))

    def forward(self, x): #pylint: disable=arguments-differ
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        #48x32x64 ==> 96x64x64
        out = F.interpolate(self.layer4(out), scale_factor=2)
        #96x64x64 ==> 192x128x128
        out = F.interpolate(self.layer5(out), scale_factor=2)
        #192x128x128 ==> 384x256x10
        out = F.interpolate(self.layer6(out), scale_factor=2)
        return out
