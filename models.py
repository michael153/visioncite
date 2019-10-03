"""The models.py module implements various neural network models."""
import torch
import torch.nn as nn


class SSM1(nn.Module):
    """Semantic Segmentation Model (SSM) Mark 1.

    A convolutional neural network that segments webpage images into classes.
    Based on http://personal.psu.edu/xuy111/projects/cvpr2017_doc.html.
    """

    def __init__(self, num_classes):
        super(SSM1, self).__init__()

        def create_conv(c1, c2, c3):
            convolution = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU())
            return convolution


        def create_deconv(c1, c2, c3):
            deconvolution = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c2),
                nn.ReLU(),
                nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c3),
                nn.ReLU())
            return deconvolution

        self.conv1 = create_conv(3, 64, 128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = create_conv(128, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3 = create_conv(128, 128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv4 = create_conv(128, 128, 128)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv4 = create_deconv(128, 128, 128)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv3 = create_deconv(128 + 128, 128, 128)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = create_deconv(128 + 128, 128, 128)
        self.deconv1 = create_deconv(128 + 128, 64, num_classes)

    def forward(self, inputs):  #pylint: disable=arguments-differ
        features1 = self.conv1(inputs)

        features2, indices1 = self.pool1(features1)
        features2 = self.conv2(features2)

        features3, indices2 = self.pool2(features2)
        features3 = self.conv3(features3)

        features4, indices3 = self.pool3(features3)
        features4 = self.conv4(features4)

        de_features4 = self.unpool3(features4, indices3)
        de_features4 = self.deconv4(de_features4)

        de_features3_1 = self.unpool2(features3, indices2)
        de_features3_2 = self.unpool2(de_features4, indices2)
        de_features3 = torch.cat([de_features3_1, de_features3_2], dim=1)
        de_features3 = self.deconv3(de_features3)

        de_features2_1 = self.unpool1(features2, indices1)
        de_features2_2 = self.unpool1(de_features3, indices1)
        de_features2 = torch.cat([de_features2_1, de_features2_2], dim=1)
        de_features2 = self.deconv2(de_features2)

        de_features1 = torch.cat([features1, de_features2], dim=1)
        de_features1 = self.deconv1(de_features1)
        return de_features1
