import collections

import torch
import torch.nn as nn


class CifarNet(nn.Module):
    """CIFAR-10 model from ege-erdogan/unsplit.

    The forward method supports split execution through the same start/end
    layer indices used by the original unsplit implementation.
    """

    def __init__(self, n_channels: int = 3):
        super().__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.features.append(self.conv11)
        self.layers["conv11"] = self.conv11

        self.ReLU11 = nn.ReLU(True)
        self.features.append(self.ReLU11)
        self.layers["ReLU11"] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        self.features.append(self.conv12)
        self.layers["conv12"] = self.conv12

        self.ReLU12 = nn.ReLU(True)
        self.features.append(self.ReLU12)
        self.layers["ReLU12"] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)
        self.layers["pool1"] = self.pool1

        self.conv21 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.features.append(self.conv21)
        self.layers["conv21"] = self.conv21

        self.ReLU21 = nn.ReLU(True)
        self.features.append(self.ReLU21)
        self.layers["ReLU21"] = self.ReLU21

        self.conv22 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.features.append(self.conv22)
        self.layers["conv22"] = self.conv22

        self.ReLU22 = nn.ReLU(True)
        self.features.append(self.ReLU22)
        self.layers["ReLU22"] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
        self.layers["pool2"] = self.pool2

        self.conv31 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.features.append(self.conv31)
        self.layers["conv31"] = self.conv31

        self.ReLU31 = nn.ReLU(True)
        self.features.append(self.ReLU31)
        self.layers["ReLU31"] = self.ReLU31

        self.conv32 = nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.features.append(self.conv32)
        self.layers["conv32"] = self.conv32

        self.ReLU32 = nn.ReLU(True)
        self.features.append(self.ReLU32)
        self.layers["ReLU32"] = self.ReLU32

        self.pool3 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool3)
        self.layers["pool3"] = self.pool3

        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)
        self.layers["fc1"] = self.fc1

        self.fc1act = nn.Sigmoid()
        self.classifier.append(self.fc1act)
        self.layers["fc1act"] = self.fc1act

        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)
        self.layers["fc2"] = self.fc2

        self.initial_params = [param.data for param in self.parameters()]

    def forward(self, x: torch.Tensor, start: int = 0, end: int = 17) -> torch.Tensor:
        if start <= len(self.features) - 1:
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 15 == end:
                    return x
        else:
            if start == 15:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 15:
                    x = layer(x)
                if idx + 15 == end:
                    return x
        return x

    def get_params(self, end: int = 17):
        params = []
        for layer in list(self.layers.values())[: end + 1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self) -> None:
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial
