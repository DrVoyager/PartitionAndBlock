import torch
import torch.nn as nn


class MobileNetV1ClientModel(nn.Module):
    def __init__(self):
        super(MobileNetV1ClientModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class ProtectedServerLayers(nn.Module):
    def __init__(self, in_channels, h, w):
        super(ProtectedServerLayers, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * h * w, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fc(x)


class OriginalServerLayers(nn.Module):
    def __init__(self):
        super(OriginalServerLayers, self).__init__()

        def depthwise_conv(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.layers = nn.Sequential(
            depthwise_conv(64, 128, 1),
            depthwise_conv(128, 128, 2),
            depthwise_conv(128, 256, 1),
            depthwise_conv(256, 256, 2),
            depthwise_conv(256, 512, 1),
            depthwise_conv(512, 512, 2),
            depthwise_conv(512, 512, 1),
            depthwise_conv(512, 512, 1),
            depthwise_conv(512, 512, 1),
            depthwise_conv(512, 512, 1),
            depthwise_conv(512, 1024, 2),
            depthwise_conv(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        return self.layers(x)


class MergingLayers(nn.Module):
    def __init__(self, protected_dim=256, original_dim=256, num_classes=10):
        super(MergingLayers, self).__init__()
        self.merge = nn.Sequential(
            nn.Linear(protected_dim + original_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, z_p, z_o):
        z_concat = torch.cat([z_p, z_o], dim=1)
        return self.merge(z_concat)


class PartitionAndBlockingModel(nn.Module):
    def __init__(self, num_classes=10):
        super(PartitionAndBlockingModel, self).__init__()

        self.client_model = MobileNetV1ClientModel()

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = self.client_model(dummy_input)
            C, H, W = dummy_output.shape[1], dummy_output.shape[2], dummy_output.shape[3]

        self.grid_h = H // 3
        self.grid_w = W // 3
        self.channels = C

        self.protected_layers = ProtectedServerLayers(C, self.grid_h, self.grid_w)
        self.original_layers = OriginalServerLayers()
        self.merging_layers = MergingLayers(protected_dim=256, original_dim=256, num_classes=num_classes)

    def extract_central_partition(self, s):
        batch_size, C, H, W = s.shape
        grid_h = self.grid_h
        grid_w = self.grid_w

        i, j = 1, 1
        h_start = i * grid_h
        h_end = min(h_start + grid_h, H)
        w_start = j * grid_w
        w_end = min(w_start + grid_w, W)

        central = s[:, :, h_start:h_end, w_start:w_end]

        if central.shape[2] != grid_h or central.shape[3] != grid_w:
            pad_h = grid_h - central.shape[2]
            pad_w = grid_w - central.shape[3]
            central = torch.nn.functional.pad(central, (0, pad_w, 0, pad_h), mode='constant', value=0)

        return central

    def forward(self, x):
        s = self.client_model(x)

        s_protected = self.extract_central_partition(s)

        z_p = self.protected_layers(s_protected)
        z_o = self.original_layers(s)

        output = self.merging_layers(z_p, z_o)

        return output


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            self._make_depthwise(32, 64, stride=1),
            self._make_depthwise(64, 128, stride=2),
            self._make_depthwise(128, 128, stride=1),
            self._make_depthwise(128, 256, stride=2),
            self._make_depthwise(256, 256, stride=1),
            self._make_depthwise(256, 512, stride=2),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 1024, stride=2),
            self._make_depthwise(1024, 1024, stride=1),

            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )

    def _make_depthwise(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                     padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
