import torch
import torch.nn as nn


STEM_CHILDREN = 3
DEPTHWISE_BLOCKS = 12
MAX_SPATIAL_SPLIT_CHILDREN = STEM_CHILDREN + DEPTHWISE_BLOCKS
SPLIT_PRESETS = {"stem": STEM_CHILDREN}
SPLIT_PRESETS.update({f"dw{idx}": STEM_CHILDREN + idx for idx in range(1, DEPTHWISE_BLOCKS + 1)})
SPLIT_PRESETS.update({
    "early64": SPLIT_PRESETS["dw1"],
    "pb-spatial": SPLIT_PRESETS["dw2"],
})


def split_children_from_depthwise_blocks(block_count: int) -> int:
    if not 1 <= block_count <= DEPTHWISE_BLOCKS:
        raise ValueError(f"split-after-depthwise must be between 1 and {DEPTHWISE_BLOCKS}.")
    return STEM_CHILDREN + block_count


def validate_split_children(split_children: int) -> int:
    if not 1 <= split_children <= MAX_SPATIAL_SPLIT_CHILDREN:
        raise ValueError(
            "split_children must select a spatial MobileNetV1 feature prefix, "
            f"between 1 and {MAX_SPATIAL_SPLIT_CHILDREN}."
        )
    return split_children


def depthwise_conv(in_ch, out_ch, stride):
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def mobilenet_v1_spatial_features():
    return [
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        depthwise_conv(32, 64, 1),
        depthwise_conv(64, 128, 2),
        depthwise_conv(128, 128, 1),
        depthwise_conv(128, 256, 2),
        depthwise_conv(256, 256, 1),
        depthwise_conv(256, 512, 2),
        depthwise_conv(512, 512, 1),
        depthwise_conv(512, 512, 1),
        depthwise_conv(512, 512, 1),
        depthwise_conv(512, 512, 1),
        depthwise_conv(512, 1024, 2),
        depthwise_conv(1024, 1024, 1),
    ]


class MobileNetV1ClientModel(nn.Module):
    """P&B client-side model: MobileNetV1 layers up to the split."""

    def __init__(self, split_children: int = STEM_CHILDREN):
        super().__init__()
        self.split_children = validate_split_children(split_children)
        self.features = nn.Sequential(*mobilenet_v1_spatial_features()[:self.split_children])

    def forward(self, x):
        return self.features(x)


class ProtectedServerLayers(nn.Module):
    """Protected branch retaining configurable projected feature maps."""

    def __init__(self, in_channels, pooled_size=4, projected_channels=None):
        super().__init__()
        self.pooled_size = pooled_size
        self.projected_channels = pooled_size if projected_channels is None else projected_channels
        if not isinstance(self.projected_channels, int) or not 1 <= self.projected_channels <= in_channels:
            raise ValueError(f"projected_channels must be an integer between 1 and {in_channels}.")
        self.spatial_filter = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.relu_spatial = nn.ReLU(inplace=True)
        self.channel_projection = nn.Conv2d(in_channels, self.projected_channels, kernel_size=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))
        self.fc = nn.Linear(self.projected_channels * pooled_size * pooled_size, 256)
        self.relu_output = nn.ReLU(inplace=True)

        with torch.no_grad():
            self.spatial_filter.weight.zero_()
            self.spatial_filter.weight[:, 0, 1, 1] = 1.0
            self.channel_projection.weight.zero_()
            for output_channel in range(self.projected_channels):
                start = output_channel * in_channels // self.projected_channels
                end = (output_channel + 1) * in_channels // self.projected_channels
                self.channel_projection.weight[output_channel, start:end].fill_(1.0 / (end - start))
            self.channel_projection.bias.zero_()

    def forward(self, x):
        features = self.relu_spatial(self.spatial_filter(x))
        projected_maps = self.channel_projection(features)
        pooled = self.pool(projected_maps)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.relu_output(self.fc(flattened))


class OriginalServerLayers(nn.Module):
    """Remaining original MobileNetV1 layers after the configured split."""

    def __init__(self, split_children: int = STEM_CHILDREN):
        super().__init__()
        self.split_children = validate_split_children(split_children)
        remaining_features = mobilenet_v1_spatial_features()[self.split_children:]
        self.layers = nn.Sequential(
            *remaining_features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256),
        )

    def forward(self, x):
        return self.layers(x)


class MergingLayers(nn.Module):
    def __init__(self, protected_dim=256, original_dim=256, num_classes=10):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Linear(protected_dim + original_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, z_p, z_o):
        z_concat = torch.cat([z_p, z_o], dim=1)
        return self.merge(z_concat)


class PartitionAndBlockingModel(nn.Module):
    """
    Split-configurable P&B MobileNetV1 model.

    By default, the split is before the first MobileNetV1 depthwise block:
    the client output is [B, 32, 32, 32], the protected branch consumes the
    centered [B, 32, 10, 10] partition, and the original branch consumes a
    same-shaped tensor with that protected partition zeroed. For later splits,
    the client contains all MobileNetV1 blocks before the split, and the
    original branch contains only the remaining original layers.
    """

    def __init__(
        self,
        num_classes=10,
        split_children: int = STEM_CHILDREN,
        mask_side=None,
        protected_pool_size: int = 4,
        protected_channels: int | None = None,
        mask_fill: str = "learned",
    ):
        super().__init__()
        self.split_children = validate_split_children(split_children)
        self.mask_side = mask_side
        self.protected_pool_size = protected_pool_size
        self.protected_channels = protected_pool_size if protected_channels is None else protected_channels
        if mask_fill not in {"learned", "zero"}:
            raise ValueError("mask_fill must be either 'learned' or 'zero'.")
        self.mask_fill_type = mask_fill
        self.client_model = MobileNetV1ClientModel(split_children=self.split_children)

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = self.client_model(dummy_input)
            channels, height, width = dummy_output.shape[1:]

        if mask_side is not None:
            if mask_side < 1 or mask_side > min(height, width):
                raise ValueError(f"mask_side must be between 1 and {min(height, width)} for this split.")
            self.grid_h = mask_side
            self.grid_w = mask_side
        else:
            self.grid_h = max(height // 3, 1)
            self.grid_w = max(width // 3, 1)
        if not isinstance(protected_pool_size, int) or protected_pool_size < 1:
            raise ValueError("protected_pool_size must be a positive integer.")
        self.channels = channels

        self.protected_layers = ProtectedServerLayers(
            channels, pooled_size=protected_pool_size, projected_channels=self.protected_channels
        )
        self.original_layers = OriginalServerLayers(split_children=self.split_children)
        self.merging_layers = MergingLayers(protected_dim=256, original_dim=256, num_classes=num_classes)
        if mask_fill == "learned":
            self.mask_fill = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_buffer("mask_fill", torch.zeros(1, channels, 1, 1))

    def central_partition_bounds(self, s):
        _, _, height, width = s.shape
        h_start = max((height - self.grid_h + 1) // 2, 0)
        h_end = min(h_start + self.grid_h, height)
        w_start = max((width - self.grid_w + 1) // 2, 0)
        w_end = min(w_start + self.grid_w, width)
        return h_start, h_end, w_start, w_end

    def extract_central_partition(self, s):
        h_start, h_end, w_start, w_end = self.central_partition_bounds(s)
        central = s[:, :, h_start:h_end, w_start:w_end]

        if central.shape[2] != self.grid_h or central.shape[3] != self.grid_w:
            pad_h = self.grid_h - central.shape[2]
            pad_w = self.grid_w - central.shape[3]
            central = torch.nn.functional.pad(central, (0, pad_w, 0, pad_h), mode="constant", value=0)

        return central

    def mask_central_partition(self, s):
        """Return the ordinary-server view with the protected block removed."""
        h_start, h_end, w_start, w_end = self.central_partition_bounds(s)
        ordinary_mask = torch.ones_like(s)
        ordinary_mask[:, :, h_start:h_end, w_start:w_end] = 0
        fill = self.mask_fill.to(dtype=s.dtype, device=s.device)
        return s * ordinary_mask + fill * (1 - ordinary_mask)

    def forward(self, x):
        s = self.client_model(x)
        z_p = self.protected_layers(self.extract_central_partition(s))
        z_o = self.original_layers(self.mask_central_partition(s))
        return self.merging_layers(z_p, z_o)
