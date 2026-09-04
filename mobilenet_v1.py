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


def scale_channels(base_channels, r, divisor=8):
    """Scale a channel count to a positive multiple of ``divisor``."""
    if r <= 0:
        raise ValueError("r must be greater than zero.")
    if not isinstance(divisor, int) or divisor < 1:
        raise ValueError("divisor must be a positive integer.")

    scaled = round(base_channels * r / divisor) * divisor
    return max(divisor, scaled)


def group_norm(num_channels):
    """Use the largest supported group count that divides ``num_channels``."""
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return nn.GroupNorm(groups, num_channels)


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


class IntermediateFeatureEncoder(nn.Module):
    """Resolution-independent encoder for a protected smashed-data block."""

    def __init__(
        self,
        input_channels=3,
        output_dim=256,
        r=1.0,
        pool_size=2,
    ):
        super().__init__()

        if not isinstance(pool_size, int) or pool_size < 1:
            raise ValueError("pool_size must be a positive integer.")

        c1 = scale_channels(32, r)
        c2 = scale_channels(64, r)
        c3 = scale_channels(128, r)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            group_norm(c1),
            nn.GELU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(c2),
            nn.GELU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            group_norm(c3),
            nn.GELU(),
            nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1, bias=False),
            group_norm(c3),
            nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3 * pool_size**2, output_dim),
        )

        self.input_channels = input_channels
        self.output_dim = output_dim
        self.r = r
        self.pool_size = pool_size
        self.feature_channels = (c1, c2, c3)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return self.projection(x)


class ProtectedServerLayers(IntermediateFeatureEncoder):
    """P&B protected branch implemented by the intermediate feature encoder."""


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
        protected_width: float = 1.0,
        protected_pool_size: int = 2,
        mask_fill: str = "learned",
    ):
        super().__init__()
        self.split_children = validate_split_children(split_children)
        self.mask_side = mask_side
        self.protected_width = protected_width
        self.protected_pool_size = protected_pool_size
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
        self.channels = channels

        self.protected_layers = ProtectedServerLayers(
            input_channels=channels,
            output_dim=256,
            r=protected_width,
            pool_size=protected_pool_size,
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
