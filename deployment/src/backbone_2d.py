"""
backbone_2d.py — OpenPCDet-compatible 2D BEV backbone.

Verified against checkpoint layer naming. The exact layout per downsampling stage:
    index 0:  ZeroPad2d(1)              (only ONE at start of stage)
    index 1:  Conv2d (downsample, stride=2)
    index 2:  BatchNorm2d
    index 3:  ReLU
    index 4:  Conv2d (regular, stride=1)
    index 5:  BatchNorm2d
    index 6:  ReLU
    index 7:  Conv2d
    index 8:  BatchNorm2d
    index 9:  ReLU
    ...
    Conv layers at indices: 1, 4, 7, 10, 13, 16
    BN layers at indices:   2, 5, 8, 11, 14, 17

Stage 0: 4 convs total (1 downsample + 3 regular) → indices 1, 4, 7, 10
Stage 1: 6 convs total (1 downsample + 5 regular) → indices 1, 4, 7, 10, 13, 16
Stage 2: 6 convs total                              → same as stage 1

Deblock layout (verified):
    index 0:  ConvTranspose2d (even when stride=1)
    index 1:  BatchNorm2d
    index 2:  ReLU
"""
import torch
import torch.nn as nn


def make_downsample_block(in_ch: int, out_ch: int, num_extra_convs: int) -> nn.Sequential:
    """Build one downsampling stage matching OpenPCDet's exact layer indexing.

    Pattern:
        ZeroPad2d (1)
        Conv2d(in_ch -> out_ch, stride=2)
        BatchNorm2d
        ReLU
        Conv2d(out_ch -> out_ch, stride=1, padding=1)  [no ZeroPad — padding built in]
        BatchNorm2d
        ReLU
        ... (extra convs as needed)
    """
    layers = []

    # Single ZeroPad at the very start of the stage
    layers.append(nn.ZeroPad2d(1))

    # First downsampling Conv (no built-in padding since ZeroPad handles it)
    layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=0, bias=False))
    layers.append(nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01))
    layers.append(nn.ReLU(inplace=True))

    # Extra regular convs — these use built-in padding (no ZeroPad)
    for _ in range(num_extra_convs):
        layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01))
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deblock(in_ch: int, out_ch: int, upsample_stride: int) -> nn.Sequential:
    """Build one upsampling stage.

    OpenPCDet uses ConvTranspose2d even when stride=1 (with kernel=1, behaves like
    a 1x1 linear projection but with transposed weight layout).

    Output Sequential indices:
        0:  ConvTranspose2d
        1:  BatchNorm2d
        2:  ReLU
    """
    deconv = nn.ConvTranspose2d(
        in_ch, out_ch,
        kernel_size=upsample_stride,
        stride=upsample_stride,
        bias=False,
    )
    return nn.Sequential(
        deconv,
        nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(inplace=True),
    )


class BaseBEVBackbone(nn.Module):
    """OpenPCDet-compatible 2D BEV backbone."""

    def __init__(
        self,
        input_channels: int = 64,
        layer_nums: list = [3, 5, 5],            # extra convs per stage (after downsample)
        num_filters: list = [64, 128, 256],
        upsample_strides: list = [1, 2, 4],
        num_upsample_filters: list = [128, 128, 128],
    ):
        super().__init__()
        self.num_levels = len(layer_nums)

        self.blocks = nn.ModuleList()
        c_in = input_channels
        for i in range(self.num_levels):
            self.blocks.append(
                make_downsample_block(c_in, num_filters[i], layer_nums[i])
            )
            c_in = num_filters[i]

        self.deblocks = nn.ModuleList()
        for i in range(self.num_levels):
            self.deblocks.append(
                make_deblock(num_filters[i], num_upsample_filters[i], upsample_strides[i])
            )

        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        cur = x
        for i in range(self.num_levels):
            cur = self.blocks[i](cur)
            outputs.append(self.deblocks[i](cur))
        return torch.cat(outputs, dim=1)
