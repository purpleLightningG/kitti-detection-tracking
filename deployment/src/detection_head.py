"""
detection_head.py — Anchor-based 3D detection head for PointPillars.

After the 2D backbone produces a multi-scale BEV feature map, the detection head
predicts three things per spatial location, per anchor:
  - Classification logits (which class is at this location?)
  - 3D box regression (7-dim: dx, dy, dz, dl, dw, dh, dyaw — encoded as deltas)
  - Direction classifier (binary: is the object facing +x or -x?)

The KITTI PointPillars uses 2 anchors per pixel per class:
  - Anchor sizes (per class): Car (3.9, 1.6, 1.56), Pedestrian (0.8, 0.6, 1.73), Cyclist (1.76, 0.6, 1.73)
  - Each anchor has rotation 0 and π/2 (forward + sideways)

Total anchors per spatial location = 6 (3 classes × 2 rotations).
"""
import torch
import torch.nn as nn


class AnchorHeadSingle(nn.Module):
    """Single-stage anchor-based detection head.

    Predicts class scores, box deltas, and direction classifier for each anchor
    at each spatial location.
    """

    def __init__(
        self,
        input_channels: int = 384,           # output channels from BaseBEVBackbone
        num_class: int = 3,                  # Car, Pedestrian, Cyclist
        num_anchors_per_location: int = 6,   # 3 classes × 2 rotations
        box_code_size: int = 7,              # [x, y, z, l, w, h, ry]
        num_dir_bins: int = 2,               # 2 direction bins
        use_direction_classifier: bool = True,
    ):
        super().__init__()
        self.num_anchors_per_location = num_anchors_per_location
        self.box_code_size = box_code_size
        self.num_dir_bins = num_dir_bins
        self.use_direction_classifier = use_direction_classifier

        # Classification head: num_class scores per anchor
        self.conv_cls = nn.Conv2d(
            input_channels,
            num_anchors_per_location * num_class,
            kernel_size=1,
        )

        # Box regression head: 7 values per anchor
        self.conv_box = nn.Conv2d(
            input_channels,
            num_anchors_per_location * box_code_size,
            kernel_size=1,
        )

        # Direction classifier
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                num_anchors_per_location * num_dir_bins,
                kernel_size=1,
            )

    def forward(self, spatial_features_2d: torch.Tensor) -> dict:
        """
        Args:
            spatial_features_2d: (B, 384, H, W) from BaseBEVBackbone

        Returns:
            dict with:
              cls_preds: (B, H, W, num_anchors_per_location * num_class)
              box_preds: (B, H, W, num_anchors_per_location * box_code_size)
              dir_cls_preds: (B, H, W, num_anchors_per_location * num_dir_bins) or None
        """
        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        # Permute to (B, H, W, C) for downstream decoding
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()

        result = {
            'cls_preds': cls_preds,
            'box_preds': box_preds,
        }

        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            result['dir_cls_preds'] = dir_cls_preds

        return result
