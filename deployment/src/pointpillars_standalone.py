"""
pointpillars_standalone.py — End-to-end standalone PointPillars in pure PyTorch.

Architecture matches OpenPCDet's pointpillar_7728.pth checkpoint exactly:
  - PFN: Linear+BN+ReLU with with_distance=True (10-dim input per point)
  - Backbone: 3 stages with ZeroPad+Conv+BN+ReLU pattern
  - Deblocks: ConvTranspose2d (1×1, 2×2, 4×4)
  - Detection head: 3 conv heads (cls, box, dir_cls) with anchors_per_loc=6
"""
import torch
import torch.nn as nn

from .pillar_encoder import PillarFeatureNet, PointPillarsScatter
from .backbone_2d import BaseBEVBackbone
from .detection_head import AnchorHeadSingle


class PointPillarsStandalone(nn.Module):
    """End-to-end PointPillars matching OpenPCDet's pointpillar_7728.pth."""

    def __init__(
        self,
        num_point_features: int = 4,
        voxel_size: tuple = (0.16, 0.16, 4.0),
        point_cloud_range: tuple = (0, -39.68, -3, 69.12, 39.68, 1),
        max_num_points_per_voxel: int = 32,
        max_num_voxels: int = 16000,
        pillar_feature_channels: tuple = (64,),
        num_class: int = 3,
        num_anchors_per_location: int = 6,
        box_code_size: int = 7,
        num_dir_bins: int = 2,
    ):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.max_num_voxels = max_num_voxels

        self.grid_size = (
            int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
            int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
            int((point_cloud_range[5] - point_cloud_range[2]) / voxel_size[2]),
        )

        self.pillar_feature_net = PillarFeatureNet(
            num_point_features=num_point_features,
            num_filters=pillar_feature_channels,
            with_distance=True,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
        )

        self.scatter = PointPillarsScatter(
            num_channels=pillar_feature_channels[-1],
            grid_size=self.grid_size,
        )

        self.backbone_2d = BaseBEVBackbone(input_channels=pillar_feature_channels[-1])

        self.detection_head = AnchorHeadSingle(
            input_channels=self.backbone_2d.num_bev_features,
            num_class=num_class,
            num_anchors_per_location=num_anchors_per_location,
            box_code_size=box_code_size,
            num_dir_bins=num_dir_bins,
        )

    def forward(
        self,
        voxel_features: torch.Tensor,
        voxel_num_points: torch.Tensor,
        coords: torch.Tensor,
        batch_size: int = 1,
    ) -> dict:
        pillar_features = self.pillar_feature_net(voxel_features, voxel_num_points, coords)
        spatial_features = self.scatter(pillar_features, coords, batch_size)
        spatial_features_2d = self.backbone_2d(spatial_features)
        return self.detection_head(spatial_features_2d)
