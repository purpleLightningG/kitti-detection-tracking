"""
pillar_encoder.py — Pillar Feature Encoder for PointPillars.

This implementation matches OpenPCDet's default config (with_distance=True):
augmented point features are 10-dimensional:
    [x, y, z, reflectance,                  # 4 raw point dims
     dx_center, dy_center, dz_center,       # 3 dist to pillar centroid
     dx_pillar, dy_pillar,                  # 2 dist to pillar voxel center XY
     distance_from_origin]                  # 1 distance to LiDAR origin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetLayer(nn.Module):
    """Per-pillar PointNet-style feature extractor (Conv1d kernel=1)."""

    def __init__(self, in_channels: int, out_channels: int, use_norm: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=not use_norm)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01) if use_norm else None
        self.use_norm = use_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        return F.relu(x)


class PillarFeatureNet(nn.Module):
    """Pillar Feature Network — 10-dim augmented input by default."""

    def __init__(
        self,
        num_point_features: int = 4,
        num_filters: tuple = (64,),
        with_distance: bool = True,
        voxel_size: tuple = (0.16, 0.16, 4.0),
        point_cloud_range: tuple = (0, -39.68, -3, 69.12, 39.68, 1),
    ):
        super().__init__()
        self.with_distance = with_distance
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        in_channels = num_point_features + 5  # +3 centroid dist +2 pillar center dist
        if with_distance:
            in_channels += 1  # +1 distance from origin

        self.pfn_layers = nn.ModuleList()
        for out_channels in num_filters:
            self.pfn_layers.append(PointNetLayer(in_channels, out_channels))
            in_channels = out_channels

    def forward(self, voxel_features, voxel_num_points, coords):
        # Distance to pillar centroid
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / \
                      voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean

        # Distance to pillar voxel center (XY only)
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_size[0]
            + self.voxel_size[0] / 2 + self.point_cloud_range[0]
        )
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_size[1]
            + self.voxel_size[1] / 2 + self.point_cloud_range[1]
        )

        features = [voxel_features, f_cluster, f_center[:, :, :2]]
        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # Mask padded points
        mask = self._get_padding_mask(voxel_num_points, features.shape[1])
        features *= mask
        features = features.permute(0, 2, 1)

        for pfn in self.pfn_layers:
            features = pfn(features)

        return torch.max(features, dim=2)[0]

    @staticmethod
    def _get_padding_mask(actual_num, max_num):
        actual_num = actual_num.view(-1, 1)
        max_indices = torch.arange(max_num, device=actual_num.device).view(1, -1)
        return (max_indices < actual_num).unsqueeze(-1).float()


class PointPillarsScatter(nn.Module):
    """Scatter pillars onto a 2D pseudo-image."""

    def __init__(self, num_channels: int, grid_size: tuple):
        super().__init__()
        self.num_channels = num_channels
        self.nx, self.ny, self.nz = grid_size

    def forward(self, pillar_features, coords, batch_size: int = 1):
        batch_canvas = []
        for batch_idx in range(batch_size):
            canvas = torch.zeros(
                self.num_channels, self.nx * self.ny,
                dtype=pillar_features.dtype, device=pillar_features.device,
            )
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask]
            indices = (this_coords[:, 2] * self.nx + this_coords[:, 3]).long()
            pillars = pillar_features[batch_mask].t()
            canvas[:, indices] = pillars
            batch_canvas.append(canvas)
        batch_canvas = torch.stack(batch_canvas, 0)
        return batch_canvas.view(batch_size, self.num_channels, self.ny, self.nx)
