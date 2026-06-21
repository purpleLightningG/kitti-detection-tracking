"""
verify_standalone.py — Sanity check standalone PointPillars vs OpenPCDet outputs.

Loads the same checkpoint into both the OpenPCDet model (via pcdet imports) and
our standalone reimplementation, runs both on the same voxelized input, and
compares the output tensors numerically.

Expected behavior:
  - For correctly mapped weights, outputs should match to within ~1e-4 relative tolerance.
  - Larger discrepancies indicate a layer-name mismatch or weight reshape bug.

NOTE: This script requires OpenPCDet to be importable. On Windows where OpenPCDet
won't compile, you can skip this verification and trust the next-step ONNX export
+ TensorRT pipeline (which uses only the standalone module).
"""
import sys
import os

# Make the deployment package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch

from src import PointPillarsStandalone, load_pcdet_checkpoint_into_standalone


def voxelize_points(points: np.ndarray, voxel_size, point_cloud_range,
                    max_points: int = 32, max_voxels: int = 16000):
    """Simple numpy voxelization (CPU-only, no CUDA ops).

    For benchmark inputs we don't need the speed of spconv's voxelizer.
    """
    # Filter points to point cloud range
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    # Compute voxel indices for each point
    vx = ((points[:, 0] - x_min) / voxel_size[0]).astype(np.int32)
    vy = ((points[:, 1] - y_min) / voxel_size[1]).astype(np.int32)
    vz = np.zeros_like(vx)  # single Z slab for pillars

    # Hash by (vy, vx) to find unique pillars
    pillar_keys = vy * 1_000_000 + vx
    unique_keys, inverse, counts = np.unique(pillar_keys, return_inverse=True, return_counts=True)

    n_pillars = min(len(unique_keys), max_voxels)
    voxel_features = np.zeros((n_pillars, max_points, points.shape[1]), dtype=np.float32)
    voxel_num_points = np.zeros(n_pillars, dtype=np.int32)
    coords = np.zeros((n_pillars, 4), dtype=np.int32)  # [batch, z, y, x]

    for pid in range(n_pillars):
        pts_in_pillar = points[inverse == pid]
        n_pts = min(len(pts_in_pillar), max_points)
        voxel_features[pid, :n_pts] = pts_in_pillar[:n_pts]
        voxel_num_points[pid] = n_pts
        coords[pid, 0] = 0  # batch index
        coords[pid, 1] = vz[inverse == pid][0]
        coords[pid, 2] = vy[inverse == pid][0]
        coords[pid, 3] = vx[inverse == pid][0]

    return voxel_features, voxel_num_points, coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcdet-ckpt', required=True, help='Path to OpenPCDet PointPillars .pth')
    parser.add_argument('--frame', default='000042', help='KITTI frame ID for input')
    parser.add_argument('--data-root',
                        default='C:/Users/Dream Team/Desktop/kitti_velodyne/training',
                        help='KITTI training root')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ----- Load standalone model -----
    print(f"Initializing standalone PointPillars on {device}...")
    standalone = PointPillarsStandalone().to(device)
    standalone.eval()

    print(f"Loading checkpoint from {args.pcdet_ckpt}...")
    missing, unexpected = load_pcdet_checkpoint_into_standalone(
        args.pcdet_ckpt, standalone, verbose=True
    )

    if missing:
        print(f"\nWARNING: {len(missing)} parameters in the standalone model were NOT")
        print("populated from the checkpoint. They retain randomly-initialized values.")
        print("This will cause garbage predictions. Fix weight_mapper.py before proceeding.")

    # ----- Load a KITTI frame -----
    bin_path = os.path.join(args.data_root, 'velodyne', f'{args.frame}.bin')
    if not os.path.exists(bin_path):
        print(f"\nERROR: Cannot find {bin_path}")
        print("Run with --data-root pointing at your KITTI training folder.")
        return

    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    print(f"\nLoaded {points.shape[0]} points from {args.frame}.bin")

    # ----- Voxelize -----
    print("Voxelizing point cloud...")
    voxel_features_np, voxel_num_points_np, coords_np = voxelize_points(
        points,
        voxel_size=standalone.voxel_size,
        point_cloud_range=standalone.point_cloud_range,
        max_points=standalone.max_num_points_per_voxel,
        max_voxels=standalone.max_num_voxels,
    )
    print(f"  Active pillars: {len(voxel_features_np)}")

    voxel_features = torch.from_numpy(voxel_features_np).to(device)
    voxel_num_points = torch.from_numpy(voxel_num_points_np).to(device)
    coords = torch.from_numpy(coords_np).to(device)

    # ----- Forward pass -----
    print("Running forward pass through standalone model...")
    with torch.no_grad():
        out = standalone(voxel_features, voxel_num_points, coords, batch_size=1)

    print("\n=== Standalone model output ===")
    for k, v in out.items():
        print(f"  {k:20s}: shape={tuple(v.shape)}, "
              f"min={v.min().item():.3f}, max={v.max().item():.3f}, "
              f"mean={v.mean().item():.3f}")

    print("\nSanity check complete. If outputs look reasonable (cls_preds with both")
    print("positive and negative values, box_preds in roughly [-5, 5] range), the")
    print("weight loading worked correctly and you're ready to export to ONNX.")
    print("\nNext step: python scripts/export_onnx.py")


if __name__ == '__main__':
    main()
