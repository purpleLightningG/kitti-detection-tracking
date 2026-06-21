"""
export_onnx.py — Export the standalone PointPillars model to ONNX.

This is the ONLY model component we export. Voxelization stays as a Python
pre-processing step (it's data-dependent and not suitable for ONNX).

The ONNX graph takes pre-voxelized tensors as input:
  voxel_features:   (N_pillars, max_points, 4)   float32
  voxel_num_points: (N_pillars,)                  int32
  coords:           (N_pillars, 4)                int32

And produces:
  cls_preds: (1, H, W, num_anchors * num_classes)
  box_preds: (1, H, W, num_anchors * 7)
  dir_cls_preds: (1, H, W, num_anchors * 2)

We use dynamic axes so the engine accepts variable N_pillars at runtime.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch

from src import PointPillarsStandalone, load_pcdet_checkpoint_into_standalone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcdet-ckpt', required=True)
    parser.add_argument('--output', default='exported/PointPillars.onnx')
    parser.add_argument('--opset', type=int, default=13)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dummy-pillars', type=int, default=8000,
                        help='Number of dummy pillars for tracing (model accepts variable count)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Build standalone model and load weights
    print("Building standalone PointPillars model...")
    model = PointPillarsStandalone().to(device)
    model.eval()

    print(f"Loading checkpoint from {args.pcdet_ckpt}...")
    load_pcdet_checkpoint_into_standalone(args.pcdet_ckpt, model)

    # Create dummy inputs for tracing
    n_pillars = args.dummy_pillars
    max_points = model.max_num_points_per_voxel
    dummy_voxel_features = torch.randn(n_pillars, max_points, 4, device=device)
    dummy_voxel_num_points = torch.randint(1, max_points, (n_pillars,),
                                            device=device, dtype=torch.int32)
    dummy_coords = torch.zeros((n_pillars, 4), device=device, dtype=torch.int32)
    # Fill with valid coords (within grid)
    nx, ny, _ = model.grid_size
    dummy_coords[:, 2] = torch.randint(0, ny, (n_pillars,), device=device, dtype=torch.int32)
    dummy_coords[:, 3] = torch.randint(0, nx, (n_pillars,), device=device, dtype=torch.int32)

    # Sanity forward pass to ensure model works
    print("Running sanity forward pass...")
    with torch.no_grad():
        out = model(dummy_voxel_features, dummy_voxel_num_points, dummy_coords)
    print(f"  cls_preds shape: {tuple(out['cls_preds'].shape)}")
    print(f"  box_preds shape: {tuple(out['box_preds'].shape)}")
    print(f"  dir_cls_preds shape: {tuple(out['dir_cls_preds'].shape)}")

    # Wrap forward to make it ONNX-friendly (no kwargs, no dict output)
    class ONNXWrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, voxel_features, voxel_num_points, coords):
            out = self.base(voxel_features, voxel_num_points, coords, batch_size=1)
            return out['cls_preds'], out['box_preds'], out['dir_cls_preds']

    wrapped = ONNXWrapper(model).to(device).eval()

    # Export
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    print(f"\nExporting to ONNX (opset {args.opset}): {args.output}")

    torch.onnx.export(
        wrapped,
        (dummy_voxel_features, dummy_voxel_num_points, dummy_coords),
        args.output,
        opset_version=args.opset,
        input_names=['voxel_features', 'voxel_num_points', 'coords'],
        output_names=['cls_preds', 'box_preds', 'dir_cls_preds'],
        dynamic_axes={
            'voxel_features': {0: 'n_pillars'},
            'voxel_num_points': {0: 'n_pillars'},
            'coords': {0: 'n_pillars'},
        },
        do_constant_folding=True,
        verbose=False,
    )

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Done. ONNX file: {args.output} ({size_mb:.1f} MB)")

    # Verify with ONNX Runtime if available
    try:
        import onnx
        onnx_model = onnx.load(args.output)
        onnx.checker.check_model(onnx_model)
        print("ONNX model passes structural check.")
    except ImportError:
        print("Install `onnx` to verify model structure: pip install onnx")
    except Exception as e:
        print(f"ONNX check failed: {e}")


if __name__ == '__main__':
    main()
