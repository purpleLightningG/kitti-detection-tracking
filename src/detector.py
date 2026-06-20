"""
detector.py — PointPillars 3D object detector wrapper.

Uses OpenPCDet as the backend for PointPillars inference.
Install: pip install openpcdet  (or clone from github.com/open-mmlab/OpenPCDet)

This wrapper handles:
  1. Loading a pretrained PointPillars checkpoint
  2. Running inference on a single KITTI LiDAR frame
  3. Returning per-frame detections as structured dicts
"""
import numpy as np
import torch
import yaml
import os
import sys


class PointPillarsDetector:
    """Wrapper around OpenPCDet's PointPillars model for KITTI inference."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        """
        Args:
            config_path: path to OpenPCDet model config YAML
            checkpoint_path: path to pretrained .pth checkpoint
            device: torch device string
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Import OpenPCDet modules
        # NOTE: You need OpenPCDet installed. See Installation section in README.
        try:
            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.models import build_network, load_data_to_gpu
            from pcdet.utils import common_utils

            # Load config
            cfg_from_yaml_file(config_path, cfg)
            self.cfg = cfg

            # Build model
            self.model = build_network(
                model_cfg=cfg.MODEL,
                num_class=len(cfg.CLASS_NAMES),
                dataset=None
            )

            # Load checkpoint
            self.model.load_params_from_file(
                filename=checkpoint_path,
                to_cpu=True
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            self.class_names = cfg.CLASS_NAMES
            print(f"[Detector] PointPillars loaded on {self.device}")
            print(f"[Detector] Classes: {self.class_names}")

        except ImportError:
            print("[Detector] WARNING: OpenPCDet not installed.")
            print("[Detector] Install via: pip install openpcdet")
            print("[Detector] Or clone: https://github.com/open-mmlab/OpenPCDet")
            print("[Detector] Running in STUB mode — will return empty detections.")
            self.model = None
            self.class_names = ["Car", "Pedestrian", "Cyclist"]

    def preprocess(self, points: np.ndarray) -> dict:
        """Convert raw point cloud to OpenPCDet input format.

        Args:
            points: (N, 4) array [x, y, z, reflectance]

        Returns:
            data_dict compatible with OpenPCDet model forward pass
        """
        # Filter points within detection range
        pc_range = np.array(self.cfg.DATA_CONFIG.POINT_CLOUD_RANGE) if self.model else np.array([0, -39.68, -3, 69.12, 39.68, 1])

        mask = (
            (points[:, 0] >= pc_range[0]) & (points[:, 0] <= pc_range[3]) &
            (points[:, 1] >= pc_range[1]) & (points[:, 1] <= pc_range[4]) &
            (points[:, 2] >= pc_range[2]) & (points[:, 2] <= pc_range[5])
        )
        points_filtered = points[mask]

        input_dict = {
            'points': points_filtered,
            'frame_id': 0,
            'batch_size': 1,
        }
        return input_dict

    @torch.no_grad()
    def detect(self, points: np.ndarray, score_thresholds: dict = None) -> list:
        """Run detection on a single frame.

        Args:
            points: (N, 4) raw LiDAR points
            score_thresholds: per-class minimum confidence, e.g. {'Car': 0.3}

        Returns:
            List of detection dicts, each with:
              type, location, dimensions, rotation_y, score
        """
        if score_thresholds is None:
            score_thresholds = {'Car': 0.3, 'Pedestrian': 0.2, 'Cyclist': 0.2}

        # Stub mode if OpenPCDet not available
        if self.model is None:
            print("[Detector] STUB mode — returning empty detections")
            return []

        # Preprocess
        data_dict = self.preprocess(points)

        # Run model forward pass
        from pcdet.models import load_data_to_gpu

        # Convert to batch format expected by OpenPCDet
        data_dict = self.model.collate_batch([data_dict])
        load_data_to_gpu(data_dict)

        pred_dicts, _ = self.model.forward(data_dict)

        # Parse predictions
        detections = []
        pred = pred_dicts[0]

        pred_boxes = pred['pred_boxes'].cpu().numpy()     # (N, 7): x,y,z,l,w,h,ry
        pred_scores = pred['pred_scores'].cpu().numpy()   # (N,)
        pred_labels = pred['pred_labels'].cpu().numpy()   # (N,) 1-indexed

        for i in range(len(pred_boxes)):
            cls_name = self.class_names[int(pred_labels[i]) - 1]
            score = float(pred_scores[i])

            # Apply per-class threshold
            thresh = score_thresholds.get(cls_name, 0.3)
            if score < thresh:
                continue

            det = {
                'type': cls_name,
                'location': pred_boxes[i, :3],        # [x, y, z]
                'dimensions': np.array([               # [h, w, l] KITTI order
                    pred_boxes[i, 5],                  # h
                    pred_boxes[i, 4],                  # w
                    pred_boxes[i, 3],                  # l
                ]),
                'rotation_y': float(pred_boxes[i, 6]),
                'score': score,
            }
            detections.append(det)

        return detections


if __name__ == "__main__":
    # Quick test — loads model and runs on a single frame
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, help="Path to a .bin LiDAR file")
    parser.add_argument("--cfg", required=True, help="Path to OpenPCDet config YAML")
    parser.add_argument("--ckpt", required=True, help="Path to pretrained checkpoint")
    args = parser.parse_args()

    from kitti_utils import load_velodyne

    # Load point cloud
    points = load_velodyne(args.bin)
    print(f"Loaded {points.shape[0]} points")

    # Run detection
    detector = PointPillarsDetector(args.cfg, args.ckpt)
    dets = detector.detect(points)
    print(f"Found {len(dets)} detections")
    for d in dets:
        print(f"  {d['type']}: score={d['score']:.2f}, loc={d['location']}")
