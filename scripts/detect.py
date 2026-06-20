"""
detect.py — Run PointPillars detection on KITTI frames.

Usage:
  python scripts/detect.py --frame 000042               # single frame
  python scripts/detect.py --all --save-dir outputs/detections/  # all frames
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import yaml
import numpy as np
from tqdm import tqdm

from kitti_utils import load_velodyne, get_frame_id, build_frame_path
from detector import PointPillarsDetector


def main():
    parser = argparse.ArgumentParser(description="Run 3D detection on KITTI frames")
    parser.add_argument("--frame", type=str, help="Single frame ID, e.g. 000042")
    parser.add_argument("--all", action="store_true", help="Run on all training frames")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--save-dir", default="outputs/detections", help="Where to save detection results")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg['data']['root']
    device = cfg.get('device', 'cuda:0')

    # Initialize detector
    det_cfg = cfg['detector']
    detector = PointPillarsDetector(
        config_path=det_cfg['config_file'],
        checkpoint_path=det_cfg['checkpoint'],
        device=device,
    )

    # Determine which frames to process
    if args.frame:
        frame_ids = [args.frame]
    elif args.all:
        # Get all frame IDs from velodyne directory
        vel_dir = os.path.join(data_root, cfg['data']['velodyne_dir'])
        frame_ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(vel_dir)
            if f.endswith('.bin')
        ])
    else:
        print("Specify --frame XXXXXX or --all")
        return

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Processing {len(frame_ids)} frames...")

    for fid in tqdm(frame_ids, desc="Detecting"):
        # Load point cloud
        bin_path = build_frame_path(data_root, cfg['data']['velodyne_dir'], fid, 'bin')
        if not os.path.exists(bin_path):
            print(f"Warning: {bin_path} not found, skipping")
            continue

        points = load_velodyne(bin_path)

        # Run detection
        detections = detector.detect(points, det_cfg.get('score_thresh', {}))

        # Save results as JSON
        save_path = os.path.join(args.save_dir, f"{fid}.json")
        serializable_dets = []
        for d in detections:
            serializable_dets.append({
                'type': d['type'],
                'location': d['location'].tolist(),
                'dimensions': d['dimensions'].tolist(),
                'rotation_y': d['rotation_y'],
                'score': d['score'],
            })

        with open(save_path, 'w') as f:
            json.dump(serializable_dets, f, indent=2)

    print(f"Done. Detections saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
