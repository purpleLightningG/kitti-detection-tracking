"""
track.py — Run ByteTrack on pre-computed detections across KITTI frames.

Usage:
  python scripts/track.py --detections outputs/detections/ --save-dir outputs/tracks/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import yaml
import numpy as np
from tqdm import tqdm

from tracker import ByteTracker3D


def main():
    parser = argparse.ArgumentParser(description="Run 3D tracking on detections")
    parser.add_argument("--detections", required=True, help="Dir with per-frame detection JSONs")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--save-dir", default="outputs/tracks", help="Where to save tracking results")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tracker_cfg = cfg['tracker']

    # Initialize tracker
    tracker = ByteTracker3D(
        high_thresh=tracker_cfg['high_thresh'],
        low_thresh=tracker_cfg['low_thresh'],
        match_thresh=tracker_cfg['match_thresh'],
        second_match_thresh=tracker_cfg['second_match_thresh'],
        max_age=tracker_cfg['max_age'],
        min_hits=tracker_cfg['min_hits'],
    )

    # Get sorted list of detection files
    det_files = sorted([f for f in os.listdir(args.detections) if f.endswith('.json')])
    if not det_files:
        print(f"No detection files found in {args.detections}")
        return

    os.makedirs(args.save_dir, exist_ok=True)
    total_tracks = 0

    print(f"Tracking across {len(det_files)} frames...")

    for det_file in tqdm(det_files, desc="Tracking"):
        frame_id = os.path.splitext(det_file)[0]

        # Load detections for this frame
        with open(os.path.join(args.detections, det_file)) as f:
            dets_raw = json.load(f)

        # Convert back to numpy arrays
        detections = []
        for d in dets_raw:
            detections.append({
                'type': d['type'],
                'location': np.array(d['location']),
                'dimensions': np.array(d['dimensions']),
                'rotation_y': d['rotation_y'],
                'score': d['score'],
            })

        # Run tracker update
        tracked = tracker.update(detections)
        total_tracks += len(tracked)

        # Save tracked results
        save_path = os.path.join(args.save_dir, f"{frame_id}.json")
        serializable_tracks = []
        for t in tracked:
            serializable_tracks.append({
                'track_id': t['track_id'],
                'type': t['type'],
                'location': t['location'].tolist(),
                'dimensions': t['dimensions'].tolist(),
                'rotation_y': t['rotation_y'],
                'score': t['score'],
            })

        with open(save_path, 'w') as f:
            json.dump(serializable_tracks, f, indent=2)

    print(f"Done. {total_tracks} tracked objects across {len(det_files)} frames.")
    print(f"Results saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
