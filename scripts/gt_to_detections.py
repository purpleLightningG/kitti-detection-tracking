"""
gt_to_detections.py — Convert KITTI ground truth labels to detection format.
Lets you run the tracking pipeline without needing OpenPCDet installed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import yaml
import numpy as np
from tqdm import tqdm
from kitti_utils import load_label, get_frame_id

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    data_root = cfg['data']['root']
    label_dir = os.path.join(data_root, cfg['data']['label_dir'])
    save_dir = "outputs/detections"
    os.makedirs(save_dir, exist_ok=True)

    # Process all label files
    label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.txt')])
    print(f"Converting {len(label_files)} ground truth files to detection format...")

    for lf in tqdm(label_files, desc="Converting"):
        fid = os.path.splitext(lf)[0]
        labels = load_label(os.path.join(label_dir, lf))

        # Filter to Car/Pedestrian/Cyclist only
        detections = []
        for l in labels:
            if l['type'] not in ['Car', 'Pedestrian', 'Cyclist']:
                continue
            # Skip heavily occluded or truncated objects
            if l.get('occluded', 0) >= 2:
                continue
            if l.get('truncated', 0) > 0.5:
                continue
            # Add a fake score (1.0 for GT, with small noise for tracker realism)
            detections.append({
                'type': l['type'],
                'location': l['location'].tolist(),
                'dimensions': l['dimensions'].tolist(),
                'rotation_y': l['rotation_y'],
                'score': float(np.clip(0.85 + np.random.uniform(-0.1, 0.15), 0.5, 1.0)),
            })

        with open(os.path.join(save_dir, f"{fid}.json"), 'w') as f:
            json.dump(detections, f, indent=2)

    print(f"Done. {len(label_files)} detection files saved to {save_dir}/")

if __name__ == "__main__":
    main()