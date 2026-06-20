"""
evaluate.py — Compute detection mAP and tracking metrics on KITTI.

Usage:
  python scripts/evaluate.py --mode detection --detections outputs/detections/
  python scripts/evaluate.py --mode tracking --tracks outputs/tracks/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import yaml
import numpy as np
from tqdm import tqdm

from kitti_utils import load_label, get_frame_id, build_frame_path
from box_utils import detection_to_box7, iou_3d_axis_aligned


def compute_ap(recall, precision):
    """Compute Average Precision (AP) using 11-point interpolation (KITTI standard)."""
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        p = precision[recall >= t]
        if len(p) > 0:
            ap += np.max(p)
    return ap / 11.0


def evaluate_detection(det_dir, data_root, label_subdir, iou_thresholds, val_split):
    """Compute per-class mAP for detections vs. ground truth labels."""
    print("\n=== Detection Evaluation ===\n")

    classes = list(iou_thresholds.keys())

    for cls in classes:
        iou_thresh = iou_thresholds[cls]
        all_scores = []
        all_matches = []
        total_gt = 0

        # Process each frame in validation split
        for idx in tqdm(range(val_split[0], val_split[1] + 1), desc=f"{cls} eval"):
            fid = get_frame_id(idx)

            # Load ground truth
            label_path = build_frame_path(data_root, label_subdir, fid, 'txt')
            if not os.path.exists(label_path):
                continue
            gt_labels = [l for l in load_label(label_path) if l['type'] == cls]
            total_gt += len(gt_labels)

            # Load detections
            det_path = os.path.join(det_dir, f"{fid}.json")
            if not os.path.exists(det_path):
                continue
            with open(det_path) as f:
                dets = json.load(f)
            dets = [d for d in dets if d['type'] == cls]

            # Sort detections by score descending
            dets.sort(key=lambda x: x['score'], reverse=True)

            # Match detections to ground truth
            matched_gt = set()
            for det in dets:
                det_box = detection_to_box7({
                    'location': np.array(det['location']),
                    'dimensions': np.array(det['dimensions']),
                    'rotation_y': det['rotation_y'],
                })

                best_iou = 0
                best_gt_idx = -1
                for gi, gt in enumerate(gt_labels):
                    if gi in matched_gt:
                        continue
                    gt_box = detection_to_box7(gt)
                    iou = iou_3d_axis_aligned(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gi

                all_scores.append(det['score'])
                if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
                    all_matches.append(1)  # true positive
                    matched_gt.add(best_gt_idx)
                else:
                    all_matches.append(0)  # false positive

        # Compute precision-recall
        if total_gt == 0:
            print(f"  {cls}: no ground truth objects found")
            continue

        scores = np.array(all_scores)
        matches = np.array(all_matches)
        sorted_idx = np.argsort(-scores)
        matches = matches[sorted_idx]

        tp_cum = np.cumsum(matches)
        fp_cum = np.cumsum(1 - matches)
        precision = tp_cum / (tp_cum + fp_cum)
        recall = tp_cum / total_gt

        ap = compute_ap(recall, precision)
        print(f"  {cls}: AP = {ap:.4f} (IoU @ {iou_thresh}, {total_gt} GT objects)")

    print()


def evaluate_tracking(tracks_dir, data_root, label_subdir, val_split):
    """Compute basic tracking statistics."""
    print("\n=== Tracking Evaluation ===\n")

    total_frames = 0
    total_tracks = 0
    all_track_ids = set()

    for idx in range(val_split[0], val_split[1] + 1):
        fid = get_frame_id(idx)
        track_path = os.path.join(tracks_dir, f"{fid}.json")
        if not os.path.exists(track_path):
            continue

        with open(track_path) as f:
            tracks = json.load(f)

        total_frames += 1
        total_tracks += len(tracks)
        for t in tracks:
            all_track_ids.add(t['track_id'])

    avg_tracks = total_tracks / max(total_frames, 1)
    print(f"  Frames processed: {total_frames}")
    print(f"  Unique track IDs: {len(all_track_ids)}")
    print(f"  Avg tracks/frame: {avg_tracks:.1f}")
    print(f"  Total tracked objects: {total_tracks}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["detection", "tracking"], required=True)
    parser.add_argument("--detections", default="outputs/detections")
    parser.add_argument("--tracks", default="outputs/tracks")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg['data']['root']
    label_dir = cfg['data']['label_dir']
    val_split = cfg['data']['val_split']

    if args.mode == "detection":
        evaluate_detection(
            args.detections, data_root, label_dir,
            cfg['eval']['iou_thresh'], val_split
        )
    elif args.mode == "tracking":
        evaluate_tracking(args.tracks, data_root, label_dir, val_split)


if __name__ == "__main__":
    main()
