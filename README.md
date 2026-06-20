# 3D Object Detection \& Tracking on KITTI

<p align="center">
  <img src="assets/demo.gif" alt="Detection and tracking demo" width="800"/>
</p>

<p align="center">
  <b>PointPillars 3D detection + ByteTrack multi-object tracking on KITTI.</b><br>
  End-to-end pipeline from raw LiDAR point clouds to tracked 3D bounding boxes with persistent IDs.
</p>

<p align="center">
  <a href="#highlights">Highlights</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#results">Results</a> ·
  <a href="#installation">Install</a> ·
  <a href="#usage">Usage</a>
</p>


## Highlights

* **PointPillars detector** — converts raw LiDAR point clouds into a pseudo-image of pillar features, then runs a 2D detection head for 3D bounding box prediction
* **ByteTrack integration** — assigns persistent track IDs across frames, handles low-confidence detections via two-stage association
* **Full KITTI evaluation** — computes mAP at IoU 0.7 (Car) / 0.5 (Pedestrian, Cyclist) using the official KITTI metric
* **End-to-end pipeline** — single script takes raw `.bin` LiDAR scans → outputs tracked 3D boxes with class, confidence, and track ID
* **Visualization** — BEV (bird's-eye view) and camera-projected overlays with color-coded track IDs

## Architecture

<p align="center">
  <img src="assets/architecture.png" alt="Pipeline architecture" width="390"/>
</p>

```
KITTI LiDAR frames (.bin)
        ↓
  PointPillars Detector
  (pillarize → pseudo-image → 2D backbone → 3D box head)
        ↓
  Per-frame 3D detections
  (class, bbox\_3d, confidence)
        ↓
  ByteTrack
  (IoU-based association, two-stage matching, Kalman filter)
        ↓
  Tracked 3D boxes with persistent IDs
        ↓
  Evaluation (mAP + visual tracking)
```

## Results

> **Evaluation methodology:** Currently running the pipeline with filtered KITTI ground truth labels as the detection source (filtering out occluded/truncated objects), since OpenPCDet CUDA extensions don't compile on Windows MSVC. This serves as a pipeline correctness check rather than a model benchmark. Real PointPillars inference numbers will replace these once the model runs on Linux/WSL.

### Pipeline Sanity Check on KITTI val split (3,769 frames)

| Class | AP @ KITTI IoU | GT Objects |
|---|---|---|
| Car | 0.636 (IoU 0.7) | 14,661 |
| Pedestrian | 0.818 (IoU 0.5) | 2,215 |
| Cyclist | 0.727 (IoU 0.5) | 790 |

The gap from 1.0 reflects the occlusion/truncation filter applied to inputs (dropping heavily-occluded objects that the model wouldn't be expected to recover) — a useful proxy for how well the pipeline can handle visible-only detections.

### Tracking on KITTI val split

| Metric | Value |
|---|---|
| Frames processed | 3,769 |
| Unique track IDs | 76 |
| Total tracked objects | 80 |
| Avg tracks/frame | < 1 |

Low tracking density is expected here — KITTI's 3D Object Detection benchmark contains shuffled single-frame snapshots from different sequences, not continuous video. The tracker's two-stage IoU association and Kalman prediction still run correctly, but cross-frame matching has nothing meaningful to associate. The tracker is designed for the **KITTI Tracking benchmark** (continuous sequences), which is the proper test set for ID persistence metrics.

## Installation

### Prerequisites

* Python 3.10+
* CUDA 11.8+ (tested with RTX 3080)
* \~15 GB disk for KITTI training split

### Setup

```bash
git clone https://github.com/purpleLightningG/kitti-detection-tracking.git
cd kitti-detection-tracking
python -m venv .venv \&\& source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### KITTI Data

Download the [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) training split. Organize as:

```
data/kitti/training/
├── calib/           # calibration files
├── image\_2/         # left camera images
├── label\_2/         # 3D bounding box annotations
├── velodyne/        # LiDAR point clouds (.bin)
└── velodyne\_reduced/
```

Update `configs/config.yaml` with your data path.

## Usage

### Run detection on a single frame

```bash
python scripts/detect.py --frame 000042
```

### Run detection on all frames

```bash
python scripts/detect.py --all --save-dir outputs/detections/
```

### Run tracking across a sequence

```bash
python scripts/track.py \\
  --detections outputs/detections/ \\
  --save-dir outputs/tracks/
```

### Visualization

```bash
# BEV (bird's-eye view) with tracked boxes
python scripts/visualize.py --mode bev --tracks outputs/tracks/

# Camera overlay with projected 3D boxes
python scripts/visualize.py --mode camera --tracks outputs/tracks/ --frame 000042

# Generate demo GIF
python scripts/visualize.py --mode gif --tracks outputs/tracks/ --start 0 --end 100
```

### Evaluation

```bash
# Detection mAP (official KITTI metric)
python scripts/evaluate.py --mode detection --detections outputs/detections/

# Tracking metrics
python scripts/evaluate.py --mode tracking --tracks outputs/tracks/
```

## Repository Structure

```
kitti-detection-tracking/
├── src/
│   ├── detector.py          # PointPillars wrapper (OpenPCDet backend)
│   ├── tracker.py           # ByteTrack 3D adapter
│   ├── kitti\_utils.py       # KITTI data loading, calibration parsing
│   ├── box\_utils.py         # 3D IoU, NMS, box conversions
│   └── vis\_utils.py         # BEV and camera projection visualization
├── scripts/
│   ├── detect.py            # Run detection
│   ├── track.py             # Run tracking on detections
│   ├── evaluate.py          # Compute mAP and tracking metrics
│   └── visualize.py         # Generate visualizations and GIFs
├── configs/
│   └── config.yaml          # All paths, hyperparameters, thresholds
├── assets/                  # Architecture diagram, demo GIF
├── outputs/                 # Detections, tracks, visualizations
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

```bibtex
@misc{hossain2026kittidettrack,
  author = {Hossain, Shahriar},
  title  = {3D Object Detection and Tracking on KITTI},
  year   = {2026},
  url    = {https://github.com/purpleLightningG/kitti-detection-tracking}
}
```

## License

MIT — see [LICENSE](LICENSE).

\---

<p align="center">
  Built by <a href="https://purpleLightningG.github.io/portfolio-website">Shahriar Hossain</a> · PhD Researcher, George Mason University
</p>

