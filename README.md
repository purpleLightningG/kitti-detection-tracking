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

\---

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

### Detection (PointPillars on KITTI val split)

|Class|mAP @ IoU 0.7/0.5|Easy|Moderate|Hard|
|-|-|-|-|-|
|Car|—|—|—|—|
|Pedestrian|—|—|—|—|
|Cyclist|—|—|—|—|

### Tracking (ByteTrack on KITTI sequences)

|Metric|Value|
|-|-|
|Inference FPS|—|
|Avg tracks/frame|—|
|ID switches|—|

> Numbers will be filled after running evaluation. See \[Usage](#evaluation) for commands.

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

