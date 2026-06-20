"""
tracker.py — ByteTrack-based 3D multi-object tracker for KITTI.

Adapts the ByteTrack algorithm (Zhang et al., ECCV 2022) for 3D bounding boxes:
  - Two-stage association: high-confidence + low-confidence detections
  - 3D IoU-based cost matrix (axis-aligned approximation)
  - Simple Kalman filter for state prediction (constant velocity model)
  - Track lifecycle: tentative → confirmed → lost → deleted

Reference: https://arxiv.org/abs/2110.06864
"""
import numpy as np
from typing import List, Dict, Optional
from scipy.optimize import linear_sum_assignment


class KalmanFilter3D:
    """Simple constant-velocity Kalman filter for 3D box tracking.
    State: [x, y, z, l, w, h, ry, vx, vy, vz]
    """

    def __init__(self, box7: np.ndarray):
        """Initialize with first detection [x, y, z, l, w, h, ry]."""
        # State: position + dims + rotation + velocity
        self.state = np.zeros(10)
        self.state[:7] = box7
        # Covariance
        self.P = np.eye(10) * 10.0
        self.P[7:, 7:] *= 100.0  # high uncertainty on initial velocity

    def predict(self):
        """Predict next state using constant velocity model."""
        # x_{t+1} = x_t + vx, etc.
        self.state[0] += self.state[7]   # x += vx
        self.state[1] += self.state[8]   # y += vy
        self.state[2] += self.state[9]   # z += vz
        # Increase uncertainty
        self.P += np.eye(10) * 0.1
        return self.state[:7].copy()

    def update(self, box7: np.ndarray):
        """Update state with a matched detection."""
        # Simple alpha-blended update (simplified Kalman gain)
        alpha = 0.7  # weight on measurement
        innovation = box7 - self.state[:7]

        # Update velocity estimate from position change
        self.state[7:10] = alpha * innovation[:3] + (1 - alpha) * self.state[7:10]

        # Update position and dimensions
        self.state[:7] = alpha * box7 + (1 - alpha) * self.state[:7]

        # Reduce uncertainty
        self.P *= 0.8

    @property
    def box(self) -> np.ndarray:
        """Current 3D box estimate [x, y, z, l, w, h, ry]."""
        return self.state[:7].copy()


class Track:
    """Single tracked object with lifecycle management."""

    _next_id = 1  # class-level track ID counter

    def __init__(self, detection: dict, frame_idx: int):
        """Create a new track from a detection."""
        from box_utils import detection_to_box7
        box7 = detection_to_box7(detection)

        self.track_id = Track._next_id
        Track._next_id += 1

        self.kf = KalmanFilter3D(box7)
        self.cls = detection['type']
        self.score = detection['score']

        self.hits = 1              # number of matched frames
        self.age = 0               # frames since creation
        self.time_since_update = 0 # frames since last match

        self.start_frame = frame_idx
        self.history = [box7.copy()]

    @property
    def is_confirmed(self) -> bool:
        """Track is confirmed after min_hits consecutive detections."""
        return self.hits >= 3

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    def predict(self):
        """Propagate track state to next frame."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: dict):
        """Update track with a matched detection."""
        from box_utils import detection_to_box7
        box7 = detection_to_box7(detection)

        self.kf.update(box7)
        self.cls = detection['type']
        self.score = detection['score']
        self.hits += 1
        self.time_since_update = 0
        self.history.append(box7.copy())


class ByteTracker3D:
    """ByteTrack-style multi-object tracker for 3D detections."""

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        second_match_thresh: float = 0.5,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        """
        Args:
            high_thresh: confidence threshold for first-stage association
            low_thresh: minimum confidence to consider for second-stage
            match_thresh: IoU threshold for first-stage matching
            second_match_thresh: IoU threshold for second-stage matching
            max_age: max frames to keep a lost track before deletion
            min_hits: minimum consecutive hits to confirm a track
        """
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.second_match_thresh = second_match_thresh
        self.max_age = max_age
        self.min_hits = min_hits

        self.tracks: List[Track] = []
        self.frame_count = 0

        # Reset track ID counter
        Track._next_id = 1

    def update(self, detections: List[dict]) -> List[dict]:
        """Process one frame of detections and return tracked objects.

        Args:
            detections: list of detection dicts with 'type', 'location',
                        'dimensions', 'rotation_y', 'score'

        Returns:
            List of tracked detection dicts, each augmented with 'track_id'
        """
        from box_utils import detection_to_box7, iou_3d_axis_aligned

        self.frame_count += 1

        # Predict existing tracks forward
        for track in self.tracks:
            track.predict()

        # Split detections into high and low confidence
        dets_high = [d for d in detections if d['score'] >= self.high_thresh]
        dets_low = [d for d in detections if self.low_thresh <= d['score'] < self.high_thresh]

        # ── Stage 1: match high-confidence detections to tracks ──
        matched_tracks_1, unmatched_tracks_1, unmatched_dets_1 = self._associate(
            self.tracks, dets_high, self.match_thresh
        )

        # Update matched tracks
        for t_idx, d_idx in matched_tracks_1:
            self.tracks[t_idx].update(dets_high[d_idx])

        # ── Stage 2: match low-confidence detections to remaining tracks ──
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks_1]
        matched_tracks_2, unmatched_tracks_2, _ = self._associate(
            remaining_tracks, dets_low, self.second_match_thresh
        )

        # Update matched tracks from stage 2
        for t_idx, d_idx in matched_tracks_2:
            remaining_tracks[t_idx].update(dets_low[d_idx])

        # ── Create new tracks from unmatched high-confidence detections ──
        for d_idx in unmatched_dets_1:
            new_track = Track(dets_high[d_idx], self.frame_count)
            self.tracks.append(new_track)

        # ── Delete old lost tracks ──
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # ── Return confirmed tracks ──
        results = []
        for track in self.tracks:
            if track.is_confirmed and track.time_since_update == 0:
                result = {
                    'track_id': track.track_id,
                    'type': track.cls,
                    'location': track.box[:3],
                    'dimensions': np.array([track.box[5], track.box[4], track.box[3]]),
                    'rotation_y': float(track.box[6]),
                    'score': track.score,
                }
                results.append(result)

        return results

    def _associate(
        self, tracks: List[Track], detections: List[dict], iou_threshold: float
    ):
        """Hungarian matching between tracks and detections using 3D IoU.

        Returns:
            matched: list of (track_idx, det_idx) pairs
            unmatched_tracks: list of track indices
            unmatched_dets: list of detection indices
        """
        from box_utils import detection_to_box7, iou_3d_axis_aligned

        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Build cost matrix (negative IoU for Hungarian minimization)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                det_box = detection_to_box7(det)
                iou = iou_3d_axis_aligned(track.box, det_box)
                cost_matrix[i, j] = -iou  # negative because we minimize

        # Hungarian assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matched, unmatched_tracks, unmatched_dets = [], [], []

        # Filter by IoU threshold
        for r, c in zip(row_indices, col_indices):
            if -cost_matrix[r, c] < iou_threshold:
                unmatched_tracks.append(r)
                unmatched_dets.append(c)
            else:
                matched.append((r, c))

        # Add unassigned tracks and detections
        for i in range(len(tracks)):
            if i not in row_indices and i not in [m[0] for m in matched]:
                unmatched_tracks.append(i)
        for j in range(len(detections)):
            if j not in col_indices and j not in [m[1] for m in matched]:
                unmatched_dets.append(j)

        return matched, unmatched_tracks, unmatched_dets


if __name__ == "__main__":
    print("ByteTracker3D ready.")
    print("Usage: instantiate ByteTracker3D(), call .update(detections) per frame.")
