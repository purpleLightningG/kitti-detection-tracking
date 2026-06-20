"""
vis_utils.py — Visualization utilities for BEV plots and camera overlays.
"""
import numpy as np
import cv2
from typing import List, Optional

# Color palette for track IDs (20 distinct colors)
# Using the tab20 colormap as RGB 0-255
TRACK_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207), (174, 199, 232), (255, 187, 120),
    (152, 223, 138), (255, 152, 150), (197, 176, 213), (196, 156, 148),
    (247, 182, 210), (199, 199, 199), (219, 219, 141), (158, 218, 229),
]


def get_track_color(track_id: int) -> tuple:
    """Get a consistent color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


def draw_bev(
    points: np.ndarray,
    detections: List[dict],
    bev_range: list = [-40, -40, 40, 40],
    resolution: float = 0.05,
    show_points: bool = True,
) -> np.ndarray:
    """Draw bird's-eye view with point cloud and 3D boxes.

    Args:
        points: (N, 3+) LiDAR points
        detections: list of detection/track dicts
        bev_range: [x_min, y_min, x_max, y_max] in meters
        resolution: meters per pixel

    Returns:
        BEV image as (H, W, 3) uint8 array
    """
    x_min, y_min, x_max, y_max = bev_range
    W = int((x_max - x_min) / resolution)
    H = int((y_max - y_min) / resolution)

    # Create black background
    bev = np.zeros((H, W, 3), dtype=np.uint8)

    # Draw point cloud in gray
    if show_points and len(points) > 0:
        px = ((points[:, 0] - x_min) / resolution).astype(int)
        py = ((points[:, 1] - y_min) / resolution).astype(int)
        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        bev[py[valid], px[valid]] = (200, 200, 200)

    # Draw detection boxes
    for det in detections:
        loc = det['location']
        dims = det['dimensions']  # [h, w, l]
        ry = det.get('rotation_y', 0)
        track_id = det.get('track_id', 0)

        l, w = dims[2], dims[1]  # length, width
        color = get_track_color(track_id) if track_id else (0, 255, 0)

        # Convert from camera coords (x=right, z=forward) to LiDAR (x=forward, y=left)
        bev_x = loc[2]   # camera z → LiDAR x (forward)
        bev_y = -loc[0]   # camera x → LiDAR -y (left)

        # 4 corners of box in BEV (rotated rectangle)
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
        ])

        # Rotation matrix
        R = np.array([[np.cos(ry), -np.sin(ry)], [np.sin(ry), np.cos(ry)]])
        corners = (R @ corners.T).T + np.array([bev_x, bev_y])

        # Convert to pixel coordinates
        corners_px = ((corners - np.array([x_min, y_min])) / resolution).astype(int)

        # Draw rotated rectangle
        for i in range(4):
            p1 = tuple(corners_px[i])
            p2 = tuple(corners_px[(i + 1) % 4])
            cv2.line(bev, p1, p2, color, 2)

        # Draw track ID label
        if 'track_id' in det:
            label = f"ID:{det['track_id']}"
            center_px = ((np.array([bev_x, bev_y]) - np.array([x_min, y_min])) / resolution).astype(int)
            cv2.putText(bev, label, tuple(center_px), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
            
        color = get_track_color(track_id) if track_id else (0, 255, 0)

        # 4 corners of box in BEV (rotated rectangle)
        corners = np.array([
            [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
        ])

        # Rotation matrix (around z-axis for BEV)
        R = np.array([[np.cos(ry), -np.sin(ry)], [np.sin(ry), np.cos(ry)]])
        corners = (R @ corners.T).T + loc[:2]

        # Convert to pixel coordinates
        corners_px = ((corners - np.array([x_min, y_min])) / resolution).astype(int)

        # Draw rotated rectangle
        for i in range(4):
            p1 = tuple(corners_px[i])
            p2 = tuple(corners_px[(i + 1) % 4])
            cv2.line(bev, p1, p2, color, 4)

        # Draw track ID label
        label = det.get('type', '')
        if 'track_id' in det:
            label = f"ID:{det['track_id']}"
            center_px = ((loc[:2] - np.array([x_min, y_min])) / resolution).astype(int)
            cv2.putText(bev, label, tuple(center_px), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)

    return bev


def draw_boxes_on_image(
    image: np.ndarray,
    detections: List[dict],
    calib: dict,
) -> np.ndarray:
    """Project 3D boxes onto camera image and draw them.

    Args:
        image: (H, W, 3) RGB image
        detections: list of detection/track dicts
        calib: KITTI calibration dict

    Returns:
        Image with projected 3D boxes drawn
    """
    from kitti_utils import corners_3d_from_label

    img = image.copy()

    for det in detections:
        track_id = det.get('track_id', 0)
        # Color by class type
        class_colors = {'Car': (0, 255, 0), 'Pedestrian': (255, 255, 0), 'Cyclist': (0, 165, 255)}
        color = class_colors.get(det.get('type', ''), (0, 255, 0))
        # Convert color from RGB to BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])

        # Get 3D box corners in camera frame
        corners_3d = corners_3d_from_label(det)

        # Project to image using calibration
        P2 = calib['P2']
        corners_hom = np.hstack([corners_3d, np.ones((8, 1))])
        proj = (P2 @ corners_hom.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        proj = proj.astype(int)

        # Draw 3D box edges
        # Bottom face
        for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            cv2.line(img, tuple(proj[i]), tuple(proj[j]), color_bgr, 2)
        # Top face
        for i, j in [(4, 5), (5, 6), (6, 7), (7, 4)]:
            cv2.line(img, tuple(proj[i]), tuple(proj[j]), color_bgr, 2)
        # Vertical edges
        for i, j in [(0, 4), (1, 5), (2, 6), (3, 7)]:
            cv2.line(img, tuple(proj[i]), tuple(proj[j]), color_bgr, 1)

        # Label
        label = f"{det['type']}"
        if 'track_id' in det:
            label += f" #{det['track_id']}"
        if 'score' in det:
            label += f" {det['score']:.1f}"
        cv2.putText(img, label, tuple(proj[4]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color_bgr, 1, cv2.LINE_AA)

    return img
