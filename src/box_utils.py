"""
box_utils.py — 3D bounding box utilities: IoU computation, NMS, format conversions.
"""
import numpy as np
from typing import List, Tuple


def iou_3d_axis_aligned(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute axis-aligned 3D IoU between two boxes.
    Each box: [x, y, z, l, w, h, ry] (center coords + dimensions + rotation).
    Simplified: ignores rotation, uses axis-aligned bounding boxes.
    """
    # Convert to min/max corners (ignoring rotation for speed)
    def to_minmax(box):
        x, y, z, l, w, h = box[0], box[1], box[2], box[3], box[4], box[5]
        return np.array([x - l/2, y - h/2, z - w/2, x + l/2, y + h/2, z + w/2])

    a = to_minmax(box_a)
    b = to_minmax(box_b)

    # Intersection
    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:], b[3:])
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)

    # Union
    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]
    union_vol = vol_a + vol_b - inter_vol

    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol


def iou_matrix_3d(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise 3D IoU between two sets of boxes.
    boxes_a: (M, 7), boxes_b: (N, 7)
    Returns: (M, N) IoU matrix.
    """
    M, N = len(boxes_a), len(boxes_b)
    iou_mat = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            iou_mat[i, j] = iou_3d_axis_aligned(boxes_a[i], boxes_b[j])
    return iou_mat


def nms_3d(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
    """3D Non-Maximum Suppression.
    boxes: (N, 7), scores: (N,)
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    # Sort by score descending
    order = np.argsort(-scores)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        # Compute IoU of top box with remaining
        remaining = order[1:]
        ious = np.array([iou_3d_axis_aligned(boxes[i], boxes[j]) for j in remaining])

        # Keep boxes with IoU below threshold
        mask = ious < thresh
        order = remaining[mask]

    return keep


def detection_to_box7(det: dict) -> np.ndarray:
    """Convert a detection dict to [x, y, z, l, w, h, ry] array."""
    loc = det['location']
    dims = det['dimensions']   # [h, w, l] in KITTI format
    ry = det['rotation_y']
    # Reorder from KITTI [h, w, l] to [l, w, h]
    return np.array([loc[0], loc[1], loc[2], dims[2], dims[1], dims[0], ry])


def box7_to_detection(box7: np.ndarray, cls: str, score: float) -> dict:
    """Convert [x, y, z, l, w, h, ry] back to detection dict."""
    return {
        'type': cls,
        'location': box7[:3],
        'dimensions': np.array([box7[5], box7[4], box7[3]]),  # [h, w, l]
        'rotation_y': box7[6],
        'score': score,
    }
