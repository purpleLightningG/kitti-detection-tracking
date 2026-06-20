"""
kitti_utils.py — KITTI data loading, calibration parsing, and coordinate transforms.
"""
import numpy as np
import os


def load_velodyne(bin_path: str) -> np.ndarray:
    """Load KITTI velodyne point cloud from .bin file.
    Returns: (N, 4) array of [x, y, z, reflectance]
    """
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return points


def load_image(img_path: str) -> np.ndarray:
    """Load RGB image from KITTI image_2 folder."""
    import cv2
    img = cv2.imread(img_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_calib(calib_path: str) -> dict:
    """Parse KITTI calibration file into a dict of numpy arrays.
    Returns dict with keys: P0, P1, P2, P3, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo
    """
    raw = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, val = line.strip().split(':', 1)
            raw[key.strip()] = np.array([float(x) for x in val.strip().split()])

    calib = {}

    # Projection matrices (3x4)
    for k in ['P0', 'P1', 'P2', 'P3']:
        if k in raw:
            calib[k] = raw[k].reshape(3, 4)

    # Rectification rotation (3x3 -> 4x4 homogeneous)
    R0 = np.eye(4)
    if 'R0_rect' in raw and len(raw['R0_rect']) >= 9:
        R0[:3, :3] = raw['R0_rect'][:9].reshape(3, 3)
    calib['R0_rect'] = R0

    # Velodyne to camera (3x4 -> 4x4 homogeneous)
    Tr = np.eye(4)
    if 'Tr_velo_to_cam' in raw and len(raw['Tr_velo_to_cam']) >= 12:
        Tr[:3, :4] = raw['Tr_velo_to_cam'][:12].reshape(3, 4)
    calib['Tr_velo_to_cam'] = Tr

    # IMU to velodyne (optional)
    if 'Tr_imu_to_velo' in raw and len(raw['Tr_imu_to_velo']) >= 12:
        Tr_imu = np.eye(4)
        Tr_imu[:3, :4] = raw['Tr_imu_to_velo'][:12].reshape(3, 4)
        calib['Tr_imu_to_velo'] = Tr_imu

    return calib

def load_label(label_path: str) -> list:
    """Parse KITTI label_2 file.
    Returns list of dicts with keys:
      type, truncated, occluded, alpha, bbox_2d, dimensions, location, rotation_y
    """
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            label = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox_2d': np.array([float(x) for x in parts[4:8]]),   # [left, top, right, bottom]
                'dimensions': np.array([float(x) for x in parts[8:11]]),  # [h, w, l]
                'location': np.array([float(x) for x in parts[11:14]]),   # [x, y, z] in camera coords
                'rotation_y': float(parts[14]),
            }
            labels.append(label)
    return labels


def project_velo_to_image(points_3d: np.ndarray, calib: dict) -> np.ndarray:
    """Project velodyne points (N, 3) onto image plane.
    Returns (N, 2) pixel coordinates.
    """
    # Add homogeneous coordinate
    pts_hom = np.hstack([points_3d[:, :3], np.ones((points_3d.shape[0], 1))])

    # Velodyne -> camera -> rectified camera -> image
    P2 = calib['P2']                    # (3, 4)
    R0 = calib['R0_rect'][:3, :3]       # (3, 3)
    Tr = calib['Tr_velo_to_cam'][:3, :4]  # (3, 4)

    # Transform: image_coords = P2 @ R0 @ Tr @ pts
    cam_pts = Tr @ pts_hom.T            # (3, N)
    rect_pts = R0 @ cam_pts             # (3, N)
    img_pts = P2 @ np.vstack([rect_pts, np.ones((1, rect_pts.shape[1]))])  # (3, N)

    # Normalize by depth
    depth = img_pts[2, :]
    valid = depth > 0
    img_pts[:2, valid] /= depth[valid]

    return img_pts[:2, :].T, depth, valid


def get_frame_id(idx: int) -> str:
    """Convert integer frame index to KITTI 6-digit string."""
    return f"{idx:06d}"


def build_frame_path(data_root: str, subdir: str, frame_id: str, ext: str) -> str:
    """Build full path to a KITTI data file."""
    return os.path.join(data_root, subdir, f"{frame_id}.{ext}")


def corners_3d_from_label(label: dict) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box from KITTI label.
    Returns (8, 3) array of corner coordinates in camera frame.
    """
    h, w, l = label['dimensions']
    x, y, z = label['location']
    ry = label['rotation_y']

    # Rotation matrix around Y axis
    R = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # 3D box corners in object frame (centered at bottom-center)
    corners = np.array([
        [ l/2,  0,  w/2],
        [ l/2,  0, -w/2],
        [-l/2,  0, -w/2],
        [-l/2,  0,  w/2],
        [ l/2, -h,  w/2],
        [ l/2, -h, -w/2],
        [-l/2, -h, -w/2],
        [-l/2, -h,  w/2],
    ])

    # Rotate and translate to camera frame
    corners = (R @ corners.T).T + np.array([x, y, z])
    return corners
