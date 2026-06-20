"""
make_demo_gif.py — Generate a side-by-side rotating demo GIF for KITTI detection.
Left: static camera image with projected 3D boxes.
Right: rotating 3D LiDAR point cloud with 3D detection boxes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import yaml
import time
import numpy as np
import cv2
import open3d as o3d
import imageio
from tqdm import tqdm

from kitti_utils import (
    load_velodyne, load_image, load_calib,
    get_frame_id, build_frame_path, corners_3d_from_label
)
from vis_utils import draw_boxes_on_image, get_track_color


def make_camera_panel(frame_id, data_root, cfg, tracks_dir):
    """Render the camera image with projected 3D boxes."""
    img_path = build_frame_path(data_root, cfg['data']['image_dir'], frame_id, 'png')
    calib_path = build_frame_path(data_root, cfg['data']['calib_dir'], frame_id, 'txt')

    img = load_image(img_path)  # RGB
    calib = load_calib(calib_path)

    # Load detections
    track_path = os.path.join(tracks_dir, f"{frame_id}.json")
    tracks = []
    if os.path.exists(track_path):
        with open(track_path) as f:
            tracks = json.load(f)
        for t in tracks:
            t['location'] = np.array(t['location'])
            t['dimensions'] = np.array(t['dimensions'])

    # Draw boxes on the camera image
    img_with_boxes = draw_boxes_on_image(img, tracks, calib)

    # Convert RGB to BGR for cv2
    return cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)


def build_o3d_geometry(frame_id, data_root, cfg, tracks_dir):
    """Build Open3D point cloud + 3D bounding box geometries."""
    # Load LiDAR
    bin_path = build_frame_path(data_root, cfg['data']['velodyne_dir'], frame_id, 'bin')
    points = load_velodyne(bin_path)[:, :3]

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Color by height (z) for depth perception
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.stack([z_norm * 0.6 + 0.2,
                       np.ones_like(z_norm) * 0.7,
                       (1 - z_norm) * 0.7 + 0.3], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Load detections
    track_path = os.path.join(tracks_dir, f"{frame_id}.json")
    boxes = []
    if os.path.exists(track_path):
        with open(track_path) as f:
            tracks = json.load(f)

        # Load calibration to convert camera coords -> lidar coords
        calib_path = build_frame_path(data_root, cfg['data']['calib_dir'], frame_id, 'txt')
        calib = load_calib(calib_path)
        Tr = calib['Tr_velo_to_cam'][:3, :4]   # 3x4
        # Invert: lidar = R^T (cam - t)
        R = Tr[:, :3]
        t = Tr[:, 3]
        R_inv = R.T
        t_inv = -R_inv @ t

        for tr in tracks:
            cls = tr['type']
            loc_cam = np.array(tr['location'])
            dims = np.array(tr['dimensions'])  # [h, w, l] in KITTI
            ry = tr['rotation_y']

            # Convert location to LiDAR frame
            loc_lidar = R_inv @ loc_cam + t_inv
            # Object center is at bottom of bbox in camera frame, lift up by h/2
            loc_lidar[2] += dims[0] / 2

            # Build Open3D oriented bounding box
            extent = np.array([dims[2], dims[1], dims[0]])  # [l, w, h] for o3d

            # Rotation around z-axis in lidar frame
            R_box = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_xyz(
                [0, 0, -ry]
            )

            box = o3d.geometry.OrientedBoundingBox(
                center=loc_lidar,
                R=R_box,
                extent=extent
            )

            # Color by class
            color_map = {'Car': [0, 1, 0], 'Pedestrian': [1, 1, 0], 'Cyclist': [1, 0.6, 0]}
            box.color = color_map.get(cls, [0, 1, 0])
            boxes.append(box)

    return pcd, boxes


def capture_rotating_frames(pcd, boxes, num_frames=60, width=800, height=600):
    """Capture frames as the point cloud rotates around vertical axis."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    vis.add_geometry(pcd)
    for box in boxes:
        vis.add_geometry(box)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.04, 0.04, 0.05])
    opt.point_size = 1.5

    ctr = vis.get_view_control()
    ctr.set_zoom(0.18)
    ctr.set_front([-1, 0, 0.5])
    ctr.set_up([0, 0, 1])

    frames = []
    rotation_per_frame = 360 / num_frames  # full rotation across all frames

    for i in tqdm(range(num_frames), desc="Capturing rotation"):
        # Open3D's rotate function uses internal units (~0.003 deg per unit)
        ctr.rotate(rotation_per_frame * 1, 0.0)
        vis.poll_events()
        vis.update_renderer()

        # Capture
        img = vis.capture_screen_float_buffer(False)
        img_np = (np.asarray(img) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        frames.append(img_bgr)

    vis.destroy_window()
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Frame ID, e.g. 004139")
    parser.add_argument("--tracks", default="outputs/detections", help="Detections or tracks dir")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-frames", type=int, default=60, help="GIF frame count")
    parser.add_argument("--output", default="outputs/vis/demo.gif")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_root = cfg['data']['root']

    print(f"Building demo GIF for frame {args.frame}...")

    # 1. Render camera image (static)
    print("Rendering camera panel...")
    cam_img = make_camera_panel(args.frame, data_root, cfg, args.tracks)

    # 2. Build Open3D scene and capture rotating frames
    print("Building Open3D scene...")
    pcd, boxes = build_o3d_geometry(args.frame, data_root, cfg, args.tracks)
    print(f"  Point cloud: {len(np.asarray(pcd.points))} points")
    print(f"  Boxes: {len(boxes)}")

    lidar_frames = capture_rotating_frames(pcd, boxes, num_frames=args.num_frames,
                                            width=600, height=480)

    # 3. Stitch camera image + rotating lidar side-by-side
    # Camera shrunk, LiDAR enlarged so the rotating 3D view gets more visual weight
    print("Stitching frames...")
    target_h = 480

    # Camera at HALF height, padded with black to align vertically
    cam_h, cam_w = cam_img.shape[:2]
    cam_target_h = target_h // 2
    cam_scale = cam_target_h / cam_h
    cam_resized = cv2.resize(cam_img, (int(cam_w * cam_scale), cam_target_h))
    # Pad camera with black on top and bottom to match target_h
    pad_top = (target_h - cam_target_h) // 2
    pad_bottom = target_h - cam_target_h - pad_top
    cam_padded = cv2.copyMakeBorder(
        cam_resized, pad_top, pad_bottom, 0, 0,
        cv2.BORDER_CONSTANT, value=(10, 10, 13)
    )

    combined_frames = []
    for lidar_frame in lidar_frames:
        lh, lw = lidar_frame.shape[:2]
        l_scale = target_h / lh
        lidar_resized = cv2.resize(lidar_frame, (int(lw * l_scale), target_h))

        combined = np.hstack([cam_padded, lidar_resized])
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        combined_frames.append(combined_rgb)
    # 4. Save GIF
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Writing GIF to {args.output}...")
    imageio.mimsave(args.output, combined_frames, fps=12, loop=0, palettesize=64)

    # Report size
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done. GIF size: {size_mb:.2f} MB")
    if size_mb > 8:
        print("  WARNING: GIF is large. Reduce --num-frames or window size.")


if __name__ == "__main__":
    main()