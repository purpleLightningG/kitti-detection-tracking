"""
visualize.py — Generate BEV, camera overlay, or GIF visualizations.

Usage:
  python scripts/visualize.py --mode bev --tracks outputs/tracks/ --frame 000042
  python scripts/visualize.py --mode camera --tracks outputs/tracks/ --frame 000042
  python scripts/visualize.py --mode gif --tracks outputs/tracks/ --start 0 --end 100
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import json
import yaml
import numpy as np
import cv2
from tqdm import tqdm

from kitti_utils import load_velodyne, load_image, load_calib, get_frame_id, build_frame_path
from vis_utils import draw_bev, draw_boxes_on_image


def main():
    parser = argparse.ArgumentParser(description="Visualize detection/tracking results")
    parser.add_argument("--mode", choices=["bev", "camera", "gif"], required=True)
    parser.add_argument("--tracks", default="outputs/tracks", help="Dir with tracked results")
    parser.add_argument("--frame", type=str, help="Frame ID for single-frame modes")
    parser.add_argument("--start", type=int, default=0, help="Start frame for GIF mode")
    parser.add_argument("--end", type=int, default=100, help="End frame for GIF mode")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--save-dir", default="outputs/vis")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = cfg['data']['root']
    vis_cfg = cfg.get('vis', {})
    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == "bev":
        # Single-frame BEV visualization
        assert args.frame, "Specify --frame for BEV mode"
        visualize_bev(args.frame, args.tracks, data_root, cfg, vis_cfg, args.save_dir)

    elif args.mode == "camera":
        # Single-frame camera overlay
        assert args.frame, "Specify --frame for camera mode"
        visualize_camera(args.frame, args.tracks, data_root, cfg, vis_cfg, args.save_dir)

    elif args.mode == "gif":
        # Generate GIF from a range of frames
        generate_gif(args.start, args.end, args.tracks, data_root, cfg, vis_cfg, args.save_dir)


def visualize_bev(frame_id, tracks_dir, data_root, cfg, vis_cfg, save_dir):
    """Generate and save a BEV visualization for one frame."""
    # Load point cloud
    bin_path = build_frame_path(data_root, cfg['data']['velodyne_dir'], frame_id, 'bin')
    points = load_velodyne(bin_path)

    # Load tracked detections
    track_path = os.path.join(tracks_dir, f"{frame_id}.json")
    tracks = []
    if os.path.exists(track_path):
        with open(track_path) as f:
            tracks = json.load(f)
        # Convert to numpy
        for t in tracks:
            t['location'] = np.array(t['location'])
            t['dimensions'] = np.array(t['dimensions'])

    # Draw BEV
    bev_range = vis_cfg.get('bev_range', [-40, -40, 40, 40])
    resolution = vis_cfg.get('bev_resolution', 0.05)
    bev = draw_bev(points, tracks, bev_range, resolution)

    # Save
    save_path = os.path.join(save_dir, f"bev_{frame_id}.png")
    cv2.imwrite(save_path, bev)
    print(f"BEV saved to {save_path}")

    # Display
    cv2.imshow("BEV", bev)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_camera(frame_id, tracks_dir, data_root, cfg, vis_cfg, save_dir):
    """Generate and save a camera overlay visualization for one frame."""
    # Load image and calibration
    img_path = build_frame_path(data_root, cfg['data']['image_dir'], frame_id, 'png')
    calib_path = build_frame_path(data_root, cfg['data']['calib_dir'], frame_id, 'txt')

    img = load_image(img_path)
    calib = load_calib(calib_path)

    # Load tracked detections
    track_path = os.path.join(tracks_dir, f"{frame_id}.json")
    tracks = []
    if os.path.exists(track_path):
        with open(track_path) as f:
            tracks = json.load(f)
        for t in tracks:
            t['location'] = np.array(t['location'])
            t['dimensions'] = np.array(t['dimensions'])

    # Draw boxes on image
    img_with_boxes = draw_boxes_on_image(img, tracks, calib)

    # Save (convert RGB to BGR for cv2)
    save_path = os.path.join(save_dir, f"cam_{frame_id}.png")
    cv2.imwrite(save_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    print(f"Camera overlay saved to {save_path}")

    # Display
    cv2.imshow("Camera", cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_gif(start, end, tracks_dir, data_root, cfg, vis_cfg, save_dir):
    """Generate a GIF from a sequence of BEV frames."""
    import imageio

    bev_range = vis_cfg.get('bev_range', [-40, -40, 40, 40])
    resolution = vis_cfg.get('bev_resolution', 0.05)

    frames = []
    for idx in tqdm(range(start, end + 1), desc="Generating GIF frames"):
        fid = get_frame_id(idx)

        # Load point cloud
        bin_path = build_frame_path(data_root, cfg['data']['velodyne_dir'], fid, 'bin')
        if not os.path.exists(bin_path):
            continue
        points = load_velodyne(bin_path)

        # Load tracks
        track_path = os.path.join(tracks_dir, f"{fid}.json")
        tracks = []
        if os.path.exists(track_path):
            with open(track_path) as f:
                tracks = json.load(f)
            for t in tracks:
                t['location'] = np.array(t['location'])
                t['dimensions'] = np.array(t['dimensions'])

        # Draw BEV
        bev = draw_bev(points, tracks, bev_range, resolution)

        # Add frame number label
        cv2.putText(bev, f"Frame {fid}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        frames.append(bev)

    # Save as GIF
    gif_path = os.path.join(save_dir, "tracking_demo.gif")
    imageio.mimsave(gif_path, frames, fps=10, loop=0)
    print(f"GIF saved to {gif_path} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
