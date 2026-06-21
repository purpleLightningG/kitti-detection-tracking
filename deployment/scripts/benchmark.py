"""
benchmark.py — Compare inference latency across PyTorch, ONNX Runtime, and TensorRT.

Loads multiple KITTI frames, runs each backend over them with warmup + multiple
trials, and reports the average per-frame latency and throughput.

Backends tested (whichever are available):
  - PyTorch (standalone)         — baseline FP32 CUDA inference
  - ONNX Runtime CUDA            — graph-optimized FP32
  - TensorRT FP32                — optimized engine, full precision
  - TensorRT FP16                — half-precision, fastest

Reports a table you can paste directly into the README.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import json
import numpy as np
import torch

from src import PointPillarsStandalone, load_pcdet_checkpoint_into_standalone


def voxelize_points(points, voxel_size, point_cloud_range, max_points=32, max_voxels=16000):
    """Simple numpy voxelization (reused from verify_standalone.py)."""
    x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]

    vx = ((points[:, 0] - x_min) / voxel_size[0]).astype(np.int32)
    vy = ((points[:, 1] - y_min) / voxel_size[1]).astype(np.int32)
    pillar_keys = vy * 1_000_000 + vx
    unique_keys, inverse, counts = np.unique(pillar_keys, return_inverse=True, return_counts=True)

    n_pillars = min(len(unique_keys), max_voxels)
    voxel_features = np.zeros((n_pillars, max_points, 4), dtype=np.float32)
    voxel_num_points = np.zeros(n_pillars, dtype=np.int32)
    coords = np.zeros((n_pillars, 4), dtype=np.int32)

    for pid in range(n_pillars):
        pts_in_pillar = points[inverse == pid]
        n_pts = min(len(pts_in_pillar), max_points)
        voxel_features[pid, :n_pts] = pts_in_pillar[:n_pts]
        voxel_num_points[pid] = n_pts
        coords[pid, 2] = vy[inverse == pid][0]
        coords[pid, 3] = vx[inverse == pid][0]

    return voxel_features, voxel_num_points, coords


def benchmark_pytorch(model, frames, warmup=10):
    """Benchmark the standalone PyTorch model."""
    device = next(model.parameters()).device
    torch.cuda.synchronize()

    # Warmup
    for i in range(warmup):
        vf, vnp, c = frames[i % len(frames)]
        vf_t = torch.from_numpy(vf).to(device)
        vnp_t = torch.from_numpy(vnp).to(device)
        c_t = torch.from_numpy(c).to(device)
        with torch.no_grad():
            _ = model(vf_t, vnp_t, c_t)
    torch.cuda.synchronize()

    # Time
    times = []
    for vf, vnp, c in frames:
        vf_t = torch.from_numpy(vf).to(device)
        vnp_t = torch.from_numpy(vnp).to(device)
        c_t = torch.from_numpy(c).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(vf_t, vnp_t, c_t)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000  # ms


def benchmark_onnxruntime(onnx_path, frames, warmup=10):
    """Benchmark with ONNX Runtime CUDA provider."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  ONNX Runtime not installed (pip install onnxruntime-gpu)")
        return None

    print(f"  Loading ONNX Runtime session...")
    sess = ort.InferenceSession(
        onnx_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    )
    inp_names = [i.name for i in sess.get_inputs()]

    # Warmup
    for i in range(warmup):
        vf, vnp, c = frames[i % len(frames)]
        sess.run(None, dict(zip(inp_names, [vf, vnp, c])))

    times = []
    for vf, vnp, c in frames:
        t0 = time.perf_counter()
        sess.run(None, dict(zip(inp_names, [vf, vnp, c])))
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000


def benchmark_tensorrt(engine_path, frames, warmup=10):
    """Benchmark a TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except ImportError:
        print(f"  Skipping {engine_path}: TensorRT + pycuda not available")
        return None

    if not os.path.exists(engine_path):
        print(f"  Skipping {engine_path}: file not found")
        return None

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # Pre-allocate output buffers for the largest expected output
    # (we infer shapes at runtime, but allocate generously)
    # In practice you'd query the engine's output shapes per input shape; keeping
    # this simple for the benchmark.

    # Warmup
    for i in range(warmup):
        vf, vnp, c = frames[i % len(frames)]
        _run_trt_inference(context, engine, vf, vnp, c)

    times = []
    for vf, vnp, c in frames:
        t0 = time.perf_counter()
        _run_trt_inference(context, engine, vf, vnp, c)
        times.append(time.perf_counter() - t0)
    return np.array(times) * 1000


def _run_trt_inference(context, engine, vf, vnp, c):
    """Helper for one TRT inference call. Simplified — real prod code would reuse buffers."""
    import pycuda.driver as cuda

    bindings = []
    inputs_outputs = []

    # Inputs
    for i, arr in enumerate([vf, vnp, c]):
        d_in = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_in, arr)
        bindings.append(int(d_in))
        inputs_outputs.append(d_in)
        # Set shape for dynamic axis
        context.set_input_shape(engine.get_tensor_name(i), arr.shape)

    # Outputs
    output_shapes = []
    for i in range(3, engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(context.get_tensor_shape(name))
        size = int(np.prod(shape)) * 4  # float32
        d_out = cuda.mem_alloc(size)
        bindings.append(int(d_out))
        inputs_outputs.append(d_out)
        output_shapes.append(shape)

    context.execute_v2(bindings=bindings)

    for d in inputs_outputs:
        d.free()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcdet-ckpt', required=True)
    parser.add_argument('--onnx', default='exported/PointPillars.onnx')
    parser.add_argument('--engines-dir', default='exported')
    parser.add_argument('--data-root',
                        default='C:/Users/Dream Team/Desktop/kitti_velodyne/training')
    parser.add_argument('--num-frames', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', default='benchmark_results.json')
    args = parser.parse_args()

    # Build standalone model
    print("Loading standalone PointPillars...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PointPillarsStandalone().to(device).eval()
    load_pcdet_checkpoint_into_standalone(args.pcdet_ckpt, model, verbose=False)

    # Pre-voxelize a batch of KITTI frames
    print(f"Pre-voxelizing {args.num_frames} KITTI frames...")
    velodyne_dir = os.path.join(args.data_root, 'velodyne')
    frame_files = sorted(os.listdir(velodyne_dir))[:args.num_frames]
    frames = []
    for f in frame_files:
        points = np.fromfile(os.path.join(velodyne_dir, f), dtype=np.float32).reshape(-1, 4)
        vf, vnp, c = voxelize_points(
            points,
            voxel_size=model.voxel_size,
            point_cloud_range=model.point_cloud_range,
        )
        frames.append((vf, vnp, c))
    avg_pillars = np.mean([len(f[0]) for f in frames])
    print(f"  Avg pillars per frame: {avg_pillars:.0f}")

    results = {}

    print("\n=== PyTorch (standalone, FP32) ===")
    times = benchmark_pytorch(model, frames, warmup=args.warmup)
    results['pytorch_fp32'] = {'mean_ms': float(times.mean()),
                                'std_ms': float(times.std()),
                                'fps': float(1000 / times.mean())}
    print(f"  Mean: {times.mean():.2f} ms | FPS: {1000 / times.mean():.1f}")

    if os.path.exists(args.onnx):
        print("\n=== ONNX Runtime (CUDA, FP32) ===")
        times = benchmark_onnxruntime(args.onnx, frames, warmup=args.warmup)
        if times is not None:
            results['onnxruntime_fp32'] = {'mean_ms': float(times.mean()),
                                            'std_ms': float(times.std()),
                                            'fps': float(1000 / times.mean())}
            print(f"  Mean: {times.mean():.2f} ms | FPS: {1000 / times.mean():.1f}")

    for prec in ('fp32', 'fp16'):
        engine_path = os.path.join(args.engines_dir, f'PointPillars_{prec}.engine')
        print(f"\n=== TensorRT {prec.upper()} ===")
        times = benchmark_tensorrt(engine_path, frames, warmup=args.warmup)
        if times is not None:
            results[f'tensorrt_{prec}'] = {'mean_ms': float(times.mean()),
                                            'std_ms': float(times.std()),
                                            'fps': float(1000 / times.mean())}
            print(f"  Mean: {times.mean():.2f} ms | FPS: {1000 / times.mean():.1f}")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary table for README
    print("\n=== Summary table for README ===\n")
    print("| Backend | Precision | Latency (ms) | Throughput (FPS) |")
    print("|---|---|---|---|")
    for k, v in results.items():
        backend, prec = k.rsplit('_', 1)
        print(f"| {backend} | {prec.upper()} | {v['mean_ms']:.2f} | {v['fps']:.1f} |")


if __name__ == '__main__':
    main()
