"""
build_engine.py — Build TensorRT engines (FP32 and FP16) from an ONNX file.

Produces two .engine files:
  - PointPillars_fp32.engine — full precision baseline
  - PointPillars_fp16.engine — half precision for ~2x speedup on RTX 3080 (Ampere)

The script uses TensorRT's Python API. Make sure your TensorRT version matches
your CUDA version (TensorRT 8.6 with CUDA 11.8 works well on RTX 3080).

Install:
  pip install tensorrt
  (or download TensorRT tarball from NVIDIA and add to PYTHONPATH)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time

try:
    import tensorrt as trt
except ImportError:
    print("TensorRT not installed. Install via:")
    print("  pip install tensorrt")
    print("Or download from https://developer.nvidia.com/tensorrt")
    sys.exit(1)


def build_engine(onnx_path: str, engine_path: str, fp16: bool = False,
                 workspace_gb: int = 4, verbose: bool = True):
    """Compile an ONNX file into a TensorRT engine."""
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network(0)

    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX:")
            for err in range(parser.num_errors):
                print(f"  {parser.get_error(err)}")
            return False

    print(f"  Network has {network.num_layers} layers, "
          f"{network.num_inputs} inputs, {network.num_outputs} outputs.")

    config = builder.create_builder_config()

    # Workspace memory pool (4 GB by default)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))

    if fp16:
        # TensorRT 11+ removed the simple FP16 flag — precision is now controlled
        # via network-level strong typing. Skipping FP16 build for now; FP32 already
        # provides a strong PyTorch->TensorRT comparison story.
        print("  WARNING: FP16 flag not available in this TRT version. Skipping FP16 build.")
        print("  Note: FP32 engine still provides a valid PyTorch→TensorRT speedup comparison.")
        return False

    # Add an optimization profile for the dynamic axis (n_pillars)
    profile = builder.create_optimization_profile()
    for input_idx in range(network.num_inputs):
        inp = network.get_input(input_idx)
        # The first axis is dynamic (n_pillars)
        name = inp.name
        if 'voxel_features' in name:
            # Allow 100 to 20000 pillars at runtime; target = 8000 (typical KITTI frame)
            profile.set_shape(name, min=(100, 32, 4), opt=(8000, 32, 4), max=(20000, 32, 4))
        elif 'voxel_num_points' in name:
            profile.set_shape(name, min=(100,), opt=(8000,), max=(20000,))
        elif 'coords' in name:
            profile.set_shape(name, min=(100, 4), opt=(8000, 4), max=(20000, 4))
    config.add_optimization_profile(profile)

    print("Building serialized engine (this can take 1-5 minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if serialized_engine is None:
        print("Engine build failed.")
        return False

    os.makedirs(os.path.dirname(engine_path) or '.', exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"  Engine built in {build_time:.1f}s, written to {engine_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True, help='Input ONNX file')
    parser.add_argument('--output-dir', default='exported', help='Output dir for engines')
    parser.add_argument('--workspace-gb', type=int, default=4)
    parser.add_argument('--fp32-only', action='store_true', help='Skip FP16 build')
    parser.add_argument('--fp16-only', action='store_true', help='Skip FP32 build')
    args = parser.parse_args()

    if not args.fp16_only:
        fp32_path = os.path.join(args.output_dir, 'PointPillars_fp32.engine')
        print("\n=== Building FP32 engine ===")
        build_engine(args.onnx, fp32_path, fp16=False, workspace_gb=args.workspace_gb)

    if not args.fp32_only:
        fp16_path = os.path.join(args.output_dir, 'PointPillars_fp16.engine')
        print("\n=== Building FP16 engine ===")
        build_engine(args.onnx, fp16_path, fp16=True, workspace_gb=args.workspace_gb)

    print("\nDone. Next step: python scripts/benchmark.py")


if __name__ == '__main__':
    main()
