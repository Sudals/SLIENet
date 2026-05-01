import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision


def load_engine(path):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--engine", required=True,
                    help="TRT engine path, or comma-separated list for "
                         "multi/partial-fused. Latency sums over engines; "
                         "softmax outputs averaged.")
    pa.add_argument("--test-samples", type=int, default=10000)
    pa.add_argument("--batch-size", type=int, default=1)
    pa.add_argument("--data-dir", default="./data")
    pa.add_argument("--warmup", type=int, default=200)
    pa.add_argument("--out-json", default=None)
    args = pa.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT inference.")
    dev = torch.device("cuda")

    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    ds = torchvision.datasets.CIFAR100(args.data_dir, False,
                                       transform=tf, download=True)
    n = (min(args.test_samples, len(ds)) // args.batch_size) * args.batch_size
    images = torch.stack([ds[i][0] for i in range(n)]).to(dev).contiguous()
    labels = torch.tensor([ds[i][1] for i in range(n)], device=dev)

    engine_paths = [p.strip() for p in args.engine.split(",") if p.strip()]
    contexts, out_bufs = [], []
    for p in engine_paths:
        eng = load_engine(p)
        ctx = eng.create_execution_context()
        in_idx = eng.get_binding_index("input")
        ctx.set_binding_shape(in_idx, (args.batch_size, 3, 32, 32))
        contexts.append(ctx)
        out_bufs.append(torch.empty((args.batch_size, 100),
                                     dtype=torch.float32, device=dev))

    def run_all(batch):
        sm = None
        for ctx, ob in zip(contexts, out_bufs):
            bindings = [int(batch.data_ptr()), int(ob.data_ptr())]
            ctx.execute_v2(bindings)
            sm = ob.clone() if sm is None else sm + ob
        return sm / float(len(contexts))

    warm = images[:args.batch_size]
    for _ in range(args.warmup):
        run_all(warm)
    torch.cuda.synchronize()

    latencies = []
    correct = 0
    for i in range(0, n, args.batch_size):
        batch = images[i:i + args.batch_size]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sm = run_all(batch)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        preds = sm.argmax(dim=1)
        correct += int((preds == labels[i:i + args.batch_size]).sum().item())

    lats = np.array(latencies)
    p50 = float(np.percentile(lats, 50))
    p99 = float(np.percentile(lats, 99))
    pmax = float(lats.max())
    fps = (1000.0 / p50) * args.batch_size
    acc = 100.0 * correct / n

    print(f"  Engines    : {len(engine_paths)}  ({', '.join(engine_paths)})")
    print(f"  Samples    : {n}  (batch={args.batch_size})")
    print(f"  Latency    : p50={p50:.3f} ms   p99={p99:.3f} ms   max={pmax:.3f} ms")
    print(f"  Throughput : {fps:.1f} FPS")
    print(f"  Accuracy   : {acc:.2f}%")
    print(f"  (Power/energy: capture tegrastats during this run; see paper appendix.)")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({
                "engines": engine_paths, "samples": n,
                "batch_size": args.batch_size,
                "p50_ms": p50, "p99_ms": p99, "max_ms": pmax,
                "fps": fps, "accuracy_pct": acc,
            }, f, indent=2)
        print(f"  Saved      : {args.out_json}")


if __name__ == "__main__":
    main()
