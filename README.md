# SLIENet: Beyond FLOPs — Train-Full, Deploy-Partial Multi-Exit Inference

Anonymous repository for double-blind NeurIPS 2026 submission. Author and
citation information will be added upon acceptance.

## Overview

This repository contains training, calibration, and deployment code for SLIENet,
a multi-exit framework that compiles into a single static TensorRT engine via
calibration-based selective ensemble of internal classifiers (ICs). Workstation
training is in PyTorch; on-device evaluation runs on NVIDIA Jetson Orin Nano
with TensorRT FP16 engines.

## Repository Structure

```
.
├── slienet/                  # Library code (importable modules)
│   ├── data.py               # CIFAR-100 loaders + train/calibration/test split
│   ├── heads.py              # MixedPool IC head
│   ├── slie.py               # Calibration-based exhaustive subset search
│   ├── train.py              # Training loop with light self-distillation (LSD)
│   └── models/
│       └── resnet56.py       # ResNet-56 backbone with 6 IC heads
├── scripts/                  # Command-line entry points
│   ├── train.py              # CLI: train a backbone with given seed
│   ├── eval.py               # CLI: per-IC evaluation on the test set
│   ├── export_onnx.py        # CLI: export a truncated checkpoint to ONNX
│   ├── build_trt.sh          # Shell: compile ONNX → TensorRT FP16 engine
│   └── measure_trt.py        # CLI: latency / power / energy on Jetson
├── README.md
└── LICENSES.md               # Third-party assets and licenses
```

The current snapshot includes the full training, calibration, and deployment
pipeline for **ResNet-56**, the primary backbone used in the paper for all main
results: Table 1 (per-IC accuracy), Tables 4–5 (Jetson Orin Nano latency,
power, and energy), and Figure 4 (engine-call overhead). The training and
calibration code is backbone-agnostic — adapting to a new backbone requires
only adding a new backbone class following the pattern in
`slienet/models/resnet56.py`, with no changes to `slienet/train.py`,
`slienet/slie.py`, or `slienet/heads.py`.

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA 11.8+
- torchvision (for CIFAR-100)
- NumPy
- (Deployment only) TensorRT 8.x and ONNX, on a Jetson Orin Nano with JetPack 5.x

```bash
pip install -r requirements.txt
```

## Dataset

CIFAR-100 is downloaded automatically to `./data/` on first run by `torchvision`.
No manual setup required.

## Reproducing Results

### Workstation training (ResNet-56)

Train SLIENet on ResNet-56 with one seed:

```bash
python scripts/train.py --arch resnet56 --epochs 100 --batch-size 128 --seed 0
```

Reproduce the 5-seed mean reported in Table 1 of the paper:

```bash
for seed in 0 1 2 3 4; do
  python scripts/train.py --arch resnet56 --seed $seed
done
```

Each run takes approximately 1–2 hours on an NVIDIA RTX 4090.

### Per-IC evaluation

```bash
python scripts/eval.py --checkpoint checkpoints/resnet56_seed0.pth
```

### Calibration and SLIE subset selection

The calibration-based subset search is invoked from the same evaluation script
by selecting the corresponding mode:

```bash
python scripts/eval.py --checkpoint checkpoints/resnet56_seed0.pth --mode slie
```

The exhaustive search over 2^k − 1 subsets completes in under 1 s on a CPU
for k ≤ 6.

### Deployment (Jetson Orin Nano)

Three-step pipeline: export to ONNX, compile to TensorRT FP16, then measure
on-device latency, power, and energy.

```bash
# 1. Export a trained checkpoint truncated at depth k
#    (example: k=5, the recommended SLIE5 configuration)
python scripts/export_onnx.py \
    --checkpoint checkpoints/resnet56_seed0.pth \
    --depth 5 \
    --output slienet_d5.onnx

# 2. Compile ONNX to a single TensorRT FP16 engine on Jetson Orin Nano
bash scripts/build_trt.sh slienet_d5.onnx slienet_d5.engine

# 3. Run on-device latency / power / energy measurement
#    (10,000 CIFAR-100 test samples; p50 / p99 / max latency, FPS, mJ/inference)
python scripts/measure_trt.py \
    --engine slienet_d5.engine \
    --test-samples 10000
```

Replace `--depth 5` with `4` or `6` to compile SLIE4 or SLIE6 configurations
reported in Tables 4 and 5.

## Reproducibility Map

This repository targets reproduction of the **main results** of the paper:

- **Table 1** (ResNet-56 per-IC accuracy, 5 seeds): full training + calibration
  via `scripts/train.py` and `scripts/eval.py`.
- **Tables 4 and 5** (Jetson Orin Nano deployment for SLIE4/5/6): on-device
  measurement via the three-step deployment pipeline above.
- **Figure 4** (engine-call count vs latency): TensorRT compilation in
  single-fused, partial-fused, and multi-engine variants of SLIE5/SLIE6 via
  `scripts/build_trt.sh` and `scripts/measure_trt.py`.
- **Table 12 (Hyperparameters), Appendix H.2 (Compute)**: documented directly
  in the paper appendix.

The VGG-16-BN and MobileNetV1 results in Tables 2, 10, and 11 are
**generalization evidence** showing that the same training and calibration
pipeline applies to other CIFAR-100 backbones. Reproducing them requires only
adding a backbone class under `slienet/models/`; the rest of the pipeline
(training loop, self-distillation loss, calibration search, deployment export)
is unchanged.

## Third-Party Assets

External datasets, model architectures, and frameworks used in this repository
are documented with their sources and licenses in `LICENSES.md`.

## License

The original code in this repository is released under the MIT License
(see `LICENSE`). Third-party assets retain their respective licenses; see
`LICENSES.md` for details.
