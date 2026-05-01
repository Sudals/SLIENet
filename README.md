# SLIENet: Beyond FLOPs Train-Full, Deploy-Partial Multi-Exit Inference

Anonymous repository for double-blind NeurIPS 2026 submission. Author and citation
information will be added upon acceptance.

## Overview

This repository contains training, calibration, and deployment code for SLIENet,
a multi-exit framework that compiles into a single static TensorRT engine via
calibration-based selective ensemble of internal classifiers (ICs). Workstation
training is in PyTorch; on-device evaluation runs on NVIDIA Jetson Orin Nano with
TensorRT FP16 engines.

## Repository Structure

```
slienet/
├── data.py              # CIFAR-100 loaders + train/calibration/test split
├── heads.py             # MixedPool IC head
├── slie.py              # Calibration-based exhaustive subset search
├── train.py             # Joint training with light self-distillation (LSD) loss
├── models/
│   ├── __init__.py
│   └── resnet56.py      # ResNet-56 backbone with 6 IC heads
└── deploy/              # ONNX export + TensorRT engine compilation scripts
```

The current snapshot includes the full training, calibration, and deployment
pipeline for **ResNet-56**, which is the primary backbone used in the paper for
all main results: Table 1 (per-IC accuracy), Tables 4–5 (Jetson Orin Nano
latency, power, and energy), and Figure 4 (engine-call overhead). VGG-16-BN
and MobileNetV1 results in Tables 2, 10, and 11 are reported as
**generalization evidence** that the same SLIENet pipeline applies to other
backbones; their backbone definitions follow standard CIFAR-100 implementations
and are not part of the core contribution. The training and calibration code
in this repository is backbone-agnostic — adapting to a new backbone requires
only adding a new backbone class following the pattern in `models/resnet56.py`,
with no changes to `train.py`, `slie.py`, or `heads.py`.

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
python -m slienet.train --arch resnet56 --epochs 100 --batch-size 128 --seed 0
```

Reproduce the 5-seed mean reported in Table 1 of the paper:

```bash
for seed in 0 1 2 3 4; do
  python -m slienet.train --arch resnet56 --seed $seed
done
```

Each run takes approximately 1–2 hours on an NVIDIA RTX 4090.

### Calibration and SLIE subset selection

After training, compute the optimal IC subset on the held-out calibration split
(2,000 samples, disjoint from the test set):

```bash
python -m slienet.slie --checkpoint checkpoints/resnet56_seed0.pth
```

The exhaustive search over $2^k - 1$ subsets completes in under 1 s on a CPU
for $k \le 6$.

### Deployment (Jetson Orin Nano)

Export a trained checkpoint to ONNX truncated at depth $k$, then compile to
a single TensorRT FP16 engine:

```bash
# Export to ONNX (example: depth k=5, the recommended SLIE5 configuration)
python -m slienet.deploy.export_onnx \
    --checkpoint checkpoints/resnet56_seed0.pth \
    --depth 5 \
    --output slienet_d5.onnx

# Compile to TensorRT FP16 engine on Jetson Orin Nano
trtexec --onnx=slienet_d5.onnx --fp16 --saveEngine=slienet_d5.engine

# Run on-device latency / energy measurement over 10,000 test samples
python -m slienet.deploy.measure \
    --engine slienet_d5.engine \
    --test-samples 10000
```

Replace `--depth 5` with `4` or `6` to compile SLIE4 / SLIE6 configurations.

## Reproducibility Map

This repository targets reproduction of the **main results** of the paper:

- **Table 1** (ResNet-56 per-IC accuracy, 5 seeds): full training + calibration.
- **Tables 4 and 5** (Jetson Orin Nano deployment for SLIE4/5/6): on-device
  measurement via the deployment scripts above.
- **Figure 4** (engine-call count vs latency): TensorRT compilation in
  single-fused, partial-fused, and multi-engine variants of SLIE5/SLIE6.
- **Tables H.1, H.2** (Appendix H — hyperparameters and compute): documented
  directly in the paper appendix.

The VGG-16-BN and MobileNetV1 results in Tables 2, 10, and 11 are
**generalization evidence** showing that the same training and calibration
pipeline applies to other CIFAR-100 backbones. Reproducing them requires only
plugging in a standard backbone class; the rest of the pipeline (training loop,
self-distillation loss, calibration search, deployment export) is unchanged.

## Third-Party Assets

External datasets, model architectures, and frameworks used in this repository
are documented with their sources and licenses in `LICENSES.md`.

## License

The original code in this repository is released under the MIT License
(see `LICENSE`). Third-party assets retain their respective licenses; see
`LICENSES.md` for details.
