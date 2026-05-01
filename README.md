# SLIE-Net

**Selective Lightweight IC Ensemble** on a self-distilled multi-exit ResNet-56
for CIFAR-100.

The multi-exit backbone is trained once with light self-distillation (the
final classifier acts as a stop-gradient teacher for each IC); at inference,
SLIE picks a subset of the already-computed internal classifiers (ICs) via
exhaustive search on a held-out calibration split and averages their
softmaxes. No backbone or IC weights are modified post-training — mid-IC
accuracy is byte-preserved, final accuracy improves.

## Backbone

| arch | IC head | params |
|---|---|---|
| `resnet56` | MixedPoolIC | ~0.9 M |

Six IC heads are attached after residual blocks `[4, 8, 12, 16, 20, 24]`,
i.e. cumulative-FLOPs positions of approximately
`{0.19, 0.34, 0.48, 0.63, 0.77, 0.92}` of the backbone. Training recipe:
SGD + cosine, 100 epochs, CE + self-distillation from the final classifier
(α=0.3, T=4).

## Install

```bash
pip install -r requirements.txt
```

CIFAR-100 is downloaded automatically to `./data/` on first run.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --seed 0
```

Checkpoints land in `./checkpoints/slienet_resnet56_s<seed>.pth` along with
a `_history.json` of per-epoch IC accuracies.

Flags: `--epochs`, `--kd-alpha`, `--kd-temp`, `--lr`, `--wd`, `--batch-size`,
`--workers`, `--data-dir`, `--save-dir`, `--tag`.

## Evaluate (SLIE per-budget table)

```bash
python scripts/eval.py --checkpoint checkpoints/slienet_resnet56_s0.pth
```

Example output:

```
  SLIE per-budget results: resnet56 (checkpoints/slienet_resnet56_s0.pth)
  ----------------------------------------------------------------------
   k   cost   single     SLIE    gain   subset
  ----------------------------------------------------------------------
   1  0.188   46.11    46.11   +0.00   {IC1}
   2  0.339   50.94    53.12   +2.18   {IC1,IC2}
   ...
   6  0.925   70.10    71.89   +1.79   {IC2,IC5,IC6}
  ----------------------------------------------------------------------
```

`single` = plain IC_k hard-cap accuracy; `SLIE` = calibration-exhaustive
subset average from IC_1..IC_k (enumerates all 2^k − 1 non-empty subsets);
`subset` = the indices SLIE picked for that budget.

## Package layout

```
release/
├── requirements.txt
├── README.md
├── slienet/
│   ├── __init__.py
│   ├── data.py              # CIFAR-100 loaders + train-split calib carve-out
│   ├── heads.py             # MixedPoolIC
│   ├── train.py             # CE + self-distillation trainer
│   ├── slie.py              # exhaustive subset selection + per-budget eval
│   └── models/
│       ├── __init__.py
│       └── resnet56.py      # SLIE_ResNet56
└── scripts/
    ├── train.py             # CLI: train ResNet-56
    └── eval.py              # CLI: per-budget SLIE table
```

## Reproducibility

- Calibration split: 2,000 samples carved out of the CIFAR-100 **training**
  set with a fixed split seed (`slienet.data.get_loaders`); the 10K test set
  is used intact for final evaluation. Dcal ⊂ Dtrain and Dcal ∩ Dtest = ∅.
- Training: deterministic cuDNN, seeded via `slienet.set_seed(seed)`.
- Three seeds (`--seed 0/1/2`) are sufficient for the headline numbers.
