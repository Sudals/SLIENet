import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from slienet import build_model, get_loaders, set_seed
from slienet.slie import slie_per_budget


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--batch-size", type=int, default=256)
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--data-dir", default="./data")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--out-json", default=None)
    args = pa.parse_args()

    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    head = ckpt.get("head", "mixedpool")
    model = build_model(arch, num_classes=100, head=head).to(dev)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    _, calib, te = get_loaders(args.batch_size, args.workers, args.data_dir)
    rows = slie_per_budget(model, calib, te, dev)

    print(f"\n  SLIE per-budget results: {arch} ({args.checkpoint})")
    print("  " + "-" * 70)
    print(f"  {'k':>2s} {'cost':>6s} {'single':>8s} {'SLIE':>8s} {'gain':>7s}  subset")
    print("  " + "-" * 70)
    for r in rows:
        gain = r["slie_acc"] - r["single_acc"]
        subset_str = "{" + ",".join(f"IC{i}" for i in r["slie_subset"]) + "}"
        print(f"  {r['k']:>2d} {r['cost']:6.3f} {r['single_acc']:8.3f} "
              f"{r['slie_acc']:8.3f} {gain:+7.3f}  {subset_str}")
    print("  " + "-" * 70)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"arch": arch, "checkpoint": args.checkpoint, "rows": rows},
                      f, indent=2)
        print(f"  Saved: {args.out_json}")


if __name__ == "__main__":
    main()
