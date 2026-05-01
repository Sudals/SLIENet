import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from slienet import build_model, get_loaders, set_seed
from slienet.slie import acc_subset, collect_logits, slie_per_budget


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--mode", choices=["per_ic", "slie"], default="per_ic",
                    help="per_ic: per-IC test accuracy. "
                         "slie: calibration subset search + per-budget table.")
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

    if args.mode == "per_ic":
        ev_ic, _, ev_y = collect_logits(model, te, dev)
        K = len(model.ic_indices)
        rows = []
        print(f"\n  Per-IC accuracy: {arch} ({args.checkpoint})")
        print("  " + "-" * 36)
        print(f"  {'IC':>3s}  {'cost':>6s}  {'acc':>8s}")
        print("  " + "-" * 36)
        for k in range(K):
            a = acc_subset(ev_ic, [k], ev_y)
            rows.append({"k": k + 1, "cost": float(model.ic_costs[k]),
                         "acc": round(a, 3)})
            print(f"  {k+1:>3d}  {model.ic_costs[k]:6.3f}  {a:8.3f}")
        print("  " + "-" * 36)
    else:
        rows = slie_per_budget(model, calib, te, dev)
        print(f"\n  SLIE per-budget results: {arch} ({args.checkpoint})")
        print("  " + "-" * 70)
        print(f"  {'k':>2s}  {'cost':>6s}  {'single':>8s}  {'SLIE':>8s}  {'gain':>7s}  subset")
        print("  " + "-" * 70)
        for r in rows:
            gain = r["slie_acc"] - r["single_acc"]
            subset_str = "{" + ",".join(f"IC{i}" for i in r["slie_subset"]) + "}"
            print(f"  {r['k']:>2d}  {r['cost']:6.3f}  {r['single_acc']:8.3f}  "
                  f"{r['slie_acc']:8.3f}  {gain:+7.3f}  {subset_str}")
        print("  " + "-" * 70)

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump({"arch": arch, "checkpoint": args.checkpoint,
                       "mode": args.mode, "rows": rows}, f, indent=2)
        print(f"  Saved: {args.out_json}")


if __name__ == "__main__":
    main()
