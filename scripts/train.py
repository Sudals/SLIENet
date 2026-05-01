import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from slienet import set_seed
from slienet.train import train_slienet


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--arch", default="resnet56", choices=["resnet56"])
    pa.add_argument("--head", default="mixedpool", choices=["mixedpool"])
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch-size", type=int, default=128)
    pa.add_argument("--lr", type=float, default=0.1)
    pa.add_argument("--wd", type=float, default=1e-4)
    pa.add_argument("--kd-alpha", type=float, default=0.3)
    pa.add_argument("--kd-temp", type=float, default=4.0)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--data-dir", default="./data")
    pa.add_argument("--save-dir", default="./checkpoints")
    pa.add_argument("--tag", default=None)
    args = pa.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    best_state, best_score, history, model = train_slienet(
        arch=args.arch,
        head=args.head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wd=args.wd,
        kd_alpha=args.kd_alpha,
        kd_temp=args.kd_temp,
        workers=args.workers,
        data_dir=args.data_dir,
    )

    name = args.tag or f"{args.arch}_seed{args.seed}"
    out = os.path.join(args.save_dir, f"{name}.pth")
    torch.save({
        "state_dict": best_state,
        "acc": best_score,
        "arch": args.arch,
        "head": args.head,
        "ic_indices": list(model.ic_indices),
        "ic_costs": list(model.ic_costs),
        "args": vars(args),
    }, out)
    print(f"  Saved: {out}  (best score: {best_score:.2f}%)")

    log_path = out.replace(".pth", "_history.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History: {log_path}")


if __name__ == "__main__":
    main()
