import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from slienet import build_model


class FusedSLIE(nn.Module):
    def __init__(self, model, depth):
        super().__init__()
        K = len(model.ic_indices)
        if depth < 1 or depth > K:
            raise ValueError(f"--depth must be in [1, {K}]")
        self.depth = depth
        self.ic_indices = model.ic_indices[:depth]
        last_block = self.ic_indices[-1]
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.blocks = nn.ModuleList(list(model.blocks[: last_block + 1]))
        self.ics = nn.ModuleList(list(model.ics[:depth]))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        sm_sum = None
        ptr = 0
        for i, blk in enumerate(self.blocks):
            out = blk(out)
            if ptr < len(self.ic_indices) and i == self.ic_indices[ptr]:
                p = F.softmax(self.ics[ptr](out), dim=1)
                sm_sum = p if sm_sum is None else sm_sum + p
                ptr += 1
        return sm_sum / float(self.depth)


class SubFusedSLIE(nn.Module):
    def __init__(self, model, ic_subset_indices):
        super().__init__()
        ic_subset_indices = sorted(ic_subset_indices)
        last_block = ic_subset_indices[-1]
        self.ic_indices = ic_subset_indices
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.blocks = nn.ModuleList(list(model.blocks[: last_block + 1]))
        # map each global IC index -> position in model.ics
        full_ic = list(model.ic_indices)
        # remap global block index -> position in model.ics
        self.ics = nn.ModuleList([model.ics[full_ic.index(i)]
                                   for i in ic_subset_indices])

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        sm_sum = None
        ptr = 0
        for i, blk in enumerate(self.blocks):
            out = blk(out)
            if ptr < len(self.ic_indices) and i == self.ic_indices[ptr]:
                p = F.softmax(self.ics[ptr](out), dim=1)
                sm_sum = p if sm_sum is None else sm_sum + p
                ptr += 1
        return sm_sum


def _shard_indices(ic_indices, num_shards):
    n = len(ic_indices)
    if num_shards < 1 or num_shards > n:
        raise ValueError(f"--num-engines must be in [1, {n}]")
    base, rem = divmod(n, num_shards)
    out, start = [], 0
    for s in range(num_shards):
        size = base + (1 if s < rem else 0)
        out.append(ic_indices[start:start + size])
        start += size
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--depth", type=int, required=True,
                    help="SLIE depth k: backbone is truncated after IC_k.")
    pa.add_argument("--output", required=True,
                    help="ONNX path. With --num-engines > 1, used as base "
                         "for <base>_e<i>.onnx (i = 1..N).")
    pa.add_argument("--num-engines", type=int, default=1,
                    help="1 = single-fused; k = multi-engine; "
                         "1 < N < k = partial-fused.")
    pa.add_argument("--batch-size", type=int, default=1)
    pa.add_argument("--opset", type=int, default=17)
    args = pa.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    head = ckpt.get("head", "mixedpool")
    model = build_model(arch, num_classes=100, head=head)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    K = len(model.ic_indices)
    if args.depth < 1 or args.depth > K:
        raise SystemExit(f"--depth must be in [1, {K}]")

    selected = list(model.ic_indices[: args.depth])

    if args.num_engines == 1:
        deploy = FusedSLIE(model, args.depth).eval()
        dummy = torch.zeros(args.batch_size, 3, 32, 32)
        torch.onnx.export(
            deploy, dummy, args.output,
            input_names=["input"], output_names=["softmax"],
            opset_version=args.opset,
            dynamic_axes={"input": {0: "batch"}, "softmax": {0: "batch"}},
        )
        print(f"  Variant  : single-fused ({args.depth} ICs averaged in 1 engine)")
        print(f"  Exported : {args.output}")
    else:
        shards = _shard_indices(selected, args.num_engines)
        base, _ = os.path.splitext(args.output)
        for i, sub in enumerate(shards, start=1):
            mod = SubFusedSLIE(model, sub).eval()
            dummy = torch.zeros(args.batch_size, 3, 32, 32)
            path = f"{base}_e{i}.onnx"
            torch.onnx.export(
                mod, dummy, path,
                input_names=["input"], output_names=["sm_sum"],
                opset_version=args.opset,
                dynamic_axes={"input": {0: "batch"}, "sm_sum": {0: "batch"}},
            )
            print(f"  Engine {i}/{args.num_engines}: ICs {sub} -> {path}")
        variant = ("multi-engine" if args.num_engines == args.depth
                   else "partial-fused")
        print(f"  Variant  : {variant} ({args.num_engines} shards over "
              f"{args.depth} ICs)")
    print(f"  Backbone : {arch}")


if __name__ == "__main__":
    main()
