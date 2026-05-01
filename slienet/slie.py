from itertools import combinations

import torch
import torch.nn.functional as F


@torch.no_grad()
def collect_logits(model, loader, dev):
    model.eval()
    K = len(model.ic_indices)
    ic = [[] for _ in range(K)]
    fin, ys = [], []
    for x, y in loader:
        x = x.to(dev)
        f, ics = model(x)
        for i, lg in enumerate(ics):
            ic[i].append(lg.cpu())
        fin.append(f.cpu())
        ys.append(y)
    return ([torch.cat(l, 0) for l in ic],
            torch.cat(fin, 0),
            torch.cat(ys, 0))


def acc_subset(ic_logits, indices, labels):
    if not indices:
        return 0.0
    avg = sum(F.softmax(ic_logits[i], dim=1) for i in indices) / len(indices)
    return (avg.argmax(1) == labels).float().mean().item() * 100.0


def exhaustive_subset(cal_ic, cal_y, k_inclusive):
    n = k_inclusive + 1
    best_subset, best_acc = None, -1.0
    for r in range(1, n + 1):
        for subset in combinations(range(n), r):
            a = acc_subset(cal_ic, list(subset), cal_y)
            if a > best_acc:
                best_acc = a
                best_subset = list(subset)
    return best_subset, best_acc


def slie_per_budget(model, calib_loader, eval_loader, dev):
    cal_ic, _, cal_y = collect_logits(model, calib_loader, dev)
    ev_ic, _, ev_y = collect_logits(model, eval_loader, dev)
    K = len(model.ic_indices)

    rows = []
    for k in range(K):
        single = acc_subset(ev_ic, [k], ev_y)
        subset, _ = exhaustive_subset(cal_ic[:K], cal_y, k)
        slie_ev = acc_subset(ev_ic, subset, ev_y)
        rows.append({
            "k": k + 1,
            "cost": float(model.ic_costs[k]),
            "single_acc": round(single, 3),
            "slie_acc": round(slie_ev, 3),
            "slie_subset": [i + 1 for i in subset],
        })
    return rows
