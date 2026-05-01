import copy
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from .data import get_loaders
from .models import build_model


@torch.no_grad()
def quick_eval(model, loader, dev):
    model.eval()
    K = len(model.ic_indices)
    ic_ok = [0] * K
    f_ok = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        final, ics = model(x)
        for i, lg in enumerate(ics):
            ic_ok[i] += lg.argmax(1).eq(y).sum().item()
        f_ok += final.argmax(1).eq(y).sum().item()
        n += y.size(0)
    return {
        "ic_accs": [100.0 * c / n for c in ic_ok],
        "final_acc": 100.0 * f_ok / n,
        "ic_costs": list(model.ic_costs),
    }


def train_slienet(
    arch="resnet56",
    *,
    num_classes=100,
    head="mixedpool",
    epochs=100,
    batch_size=128,
    lr=0.1,
    wd=1e-4,
    kd_alpha=0.3,
    kd_temp=4.0,
    workers=4,
    data_dir="./data",
    log_every=5,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, num_classes=num_classes, head=head).to(dev)
    K = len(model.ic_indices)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  SLIE-Net {arch.upper()} | head={head} | ICs at {model.ic_indices}")
    print(f"  IC costs: {[f'{c:.3f}' for c in model.ic_costs]}")
    print(f"  Params: {nparam:,} ({nparam/1e6:.2f}M)")
    print(f"  Self-distill: alpha={kd_alpha} T={kd_temp} | epochs={epochs}")

    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                    weight_decay=wd, nesterov=True)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr, _, te = get_loaders(batch_size, workers, data_dir)

    best, best_state = 0.0, None
    history = []
    T = kd_temp

    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        progress = ep / epochs

        for x, y in tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad()
            final, ics = model(x)
            loss = F.cross_entropy(final, y)

            # detached teacher
            soft_t = F.softmax(final.detach() / T, dim=1)
            for j, lg in enumerate(ics):
                cost = model.ic_costs[j]
                tau = 0.01 + progress * (cost - 0.01)
                ce_term = F.cross_entropy(lg, y)
                if kd_alpha > 0:
                    kd_term = F.kl_div(
                        F.log_softmax(lg / T, dim=1),
                        soft_t,
                        reduction="batchmean",
                    ) * (T * T)
                    ic_loss = (1.0 - kd_alpha) * ce_term + kd_alpha * kd_term
                else:
                    ic_loss = ce_term
                loss = loss + tau * ic_loss

            loss.backward()
            opt.step()
        sched.step()

        r = quick_eval(model, te, dev)
        score = max(r["ic_accs"] + [r["final_acc"]])
        history.append({"epoch": ep, **r, "score": score})
        if score > best:
            best = score
            best_state = copy.deepcopy(model.state_dict())

        if ep % log_every == 0 or ep <= 3 or ep == epochs:
            ic_str = " ".join(f"IC{i+1}:{a:.2f}" for i, a in enumerate(r["ic_accs"]))
            print(f"  [{ep:3d}/{epochs}] {ic_str} | Final:{r['final_acc']:.2f} "
                  f"Best:{best:.2f} lr:{opt.param_groups[0]['lr']:.4f} "
                  f"({time.time()-t0:.1f}s)")

    return best_state, best, history, model
