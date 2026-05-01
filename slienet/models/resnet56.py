import torch
import torch.nn as nn
import torch.nn.functional as F

from ..heads import build_head


class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.skip = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                          nn.BatchNorm2d(out_c))
            if stride != 1 or in_c != out_c else nn.Sequential()
        )

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.skip(x))


class SLIE_ResNet56(nn.Module):
    def __init__(self, num_classes=100, ic_indices=(4, 8, 12, 16, 20, 24),
                 head="mixedpool"):
        super().__init__()
        self.num_classes = num_classes
        self.head_type = head

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.blocks = nn.ModuleList()
        self.block_ch, self.block_fs = [], []
        ip, fs = 16, 32
        # 3 stage × 9 block
        for out_c, st in [(16, 1), (32, 2), (64, 2)]:
            for i in range(9):
                s = st if i == 0 else 1
                self.blocks.append(_ResBlock(ip, out_c, s))
                ip = out_c
                if s == 2:
                    fs //= 2
                self.block_ch.append(out_c)
                self.block_fs.append(fs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self.ic_indices = list(ic_indices)
        self.ics = nn.ModuleList()
        self._compute_ic_costs()
        for i in self.ic_indices:
            self.ics.append(build_head(head, self.block_ch[i],
                                       self.block_fs[i], num_classes))
        self._init_weights()

    def _compute_ic_costs(self):
        flops, in_c = [], 16
        for i in range(len(self.blocks)):
            oc, fs = self.block_ch[i], self.block_fs[i]
            f = 3 * 3 * in_c * oc * fs * fs + 3 * 3 * oc * oc * fs * fs
            if in_c != oc:
                f += in_c * oc * fs * fs
            flops.append(f)
            in_c = oc
        cum, s = [], 0
        for f in flops:
            s += f
            cum.append(s)
        total = cum[-1]
        self.ic_costs = [cum[i] / total for i in self.ic_indices]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        ic_logits = []
        ptr = 0
        for i, blk in enumerate(self.blocks):
            out = blk(out)
            if ptr < len(self.ic_indices) and i == self.ic_indices[ptr]:
                ic_logits.append(self.ics[ptr](out))
                ptr += 1
        feat = self.pool(out).flatten(1)
        final = self.fc(feat)
        if return_feat:
            return final, ic_logits, feat
        return final, ic_logits
