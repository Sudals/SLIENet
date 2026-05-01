import torch
import torch.nn as nn


class MixedPoolIC(nn.Module):
    def __init__(self, in_channels, feature_size, num_classes):
        super().__init__()
        target = min(feature_size, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(target)
        self.max_pool = nn.AdaptiveMaxPool2d(target)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.fc = nn.Linear(in_channels * target * target, num_classes)

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        p = a * self.avg_pool(x) + (1 - a) * self.max_pool(x)
        return self.fc(p.view(p.size(0), -1))


def build_head(head_type, in_channels, feature_size, num_classes):
    if head_type == "mixedpool":
        return MixedPoolIC(in_channels, feature_size, num_classes)
    raise ValueError(f"unknown head type: {head_type}")
