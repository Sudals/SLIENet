from .resnet56 import SLIE_ResNet56


def build_model(arch, num_classes=100, head="mixedpool"):
    if arch == "resnet56":
        return SLIE_ResNet56(num_classes=num_classes, head=head)
    raise ValueError(f"unknown arch: {arch}")


__all__ = ["build_model", "SLIE_ResNet56"]
