import random
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(bs=128, workers=4, data_dir="./data", calib_size=2000,
                split_seed=0):
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    D = torchvision.datasets.CIFAR100
    train_set = D(data_dir, True, download=True, transform=train_tf)
    # second copy of train without augmentation, used for calib subset
    calib_src = D(data_dir, True, download=False, transform=test_tf)
    test_set = D(data_dir, False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(split_seed)
    idx = torch.randperm(len(train_set), generator=g).tolist()
    calib_idx, train_idx = idx[:calib_size], idx[calib_size:]

    return (
        DataLoader(Subset(train_set, train_idx), bs, True,
                   num_workers=workers, pin_memory=True, drop_last=True),
        DataLoader(Subset(calib_src, calib_idx), 256, False,
                   num_workers=workers, pin_memory=True),
        DataLoader(test_set, 256, False,
                   num_workers=workers, pin_memory=True),
    )
