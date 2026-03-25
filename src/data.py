from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def _flatten_to_vector(x):
    return x.view(-1)


def _mnist_transform(flatten: bool = True) -> transforms.Compose:
    tfms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [0, 1] -> [-1, 1]
    ]
    if flatten:
        tfms.append(transforms.Lambda(_flatten_to_vector))
    return transforms.Compose(tfms)


def get_mnist_dataloaders(
    root: str,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/test dataloaders for MNIST flattened to 784."""
    transform = _mnist_transform(flatten=True)
    train_ds = datasets.MNIST(root=root, train=True, transform=transform, download=download)
    test_ds = datasets.MNIST(root=root, train=False, transform=transform, download=download)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader
