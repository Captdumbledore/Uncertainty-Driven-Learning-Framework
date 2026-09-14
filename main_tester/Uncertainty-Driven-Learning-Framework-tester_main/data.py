"""
data.py
-------
Dataset loading utilities for the Shishya module.

Supports CIFAR-10, MNIST, and FashionMNIST with reproducible
train/validation splits.
"""

import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


STATS = {
    "FashionMNIST": {
        "mean": (0.2860,),
        "std": (0.3530,),
    },
    "MNIST": {
        "mean": (0.1307,),
        "std": (0.3081,),
    },
    "CIFAR10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
    },
}


DATASET_META = {
    "FashionMNIST": {
        "in_channels": 1,
        "image_size": 28,
    },
    "MNIST": {
        "in_channels": 1,
        "image_size": 28,
    },
    "CIFAR10": {
        "in_channels": 3,
        "image_size": 32,
    },
}


def get_dataloaders(
    dataset_name: str = "CIFAR10",
    batch_size: int = 64,
    val_split: float = 0.10,
    seed: int = 42,
    data_root: str = "./data",
):
    """
    Load a supported dataset and return train, validation,
    test DataLoaders and the training dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name: CIFAR10, MNIST, or FashionMNIST.
    batch_size : int
        Batch size for the DataLoaders.
    val_split : float
        Fraction of the training data used for validation.
    seed : int
        Random seed for reproducible splitting.
    data_root : str
        Root directory for downloaded or stored datasets.
    """

    if dataset_name not in DATASET_META:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Choose from {list(DATASET_META.keys())}."
        )

    meta = DATASET_META[dataset_name]
    stats = STATS[dataset_name]

    image_size = meta["image_size"]

    transform = transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            stats["mean"],
            stats["std"]
        ),
    ])

    # Existing CIFAR-10 ImageFolder dataset
    cifar_folder = "./CIFAR-10-images-master"

    if (
        dataset_name == "CIFAR10"
        and os.path.isdir(
            os.path.join(cifar_folder, "train")
        )
        and os.path.isdir(
            os.path.join(cifar_folder, "test")
        )
    ):

        full_train_dataset = datasets.ImageFolder(
            root=os.path.join(
                cifar_folder,
                "train"
            ),
            transform=transform,
        )

        test_dataset = datasets.ImageFolder(
            root=os.path.join(
                cifar_folder,
                "test"
            ),
            transform=transform,
        )

    else:

        dataset_class = getattr(
            datasets,
            dataset_name
        )

        full_train_dataset = dataset_class(
            root=data_root,
            train=True,
            download=True,
            transform=transform,
        )

        test_dataset = dataset_class(
            root=data_root,
            train=False,
            download=True,
            transform=transform,
        )

    total_size = len(
        full_train_dataset
    )

    validation_size = int(
        total_size * val_split
    )

    train_size = (
        total_size - validation_size
    )

    generator = torch.Generator().manual_seed(
        seed
    )

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    print(
        f"  Dataset     : {dataset_name}"
    )
    print(
        f"  Train size  : {len(train_dataset):,}"
    )
    print(
        f"  Val size    : {len(validation_dataset):,}"
    )
    print(
        f"  Test size   : {len(test_dataset):,}"
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset,
    )