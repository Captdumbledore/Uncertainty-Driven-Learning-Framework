import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size=64,
    validation_split=0.1,
    seed=42,
):
    # Path to CIFAR-10 dataset
    dataset_path = "./CIFAR-10-images-master"

    # CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # Load training images
    full_train_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "train"),
        transform=transform
    )

    # Calculate split sizes
    total_size = len(full_train_dataset)

    validation_size = int(
        total_size * validation_split
    )

    train_size = total_size - validation_size

    # Reproducible split
    generator = torch.Generator().manual_seed(seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        [train_size, validation_size],
        generator=generator
    )

    # Load test images
    test_dataset = datasets.ImageFolder(
        root=os.path.join(dataset_path, "test"),
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
        train_dataset
    )