"""
train.py
--------
Standard supervised training loop — Phase 1.

Responsibilities
----------------
- Train SimpleCNN using cross-entropy loss and Adam.
- Track per-epoch training and validation loss and accuracy.
- Return the trained model and history for later evaluation and plotting.
"""


import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model:        nn.Module,
    train_loader,
    val_loader,
    epochs:       int   = 15,
    lr:           float = 0.001,
    device              = "cpu",
):
    """
    Train the CNN and return the trained model and history.

    Parameters
    ----------
    model : nn.Module
        CNN model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader. Used only for evaluation during training;
        never used for augmentation or retraining.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam.
    device : str or torch.device
        Device used for training.

    Returns
    -------
    model : nn.Module
        Trained CNN.
    history : dict
        Per-epoch training and validation metrics.
    """

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):

        # Training
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        train_loss = (
            running_loss / len(train_loader)
        )

        train_acc = (
            correct / total
        )

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                val_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_total += labels.size(0)

                val_correct += (
                    predictions == labels
                ).sum().item()

        val_loss /= len(val_loader)

        val_acc = (
            val_correct / val_total
        )

        # Save history
        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        print(
            f"    Epoch {epoch:>2}/{epochs}  "
            f"| Train Loss {train_loss:.4f}  Acc {train_acc:.4f}"
            f"  | Val Loss {val_loss:.4f}  Acc {val_acc:.4f}"
        )

    return model, history