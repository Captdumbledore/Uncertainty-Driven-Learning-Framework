import torch

from data import get_dataloaders
from model import SimpleCNN
from train import train_model


def main():
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load CIFAR-10 data
    train_loader, validation_loader, test_loader, train_dataset = get_dataloaders(
        batch_size=64
    )

    # Create the CNN
    model = SimpleCNN(
        num_classes=10,
        dropout_p=0.3,
        in_channels=3,
        image_size=32
    )

    # Train CNN
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        epochs=10,
        lr=0.001,
        device=device,
    )

    # Save trained model
    torch.save(
        trained_model.state_dict(),
        "shishya_cnn.pth"
    )

    print("Trained model saved successfully!")


if __name__ == "__main__":
    main()