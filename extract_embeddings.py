import torch

from data import get_dataloaders
from model import SimpleCNN


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load CIFAR-10 data
    train_loader, validation_loader, test_loader, train_dataset = get_dataloaders(
        batch_size=64
    )

    # Create the trained CNN architecture
    model = SimpleCNN(
        num_classes=10,
        dropout_p=0.3,
        in_channels=3,
        image_size=32
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            "shishya_cnn.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    # Store embeddings and labels
    all_embeddings = []
    all_labels = []

    # Extract penultimate-layer embeddings
    with torch.no_grad():

        for images, labels in train_loader:

            images = images.to(device)

            embeddings = model.get_embedding(images)

            all_embeddings.append(
                embeddings.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    # Combine batches
    all_embeddings = torch.cat(
        all_embeddings,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    print("Embedding shape:", all_embeddings.shape)
    print("Labels shape:", all_labels.shape)

    # Save embeddings
    torch.save(
        {
            "embeddings": all_embeddings,
            "labels": all_labels
        },
        "train_embeddings.pt"
    )

    print("Training embeddings saved successfully!")


if __name__ == "__main__":
    main()