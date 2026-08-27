import torch

from data import get_dataloaders
from model import CNN


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Create the CNN model
model = CNN(
    num_classes=10,
    embedding_dim=128
)


# Load the trained model
model.load_state_dict(
    torch.load(
        "shishya_cnn.pth",
        map_location=device
    )
)

model = model.to(device)

# Evaluation mode
model.eval()


# Load the dataset
train_loader, validation_loader, test_loader = get_dataloaders(
    batch_size=64
)


# Store embeddings and labels
all_embeddings = []
all_labels = []


# Extract embeddings from the training data
with torch.no_grad():

    for images, labels in train_loader:

        images = images.to(device)

        # Extract penultimate-layer embeddings
        embeddings = model.get_embedding(images)

        # Move embeddings to CPU before storing
        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(
            labels.cpu()
        )


# Combine all batches
all_embeddings = torch.cat(
    all_embeddings,
    dim=0
)

all_labels = torch.cat(
    all_labels,
    dim=0
)


# Display shapes
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