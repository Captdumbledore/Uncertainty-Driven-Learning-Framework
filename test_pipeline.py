import torch

from data import get_dataloaders
from model import CNN
from uncertainty import compute_uncertainty


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load CIFAR-10 data
train_loader, validation_loader, test_loader = get_dataloaders(
    batch_size=64
)


# Create the CNN model
model = CNN(
    num_classes=10,
    embedding_dim=128
)


# Load the trained CNN weights
model.load_state_dict(
    torch.load(
        "shishya_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# Get one batch from test data
images, labels = next(iter(test_loader))

images = images.to(device)


print("Input images:", images.shape)
print("Labels:", labels.shape)


# Run images through the trained CNN
with torch.no_grad():

    # Get penultimate-layer embeddings
    embeddings = model.get_embedding(images)

    # Get classification logits
    logits = model(images)


# Compute predictions and uncertainty
results = compute_uncertainty(logits)

predictions = results["predictions"]
uncertainty_scores = results["uncertainty"]


print("\nEmbeddings:", embeddings.shape)
print("Logits:", logits.shape)
print("Predictions:", predictions.shape)
print("Uncertainty scores:", uncertainty_scores.shape)

print("\nFirst 10 actual labels:")
print(labels[:10])

print("\nFirst 10 predictions:")
print(predictions[:10].cpu())

print("\nFirst 10 uncertainty scores:")
print(uncertainty_scores[:10].cpu())

print("\nAverage uncertainty:")
print(uncertainty_scores.mean().item())