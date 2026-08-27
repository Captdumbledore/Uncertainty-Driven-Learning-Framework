import torch

from data import get_dataloaders
from model import CNN


# Load CIFAR-10 data
train_loader, validation_loader, test_loader = get_dataloaders(
    batch_size=64
)


# Create the CNN model
model = CNN(
    num_classes=10,
    embedding_dim=128
)


# Get one real batch of CIFAR-10 images
images, labels = next(iter(train_loader))


print("Input images:", images.shape)
print("Labels:", labels.shape)


# Run the real images through the CNN
with torch.no_grad():

    # Get penultimate-layer features
    embeddings = model.get_embedding(images)

    # Get classification output
    logits = model(images)


print("Embeddings:", embeddings.shape)
print("Logits:", logits.shape)


# Convert logits into predicted class indices
predictions = torch.argmax(logits, dim=1)

print("Predictions:", predictions.shape)

print("First 10 actual labels:", labels[:10])
print("First 10 predictions:", predictions[:10])