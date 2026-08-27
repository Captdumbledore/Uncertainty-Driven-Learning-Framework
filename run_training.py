import torch

from data import get_dataloaders
from model import CNN
from train import train_model


# Load CIFAR-10 data
train_loader, validation_loader, test_loader = get_dataloaders(
    batch_size=64
)

# Create CNN
model = CNN(
    num_classes=10,
    embedding_dim=128
)

# Train CNN
trained_model = train_model(
    model=model,
    train_loader=train_loader,
    validation_loader=validation_loader,
    num_epochs=10,
    learning_rate=0.001,
)

# Save trained model
torch.save(
    trained_model.state_dict(),
    "shishya_cnn.pth"
)

print("Trained model saved successfully!")