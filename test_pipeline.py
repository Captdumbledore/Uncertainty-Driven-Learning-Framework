import torch

from data import get_dataloaders
from model import SimpleCNN
from uncertainty import (
    mc_dropout_predict,
    select_top_uncertain,
    save_uncertainty_csv
)


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load CIFAR-10 data
train_loader, validation_loader, test_loader, train_dataset = get_dataloaders(
    batch_size=64
)


# Create CNN
model = SimpleCNN(
    num_classes=10,
    dropout_p=0.3,
    in_channels=3,
    image_size=32
)


# Load trained model
model.load_state_dict(
    torch.load(
        "shishya_cnn.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# MC Dropout prediction
results = mc_dropout_predict(
    model=model,
    data_loader=test_loader,
    device=device,
    n_samples=30
)


# Select top 10% uncertain samples
selected_indices, selected_scores = select_top_uncertain(
    results["entropy"],
    fraction=0.10
)


# Save uncertainty results
save_uncertainty_csv(
    results,
    selected_indices,
    output_path="uncertainty_scores.csv"
)


print("\nMean probabilities:", results["mean_probs"].shape)
print("Predicted classes:", results["predicted_classes"].shape)
print("Confidence:", results["confidence"].shape)
print("Entropy:", results["entropy"].shape)
print("True labels:", results["true_labels"].shape)

print("\nNumber of selected uncertain samples:")
print(len(selected_indices))

print("\nFirst 10 selected indices:")
print(selected_indices[:10])

print("\nFirst 10 selected uncertainty scores:")
print(selected_scores[:10])

print("\nAverage uncertainty:")
print(results["entropy"].mean())