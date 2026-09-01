import torch

from data import get_dataloaders
from model import SimpleCNN
from uncertainty import (
    mc_dropout_predict,
    get_selection_indices,
    save_uncertainty_csv,
    generate_uncertainty_report,
)
from evaluate import evaluate_model


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Load data
train_loader, validation_loader, test_loader, train_dataset = get_dataloaders(
    dataset_name="CIFAR10",
    batch_size=64,
)


# Create model
model = SimpleCNN(
    num_classes=10,
    dropout_p=0.3,
    in_channels=3,
    image_size=32,
)


# Load trained model
model.load_state_dict(
    torch.load(
        "shishya_cnn.pth",
        map_location=device,
    )
)

model = model.to(device)


# Evaluate model
evaluation = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device,
)

print("\nEvaluation Results:")
print("Accuracy:", evaluation["accuracy"])
print("Precision:", evaluation["precision"])
print("Recall:", evaluation["recall"])
print("F1:", evaluation["f1"])


# MC Dropout uncertainty
results = mc_dropout_predict(
    model=model,
    data_loader=test_loader,
    n_samples=30,
    device=device,
)


# Select top 10% uncertain samples
selected_indices = get_selection_indices(
    results["entropy"],
    top_fraction=0.10,
    mode="highest",
)


# Save uncertainty CSV
save_uncertainty_csv(
    results,
    save_path="uncertainty_scores.csv",
)


# CIFAR-10 class names
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Generate uncertainty report
generate_uncertainty_report(
    results=results,
    class_names=class_names,
    save_path="uncertainty_report.txt",
    top_fraction=0.10,
)


print("\nMean probabilities:", results["mean_probs"].shape)
print("Predicted classes:", results["predicted_classes"].shape)
print("Confidence:", results["confidence"].shape)
print("Entropy:", results["entropy"].shape)
print("True labels:", results["true_labels"].shape)

print("\nSelected uncertain samples:")
print(len(selected_indices))

print("\nUncertainty pipeline completed successfully!")