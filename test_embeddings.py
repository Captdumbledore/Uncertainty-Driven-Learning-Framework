from akrm.diagnosis.knn import find_knn
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from akrm.diagnosis.embeddings import extract_embeddings


# Small dummy CNN
class DummyCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 4, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Penultimate layer
        self.fc = nn.Linear(4, 8)

        # Final classification layer
        self.classifier = nn.Linear(8, 2)

    def extract_features(self, x):

        x = self.conv(x)
        x = torch.relu(x)

        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


# Create dummy images
x = torch.randn(10, 1, 28, 28)

# Create dummy labels
y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

dataset = TensorDataset(x, y)

# Use CPU
device = torch.device("cpu")

# Create model
model = DummyCNN()

# Extract embeddings
embeddings, labels = extract_embeddings(
    model,
    dataset,
    device,
    batch_size=5
)

query = embeddings[0]

indices, distances, neighbor_labels = find_knn(
    query,
    embeddings,
    labels,
    k=3
)

print("K-NN neighbor indices:", indices)
print("K-NN neighbor distances:", distances)
print("K-NN neighbor labels:", neighbor_labels)

class_counts = np.bincount(neighbor_labels)

print("Neighbor class counts:", class_counts)

print("Embedding shape:", embeddings.shape)
print("Labels shape:", labels.shape)
print("First embedding:", embeddings[0])
print("Labels:", labels)
