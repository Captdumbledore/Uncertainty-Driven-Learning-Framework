import torch
import torch.nn as nn


class Model:
    """Base model wrapper."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("fit() must be implemented in a subclass.")

    def predict(self, X):
        raise NotImplementedError("predict() must be implemented in a subclass.")


class CNN(nn.Module):
    """
    CNN for CIFAR-10 classification.

    The penultimate feature layer produces a 128-dimensional
    embedding that can later be used by the uncertainty/diagnosis modules.
    """

    def __init__(self, num_classes=10, embedding_dim=128):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()

        # Penultimate layer / feature embedding
        self.feature_layer = nn.Linear(
            128 * 4 * 4,
            embedding_dim
        )

        # Final classification layer
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def get_embedding(self, x):
        """
        Return the penultimate-layer feature representation.
        """
        x = self.features(x)
        x = self.flatten(x)
        embedding = self.feature_layer(x)

        return embedding

    def forward(self, x):
        """
        Return classification logits.
        """
        embedding = self.get_embedding(x)
        logits = self.classifier(embedding)

        return logits