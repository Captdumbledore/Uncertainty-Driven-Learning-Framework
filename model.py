import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes=10, embedding_dim=128):
        super(CNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # After 3 pooling operations:
        # 32x32 → 16x16 → 8x8 → 4x4
        self.flatten = nn.Flatten()

        self.feature_layer = nn.Linear(
             128 * 4 * 4,
             embedding_dim
       )

        # Final classifier
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )


    def get_embedding(self, x):
        """Return the 128-D penultimate-layer representation."""

        x = self.features(x)
        x = self.flatten(x)
        x = self.feature_layer(x)

        return x


    def forward(self, x):
        """Return classification logits."""

        embedding = self.get_embedding(x)

        logits = self.classifier(embedding)

        return logits