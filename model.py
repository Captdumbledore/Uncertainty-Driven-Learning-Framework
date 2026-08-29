import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Baseline CNN used by the Shishya module.

    Supports CIFAR-10, MNIST, and FashionMNIST through
    configurable input channels and image size.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        in_channels: int = 1,
        image_size: int = 28,
    ):
        super().__init__()

        # Channel sizes
        c1 = 32 if in_channels == 3 else 16
        c2 = 64 if in_channels == 3 else 32

        # Hidden size of penultimate fully-connected layer
        fc1_hidden = 256 if in_channels == 3 else 128

        # Two 2x2 pooling operations
        after_pool = image_size // 4

        # Input size to FC layer
        fc1_in = c2 * after_pool * after_pool

        # Named convolution layers
        self.conv1 = nn.Conv2d(
            in_channels,
            c1,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            c1,
            c2,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(2, 2)

        # Dropout for MC Dropout uncertainty
        self.dropout = nn.Dropout(p=dropout_p)

        # Penultimate feature layer
        self.fc1 = nn.Linear(
            fc1_in,
            fc1_hidden
        )

        # Final classification layer
        self.fc2 = nn.Linear(
            fc1_hidden,
            num_classes
        )

    def extract_features(self, x):
        """
        Extract convolutional features before the fully connected layers.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        return x

    def get_embedding(self, x):
        """
        Return the penultimate-layer representation.
        """

        x = self.extract_features(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        return x

    def forward(self, x):
        """
        Return classification logits.
        """

        x = self.extract_features(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.fc2(x)

        return x


def enable_dropout(model: nn.Module) -> None:
    """
    Put the model in evaluation mode while keeping
    Dropout layers active for MC Dropout inference.
    """

    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def get_model(
    num_classes: int = 10,
    dropout_p: float = 0.3,
    in_channels: int = 1,
    image_size: int = 28,
) -> SimpleCNN:
    """
    Factory function returning a freshly initialized SimpleCNN.
    """

    return SimpleCNN(
        num_classes=num_classes,
        dropout_p=dropout_p,
        in_channels=in_channels,
        image_size=image_size,
    )