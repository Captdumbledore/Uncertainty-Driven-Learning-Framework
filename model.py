"""
model.py
--------
Simple CNN architecture shared by the learning pipelines.

The model supports grayscale 28x28 datasets and RGB 32x32 datasets
through configurable input channels and image size.

The penultimate fully-connected representation is used as the
embedding for downstream uncertainty and diagnostic analysis.
"""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Minimal two-convolution-block CNN with a single dropout layer.

    Layer names (conv1, conv2, pool, fc1, fc2) are kept stable across
    configurations so downstream embedding extraction requires no
    modifications when switching datasets.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_p: float = 0.3,
        in_channels: int = 1,
        image_size: int = 28,
    ):
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes.
        dropout_p : float
            Dropout probability used for MC Dropout.
        in_channels : int
            Number of input image channels.
        image_size : int
            Height and width of the input image.
        """

        super().__init__()

        c1 = 32 if in_channels == 3 else 16
        c2 = 64 if in_channels == 3 else 32
        fc1_hidden = 256 if in_channels == 3 else 128

        after_pool = image_size // 4
        fc1_in = c2 * after_pool * after_pool

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

        self.dropout = nn.Dropout(
            p=dropout_p
        )

        self.fc1 = nn.Linear(
            fc1_in,
            fc1_hidden
        )

        self.fc2 = nn.Linear(
            fc1_hidden,
            num_classes
        )

    def forward(self, x):
        """
        Return classification logits.
        """

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = self.dropout(
            F.relu(self.fc1(x))
        )

        x = self.fc2(x)

        return x

    def extract_features(self, x):
        """
        Extract the dense representation before the final classifier.
        """

        x = self.pool(
            F.relu(self.conv1(x))
        )

        x = self.pool(
            F.relu(self.conv2(x))
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = F.relu(
            self.fc1(x)
        )

        return x

    def get_embedding(self, x):
        """
        Return the penultimate-layer representation.
        """

        return self.extract_features(x)


def enable_dropout(model: nn.Module) -> None:
    """
    Switch only Dropout layers to train mode while keeping everything
    else in eval mode for Monte Carlo Dropout inference.
    """

    model.eval()

    for module in model.modules():

        if isinstance(
            module,
            nn.Dropout
        ):
            module.train()


def get_model(
    num_classes: int = 10,
    dropout_p: float = 0.3,
    in_channels: int = 1,
    image_size: int = 28,
) -> SimpleCNN:
    """
    Factory — returns a freshly initialised SimpleCNN.
    """

    return SimpleCNN(
        num_classes=num_classes,
        dropout_p=dropout_p,
        in_channels=in_channels,
        image_size=image_size,
    )