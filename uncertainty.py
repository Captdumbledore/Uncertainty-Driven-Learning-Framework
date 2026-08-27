import torch
import torch.nn.functional as F


def compute_uncertainty(logits):
    """
    Compute predictive uncertainty from CNN output logits.

    Higher entropy means higher uncertainty.
    Lower entropy means lower uncertainty.
    """

    # Convert logits into class probabilities
    probabilities = F.softmax(logits, dim=1)

    # Prevent log(0)
    probabilities = torch.clamp(
        probabilities,
        min=1e-10
    )

    # Predictive entropy
    uncertainty = -torch.sum(
        probabilities * torch.log(probabilities),
        dim=1
    )

    # Predicted class
    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    return {
        "predictions": predictions,
        "uncertainty": uncertainty
    }