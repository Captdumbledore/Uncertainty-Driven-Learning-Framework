import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import enable_dropout


def mc_dropout_predict(
    model,
    data_loader,
    device="cpu",
    n_samples=30,
):
    """
    Perform Monte Carlo Dropout prediction.
    """

    model = model.to(device)

    stochastic_probs = []
    true_labels = []

    for sample in range(n_samples):

        enable_dropout(model)

        batch_probs = []
        batch_labels = []

        with torch.no_grad():

            for images, labels in data_loader:

                images = images.to(device)

                logits = model(images)

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                batch_probs.append(
                    probabilities.cpu().numpy()
                )

                if sample == 0:
                    batch_labels.append(
                        labels.numpy()
                    )

        stochastic_probs.append(
            np.concatenate(
                batch_probs,
                axis=0
            )
        )

        if sample == 0:
            true_labels = np.concatenate(
                batch_labels,
                axis=0
            )

    # Shape:
    # (number of MC samples, number of images, number of classes)
    stochastic_probs = np.stack(
        stochastic_probs,
        axis=0
    )

    # Average predictions across MC samples
    mean_probs = np.mean(
        stochastic_probs,
        axis=0
    )

    # Predicted class
    predicted_classes = np.argmax(
        mean_probs,
        axis=1
    )

    # Confidence
    confidence = np.max(
        mean_probs,
        axis=1
    )

    # Predictive entropy
    entropy = -np.sum(
        mean_probs * np.log(
            mean_probs + 1e-8
        ),
        axis=1
    )

    return {
        "mean_probs": mean_probs,
        "predicted_classes": predicted_classes,
        "confidence": confidence,
        "entropy": entropy,
        "true_labels": true_labels,
    }


def select_top_uncertain(
    uncertainty_scores,
    fraction=0.10
):
    """
    Select the most uncertain samples.
    """

    uncertainty_scores = np.asarray(
        uncertainty_scores
    )

    n_samples = len(
        uncertainty_scores
    )

    n_select = max(
        1,
        int(n_samples * fraction)
    )

    sorted_indices = np.argsort(
        uncertainty_scores
    )[::-1]

    selected_indices = sorted_indices[
        :n_select
    ]

    selected_scores = uncertainty_scores[
        selected_indices
    ]

    return selected_indices, selected_scores


def save_uncertainty_csv(
    results,
    selected_indices,
    output_path="uncertainty_scores.csv"
):
    """
    Save uncertainty results to a CSV file.
    """

    df = pd.DataFrame({
        "sample_index": np.arange(
            len(results["entropy"])
        ),
        "true_label": results["true_labels"],
        "predicted_class": results["predicted_classes"],
        "confidence": results["confidence"],
        "entropy": results["entropy"],
    })

    df["selected"] = False

    df.loc[
        selected_indices,
        "selected"
    ] = True

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Uncertainty CSV saved to: {output_path}"
    )