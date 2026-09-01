"""
evaluate.py
-----------
Evaluation utilities for classification, calibration, uncertainty,
error comparison, plotting, and result summaries.
"""

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(
    model,
    test_loader,
    device="cpu",
):
    """
    Evaluate a trained model on test data.

    Returns
    -------
    dict
        accuracy, precision, recall, f1, confusion_matrix,
        predictions, and true_labels.
    """

    model = model.to(device)
    model.eval()

    predictions = []
    true_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                labels.numpy()
            )

    predictions = np.asarray(
        predictions
    )

    true_labels = np.asarray(
        true_labels
    )

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    precision = precision_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(
        true_labels,
        predictions
    )

    print(
        f"Test Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1 Score       : {f1:.4f}"
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
        "predictions": predictions,
        "true_labels": true_labels,
    }


def compute_ece(
    probabilities,
    true_labels,
    n_bins=10,
):
    """
    Compute Expected Calibration Error.
    """

    probabilities = np.asarray(
        probabilities
    )

    true_labels = np.asarray(
        true_labels
    )

    confidences = np.max(
        probabilities,
        axis=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )

        else:

            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        if not np.any(mask):
            continue

        accuracy = np.mean(
            predictions[mask]
            == true_labels[mask]
        )

        confidence = np.mean(
            confidences[mask]
        )

        ece += (
            np.sum(mask)
            / len(true_labels)
        ) * abs(
            accuracy - confidence
        )

    return float(ece)


def compute_mean_entropy(
    model,
    data_loader,
    device="cpu",
    n_samples=30,
):
    """
    Compute mean predictive entropy using MC Dropout.
    """

    from uncertainty import mc_dropout_predict

    results = mc_dropout_predict(
        model=model,
        data_loader=data_loader,
        n_samples=n_samples,
        device=device,
    )

    return float(
        np.mean(
            results["entropy"]
        )
    )


def count_corrected_errors(
    predictions_before,
    predictions_after,
    true_labels,
):
    """
    Count samples that changed from incorrect to correct
    after retraining.
    """

    predictions_before = np.asarray(
        predictions_before
    )

    predictions_after = np.asarray(
        predictions_after
    )

    true_labels = np.asarray(
        true_labels
    )

    incorrect_before = (
        predictions_before != true_labels
    )

    correct_after = (
        predictions_after == true_labels
    )

    corrected = (
        incorrect_before
        & correct_after
    )

    return int(
        corrected.sum()
    )


def plot_training_curves(
    histories,
    save_path="training_curves.png",
):
    """
    Plot training and validation loss/accuracy curves.

    histories may be one history dictionary or a list of histories.
    """

    if isinstance(histories, dict):
        histories = [histories]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )

    for history in histories:

        epochs = range(
            1,
            len(history["train_loss"]) + 1
        )

        axes[0].plot(
            epochs,
            history["train_loss"]
        )

        axes[0].plot(
            epochs,
            history["val_loss"]
        )

        axes[1].plot(
            epochs,
            history["train_acc"]
        )

        axes[1].plot(
            epochs,
            history["val_acc"]
        )

    axes[0].set_title(
        "Training and Validation Loss"
    )

    axes[0].set_xlabel(
        "Epoch"
    )

    axes[0].set_ylabel(
        "Loss"
    )

    axes[0].legend([
        "Train",
        "Validation"
    ])

    axes[1].set_title(
        "Training and Validation Accuracy"
    )

    axes[1].set_xlabel(
        "Epoch"
    )

    axes[1].set_ylabel(
        "Accuracy"
    )

    axes[1].legend([
        "Train",
        "Validation"
    ])

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def plot_metrics_bar(
    results,
    save_path="metrics.png",
):
    """
    Plot classification metrics.
    """

    names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
    ]

    values = [
        results["accuracy"],
        results["precision"],
        results["recall"],
        results["f1"],
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        names,
        values
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Classification Metrics"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def plot_confusion_matrices(
    confusion_matrices,
    class_names,
    save_path="confusion_matrices.png",
):
    """
    Plot one or more row-normalised confusion matrices.
    """

    if isinstance(
        confusion_matrices,
        np.ndarray
    ):
        confusion_matrices = [
            confusion_matrices
        ]

    n = len(
        confusion_matrices
    )

    fig, axes = plt.subplots(
        1,
        n,
        figsize=(6 * n, 5)
    )

    if n == 1:
        axes = [axes]

    for ax, cm in zip(
        axes,
        confusion_matrices
    ):

        cm = cm.astype(float)

        row_sums = cm.sum(
            axis=1,
            keepdims=True
        )

        normalized = np.divide(
            cm,
            row_sums,
            out=np.zeros_like(cm),
            where=row_sums != 0
        )

        ax.imshow(
            normalized
        )

        ax.set_xticks(
            range(len(class_names))
        )

        ax.set_yticks(
            range(len(class_names))
        )

        ax.set_xticklabels(
            class_names,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            class_names
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "True"
        )

        ax.set_title(
            "Normalised Confusion Matrix"
        )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def plot_uncertainty_distribution(
    entropy,
    save_path="uncertainty_distribution.png",
):
    """
    Plot entropy distribution.
    """

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        entropy,
        bins=30
    )

    ax.set_xlabel(
        "Predictive Entropy"
    )

    ax.set_ylabel(
        "Number of Samples"
    )

    ax.set_title(
        "Uncertainty Distribution"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def plot_uncertainty_before_after(
    before_entropy,
    after_entropy,
    save_path="uncertainty_before_after.png",
):
    """
    Plot mean uncertainty before and after retraining.
    """

    values = [
        np.mean(before_entropy),
        np.mean(after_entropy),
    ]

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        ["Before", "After"],
        values
    )

    ax.set_ylabel(
        "Mean Predictive Entropy"
    )

    ax.set_title(
        "Uncertainty Before vs After Retraining"
    )

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=150
    )

    plt.close(fig)


def save_results_summary(
    results,
    save_path="results_summary.txt",
):
    """
    Save evaluation metrics and hypothesis summary.
    """

    lines = [
        "RESULTS SUMMARY",
        "=" * 50,
        f"Accuracy  : {results['accuracy']:.4f}",
        f"Precision : {results['precision']:.4f}",
        f"Recall    : {results['recall']:.4f}",
        f"F1        : {results['f1']:.4f}",
    ]

    text = "\n".join(
        lines
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as fh:

        fh.write(text)

    return text


def save_cross_dataset_comparison(
    results,
    save_path="cross_dataset_comparison.txt",
):
    """
    Save a simple cross-dataset comparison.
    """

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as fh:

        fh.write(
            "CROSS-DATASET COMPARISON\n"
            "========================\n\n"
        )

        for dataset_name, dataset_results in results.items():

            fh.write(
                f"{dataset_name}\n"
            )

            fh.write(
                f"Accuracy: "
                f"{dataset_results.get('accuracy', 'N/A')}\n"
            )

            fh.write(
                f"F1: "
                f"{dataset_results.get('f1', 'N/A')}\n\n"
            )