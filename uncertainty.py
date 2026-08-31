import csv
import os

import numpy as np
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

    stochastic_probs = np.stack(
        stochastic_probs,
        axis=0
    )

    mean_probs = np.mean(
        stochastic_probs,
        axis=0
    )

    predicted_classes = np.argmax(
        mean_probs,
        axis=1
    )

    confidence = np.max(
        mean_probs,
        axis=1
    )

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


def get_selection_indices(
    entropy_scores: np.ndarray,
    top_fraction: float = 0.10,
    mode: str = "highest",
    seed: int = 42,
) -> np.ndarray:
    """
    Select sample indices based on entropy.

    mode:
        highest = most uncertain
        lowest  = least uncertain
        random  = random selection
    """

    n_total = len(entropy_scores)

    n_select = max(
        1,
        int(n_total * top_fraction)
    )

    if mode == "highest":

        indices = np.argsort(
            entropy_scores
        )[-n_select:]

    elif mode == "lowest":

        indices = np.argsort(
            entropy_scores
        )[:n_select]

    elif mode == "random":

        rng = np.random.default_rng(seed)

        indices = rng.choice(
            n_total,
            size=n_select,
            replace=False
        )

    else:

        raise ValueError(
            f"Unknown mode '{mode}'. "
            "Use 'highest', 'lowest', or 'random'."
        )

    return indices


def save_uncertainty_csv(
    results: dict,
    save_path: str
) -> None:
    """
    Save per-sample uncertainty analysis to CSV.
    """

    dir_path = os.path.dirname(save_path)

    if dir_path:
        os.makedirs(
            dir_path,
            exist_ok=True
        )

    correct = (
        results["true_labels"]
        == results["predicted_classes"]
    )

    with open(
        save_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as fh:

        writer = csv.writer(fh)

        writer.writerow([
            "sample_id",
            "true_label",
            "predicted_label",
            "confidence",
            "predictive_entropy",
            "correct_prediction",
        ])

        n = len(
            results["true_labels"]
        )

        for i in range(n):

            writer.writerow([
                i,
                int(results["true_labels"][i]),
                int(results["predicted_classes"][i]),
                f"{results['confidence'][i]:.6f}",
                f"{results['entropy'][i]:.6f}",
                bool(correct[i]),
            ])

    print(
        f"Saved uncertainty CSV ({n:,} rows) → {save_path}"
    )


def generate_uncertainty_report(
    results: dict,
    class_names: list,
    save_path: str,
    top_fraction: float = 0.10,
) -> str:
    """
    Generate a pre-retraining uncertainty analysis report.
    """

    from collections import Counter

    entropy = results["entropy"]
    true_labels = results["true_labels"]
    predicted = results["predicted_classes"]

    correct_mask = (
        true_labels == predicted
    )

    errors = (
        ~correct_mask
    ).astype(int)

    n_total = len(entropy)

    n_select = max(
        1,
        int(n_total * top_fraction)
    )

    top_idx = np.argsort(
        entropy
    )[-n_select:]

    top_correct = correct_mask[
        top_idx
    ]

    pct_misclf = (
        1 - top_correct.mean()
    ) * 100

    pct_already_ok = (
        top_correct.mean()
    ) * 100

    top_labels = true_labels[
        top_idx
    ]

    class_counts = Counter(
        int(label)
        for label in top_labels
    )

    corr = float(
        np.corrcoef(
            entropy,
            errors
        )[0, 1]
    )

    ent_correct = entropy[
        correct_mask
    ]

    ent_incorrect = entropy[
        ~correct_mask
    ]

    lines = [
        "=" * 64,
        "  PRE-RETRAINING UNCERTAINTY ANALYSIS REPORT",
        "=" * 64,
        "",
        f"  Total samples                    : {n_total:,}",
        f"  Accuracy                         : {correct_mask.mean() * 100:.2f}%",
        f"  Top-{top_fraction * 100:.0f}% uncertain samples : {n_select:,}",
        "",
        "-" * 60,
        "  Q1  Are the most uncertain samples misclassified?",
        "-" * 60,
        f"      Misclassified : {pct_misclf:.2f}%",
        f"      Correct       : {pct_already_ok:.2f}%",
        "",
        "-" * 60,
        "  Q2  Which classes appear most among uncertain samples?",
        "-" * 60,
    ]

    for c in sorted(
        class_counts,
        key=lambda x: class_counts[x],
        reverse=True
    ):

        name = (
            class_names[c]
            if c < len(class_names)
            else str(c)
        )

        count = class_counts[c]

        pct = (
            count / n_select
        ) * 100

        lines.append(
            f"      {name:<24} {count:>5} ({pct:5.1f}%)"
        )

    if corr > 0.15:

        interpretation = (
            "Positive — entropy is a meaningful "
            "signal for prediction errors."
        )

    elif corr > 0.0:

        interpretation = (
            "Weak positive — entropy imperfectly "
            "predicts errors."
        )

    else:

        interpretation = (
            "Near-zero or negative — entropy does "
            "not reliably predict errors."
        )

    lines += [
        "",
        "-" * 60,
        "  Q3  Correlation: entropy vs prediction error",
        "-" * 60,
        f"      Pearson r = {corr:.4f}",
        f"      Interpretation: {interpretation}",
        "",
        "-" * 60,
        "  Q4  Entropy by prediction correctness",
        "-" * 60,
    ]

    if correct_mask.sum() > 0:

        lines.append(
            f"      Correct predictions   : "
            f"{ent_correct.mean():.4f}"
        )

    if (~correct_mask).sum() > 0:

        lines.append(
            f"      Incorrect predictions : "
            f"{ent_incorrect.mean():.4f}"
        )

    lines += [
        "",
        "=" * 64,
    ]

    report = "\n".join(lines)

    dir_path = os.path.dirname(save_path)

    if dir_path:
        os.makedirs(
            dir_path,
            exist_ok=True
        )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as fh:

        fh.write(report)

    print(
        f"Saved uncertainty report → {save_path}"
    )

    return report