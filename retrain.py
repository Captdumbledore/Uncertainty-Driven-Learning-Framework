"""
retrain.py
----------
Knowledge distillation retraining for Shishya.

Implements:
- Hard Cross-Entropy loss
- Soft KL-Divergence loss
- Combined distillation loss
- Adaptive distillation weighting based on knowledge-gap difficulty
- Experience Pool validation fail-safe
"""

import copy

import torch
import torch.nn.functional as F


def compute_adaptive_alpha(
    entropy,
    min_alpha=0.2,
    max_alpha=0.8,
):
    """
    Convert knowledge-gap difficulty into an adaptive
    hard-loss weight.

    Higher entropy = harder knowledge gap = higher alpha.

    alpha controls the hard-loss contribution:

        Total Loss =
            alpha * Hard Loss
            + (1 - alpha) * Soft Loss
    """

    max_entropy = torch.log(
        torch.tensor(10.0, dtype=torch.float32)
    )

    if not torch.is_tensor(entropy):
        entropy = torch.tensor(
            entropy,
            dtype=torch.float32,
        )

    difficulty = torch.clamp(
        entropy / max_entropy,
        min=0.0,
        max=1.0,
    )

    alpha = (
        min_alpha
        + difficulty * (max_alpha - min_alpha)
    )

    return alpha


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    alpha=0.5,
    temperature=4.0,
):
    """
    Calculate the combined knowledge-distillation loss.

    Total loss =
        alpha * Hard Loss
        + (1 - alpha) * Soft KL Loss
    """

    hard_loss = F.cross_entropy(
        student_logits,
        labels,
    )

    soft_loss = F.kl_div(
        F.log_softmax(
            student_logits / temperature,
            dim=1,
        ),
        F.softmax(
            teacher_logits / temperature,
            dim=1,
        ),
        reduction="batchmean",
    ) * (temperature ** 2)

    total_loss = (
        alpha * hard_loss
        + (1.0 - alpha) * soft_loss
    )

    return total_loss, hard_loss, soft_loss


def retrain_model(
    model,
    data,
    epochs=5,
    lr=0.001,
    alpha=0.5,
    temperature=4.0,
    device=None,
    entropy_scores=None,
    experience_pool=None,
    pool_tolerance=1.5,
):
    """
    Retrain a student model using knowledge distillation.

    If entropy_scores are provided, alpha is calculated
    adaptively for each batch.

    If an Experience Pool is provided, it is validated
    before retraining begins.

    Parameters
    ----------
    model : nn.Module
        Original trained model.

    data : DataLoader
        Training/experience-pool data.

    epochs : int
        Number of retraining epochs.

    lr : float
        Learning rate.

    alpha : float
        Fixed alpha used when entropy_scores are not provided.

    temperature : float
        Temperature used for soft distillation.

    device : torch.device
        CPU or CUDA device.

    entropy_scores : array-like or None
        Per-sample predictive entropy values.

    experience_pool : ExperiencePool or None
        Optional Experience Pool to validate before
        retraining.

    pool_tolerance : float
        Maximum allowed class imbalance ratio.

    Returns
    -------
    dict
        Retrained model, teacher model, training history,
        and adaptive-alpha statistics.
    """

    # ---------------------------------------------------------
    # Experience Pool fail-safe
    # ---------------------------------------------------------
    if experience_pool is not None:

        print()
        print(
            "Validating Experience Pool "
            "before retraining..."
        )

        experience_pool.print_statistics()

        experience_pool.validate(
            tolerance=pool_tolerance
        )

        print(
            "Experience Pool accepted. "
            "Retraining can begin."
        )
        print()

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    # ---------------------------------------------------------
    # Teacher and Student
    # ---------------------------------------------------------
    teacher = copy.deepcopy(model)
    student = copy.deepcopy(model)

    teacher = teacher.to(device)
    student = student.to(device)

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student.train()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=lr,
    )

    # ---------------------------------------------------------
    # Training history
    # ---------------------------------------------------------
    history = {
        "total_loss": [],
        "hard_loss": [],
        "soft_loss": [],
        "accuracy": [],
        "mean_alpha": [],
    }

    # ---------------------------------------------------------
    # Entropy scores
    # ---------------------------------------------------------
    if entropy_scores is not None:

        entropy_scores = torch.as_tensor(
            entropy_scores,
            dtype=torch.float32,
        )

    # ---------------------------------------------------------
    # Retraining loop
    # ---------------------------------------------------------
    for epoch in range(epochs):

        student.train()

        running_total = 0.0
        running_hard = 0.0
        running_soft = 0.0

        correct = 0
        total = 0

        alpha_sum = 0.0
        alpha_count = 0

        sample_position = 0

        for images, labels in data:

            images = images.to(device)
            labels = labels.to(device)

            batch_size = labels.size(0)

            optimizer.zero_grad()

            # -------------------------------------------------
            # Teacher prediction
            # -------------------------------------------------
            with torch.no_grad():
                teacher_logits = teacher(images)

            # -------------------------------------------------
            # Student prediction
            # -------------------------------------------------
            student_logits = student(images)

            # -------------------------------------------------
            # Adaptive alpha
            # -------------------------------------------------
            if entropy_scores is not None:

                batch_entropy = entropy_scores[
                    sample_position:
                    sample_position + batch_size
                ]

                if len(batch_entropy) != batch_size:
                    raise ValueError(
                        "Entropy scores are not aligned "
                        "with the Experience Pool/DataLoader. "
                        "Use shuffle=False when adaptive "
                        "entropy scores are supplied."
                    )

                batch_alpha = compute_adaptive_alpha(
                    batch_entropy
                )

                # One alpha value for the whole batch.
                batch_alpha = (
                    batch_alpha.mean().item()
                )

                sample_position += batch_size

            else:

                batch_alpha = alpha

            # -------------------------------------------------
            # Distillation loss
            # -------------------------------------------------
            total_loss, hard_loss, soft_loss = (
                distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    alpha=batch_alpha,
                    temperature=temperature,
                )
            )

            # -------------------------------------------------
            # Backpropagation
            # -------------------------------------------------
            total_loss.backward()

            optimizer.step()

            # -------------------------------------------------
            # Loss statistics
            # -------------------------------------------------
            running_total += (
                total_loss.item()
                * batch_size
            )

            running_hard += (
                hard_loss.item()
                * batch_size
            )

            running_soft += (
                soft_loss.item()
                * batch_size
            )

            # -------------------------------------------------
            # Accuracy
            # -------------------------------------------------
            predictions = torch.argmax(
                student_logits,
                dim=1,
            )

            correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total += batch_size

            # -------------------------------------------------
            # Alpha statistics
            # -------------------------------------------------
            alpha_sum += batch_alpha
            alpha_count += 1

        # -----------------------------------------------------
        # Epoch statistics
        # -----------------------------------------------------
        if total == 0:
            raise ValueError(
                "Retraining DataLoader is empty. "
                "Training cannot continue."
            )

        epoch_total = (
            running_total / total
        )

        epoch_hard = (
            running_hard / total
        )

        epoch_soft = (
            running_soft / total
        )

        epoch_accuracy = (
            correct / total
        )

        epoch_alpha = (
            alpha_sum / alpha_count
            if alpha_count > 0
            else alpha
        )

        history["total_loss"].append(
            epoch_total
        )

        history["hard_loss"].append(
            epoch_hard
        )

        history["soft_loss"].append(
            epoch_soft
        )

        history["accuracy"].append(
            epoch_accuracy
        )

        history["mean_alpha"].append(
            epoch_alpha
        )

        print(
            f"Retraining Epoch "
            f"{epoch + 1}/{epochs} | "
            f"Total Loss: {epoch_total:.4f} | "
            f"Hard Loss: {epoch_hard:.4f} | "
            f"Soft Loss: {epoch_soft:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f} | "
            f"Mean Alpha: {epoch_alpha:.4f}"
        )

    # ---------------------------------------------------------
    # Final entropy alignment check
    # ---------------------------------------------------------
    if entropy_scores is not None:

        if sample_position != len(
            entropy_scores
        ):
            raise ValueError(
                "Number of entropy scores does not "
                "match the number of samples processed "
                "by the retraining DataLoader."
            )

    # ---------------------------------------------------------
    # Return results
    # ---------------------------------------------------------
    return {
        "model": student,
        "teacher": teacher,
        "history": history,
        "alpha": alpha,
        "temperature": temperature,
        "adaptive": (
            entropy_scores is not None
        ),
        "experience_pool_validated": (
            experience_pool is not None
        ),
    }