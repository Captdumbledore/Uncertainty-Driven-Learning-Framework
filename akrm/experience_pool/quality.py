from typing import Dict, Sequence

import numpy as np
import torch


class ExperienceQualityEvaluator:
    """
    Evaluates the quality of candidate experience images.

    Quality is based on:
    - valid image values
    - non-degenerate image content
    - sufficient image variation
    - usable dynamic range

    The final quality score is normalized to [0, 1].
    """

    def __init__(
        self,
        validity_weight: float = 0.4,
        variation_weight: float = 0.3,
        dynamic_range_weight: float = 0.3,
    ):
        total = (
            validity_weight
            + variation_weight
            + dynamic_range_weight
        )

        if total <= 0:
            raise ValueError(
                "Quality weights must sum to a positive value."
            )

        self.validity_weight = validity_weight / total
        self.variation_weight = variation_weight / total
        self.dynamic_range_weight = dynamic_range_weight / total

    @staticmethod
    def _to_numpy(image) -> np.ndarray:
        """Convert a dataset image to a NumPy array."""
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()

        return np.asarray(image)

    def score_image(self, image) -> float:
        """
        Compute a quality score for one image.

        Returns a value in [0, 1].
        """
        image = self._to_numpy(image)

        # Invalid or empty image
        if image.size == 0:
            return 0.0

        if not np.all(np.isfinite(image)):
            return 0.0

        image = image.astype(np.float32)

        # Variation: reject nearly constant / blank images.
        std = float(np.std(image))

        if std <= 1e-8:
            variation_score = 0.0
        else:
            variation_score = min(std / 1.0, 1.0)

        # Dynamic range: images should contain meaningful
        # variation between low and high values.
        minimum = float(np.min(image))
        maximum = float(np.max(image))
        dynamic_range = maximum - minimum

        if dynamic_range <= 1e-8:
            dynamic_range_score = 0.0
        else:
            dynamic_range_score = min(
                dynamic_range / 4.0,
                1.0,
            )

        validity_score = 0.0 if std <= 1e-8 else 1.0

        score = (
            self.validity_weight * validity_score
            + self.variation_weight * variation_score
            + self.dynamic_range_weight * dynamic_range_score
        )

        return float(np.clip(score, 0.0, 1.0))

    def score_candidates(
        self,
        candidate_indices: Sequence[int],
        dataset,
    ) -> Dict[int, float]:
        """
        Evaluate every candidate image in the dataset.
        """
        scores = {}

        for index in candidate_indices:
            if index < 0 or index >= len(dataset):
                scores[index] = 0.0
                continue

            try:
                image, _ = dataset[index]
                scores[index] = self.score_image(image)
            except Exception:
                scores[index] = 0.0

        return scores