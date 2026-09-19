from typing import Dict, List, Optional, Sequence

import numpy as np


class ExperienceSelector:
    """
    Scores and selects candidate experiences using quality,
    diversity, and novelty signals.
    """

    def __init__(
        self,
        quality_weight: float = 0.4,
        diversity_weight: float = 0.3,
        novelty_weight: float = 0.3,
    ):
        total = quality_weight + diversity_weight + novelty_weight

        if total <= 0:
            raise ValueError("Selection weights must sum to a positive value.")

        self.quality_weight = quality_weight / total
        self.diversity_weight = diversity_weight / total
        self.novelty_weight = novelty_weight / total

    @staticmethod
    def _normalize(values: Sequence[float]) -> np.ndarray:
        """Normalize scores to the range [0, 1]."""
        values = np.asarray(values, dtype=float)

        if len(values) == 0:
            return values

        minimum = np.min(values)
        maximum = np.max(values)

        if maximum == minimum:
            return np.ones(len(values))

        return (values - minimum) / (maximum - minimum)

    def quality_scores(
        self,
        candidate_indices: Sequence[int],
        embeddings: np.ndarray,
    ) -> Dict[int, float]:
        """
        Assign a quality score based on embedding validity.

        Valid, finite embeddings receive a quality score of 1.
        Invalid or non-finite embeddings receive 0.
        """
        scores = {}

        for index in candidate_indices:
            if index < 0 or index >= len(embeddings):
                scores[index] = 0.0
                continue

            embedding = np.asarray(embeddings[index])

            if embedding.size == 0 or not np.all(np.isfinite(embedding)):
                scores[index] = 0.0
            else:
                scores[index] = 1.0

        return scores

    def novelty_scores(
        self,
        candidate_indices: Sequence[int],
        embeddings: np.ndarray,
        reference_indices: Optional[Sequence[int]] = None,
    ) -> Dict[int, float]:
        """
        Score candidates according to distance from reference experiences.

        Larger distance means greater novelty.
        """
        candidates = list(candidate_indices)

        if not candidates:
            return {}

        if reference_indices is None:
            reference_indices = []

        valid_references = [
            index
            for index in reference_indices
            if 0 <= index < len(embeddings)
        ]

        scores = []

        for index in candidates:
            if index < 0 or index >= len(embeddings):
                scores.append(0.0)
                continue

            candidate_embedding = np.asarray(embeddings[index], dtype=float)

            if not np.all(np.isfinite(candidate_embedding)):
                scores.append(0.0)
                continue

            if valid_references:
                reference_embeddings = np.asarray(
                    embeddings[valid_references],
                    dtype=float,
                )
                distances = np.linalg.norm(
                    reference_embeddings - candidate_embedding,
                    axis=1,
                )
                scores.append(float(np.mean(distances)))
            else:
                scores.append(1.0)

        normalized = self._normalize(scores)

        return {
            index: float(score)
            for index, score in zip(candidates, normalized)
        }

    def diversity_scores(
        self,
        candidate_indices: Sequence[int],
        embeddings: np.ndarray,
    ) -> Dict[int, float]:
        """
        Estimate candidate diversity using distance from other candidates.

        Candidates farther from the other candidates receive higher scores.
        """
        candidates = list(candidate_indices)

        if not candidates:
            return {}

        if len(candidates) == 1:
            return {candidates[0]: 1.0}

        candidate_embeddings = np.asarray(
            embeddings[candidates],
            dtype=float,
        )

        scores = []

        for i in range(len(candidates)):
            distances = np.linalg.norm(
                candidate_embeddings - candidate_embeddings[i],
                axis=1,
            )

            distances[i] = 0.0

            non_zero_distances = distances[distances > 0]

            if len(non_zero_distances) == 0:
                scores.append(0.0)
            else:
                scores.append(float(np.mean(non_zero_distances)))

        normalized = self._normalize(scores)

        return {
            index: float(score)
            for index, score in zip(candidates, normalized)
        }

    def select_top_n(
        self,
        candidate_indices: Sequence[int],
        embeddings: np.ndarray,
        top_n: int,
        reference_indices: Optional[Sequence[int]] = None,
    ) -> List[int]:
        """
        Greedily select the highest-scoring Top-N candidates.
        """
        if top_n <= 0:
            return []

        candidates = list(dict.fromkeys(candidate_indices))

        if not candidates:
            return []

        quality = self.quality_scores(
            candidates,
            embeddings,
        )

        diversity = self.diversity_scores(
            candidates,
            embeddings,
        )

        novelty = self.novelty_scores(
            candidates,
            embeddings,
            reference_indices,
        )

        combined_scores = {}

        for index in candidates:
            combined_scores[index] = (
                self.quality_weight * quality[index]
                + self.diversity_weight * diversity[index]
                + self.novelty_weight * novelty[index]
            )

        ranked = sorted(
            candidates,
            key=lambda index: (-combined_scores[index], index),
        )

        return ranked[:top_n]