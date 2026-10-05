import numpy as np
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from akrm.objective import LearningObjective

from .base_provider import BaseProvider

class CounterexampleProvider(BaseProvider):
    """
    Retrieves examples from both the true and confused classes
    to strengthen class separation.
    """

    def __init__(
        self,
        train_labels: np.ndarray,
        embeddings: np.ndarray,
        n_retrieve: int = 8
    ):
        self.train_labels = train_labels
        self.embeddings = embeddings
        self.n_retrieve = n_retrieve

    def provide(
        self,
        sample_idx: int,
        objective: "LearningObjective",
        true_class: int = None,
        confused_class: int = None
    ) -> List[int]:
        """
        Retrieves nearby examples from the true and confused classes.

        Args:
            sample_idx: Index of the uncertain sample.
            objective: Learning objective supplied by the planner.
            true_class: Ground-truth class of the uncertain sample.
            confused_class: Class the model is confusing with the true class.

        Returns:
            List of dataset indices selected as counterexamples.
        """

        # Required class information is not available.
        if true_class is None or confused_class is None:
            return []

        candidates = []

        sample_emb = self.embeddings[sample_idx]

        # Divide the requested number between the two classes.
        n_per_class = self.n_retrieve // 2
        n_extra = self.n_retrieve - (2 * n_per_class)

        for class_id, n_requested in [
            (true_class, n_per_class + n_extra),
            (confused_class, n_per_class)
        ]:

            class_mask = (self.train_labels == class_id)

            if class_id == true_class:
                class_mask[sample_idx] = False
            class_indices = np.where(class_mask)[0]

            # No candidates available for this class.
            if len(class_indices) == 0:
                continue

            class_embs = self.embeddings[class_indices]

            # Calculate distance from the uncertain sample.
            distances = np.linalg.norm(
                class_embs - sample_emb,
                axis=1
            )

            # Select nearest examples.
            sorted_order = np.argsort(distances)

            n_take = min(
                n_requested,
                len(class_indices)
            )

            if n_take == 0:
                continue

            chosen = sorted_order[:n_take]

            candidates.extend(
                class_indices[chosen].tolist()
            )

        return candidates