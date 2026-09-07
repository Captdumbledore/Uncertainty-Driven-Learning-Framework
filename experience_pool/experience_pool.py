from typing import List, Optional
from collections import Counter

from torch.utils.data import Dataset, Subset


class ExperiencePool:
    """
    Maintains a pool of selected experience indices and constructs
    a PyTorch Subset from the original training dataset.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.indices = []

    def build(
        self,
        candidate_indices: List[int],
        max_samples: Optional[int] = None,
        max_per_class: Optional[int] = None
    ) -> Subset:
        """
        Builds the Experience Pool from candidate dataset indices.

        Args:
            candidate_indices: Candidate indices provided by Experience Providers.
            max_samples: Maximum total number of experiences.
            max_per_class: Maximum number of experiences from each class.

        Returns:
            A PyTorch Subset containing the selected experiences.
        """

        if not candidate_indices:
            return Subset(self.dataset, self.indices)

        valid_indices = []

        for index in candidate_indices:
            if 0 <= index < len(self.dataset):
                valid_indices.append(index)

        valid_indices = list(dict.fromkeys(valid_indices))

        if max_per_class is not None:
            class_counts = Counter()
            balanced_indices = []

            for index in valid_indices:
                _, label = self.dataset[index]

                if hasattr(label, "item"):
                    label = label.item()

                if class_counts[label] < max_per_class:
                    balanced_indices.append(index)
                    class_counts[label] += 1

            valid_indices = balanced_indices

        if max_samples is not None:
            valid_indices = valid_indices[:max_samples]

        existing_indices = set(self.indices)

        for index in valid_indices:
            if index not in existing_indices:
                self.indices.append(index)

        return Subset(self.dataset, self.indices)

    def get_statistics(self):
        """
        Returns basic statistics about the Experience Pool.
        """

        class_distribution = Counter()

        for index in self.indices:
            _, label = self.dataset[index]

            if hasattr(label, "item"):
                label = label.item()

            class_distribution[label] += 1

        return {
            "total_samples": len(self.indices),
            "class_distribution": dict(class_distribution)
        }

    def get_indices(self) -> List[int]:
        """
        Returns the indices currently stored in the Experience Pool.
        """

        return list(self.indices)

    def clear(self):
        """
        Clears the Experience Pool.
        """

        self.indices = []