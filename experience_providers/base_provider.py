from abc import ABC, abstractmethod
from typing import List, Any


class ExperienceProvider(ABC):
    """
    Abstract Base Class for all Experience Providers.
    """

    @abstractmethod
    def provide(self, sample_idx: int, objective) -> List[Any]:
        """
        Provides a list of task-appropriate learning experiences
        intended to satisfy the given learning objective.

        Args:
            sample_idx: Index of the uncertain sample.
            objective: The learning objective derived from diagnosis.

        Returns:
            List of experiences (can be indices into a dataset or actual generated Tensors).
        """
        pass