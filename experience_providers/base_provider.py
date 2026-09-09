from abc import ABC, abstractmethod
from typing import List, Any

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from akrm.objective import LearningObjective


class BaseProvider(ABC):
    """
    Abstract Base Class for all Experience Providers.
    """

    @abstractmethod
    def provide(
        self,
        sample_idx: int,
        objective: "LearningObjective"
    ) -> List[Any]:
        """
        Provides a list of task-appropriate learning experiences
        intended to satisfy the given learning objective.

        Args:
            sample_idx: Index of the uncertain sample.
            objective: Learning objective derived from diagnosis.

        Returns:
            List of experiences.
        """
        pass