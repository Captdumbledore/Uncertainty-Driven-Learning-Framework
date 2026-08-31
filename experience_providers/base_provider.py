from abc import ABC, abstractmethod


class ExperienceProvider(ABC):

    @abstractmethod
    def retrieve(self, query=None, k=3, **kwargs):
        """
        Retrieve experience samples based on planner input.

        Parameters:
            query: Information provided by the Knowledge-Guided Planner.
            k: Maximum number of samples to retrieve.
            kwargs: Additional strategy-specific parameters.
        """
        pass