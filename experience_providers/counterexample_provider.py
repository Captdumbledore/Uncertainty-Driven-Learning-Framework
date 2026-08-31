from .base_provider import ExperienceProvider


class CounterexampleProvider(ExperienceProvider):

    def __init__(self, dataset, confidences):
        self.dataset = dataset
        self.confidences = confidences

    def retrieve(
        self,
        query=None,
        k=3,
        lower_bound=0.50,
        upper_bound=0.95,
        **kwargs
    ):
        counterexamples = []

        for i, confidence in enumerate(self.confidences):

            if lower_bound <= confidence <= upper_bound:
                counterexamples.append(
                    (self.dataset[i], confidence)
                )

        # Limit the number of returned samples
        return counterexamples[:k]

    def get_counterexamples(
        self,
        lower_bound=0.50,
        upper_bound=0.95
    ):
        return self.retrieve(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            k=len(self.dataset)
        )