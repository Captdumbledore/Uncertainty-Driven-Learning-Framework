from .base_provider import ExperienceProvider


class RetrievalProvider(ExperienceProvider):

    def __init__(self, dataset, distances):
        self.dataset = dataset
        self.distances = distances

    def retrieve(
        self,
        query=None,
        k=3,
        query_index=None,
        **kwargs
    ):
        # If no query index is provided, there is nothing to retrieve
        if query_index is None:
            return []

        indexed_distances = list(enumerate(self.distances))

        # Do not retrieve the query/outlier itself
        indexed_distances = [
            item
            for item in indexed_distances
            if item[0] != query_index
        ]

        # Smaller distance means more similar
        indexed_distances.sort(key=lambda x: x[1])

        neighbors = []

        for index, distance in indexed_distances[:k]:
            neighbors.append(
                (self.dataset[index], distance)
            )

        return neighbors

    def get_nearest_neighbors(self, query_index, k=3):
        return self.retrieve(
            query_index=query_index,
            k=k
        )