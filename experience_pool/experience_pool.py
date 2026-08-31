from torch.utils.data import Subset, ConcatDataset


class ExperiencePool:

    def __init__(self, dataset):
        self.dataset = dataset
        self.pool = None

    def build(self, indices, max_samples=None, max_per_class=None):

        # Handle missing candidates
        if not indices:
            return self.pool

        # Remove duplicate indices
        indices = list(dict.fromkeys(indices))

        # Handle excessive candidates
        if max_samples is not None:
            indices = indices[:max_samples]

        # Handle class imbalance
        if max_per_class is not None:
            class_counts = {}
            balanced_indices = []

            for index in indices:
                _, label = self.dataset[index]

                if hasattr(label, "item"):
                    label = label.item()

                if class_counts.get(label, 0) < max_per_class:
                    balanced_indices.append(index)
                    class_counts[label] = class_counts.get(label, 0) + 1

            indices = balanced_indices

        # Check again after filtering
        if not indices:
            return self.pool

        # Create subset
        selected_data = Subset(self.dataset, indices)

        # Add to experience pool
        if self.pool is None:
            self.pool = selected_data
        else:
            self.pool = ConcatDataset(
                [self.pool, selected_data]
            )

        return self.pool

    def get_pool(self):
        return self.pool

    def get_statistics(self):

        # Handle empty pool
        if self.pool is None:
            return {
                "total_samples": 0,
                "class_distribution": {}
            }

        class_distribution = {}

        for i in range(len(self.pool)):
            _, label = self.pool[i]

            if hasattr(label, "item"):
                label = label.item()

            class_distribution[label] = (
                class_distribution.get(label, 0) + 1
            )

        return {
            "total_samples": len(self.pool),
            "class_distribution": class_distribution
        }