"""
experience_pool.py
------------------
Experience Pool management for Shishya.

Provides:
- Experience Pool creation
- Class-distribution statistics
- Real-time pool statistics
- Class-balance checking
- Automatic class balancing
- Uncertainty-based class-balanced selection
- Training fail-safe validation
"""

from collections import Counter

import numpy as np


class ExperiencePool:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.indices)

    def get_labels(self):
        labels = []

        for index in self.indices:
            _, label = self.dataset[int(index)]
            labels.append(int(label))

        return np.asarray(
            labels,
            dtype=np.int64,
        )

    def get_class_distribution(self):
        labels = self.get_labels()

        counts = Counter(
            int(label)
            for label in labels
        )

        return dict(
            sorted(counts.items())
        )

    def get_statistics(self):
        distribution = (
            self.get_class_distribution()
        )

        pool_size = len(self)

        if distribution:
            minimum = min(
                distribution.values()
            )

            maximum = max(
                distribution.values()
            )

            if minimum > 0:
                imbalance_ratio = (
                    maximum / minimum
                )
            else:
                imbalance_ratio = float(
                    "inf"
                )
        else:
            minimum = 0
            maximum = 0
            imbalance_ratio = 0.0

        return {
            "pool_size": pool_size,
            "num_classes": len(
                distribution
            ),
            "class_distribution": distribution,
            "minimum_class_count": minimum,
            "maximum_class_count": maximum,
            "imbalance_ratio": (
                imbalance_ratio
            ),
        }

    def print_statistics(self):
        stats = self.get_statistics()

        print()
        print("=" * 60)
        print("  EXPERIENCE POOL STATISTICS")
        print("=" * 60)

        print(
            f"  Pool size            : "
            f"{stats['pool_size']:,}"
        )

        print(
            f"  Classes represented  : "
            f"{stats['num_classes']}"
        )

        print(
            f"  Minimum class count  : "
            f"{stats['minimum_class_count']}"
        )

        print(
            f"  Maximum class count  : "
            f"{stats['maximum_class_count']}"
        )

        print(
            f"  Imbalance ratio      : "
            f"{stats['imbalance_ratio']:.3f}"
        )

        print()
        print("  Class distribution:")

        for label, count in stats[
            "class_distribution"
        ].items():
            print(
                f"      Class {label}: {count}"
            )

        print("=" * 60)
        print()

        return stats

    def is_balanced(self, tolerance=1.5):
        stats = self.get_statistics()

        if stats["pool_size"] == 0:
            return False

        return (
            stats["imbalance_ratio"]
            <= tolerance
        )

    def validate(self, tolerance=1.5):
        """
        Validate the Experience Pool before training.

        Raises an error if:
        - The pool is empty.
        - No classes are represented.
        - Not all 10 CIFAR-10 classes are represented.
        - The pool is too imbalanced.
        """

        stats = self.get_statistics()

        if stats["pool_size"] == 0:
            raise ValueError(
                "Experience Pool is empty. "
                "Retraining cannot start."
            )

        if stats["num_classes"] == 0:
            raise ValueError(
                "Experience Pool contains "
                "no classes."
            )

        if stats["num_classes"] < 10:
            raise ValueError(
                f"Experience Pool contains only "
                f"{stats['num_classes']} classes. "
                f"All 10 CIFAR-10 classes are "
                f"required."
            )

        if not self.is_balanced(
            tolerance=tolerance
        ):
            raise ValueError(
                "Experience Pool is too "
                "imbalanced. "
                f"Imbalance ratio: "
                f"{stats['imbalance_ratio']:.3f}, "
                f"allowed: {tolerance:.3f}"
            )

        print(
            "Experience Pool validation: PASSED"
        )

        return True

    def balance(
        self,
        num_classes=10,
        target_size=None,
        random_seed=42,
    ):
        """
        Balance the current pool by sampling
        equally from each class.

        This method can only balance using
        samples already present in the pool.
        """

        if len(self) == 0:
            raise ValueError(
                "Cannot balance an empty "
                "Experience Pool."
            )

        if target_size is None:
            target_size = len(self)

        samples_per_class = (
            target_size // num_classes
        )

        if samples_per_class == 0:
            raise ValueError(
                "target_size is too small."
            )

        labels = self.get_labels()

        rng = np.random.default_rng(
            random_seed
        )

        balanced_indices = []

        for class_id in range(
            num_classes
        ):
            class_positions = np.where(
                labels == class_id
            )[0]

            if len(class_positions) < (
                samples_per_class
            ):
                raise ValueError(
                    f"Class {class_id} has only "
                    f"{len(class_positions)} "
                    f"samples in the current "
                    f"pool. Cannot create a "
                    f"balanced pool of "
                    f"{target_size} samples."
                )

            selected_positions = (
                rng.choice(
                    class_positions,
                    size=samples_per_class,
                    replace=False,
                )
            )

            balanced_indices.extend(
                self.indices[
                    selected_positions
                ]
            )

        balanced_indices = np.asarray(
            balanced_indices,
            dtype=np.int64,
        )

        return ExperiencePool(
            self.dataset,
            balanced_indices,
        )

    @classmethod
    def from_uncertainty(
        cls,
        dataset,
        entropy_scores,
        pool_size=1000,
        num_classes=10,
    ):
        """
        Build a class-balanced Experience Pool
        directly from the complete dataset.

        For each class, the samples with the
        highest entropy are selected.

        Example:
            pool_size=1000
            num_classes=10

        gives:

            100 samples per class.
        """

        entropy_scores = np.asarray(
            entropy_scores
        )

        if len(entropy_scores) != len(
            dataset
        ):
            raise ValueError(
                "Number of entropy scores "
                "must match dataset size."
            )

        if pool_size % num_classes != 0:
            raise ValueError(
                "pool_size must be divisible "
                "by num_classes."
            )

        samples_per_class = (
            pool_size // num_classes
        )

        labels = []

        print(
            "Reading dataset labels..."
        )

        for index in range(
            len(dataset)
        ):
            _, label = dataset[index]
            labels.append(int(label))

        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        selected_indices = []

        print(
            "Selecting most uncertain "
            "samples per class..."
        )

        for class_id in range(
            num_classes
        ):
            class_indices = np.where(
                labels == class_id
            )[0]

            if len(class_indices) < (
                samples_per_class
            ):
                raise ValueError(
                    f"Class {class_id} has only "
                    f"{len(class_indices)} "
                    f"dataset samples."
                )

            class_entropies = (
                entropy_scores[
                    class_indices
                ]
            )

            ranking = np.argsort(
                class_entropies
            )[::-1]

            top_positions = ranking[
                :samples_per_class
            ]

            class_selected = (
                class_indices[
                    top_positions
                ]
            )

            selected_indices.extend(
                class_selected.tolist()
            )

        selected_indices = np.asarray(
            selected_indices,
            dtype=np.int64,
        )

        pool = cls(
            dataset,
            selected_indices,
        )

        print(
            "Class-balanced Experience "
            "Pool created."
        )

        return pool