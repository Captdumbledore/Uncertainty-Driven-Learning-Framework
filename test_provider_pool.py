import numpy as np
import torch
from torch.utils.data import TensorDataset

from experience_providers import (
    CounterexampleProvider,
    RetrievalProvider
)
from experience_pool import ExperiencePool


def main():
    train_labels = np.array([0, 0, 0, 1, 1, 1])

    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.1, 1.1],
        [1.2, 1.2]
    ])

    retrieval_provider = RetrievalProvider(
        embeddings=embeddings,
        train_labels=train_labels,
        n_retrieve=2
    )

    counterexample_provider = CounterexampleProvider(
        embeddings=embeddings,
        train_labels=train_labels,
        n_retrieve=4
    )

    retrieval_result = retrieval_provider.provide(
        sample_idx=0,
        objective=None,
        true_label=0
    )

    counterexample_result = counterexample_provider.provide(
        sample_idx=0,
        objective=None,
        true_class=0,
        confused_class=1
    )

    print("Retrieval result:", retrieval_result)
    print("Counterexample result:", counterexample_result)

    assert len(retrieval_result) == 2
    assert 0 not in retrieval_result

    assert len(counterexample_result) == 4
    assert 0 not in counterexample_result

    combined_candidates = retrieval_result + counterexample_result

    print("Combined candidates:", combined_candidates)

    assert len(combined_candidates) == 6

    features = torch.tensor([
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0]
    ])

    labels = torch.tensor([0, 0, 1, 1, 0, 1])

    dataset = TensorDataset(features, labels)

    pool = ExperiencePool(dataset)

    result = pool.build(combined_candidates)

    print("Final pool indices:", result.indices)
    print("Pool statistics:", pool.get_statistics())

    assert len(result.indices) > 0
    assert len(result.indices) == len(set(result.indices))

    stats = pool.get_statistics()

    assert stats["total_samples"] == len(result.indices)
    assert sum(stats["class_distribution"].values()) == stats["total_samples"]

    # Empty candidate test
    pool.clear()

    empty_result = pool.build([])

    print("Empty pool:", empty_result.indices)

    assert empty_result.indices == []


if __name__ == "__main__":
    main()