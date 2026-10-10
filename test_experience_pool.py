import torch
from torch.utils.data import TensorDataset

from akrm.experience_pool import ExperiencePool, ExperienceSelector


def main():
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

    # Normal pool construction
    pool = ExperiencePool(dataset)

    result = pool.build([0, 1, 2, 3])

    print("Pool indices:", result.indices)
    print("Pool statistics:", pool.get_statistics())

    assert result.indices == [0, 1, 2, 3]
    assert pool.get_statistics()["total_samples"] == 4
    assert pool.get_statistics()["class_distribution"] == {
        0: 2,
        1: 2
    }

    # Maximum number of samples
    pool.clear()

    result = pool.build(
        [0, 1, 2, 3],
        max_samples=3
    )

    print("Limited pool:", result.indices)

    assert result.indices == [0, 1, 2]

    # Duplicate handling
    pool.clear()

    result = pool.build([1, 2, 2, 3, 1])

    print("Duplicate-free pool:", result.indices)

    assert result.indices == [1, 2, 3]

    # Class balancing
    pool.clear()

    result = pool.build(
        [0, 1, 2, 3, 4, 5],
        max_per_class=1
    )

    print("Balanced pool:", result.indices)
    print("Balanced statistics:", pool.get_statistics())

    assert len(result.indices) == 2
    assert pool.get_statistics()["class_distribution"] == {
        0: 1,
        1: 1
    }

    # Empty candidate list
    pool.clear()

    result = pool.build([])

    print("Empty pool:", result.indices)

    assert result.indices == []

    # Experience quality evaluation
    embeddings = torch.tensor([
        [0.0, 0.0],
        [0.1, 0.1],
        [1.0, 1.0],
        [2.0, 2.0]
    ]).numpy()

    selector = ExperienceSelector()

    quality_scores = selector.quality_scores(
        [0, 1, 2, 3],
        embeddings
    )

    print("Quality scores:", quality_scores)

    assert len(quality_scores) == 4
    assert all(0.0 <= score <= 1.0 for score in quality_scores.values())
    assert quality_scores[0] <= quality_scores[1]
    assert quality_scores[1] <= quality_scores[2]
    assert quality_scores[2] <= quality_scores[3]
if __name__ == "__main__":
    main()
    