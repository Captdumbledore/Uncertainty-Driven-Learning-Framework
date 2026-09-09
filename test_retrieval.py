import numpy as np

from experience_providers import RetrievalProvider
from test_objective import LearningObjective


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

    provider = RetrievalProvider(
        embeddings=embeddings,
        train_labels=train_labels,
        n_retrieve=2
    )

    result = provider.provide(
        sample_idx=0,
        objective=LearningObjective.IMPROVE_CLASS_SEPARATION,
        true_label=0
    )

    print("Retrieved indices:", result)
    print("Number of retrieved samples:", len(result))

    assert len(result) == 2
    assert all(isinstance(i, (int, np.integer)) for i in result)
    assert 0 not in result

    single_class_labels = np.array([0])
    single_class_embeddings = np.array([
        [0.0, 0.0]
    ])

    single_provider = RetrievalProvider(
        embeddings=single_class_embeddings,
        train_labels=single_class_labels,
        n_retrieve=2
    )

    missing_retrieval = single_provider.provide(
        sample_idx=0,
        objective=LearningObjective.IMPROVE_CLASS_SEPARATION,
        true_label=0
    )

    print("No same-class result:", missing_retrieval)

    assert missing_retrieval == []


if __name__ == "__main__":
    main()