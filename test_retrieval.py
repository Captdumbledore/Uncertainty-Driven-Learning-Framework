import numpy as np

from experience_providers import RetrievalProvider


def main():

    # Sample labels for a small test dataset
    train_labels = np.array([
        0, 0, 0, 1, 1, 1
    ])


    # Simple 2D embeddings
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [1.0, 1.0],
        [1.1, 1.1],
        [1.2, 1.2]
    ])


    provider = RetrievalProvider(
        train_labels=train_labels,
        embeddings=embeddings,
        n_retrieve=2
    )


    # Test normal retrieval
    result = provider.provide(
        sample_idx=0,
        objective=None,
        true_label=0
    )

    print("Retrieved indices:", result)
    print("Number of retrieved samples:", len(result))

    assert len(result) == 2, \
        "Should retrieve 2 samples"

    assert all(isinstance(i, (int, np.integer)) for i in result), \
        "All indices should be integers"

    assert 0 not in result, \
        "The sample itself should not be retrieved"


    # Test when no other samples exist in the same class
    single_class_labels = np.array([0])

    single_class_embeddings = np.array([
        [0.0, 0.0]
    ])


    single_provider = RetrievalProvider(
        train_labels=single_class_labels,
        embeddings=single_class_embeddings,
        n_retrieve=2
    )


    missing_retrieval = single_provider.provide(
        sample_idx=0,
        objective=None,
        true_label=0
    )

    print("No same-class result:", missing_retrieval)

    assert missing_retrieval == [], \
        "Should return empty list when no same-class candidates exist"


if __name__ == "__main__":
    main()