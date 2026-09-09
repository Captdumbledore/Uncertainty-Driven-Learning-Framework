import numpy as np

from experience_providers import CounterexampleProvider


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


    provider = CounterexampleProvider(
        train_labels=train_labels,
        embeddings=embeddings,
        n_retrieve=4
    )


    # Test normal counterexample retrieval
    result = provider.provide(
        sample_idx=0,
        objective=None,
        true_class=0,
        confused_class=1
    )

    print("Counterexample indices:", result)
    print("Number of retrieved samples:", len(result))

    assert len(result) > 0, "Should return counterexamples"
    assert all(isinstance(i, (int, np.integer)) for i in result), \
        "All indices should be integers"


    # Test missing class information
    missing_result = provider.provide(
        sample_idx=0,
        objective=None
    )

    print("Missing class result:", missing_result)

    assert missing_result == [], \
        "Should return empty list when class information is missing"


    # Test odd number of requested experiences
    odd_labels = np.array([
        0, 0, 0, 0,
        1, 1, 1
    ])


    odd_embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [1.0, 1.0],
        [1.1, 1.1],
        [1.2, 1.2]
    ])


    odd_provider = CounterexampleProvider(
        train_labels=odd_labels,
        embeddings=odd_embeddings,
        n_retrieve=5
    )


    odd_result = odd_provider.provide(
        sample_idx=0,
        objective=None,
        true_class=0,
        confused_class=1
    )

    print("Odd n_retrieve result:", odd_result)
    print("Odd n_retrieve count:", len(odd_result))

    assert len(odd_result) == 5, \
        "Odd n_retrieve should return 5 samples"


if __name__ == "__main__":
    main()