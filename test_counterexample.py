import numpy as np

from experience_providers import CounterexampleProvider
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

    provider = CounterexampleProvider(
        embeddings=embeddings,
        train_labels=train_labels,
        n_retrieve=4
    )

    result = provider.provide(
        sample_idx=0,
        objective=LearningObjective.IMPROVE_CLASS_SEPARATION,
        true_class=0,
        confused_class=1
    )

    print("Counterexample indices:", result)
    print("Number of retrieved samples:", len(result))

    assert len(result) == 4
    assert all(isinstance(i, (int, np.integer)) for i in result)

    missing_result = provider.provide(
        sample_idx=0,
        objective=LearningObjective.IMPROVE_CLASS_SEPARATION
    )

    print("Missing class result:", missing_result)

    assert missing_result == []

    odd_labels = np.array([0, 0, 0, 0, 1, 1, 1])

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
        embeddings=odd_embeddings,
        train_labels=odd_labels,
        n_retrieve=5
    )

    odd_result = odd_provider.provide(
        sample_idx=0,
        objective=LearningObjective.IMPROVE_CLASS_SEPARATION,
        true_class=0,
        confused_class=1
    )

    print("Odd n_retrieve result:", odd_result)
    print("Odd n_retrieve count:", len(odd_result))

    assert len(odd_result) == 5


if __name__ == "__main__":
    main()