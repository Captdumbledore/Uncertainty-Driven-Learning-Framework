import numpy as np

from experience_providers import CounterexampleProvider


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


result = provider.provide(
    sample_idx=0,
    objective=None,
    true_class=0,
    confused_class=1
)


print("Counterexample indices:", result)
print("Number of retrieved samples:", len(result))

# Test missing class information
missing_result = provider.provide(
    sample_idx=0,
    objective=None
)

print("Missing class result:", missing_result)

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

odd_result = odd_provider.provide(
    sample_idx=0,
    objective=None,
    true_class=0,
    confused_class=1
)

print("Odd n_retrieve result:", odd_result)
print("Odd n_retrieve count:", len(odd_result))