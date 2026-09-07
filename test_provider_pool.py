import numpy as np
import torch
from torch.utils.data import TensorDataset

from experience_providers import (
    RetrievalProvider,
    CounterexampleProvider
)
from experience_pool import ExperiencePool


# Small test dataset
data = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
    [6.0]
])

labels = torch.tensor([
    0, 0, 0, 1, 1, 1
])

dataset = TensorDataset(data, labels)


# Simple embeddings
embeddings = np.array([
    [0.0, 0.0],
    [0.1, 0.1],
    [0.2, 0.2],
    [1.0, 1.0],
    [1.1, 1.1],
    [1.2, 1.2]
])

train_labels = np.array([
    0, 0, 0, 1, 1, 1
])


# Create providers
retrieval_provider = RetrievalProvider(
    train_labels=train_labels,
    embeddings=embeddings,
    n_retrieve=2
)

counterexample_provider = CounterexampleProvider(
    train_labels=train_labels,
    embeddings=embeddings,
    n_retrieve=4
)


# Get experiences from both providers
retrieval_indices = retrieval_provider.provide(
    sample_idx=0,
    objective=None,
    true_label=0
)

counterexample_indices = counterexample_provider.provide(
    sample_idx=0,
    objective=None,
    true_class=0,
    confused_class=1
)


# Combine provider outputs
candidate_indices = retrieval_indices + counterexample_indices


# Build Experience Pool
pool = ExperiencePool(dataset)

result = pool.build(
    candidate_indices=candidate_indices
)


print("Retrieval output:", retrieval_indices)
print("Counterexample output:", counterexample_indices)
print("Combined candidates:", candidate_indices)
print("Final pool indices:", pool.get_indices())
print("Pool statistics:", pool.get_statistics())
print("Final pool size:", len(result))

# Test empty candidate list
empty_pool = ExperiencePool(dataset)

empty_result = empty_pool.build(
    candidate_indices=[]
)

print("Empty pool indices:", empty_pool.get_indices())
print("Empty pool size:", len(empty_result))