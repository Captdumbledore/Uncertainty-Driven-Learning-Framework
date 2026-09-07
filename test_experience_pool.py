from torch.utils.data import TensorDataset
import torch

from experience_pool import ExperiencePool


# Small sample dataset
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


# Create Experience Pool
pool = ExperiencePool(dataset)


# Add candidate experiences
result = pool.build(
    candidate_indices=[1, 2, 3, 4],
    max_samples=4
)


print("Pool indices:", pool.get_indices())
print("Pool statistics:", pool.get_statistics())
print("Pool size:", len(result))

# Test maximum pool size
large_pool = ExperiencePool(dataset)

large_result = large_pool.build(
    candidate_indices=[0, 1, 2, 3, 4, 5],
    max_samples=3
)

print("Limited pool indices:", large_pool.get_indices())
print("Limited pool size:", len(large_result))

# Test duplicate candidates
duplicate_pool = ExperiencePool(dataset)

duplicate_result = duplicate_pool.build(
    candidate_indices=[1, 1, 2, 2, 3, 3]
)

print("Duplicate test indices:", duplicate_pool.get_indices())
print("Duplicate test size:", len(duplicate_result))

# Test class imbalance handling
balanced_pool = ExperiencePool(dataset)

balanced_result = balanced_pool.build(
    candidate_indices=[0, 1, 2, 3, 4, 5],
    max_per_class=1
)

print("Balanced pool indices:", balanced_pool.get_indices())
print("Balanced pool statistics:", balanced_pool.get_statistics())
print("Balanced pool size:", len(balanced_result))