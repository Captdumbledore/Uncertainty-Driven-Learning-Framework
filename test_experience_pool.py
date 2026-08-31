import torch
from torch.utils.data import TensorDataset

from experience_pool.experience_pool import ExperiencePool


# Create a small dummy dataset
data = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0]
])

labels = torch.tensor([
    0,
    0,
    1,
    1,
    1
])


dataset = TensorDataset(data, labels)


# Create Experience Pool
pool = ExperiencePool(dataset)


# Select some experiences
selected_indices = [1, 3, 4]

experience_dataset = pool.build(selected_indices)


print("Experience Pool Size:", len(experience_dataset))

print("\nExperience Samples:")

for i in range(len(experience_dataset)):
    sample, label = experience_dataset[i]
    print("Sample:", sample.item(), "Label:", label.item())