import torch
from torch.utils.data import TensorDataset

from experience_providers.counterexample_provider import CounterexampleProvider
from experience_providers.retrieval_provider import RetrievalProvider
from experience_pool.experience_pool import ExperiencePool


# ============================================================
# 1. Create dummy dataset
# ============================================================

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


# ============================================================
# 2. CounterexampleProvider
# ============================================================

confidences = [
    0.95,
    0.72,
    0.98,
    0.61,
    0.89
]

counterexample_provider = CounterexampleProvider(
    dataset,
    confidences
)

counterexamples = counterexample_provider.retrieve(
    lower_bound=0.50,
    upper_bound=0.95
)

print("Counterexamples:")
print(counterexamples)


# ============================================================
# 3. RetrievalProvider
# ============================================================

distances = [
    0.80,
    0.20,
    0.60,
    0.10,
    0.40
]

retrieval_provider = RetrievalProvider(
    dataset,
    distances
)

query_index = 2

retrieved_examples = retrieval_provider.retrieve(
    query_index=query_index,
    k=3
)

print("\nRetrieved Examples:")
print(retrieved_examples)


# ============================================================
# 4. Convert provider results to dataset indices
# ============================================================

counterexample_indices = []

for sample, confidence in counterexamples:

    sample_data, sample_label = sample

    for i in range(len(dataset)):

        dataset_sample, dataset_label = dataset[i]

        if (
            torch.equal(sample_data, dataset_sample)
            and sample_label.item() == dataset_label.item()
        ):
            counterexample_indices.append(i)
            break


retrieval_indices = []

for sample, distance in retrieved_examples:

    sample_data, sample_label = sample

    for i in range(len(dataset)):

        dataset_sample, dataset_label = dataset[i]

        if (
            torch.equal(sample_data, dataset_sample)
            and sample_label.item() == dataset_label.item()
        ):
            retrieval_indices.append(i)
            break


print("\nCounterexample Indices:")
print(counterexample_indices)

print("\nRetrieval Indices:")
print(retrieval_indices)


# ============================================================
# 5. Remove duplicates
# ============================================================

unique_indices = list(
    dict.fromkeys(
        counterexample_indices + retrieval_indices
    )
)

print("\nUnique Experience Indices:")
print(unique_indices)


# ============================================================
# 6. Build Experience Pool
# ============================================================

experience_pool = ExperiencePool(dataset)

experience_pool.build(unique_indices)

final_pool = experience_pool.get_pool()

print("\nFinal Experience Pool Size:")
print(len(final_pool))


# ============================================================
# 7. Test missing candidates
# ============================================================

empty_pool = ExperiencePool(dataset)

empty_pool.build([])

print("\nEmpty Experience Pool:")
print(empty_pool.get_pool())


# ============================================================
# 8. Test excessive candidates
# ============================================================

large_candidate_list = [0, 1, 2, 3, 4]

limited_pool = ExperiencePool(dataset)

limited_pool.build(
    large_candidate_list,
    max_samples=3
)

print("\nLimited Experience Pool Size:")
print(len(limited_pool.get_pool()))


# ============================================================
# 9. Test class imbalance
# ============================================================

imbalanced_indices = [0, 1, 2, 3, 4]

balanced_pool = ExperiencePool(dataset)

balanced_pool.build(
    imbalanced_indices,
    max_per_class=1
)

print("\nBalanced Experience Pool Size:")
print(len(balanced_pool.get_pool()))

print("\nBalanced Experience Pool Samples:")

for i in range(len(balanced_pool.get_pool())):

    sample, label = balanced_pool.get_pool()[i]

    print(
        "Sample:",
        sample.item(),
        "Label:",
        label.item()
    )


# ============================================================
# 10. Experience Pool statistics
# ============================================================

statistics = experience_pool.get_statistics()

print("\nExperience Pool Statistics:")
print("Total Samples:", statistics["total_samples"])
print(
    "Class Distribution:",
    statistics["class_distribution"]
)

# ============================================================
# 11. Test common ExperienceProvider interface
# ============================================================

from experience_providers.base_provider import ExperienceProvider

print("\nCommon Provider Interface Test:")

print(
    "CounterexampleProvider:",
    isinstance(counterexample_provider, ExperienceProvider)
)

print(
    "RetrievalProvider:",
    isinstance(retrieval_provider, ExperienceProvider)
)

# ============================================================
# 12. Test planner-style input
# ============================================================

planner_query = "boundary_confusion"

planner_result = counterexample_provider.retrieve(
    query=planner_query,
    k=2
)

print("\nPlanner Input Test:")
print("Query:", planner_query)
print("Retrieved Samples:", planner_result)

# ============================================================
# 13. Test RetrievalProvider with no query
# ============================================================

no_query_result = retrieval_provider.retrieve(
    query_index=None,
    k=3
)

print("\nRetrieval Without Query Test:")
print("Result:", no_query_result)