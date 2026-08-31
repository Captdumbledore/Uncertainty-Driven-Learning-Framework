from experience_providers.retrieval_provider import RetrievalProvider


# Dummy dataset
dataset = [
    "image_0",
    "image_1",
    "image_2",
    "image_3",
    "image_4"
]


# Dummy distance values
distances = [
    0.80,
    0.20,
    0.60,
    0.10,
    0.40
]


# Create the provider
provider = RetrievalProvider(dataset, distances)


# Assume image_2 is the outlier
query_index = 2

# Retrieve 3 nearest examples
neighbors = provider.retrieve(query_index, k=3)

print("Outlier:", dataset[query_index])
print("Retrieved Neighbors:", neighbors)