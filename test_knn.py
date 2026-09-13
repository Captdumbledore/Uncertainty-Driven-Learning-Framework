import numpy as np

from akrm.diagnosis.knn import (
    find_knn,
    diagnose_neighbors,
    diagnose_query,
    calculate_diagnosis_stats
)
from akrm.diagnosis.planner import select_action,create_learning_plan



# Training embeddings
embeddings = np.array([
    [1.0, 1.0],
    [1.2, 1.1],
    [5.0, 5.0],
    [5.2, 5.1],
    [9.0, 9.0]
])

# Corresponding class labels
labels = np.array([
    0,
    0,
    1,
    1,
    2
])

# Query embedding
query = np.array([1.1, 1.0])

# Find 3 nearest neighbors
indices, distances, neighbor_labels = find_knn(
    query,
    embeddings,
    labels,
    k=3
)

print("Neighbor indices:", indices)
print("Neighbor distances:", distances)
print("Neighbor labels:", neighbor_labels)

print("\nTesting invalid k:")

try:
    find_knn(query, embeddings, labels, k=10)
except ValueError as e:
    print("Error caught:", e)

print("\nTesting mismatched labels:")

wrong_labels = np.array([0, 1, 0])

try:
    find_knn(
        query,
        embeddings,
        wrong_labels,
        k=3
    )
except ValueError as e:
    print("Error caught:", e)
print("\nControlled Boundary Example:")

boundary_embeddings = np.array([
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [1.0, 1.0],
    [1.1, 1.0],
    [1.0, 1.1]
])

boundary_labels = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

boundary_query = np.array([0.5, 0.5])

indices, distances, neighbor_labels = find_knn(
    boundary_query,
    boundary_embeddings,
    boundary_labels,
    k=4
)

print("Boundary neighbor labels:", neighbor_labels)
print("Boundary neighbor distances:", distances)

unique, counts = np.unique(
    neighbor_labels,
    return_counts=True
)

majority_count = np.max(counts)

disagreement = (
    len(neighbor_labels) - majority_count
) / len(neighbor_labels)

print("Majority neighbor count:", majority_count)
print("Disagreement:", disagreement)
if disagreement >= 0.25:
    print("Diagnosis: Boundary Confusion")
else:
    print("Diagnosis: Not Boundary Confusion")

    print("\nControlled Outlier Example:")

outlier_embeddings = np.array([
    [0.0, 0.0],
    [0.1, 0.0],
    [0.0, 0.1],
    [1.0, 1.0],
    [1.1, 1.0],
    [1.0, 1.1]
])

outlier_labels = np.array([
    0,
    0,
    0,
    1,
    1,
    1
])

outlier_query = np.array([10.0, 10.0])

indices, distances, neighbor_labels = find_knn(
    outlier_query,
    outlier_embeddings,
    outlier_labels,
    k=3
)

print("Outlier neighbor labels:", neighbor_labels)
print("Outlier neighbor distances:", distances)

average_distance = np.mean(distances)

print("Average neighbor distance:", average_distance)
if average_distance > 5:
    print("Diagnosis: Outlier")
else:
    print("Diagnosis: Not Outlier")

print("\nTesting Diagnosis Function:")

diagnosis = diagnose_neighbors(
    neighbor_labels,
    distances
)

print("Final diagnosis:", diagnosis)
print("\nTesting Boundary Diagnosis Function:")

boundary_diagnosis = diagnose_neighbors(
    np.array([0, 0, 0, 1]),
    np.array([
        0.64031242,
        0.64031242,
        0.70710678,
        0.70710678
    ])
)

print("Boundary diagnosis:", boundary_diagnosis)
print("\nTesting Normal Diagnosis Function:")

normal_diagnosis = diagnose_neighbors(
    np.array([0, 0, 0, 0]),
    np.array([
        0.1,
        0.2,
        0.15,
        0.18
    ])
)

print("Normal diagnosis:", normal_diagnosis)

print("\nTesting Complete Query Diagnosis:")

query_diagnosis = diagnose_query(
    boundary_query,
    boundary_embeddings,
    boundary_labels,
    k=4
)
print("Query diagnosis:", query_diagnosis["diagnosis"])
print("Neighbor labels:", query_diagnosis["neighbor_labels"])
print("Neighbor distances:", query_diagnosis["neighbor_distances"])

print("\nTesting Diagnosis Statistics:")

disagreement, average_distance = calculate_diagnosis_stats(
    np.array([0, 0, 0, 1]),
    np.array([
        0.64031242,
        0.64031242,
        0.70710678,
        0.70710678
    ])
)

print("Disagreement:", disagreement)
print("Average distance:", average_distance)
print("Disagreement:", query_diagnosis["disagreement"])
print("Average distance:", query_diagnosis["average_distance"])




print("\nTesting Knowledge-Guided Planner:")

boundary_action = select_action("Boundary Confusion")
print("Boundary result:", boundary_action)

outlier_action = select_action("Outlier")
print("Outlier result:", outlier_action)

normal_action = select_action("Normal")
print("Normal result:", normal_action)

print("\nTesting Complete Learning Plan:")

learning_plan = create_learning_plan(
    boundary_query,
    boundary_embeddings,
    boundary_labels,
    k=4
)

print("Learning plan:", learning_plan)

print("\nTesting with 256-dimensional embeddings:")

realistic_embeddings = np.random.rand(20, 256)
realistic_labels = np.array([
    0, 0, 0, 0, 1,
    1, 1, 1, 2, 2,
    2, 2, 3, 3, 3,
    4, 4, 4, 5, 5
])

query_embedding = np.random.rand(256)

result = diagnose_query(
    query_embedding,
    realistic_embeddings,
    realistic_labels,
    k=5
)

print("Diagnosis:", result["diagnosis"])
print("Neighbor labels:", result["neighbor_labels"])
print("Neighbor distances:", result["neighbor_distances"])
print("Disagreement:", result["disagreement"])
print("Average distance:", result["average_distance"])

print("\nTesting complete diagnosis with 256-D query:")

query_256 = np.random.rand(256)

training_256 = np.random.rand(20, 256)

training_labels_256 = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    4, 4, 4, 4
])

result = diagnose_query(
    query_embedding=query_256,
    embeddings=training_256,
    labels=training_labels_256,
    k=5
)

print("Diagnosis:", result["diagnosis"])
print("Neighbor labels:", result["neighbor_labels"])
print("Neighbor distances:", result["neighbor_distances"])
print("Disagreement:", result["disagreement"])
print("Average distance:", result["average_distance"])

print("\nTesting complete 256-D learning plan:")

plan = create_learning_plan(
    query_embedding=query_256,
    embeddings=training_256,
    labels=training_labels_256,
    k=5
)

print("Diagnosis:", plan["diagnosis"])
print("Action:", plan["action"])
print("Disagreement:", plan["disagreement"])
print("Average distance:", plan["average_distance"])
