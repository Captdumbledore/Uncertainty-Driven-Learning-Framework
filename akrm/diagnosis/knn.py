import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_knn(query_embedding, embeddings, labels, k=5):
    """
    Find the k nearest training samples
    for a given query embedding.
    """
    if len(embeddings) != len(labels):
        raise ValueError("Number of embeddings and labels must match.")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    if k > len(embeddings):
        raise ValueError("k cannot be greater than the number of embeddings.")

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean"
    )

    knn.fit(embeddings)

    query_embedding = np.asarray(query_embedding).reshape(1, -1)

    distances, indices = knn.kneighbors(
        query_embedding
    )
    neighbor_labels = labels[indices[0]]


    return indices[0], distances[0], neighbor_labels
def calculate_diagnosis_stats(neighbor_labels, distances):
    """
    Calculate statistics used for knowledge-gap diagnosis.
    """

    _, counts = np.unique(
       neighbor_labels,
       return_counts=True
       )

    majority_count = np.max(counts)

    disagreement = (
        len(neighbor_labels) - majority_count
    ) / len(neighbor_labels)

    average_distance = np.mean(distances)

    return disagreement, average_distance

def diagnose_neighbors(
    neighbor_labels,
    distances,
    disagreement_threshold=0.25,
    outlier_threshold=5.0
):
    """
    Diagnose a query based on its nearest neighbors.

    Returns:
        "Boundary Confusion"
        "Outlier"
        "Normal"
    """

    disagreement, average_distance = calculate_diagnosis_stats(
        neighbor_labels,
        distances
    )

    if disagreement >= disagreement_threshold:
        return "Boundary Confusion"

    elif average_distance > outlier_threshold:
        return "Outlier"

    else:
        return "Normal"
def diagnose_query(
    query_embedding,
    embeddings,
    labels,
    k=5,
    disagreement_threshold=0.25,
    outlier_threshold=5.0
):
    """
    Find nearest neighbors and diagnose
    the query embedding.
    """

    indices, distances, neighbor_labels = find_knn(
        query_embedding,
        embeddings,
        labels,
        k=k
    )

    diagnosis = diagnose_neighbors(
        neighbor_labels,
        distances,
        disagreement_threshold=disagreement_threshold,
        outlier_threshold=outlier_threshold
    )

    disagreement, average_distance = calculate_diagnosis_stats(
        neighbor_labels,
        distances
    )

    return {
        "diagnosis": diagnosis,
        "neighbor_indices": indices,
        "neighbor_distances": distances,
        "neighbor_labels": neighbor_labels,
        "disagreement": disagreement,
        "average_distance": average_distance
    }
