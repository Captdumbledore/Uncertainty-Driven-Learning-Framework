import numpy as np


def calculate_boundary_weight(neighbor_labels):
    """
    Calculate the severity of a decision boundary.

    A higher value means that the neighboring samples
    contain more disagreement between classes.

    Returns:
        float: boundary weight between 0 and 1
    """

    neighbor_labels = np.asarray(neighbor_labels)

    if len(neighbor_labels) == 0:
        raise ValueError("neighbor_labels cannot be empty.")

    _, counts = np.unique(
        neighbor_labels,
        return_counts=True
    )

    majority_count = np.max(counts)

    disagreement = (
        len(neighbor_labels) - majority_count
    ) / len(neighbor_labels)

    return float(disagreement)
def calculate_distance_weight(distances):
    """
    Calculate a weight based on how close the neighbors are.

    Smaller average distance means the query is close to
    its neighbors, so the boundary can be harder.

    Returns:
        float: distance-based weight
    """

    distances = np.asarray(distances, dtype=float)

    if len(distances) == 0:
        raise ValueError("distances cannot be empty.")

    average_distance = np.mean(distances)

    weight = 1.0 / (1.0 + average_distance)

    return float(weight)
def calculate_boundary_severity(neighbor_labels, distances):
    """
    Calculate the overall severity of a decision boundary.

    The score combines:
    - class disagreement among neighbors
    - distance-based closeness of neighbors

    Returns:
        float: boundary severity score between 0 and 1
    """

    boundary_weight = calculate_boundary_weight(
        neighbor_labels
    )

    distance_weight = calculate_distance_weight(
        distances
    )

    severity = boundary_weight * distance_weight

    return float(severity)

def calculate_boundary_info(
    query_embedding,
    embeddings,
    labels,
    k=5
):
    """
    Find the nearest neighbors of a query and
    calculate boundary-related information.
    """

    from .knn import find_knn

    indices, distances, neighbor_labels = find_knn(
        query_embedding,
        embeddings,
        labels,
        k=k
    )

    boundary_weight = calculate_boundary_weight(
        neighbor_labels
    )

    distance_weight = calculate_distance_weight(
        distances
    )

    boundary_severity = calculate_boundary_severity(
        neighbor_labels,
        distances
    )

    return {
        "neighbor_indices": indices,
        "neighbor_distances": distances,
        "neighbor_labels": neighbor_labels,
        "boundary_weight": boundary_weight,
        "distance_weight": distance_weight,
        "boundary_severity": boundary_severity
    }
def is_difficult_boundary(
    boundary_severity,
    severity_threshold=0.15
    ):
    """
    Check whether a boundary should be treated
    as a difficult boundary.

    Returns:
        bool: True if the boundary is difficult.
    """

    return boundary_severity >= severity_threshold