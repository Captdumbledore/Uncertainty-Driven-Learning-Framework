import numpy as np

from akrm.diagnosis.knn import (
    find_knn,
    diagnose_neighbors,
    diagnose_query,
    calculate_diagnosis_stats,
)

from akrm.diagnosis.planner import (
    ProviderType,
    LearningObjective,
    LearningObjectiveGenerator,
    KnowledgeGuidedExperiencePlanner,
)

from akrm.diagnosis.gap_types import KnowledgeGapType


def main():

    # ---------------------------------------------------------
    # Basic K-NN test
    # ---------------------------------------------------------

    embeddings = np.array([
        [1.0, 1.0],
        [1.2, 1.1],
        [5.0, 5.0],
        [5.2, 5.1],
        [9.0, 9.0]
    ])

    labels = np.array([
        0,
        0,
        1,
        1,
        2
    ])

    query = np.array([1.1, 1.0])

    indices, distances, neighbor_labels = find_knn(
        query,
        embeddings,
        labels,
        k=3
    )

    assert len(indices) == 3
    assert len(distances) == 3
    assert len(neighbor_labels) == 3
    assert indices[0] == 0

    print("Neighbor indices:", indices)
    print("Neighbor distances:", distances)
    print("Neighbor labels:", neighbor_labels)

    # ---------------------------------------------------------
    # Invalid k
    # ---------------------------------------------------------

    print("\nTesting invalid k:")

    try:
        find_knn(query, embeddings, labels, k=10)
        assert False, "Expected ValueError for invalid k"
    except ValueError as e:
        print("Error caught:", e)

    # ---------------------------------------------------------
    # Mismatched labels
    # ---------------------------------------------------------

    print("\nTesting mismatched labels:")

    wrong_labels = np.array([0, 1, 0])

    try:
        find_knn(
            query,
            embeddings,
            wrong_labels,
            k=3
        )
        assert False, "Expected ValueError for mismatched labels"
    except ValueError as e:
        print("Error caught:", e)

    # ---------------------------------------------------------
    # Boundary Confusion
    # ---------------------------------------------------------

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

    disagreement, average_distance = calculate_diagnosis_stats(
        neighbor_labels,
        distances
    )

    print("Disagreement:", disagreement)

    boundary_diagnosis = diagnose_neighbors(
        neighbor_labels,
        distances
    )

    print("Diagnosis:", boundary_diagnosis)

    assert boundary_diagnosis == "Boundary Confusion"
    assert disagreement == 0.25

    # ---------------------------------------------------------
    # Outlier
    # ---------------------------------------------------------

    print("\nControlled Outlier Example:")

    outlier_query = np.array([10.0, 10.0])

    indices, distances, neighbor_labels = find_knn(
        outlier_query,
        boundary_embeddings,
        boundary_labels,
        k=3
    )

    average_distance = np.mean(distances)

    print("Outlier neighbor labels:", neighbor_labels)
    print("Outlier neighbor distances:", distances)
    print("Average neighbor distance:", average_distance)

    outlier_diagnosis = diagnose_neighbors(
        neighbor_labels,
        distances
    )

    print("Diagnosis:", outlier_diagnosis)

    assert outlier_diagnosis == "Outlier"
    assert average_distance > 5.0

    # ---------------------------------------------------------
    # Normal diagnosis
    # ---------------------------------------------------------

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

    assert normal_diagnosis == "Normal"

    # ---------------------------------------------------------
    # Complete query diagnosis
    # ---------------------------------------------------------

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

    assert query_diagnosis["diagnosis"] == "Boundary Confusion"
    assert len(query_diagnosis["neighbor_labels"]) == 4

    # ---------------------------------------------------------
    # Diagnosis statistics
    # ---------------------------------------------------------

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

    assert disagreement == 0.25
    assert average_distance > 0.0

    # ---------------------------------------------------------
    # Knowledge-Guided Planner
    # ---------------------------------------------------------

    print("\nTesting Knowledge-Guided Planner:")

    objective_generator = LearningObjectiveGenerator()

    boundary_objective = objective_generator.generate(
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
    )

    print("Boundary objective:", boundary_objective)

    assert (
        boundary_objective
        == LearningObjective.IMPROVE_CLASS_SEPARATION
    )

    outlier_objective = objective_generator.generate(
        KnowledgeGapType.OOD_CONCEPT
    )

    print("OOD objective:", outlier_objective)

    assert (
        outlier_objective
        == LearningObjective.IMPROVE_ROBUSTNESS
    )

    planner = KnowledgeGuidedExperiencePlanner()

    boundary_strategy = planner.select_strategy(
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION,
        margin=0.05
    )

    print("Boundary strategy:", boundary_strategy)

    assert boundary_strategy == ProviderType.COUNTEREXAMPLE

    normal_strategy = planner.select_strategy(
        KnowledgeGapType.GENERAL_UNCERTAINTY
    )

    print("General uncertainty strategy:", normal_strategy)

    assert normal_strategy == ProviderType.RETRIEVAL

    print("\nAll K-NN and planner tests passed!")


if __name__ == "__main__":
    main()