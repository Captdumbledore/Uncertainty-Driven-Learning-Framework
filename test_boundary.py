import numpy as np

from akrm.diagnosis.boundary import (
    calculate_boundary_weight,
    calculate_distance_weight,
    calculate_boundary_severity,
    calculate_boundary_info,
    is_difficult_boundary
)


def main():

    # --------------------------------------------------
    # Test 1: Boundary weight
    # --------------------------------------------------

    weight = calculate_boundary_weight([0, 0, 0, 1])

    assert abs(weight - 0.25) < 1e-6

    print("Boundary weight test passed.")


    # --------------------------------------------------
    # Test 2: Distance weight
    # --------------------------------------------------

    weight = calculate_distance_weight([0.5, 0.5])

    assert abs(weight - 0.6666666667) < 1e-6

    print("Distance weight test passed.")


    # --------------------------------------------------
    # Test 3: Boundary severity
    # --------------------------------------------------

    severity = calculate_boundary_severity(
        [0, 0, 0, 1],
        [0.5, 0.5, 0.5, 0.5]
    )

    expected = 0.25 * (1.0 / 1.5)

    assert abs(severity - expected) < 1e-6

    print("Boundary severity test passed.")


    # --------------------------------------------------
    # Test 4: Complete boundary information
    # --------------------------------------------------

    embeddings = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [10, 10]
        ],
        dtype=float
    )

    labels = np.array(
        [0, 0, 0, 1, 1]
    )

    result = calculate_boundary_info(
        np.array([0.5, 0.5]),
        embeddings,
        labels,
        k=4
    )

    assert len(result["neighbor_labels"]) == 4
    assert "boundary_weight" in result
    assert "distance_weight" in result
    assert "boundary_severity" in result

    print("Boundary information test passed.")


    # --------------------------------------------------
    # Test 5: Difficult boundary
    # --------------------------------------------------

    assert is_difficult_boundary(0.20) is True
    assert is_difficult_boundary(0.10) is False

    print("Difficult boundary test passed.")


    print("\nAll boundary weighting tests passed!")


if __name__ == "__main__":
    main()