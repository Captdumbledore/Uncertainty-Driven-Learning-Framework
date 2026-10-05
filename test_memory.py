from akrm.diagnosis.gap_types import KnowledgeGap, KnowledgeGapType
from akrm.diagnosis.memory import KnowledgeMemory


def main():
    memory = KnowledgeMemory()

    gap = KnowledgeGap(
        sample_idx=10,
        gap_type=KnowledgeGapType.DECISION_BOUNDARY_CONFUSION,
        entropy=0.8,
        confidence=0.6,
        margin=0.1,
        true_class=0,
        predicted_class=1,
        top2_class=0,
        top2_prob=0.4,
        density_dist=1.5,
        explanation="The sample is close to a decision boundary."
    )

    # Initially the memory should be empty
    assert memory.size() == 0
    assert not memory.contains(10)

    # Add the knowledge gap
    memory.add(gap)

    assert memory.size() == 1
    assert memory.contains(10)

    # Retrieve the stored gap
    stored_gap = memory.get(10)

    assert stored_gap is not None
    assert stored_gap.sample_idx == 10
    assert stored_gap.gap_type == KnowledgeGapType.DECISION_BOUNDARY_CONFUSION

    # Add the same gap again
    memory.add(gap)

    assert memory.size() == 1
    assert memory.get_visit_count(10) == 2

    # Remove the gap
    memory.remove(10)

    assert memory.size() == 0
    assert not memory.contains(10)
    assert memory.get(10) is None

    # Test get_or_add()
    memory = KnowledgeMemory()

    first_gap = memory.get_or_add(gap)

    assert memory.size() == 1
    assert memory.get_visit_count(10) == 1

    second_gap = memory.get_or_add(gap)

    assert memory.size() == 1
    assert memory.get_visit_count(10) == 2
    assert second_gap.sample_idx == first_gap.sample_idx

    # Test memory decay
    memory = KnowledgeMemory()
    memory.add(gap)

    assert memory.size() == 1
    assert memory.memory[10].strength == 1.0

    # Apply one decay cycle
    memory.decay_memory(
        decay_rate=0.2,
        min_strength=0.2
    )

    assert memory.size() == 1
    assert memory.memory[10].age == 1
    assert abs(memory.memory[10].strength - 0.8) < 1e-6

    # Apply more decay cycles until the old gap is removed
    for _ in range(20):
        memory.decay_memory(
            decay_rate=0.2,
            min_strength=0.2
        )

    assert memory.size() == 0
    assert not memory.contains(10)

    print("All Knowledge Memory tests passed!")


if __name__ == "__main__":
    main()