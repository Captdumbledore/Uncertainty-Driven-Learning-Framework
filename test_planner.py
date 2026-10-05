from akrm.diagnosis.gap_types import (
    KnowledgeGap,
    KnowledgeGapType
)

from akrm.diagnosis.planner import (
    KnowledgeGuidedExperiencePlanner,
    LearningObjective,
    ProviderType
)


def main():

    gap = KnowledgeGap(
        sample_idx=10,
        gap_type=KnowledgeGapType.DECISION_BOUNDARY_CONFUSION,
        entropy=0.8,
        confidence=0.6,
        margin=0.05,
        true_class=0,
        predicted_class=1,
        top2_class=0,
        top2_prob=0.4,
        density_dist=1.5,
        explanation="The sample is close to a decision boundary."
    )

    planner = KnowledgeGuidedExperiencePlanner()

    result = planner.plan_for_gap(gap)

    assert result["sample_idx"] == 10

    assert result["gap_type"] == (
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
    )

    assert result["learning_objective"] == (
        LearningObjective.IMPROVE_CLASS_SEPARATION
    )

    assert result["strategy"] == (
        ProviderType.COUNTEREXAMPLE
    )

    assert result["explanation"] == (
        "The sample is close to a decision boundary."
    )

    print("Planner integration test passed!")
    print("\nResult:")
    print(result)

    print("\nAll planner tests passed!")


if __name__ == "__main__":
    main()