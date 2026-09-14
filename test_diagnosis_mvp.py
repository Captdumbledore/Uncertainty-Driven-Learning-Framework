import numpy as np

from akrm.diagnosis.gap_types import (
    KnowledgeGapType,
    KnowledgeGap,
)

from akrm.diagnosis.config import AKRMConfig

from akrm.diagnosis.analysis import (
    UncertaintyAnalysisEngine,
)

from akrm.diagnosis.diagnoser import (
    KnowledgeGapDiagnoser,
)

from akrm.diagnosis.planner import (
    ProviderType,
    LearningObjective,
    LearningObjectiveGenerator,
    KnowledgeGuidedExperiencePlanner,
)


def main():

    print("Testing Knowledge Gap Types:")

    assert KnowledgeGapType.DECISION_BOUNDARY_CONFUSION.value == \
        "decision_boundary_confusion"

    assert KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION.value == \
        "sparse_concept_representation"

    assert KnowledgeGapType.GENERAL_UNCERTAINTY.value == \
        "general_uncertainty"

    assert KnowledgeGapType.OOD_CONCEPT.value == \
        "ood_concept"

    assert KnowledgeGapType.LOW_CONFIDENCE_CORRECT.value == \
        "low_confidence_correct"

    print("Knowledge gap types passed!")


    print("\nTesting AKRM Configuration:")

    config = AKRMConfig()

    assert config.policy_type == "heuristic"
    assert config.density_k == 20
    assert config.n_retrieve_retrieval == 8
    assert config.n_retrieve_counterexample == 8

    print("Configuration:", config)
    print("Configuration tests passed!")


    print("\nTesting Uncertainty Analysis:")

    unc_results = {
        "mean_probs": np.array([
            [0.70, 0.20, 0.10],
            [0.45, 0.40, 0.15],
        ]),
        "true_labels": np.array([0, 1]),
        "predicted_classes": np.array([0, 0]),
        "entropy": np.array([0.80, 0.90]),
        "confidence": np.array([0.70, 0.45]),
    }

    engine = UncertaintyAnalysisEngine(unc_results)

    analysis = engine.analyse(0)

    print("Analysis result:", analysis)

    assert analysis["sample_idx"] == 0
    assert analysis["true_label"] == 0
    assert analysis["predicted_label"] == 0
    assert analysis["top2_class"] == 1
    assert analysis["is_correct"] is True

    print("Uncertainty analysis tests passed!")


    print("\nTesting Knowledge Gap Diagnoser:")

    diagnoser = KnowledgeGapDiagnoser(config)

    gap = diagnoser.diagnose(
        analysis=analysis,
        density_dist=1.0,
        density_threshold=2.0,
    )

    print("Diagnosed gap:", gap)

    assert isinstance(gap, KnowledgeGap)
    assert gap.gap_type == KnowledgeGapType.LOW_CONFIDENCE_CORRECT
    assert gap.sample_idx == 0

    print("Knowledge gap diagnosis passed!")


    print("\nTesting Boundary Confusion Diagnosis:")

    boundary_analysis = {
        "sample_idx": 1,
        "true_label": 1,
        "predicted_label": 0,
        "entropy": 0.80,
        "confidence": 0.45,
        "margin": 0.05,
        "top2_class": 1,
        "top2_prob": 0.40,
        "is_correct": False,
    }

    boundary_gap = diagnoser.diagnose(
        analysis=boundary_analysis,
        density_dist=1.0,
        density_threshold=2.0,
    )

    print("Boundary gap:", boundary_gap)

    assert (
        boundary_gap.gap_type
        == KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
    )

    print("Boundary diagnosis passed!")


    print("\nTesting Sparse Concept Representation:")

    sparse_analysis = {
        "sample_idx": 2,
        "true_label": 1,
        "predicted_label": 2,
        "entropy": 0.30,
        "confidence": 0.40,
        "margin": 0.10,
        "top2_class": 1,
        "top2_prob": 0.35,
        "is_correct": False,
    }

    sparse_gap = diagnoser.diagnose(
        analysis=sparse_analysis,
        density_dist=5.0,
        density_threshold=2.0,
    )

    print("Sparse gap:", sparse_gap)

    assert (
        sparse_gap.gap_type
        == KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION
    )

    print("Sparse concept diagnosis passed!")


    print("\nTesting Learning Objective Generator:")

    generator = LearningObjectiveGenerator()

    objective = generator.generate(
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
    )

    print("Boundary objective:", objective)

    assert (
        objective
        == LearningObjective.IMPROVE_CLASS_SEPARATION
    )

    objective = generator.generate(
        KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION
    )

    assert (
        objective
        == LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY
    )

    objective = generator.generate(
        KnowledgeGapType.GENERAL_UNCERTAINTY
    )

    assert (
        objective
        == LearningObjective.STRENGTHEN_CONCEPT_UNDERSTANDING
    )

    objective = generator.generate(
        KnowledgeGapType.OOD_CONCEPT
    )

    assert (
        objective
        == LearningObjective.IMPROVE_ROBUSTNESS
    )

    objective = generator.generate(
        KnowledgeGapType.LOW_CONFIDENCE_CORRECT
    )

    assert (
        objective
        == LearningObjective.CALIBRATE_CONFIDENCE
    )

    print("Learning objective tests passed!")


    print("\nTesting Knowledge-Guided Experience Planner:")

    planner = KnowledgeGuidedExperiencePlanner()

    strategy = planner.select_strategy(
        KnowledgeGapType.DECISION_BOUNDARY_CONFUSION,
        margin=0.05,
    )

    print("Boundary strategy:", strategy)

    assert strategy == ProviderType.COUNTEREXAMPLE

    strategy = planner.select_strategy(
        KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION
    )

    print("Sparse strategy:", strategy)

    assert strategy == ProviderType.CONTEXTUAL

    strategy = planner.select_strategy(
        KnowledgeGapType.GENERAL_UNCERTAINTY
    )

    print("General uncertainty strategy:", strategy)

    assert strategy == ProviderType.RETRIEVAL

    strategy = planner.select_strategy(
        KnowledgeGapType.OOD_CONCEPT
    )

    print("OOD strategy:", strategy)

    assert strategy == ProviderType.CONTEXTUAL

    strategy = planner.select_strategy(
        KnowledgeGapType.LOW_CONFIDENCE_CORRECT
    )

    print("Low-confidence strategy:", strategy)

    assert strategy == ProviderType.RETRIEVAL

    print("Planner tests passed!")


    print("\nAll diagnosis MVP tests passed!")


if __name__ == "__main__":
    main()