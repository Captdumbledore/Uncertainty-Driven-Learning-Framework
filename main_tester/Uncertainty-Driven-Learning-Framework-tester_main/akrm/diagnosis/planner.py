from enum import Enum

from .gap_types import KnowledgeGapType


class ProviderType(str, Enum):
    RETRIEVAL = "retrieval"
    COUNTEREXAMPLE = "counterexample"
    CONTEXTUAL = "contextual"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"


class LearningObjective(str, Enum):
    IMPROVE_CLASS_SEPARATION = "improve_class_separation"
    INCREASE_INTRA_CLASS_DIVERSITY = "increase_intra_class_diversity"
    STRENGTHEN_CONCEPT_UNDERSTANDING = "strengthen_concept_understanding"
    IMPROVE_ROBUSTNESS = "improve_robustness"
    CALIBRATE_CONFIDENCE = "calibrate_confidence"


class LearningObjectiveGenerator:

    def generate(
        self,
        gap_type: KnowledgeGapType,
    ) -> LearningObjective:

        mapping = {
            KnowledgeGapType.DECISION_BOUNDARY_CONFUSION:
                LearningObjective.IMPROVE_CLASS_SEPARATION,

            KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION:
                LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY,

            KnowledgeGapType.GENERAL_UNCERTAINTY:
                LearningObjective.STRENGTHEN_CONCEPT_UNDERSTANDING,

            KnowledgeGapType.OOD_CONCEPT:
                LearningObjective.IMPROVE_ROBUSTNESS,

            KnowledgeGapType.LOW_CONFIDENCE_CORRECT:
                LearningObjective.CALIBRATE_CONFIDENCE,
        }

        return mapping.get(
            gap_type,
            LearningObjective.STRENGTHEN_CONCEPT_UNDERSTANDING,
        )


class KnowledgeGuidedExperiencePlanner:

    def __init__(self, policy_type: str = "heuristic"):
        self.policy_type = policy_type
        self.objective_generator = LearningObjectiveGenerator()

    def select_strategy(
        self,
        gap_type: KnowledgeGapType,
        margin: float = 1.0,
    ) -> ProviderType:

        objective = self.objective_generator.generate(gap_type)

        if self.policy_type == "random":
            if margin < 0.5:
                return ProviderType.COUNTEREXAMPLE
            return ProviderType.RETRIEVAL

        if objective == LearningObjective.IMPROVE_CLASS_SEPARATION:
            if margin < 0.10:
                return ProviderType.COUNTEREXAMPLE
            return ProviderType.SYNTHETIC

        if objective == LearningObjective.INCREASE_INTRA_CLASS_DIVERSITY:
            return ProviderType.CONTEXTUAL

        if objective == LearningObjective.STRENGTHEN_CONCEPT_UNDERSTANDING:
            return ProviderType.RETRIEVAL

        if objective == LearningObjective.IMPROVE_ROBUSTNESS:
            return ProviderType.CONTEXTUAL

        if objective == LearningObjective.CALIBRATE_CONFIDENCE:
            return ProviderType.RETRIEVAL

        return ProviderType.RETRIEVAL