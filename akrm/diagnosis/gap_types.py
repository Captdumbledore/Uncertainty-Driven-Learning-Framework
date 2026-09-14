from enum import Enum
from dataclasses import dataclass


class KnowledgeGapType(str, Enum):
    """
    Structural gap types diagnosed by the AKRM.
    """
    DECISION_BOUNDARY_CONFUSION = "decision_boundary_confusion"
    SPARSE_CONCEPT_REPRESENTATION = "sparse_concept_representation"
    GENERAL_UNCERTAINTY = "general_uncertainty"
    OOD_CONCEPT = "ood_concept"
    LOW_CONFIDENCE_CORRECT = "low_confidence_correct"


@dataclass
class KnowledgeGap:
    """
    Output of the diagnosis phase.
    """
    sample_idx: int
    gap_type: KnowledgeGapType
    entropy: float
    confidence: float
    margin: float
    true_class: int
    predicted_class: int
    top2_class: int
    top2_prob: float
    density_dist: float
    explanation: str