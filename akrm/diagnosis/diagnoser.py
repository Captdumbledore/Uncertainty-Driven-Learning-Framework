from .gap_types import KnowledgeGap, KnowledgeGapType
from .config import AKRMConfig


class KnowledgeGapDiagnoser:
    def __init__(self, config: AKRMConfig | None = None):
        self.config = config or AKRMConfig()

    def diagnose(
        self,
        analysis: dict,
        density_dist: float,
        density_threshold: float,
        class_names: dict | None = None,
    ) -> KnowledgeGap:

        entropy = analysis["entropy"]
        confidence = analysis["confidence"]
        margin = analysis["margin"]
        is_correct = analysis["is_correct"]

        true_class = analysis["true_label"]
        predicted_class = analysis["predicted_label"]
        top2_class = analysis["top2_class"]
        top2_prob = analysis["top2_prob"]

        # Rule 1: Correct but uncertain
        if (
            is_correct
            and entropy > self.config.medium_entropy_threshold
        ):
            gap_type = KnowledgeGapType.LOW_CONFIDENCE_CORRECT
            explanation = (
                "The model predicted the sample correctly "
                "but shows uncertainty."
            )

        # Rule 2: Decision boundary confusion
        elif (
            entropy > self.config.high_entropy_threshold
            and margin < self.config.low_margin_threshold
        ):
            gap_type = KnowledgeGapType.DECISION_BOUNDARY_CONFUSION
            explanation = (
                "The model is uncertain between competing classes "
                "and has a small prediction margin."
            )

        # Rule 3: Sparse concept representation
        elif (
            entropy > self.config.medium_entropy_threshold
            and density_dist > density_threshold
        ):
            gap_type = KnowledgeGapType.SPARSE_CONCEPT_REPRESENTATION
            explanation = (
                "The sample is uncertain and far from dense "
                "training representations."
            )

        # Rule 4: General uncertainty
        else:
            gap_type = KnowledgeGapType.GENERAL_UNCERTAINTY
            explanation = (
                "The sample shows general uncertainty "
                "without a stronger structural gap."
            )

        return KnowledgeGap(
            sample_idx=analysis["sample_idx"],
            gap_type=gap_type,
            entropy=entropy,
            confidence=confidence,
            margin=margin,
            true_class=true_class,
            predicted_class=predicted_class,
            top2_class=top2_class,
            top2_prob=top2_prob,
            density_dist=density_dist,
            explanation=explanation,
        )