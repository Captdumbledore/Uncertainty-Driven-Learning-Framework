from enum import Enum


class LearningObjective(str, Enum):
    IMPROVE_CLASS_SEPARATION = "improve_class_separation"
    INCREASE_INTRA_CLASS_DIVERSITY = "increase_intra_class_diversity"
    STRENGTHEN_CONCEPT_UNDERSTANDING = "strengthen_concept_understanding"
    IMPROVE_ROBUSTNESS = "improve_robustness"
    CALIBRATE_CONFIDENCE = "calibrate_confidence"