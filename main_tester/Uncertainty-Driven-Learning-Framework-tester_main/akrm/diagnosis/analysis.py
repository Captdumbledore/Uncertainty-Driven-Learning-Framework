import numpy as np


class UncertaintyAnalysisEngine:
    def __init__(self, unc_results: dict):
        self.unc_results = unc_results

    def analyse(self, sample_idx: int) -> dict:
        mean_probs = self.unc_results["mean_probs"][sample_idx]
        true_label = int(self.unc_results["true_labels"][sample_idx])
        pred_label = int(
            self.unc_results["predicted_classes"][sample_idx]
        )
        entropy = float(self.unc_results["entropy"][sample_idx])
        confidence = float(
            self.unc_results["confidence"][sample_idx]
        )

        sorted_classes = np.argsort(mean_probs)[::-1]

        top2_class = int(sorted_classes[1])
        top2_prob = float(mean_probs[top2_class])

        margin = float(
            mean_probs[sorted_classes[0]] - top2_prob
        )

        return {
            "sample_idx": sample_idx,
            "true_label": true_label,
            "predicted_label": pred_label,
            "entropy": entropy,
            "confidence": confidence,
            "margin": margin,
            "top2_class": top2_class,
            "top2_prob": top2_prob,
            "is_correct": (true_label == pred_label),
        }