from .memory import KnowledgeMemory
from .planner import KnowledgeGuidedExperiencePlanner
from .gap_types import KnowledgeGap


class KnowledgeGuidedIntegration:

    def __init__(self, policy_type: str = "heuristic"):
        self.memory = KnowledgeMemory()
        self.planner = KnowledgeGuidedExperiencePlanner(
            policy_type=policy_type
        )

        # Store the learning plan for already processed gaps
        self.plans = {}

    def process_gap(self, gap: KnowledgeGap) -> dict:
        """
        Process a diagnosed knowledge gap using memory
        and the knowledge-guided planner.

        If the same gap has already been processed,
        reuse the previously generated learning plan.
        """

        sample_idx = gap.sample_idx

        # Check whether this gap was already processed
        already_seen = self.memory.contains(sample_idx)

        # Get the stored gap or add the new gap
        stored_gap = self.memory.get_or_add(gap)

        # If this gap was already processed, reuse its plan
        if already_seen:
            plan = self.plans[sample_idx]

        else:
            # Generate a new learning plan only for a new gap
            plan = self.planner.plan_for_gap(stored_gap)

            # Store the generated plan
            self.plans[sample_idx] = plan

        return {
            "sample_idx": stored_gap.sample_idx,
            "gap_type": stored_gap.gap_type,
            "learning_objective": plan["learning_objective"],
            "strategy": plan["strategy"],
            "already_seen": already_seen,
            "visit_count": self.memory.get_visit_count(sample_idx),
            "explanation": stored_gap.explanation
        }