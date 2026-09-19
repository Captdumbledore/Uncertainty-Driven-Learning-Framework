from dataclasses import dataclass


@dataclass
class AKRMConfig:
    policy_type: str = "heuristic"
    high_entropy_threshold: float = 0.50
    medium_entropy_threshold: float = 0.20
    low_margin_threshold: float = 0.20
    density_k: int = 20
    sparse_density_percentile: float = 66.7
    n_retrieve_retrieval: int = 8
    n_retrieve_counterexample: int = 8
    experience_pool_size: int = 100
    quality_weight: float = 0.4
    diversity_weight: float = 0.3
    novelty_weight: float = 0.3