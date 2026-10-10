from dataclasses import dataclass
from typing import Dict, Optional

from .gap_types import KnowledgeGap


@dataclass
class MemoryEntry:
    gap: KnowledgeGap
    visit_count: int = 1
    age: int = 0
    strength: float = 1.0


class KnowledgeMemory:
    def __init__(self):
        self.memory: Dict[int, MemoryEntry] = {}

    def contains(self, sample_idx: int) -> bool:
        return sample_idx in self.memory

    def add(self, gap: KnowledgeGap) -> None:
        if gap.sample_idx in self.memory:
            self.memory[gap.sample_idx].visit_count += 1
        else:
            self.memory[gap.sample_idx] = MemoryEntry(gap=gap)

    def get_or_add(self, gap: KnowledgeGap) -> KnowledgeGap:
        if self.contains(gap.sample_idx):
            self.memory[gap.sample_idx].visit_count += 1
            return self.memory[gap.sample_idx].gap

        self.add(gap)
        return gap 
    
    def decay_memory(self, decay_rate: float = 0.1, min_strength: float = 0.2) -> None:
        to_remove = []

        for sample_idx, entry in self.memory.items():
            entry.age += 1
            entry.strength = (1 - decay_rate) ** entry.age

            if entry.strength < min_strength:
                to_remove.append(sample_idx)

        for sample_idx in to_remove:
            self.remove(sample_idx)

    def get(self, sample_idx: int) -> Optional[KnowledgeGap]:
        entry = self.memory.get(sample_idx)

        if entry is None:
            return None

        return entry.gap

    def get_visit_count(self, sample_idx: int) -> int:
        entry = self.memory.get(sample_idx)

        if entry is None:
            return 0

        return entry.visit_count

    def remove(self, sample_idx: int) -> None:
        self.memory.pop(sample_idx, None)

    def size(self) -> int:
        return len(self.memory)