from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .types import AlignmentResult


class AlignmentEngine(ABC):
    name = "base"

    @abstractmethod
    def align(self, source_segments: List[str], target_segments: List[str]) -> AlignmentResult:
        raise NotImplementedError
