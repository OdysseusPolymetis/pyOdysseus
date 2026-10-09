from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AlignmentBlock:
    """Canonical pyOdysseus alignment unit.

    Indices are always zero-based internally. The UI is responsible for displaying
    human-friendly one-based numbers.
    """

    source_ids: List[int]
    target_ids: List[int]
    score: Optional[float] = None
    confidence: Optional[float] = None
    kind: Optional[str] = None
    is_reordered: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def alignment_type(self) -> str:
        return f"{len(self.source_ids)}→{len(self.target_ids)}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["type"] = self.alignment_type
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AlignmentBlock":
        return cls(
            source_ids=[int(x) for x in data.get("source_ids", [])],
            target_ids=[int(x) for x in data.get("target_ids", [])],
            score=data.get("score"),
            confidence=data.get("confidence"),
            kind=data.get("kind"),
            is_reordered=bool(data.get("is_reordered", False)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AlignmentResult:
    engine: str
    blocks: List[AlignmentBlock]
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def legacy_beads(self):
        """Compatibility representation expected by the current V2 UI."""
        return [
            ([int(x) for x in block.source_ids], [int(x) for x in block.target_ids])
            for block in self.blocks
        ]
