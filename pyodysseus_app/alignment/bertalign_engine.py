from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

from .base import AlignmentEngine
from .types import AlignmentBlock, AlignmentResult


@dataclass
class BertAlignSettings:
    max_align: int = 5
    top_k: int = 3
    win: int = 5
    skip: float = -0.1
    margin: bool = True
    len_penalty: bool = True


class BertAlignEngine(AlignmentEngine):
    """Thin adapter around the bundled Bertalign implementation.

    Keeping this adapter small is intentional: the rest of pyOdysseus now depends
    on AlignmentResult/AlignmentBlock, not on Bertalign's native tuple format.
    """

    name = "bertalign"

    def __init__(self, bertalign_module: Any, settings: BertAlignSettings, encoder_override: Any = None):
        self.bertalign_module = bertalign_module
        self.settings = settings
        self.encoder_override = encoder_override

    def align(self, source_segments: List[str], target_segments: List[str]) -> AlignmentResult:
        Bertalign = self.bertalign_module.Bertalign
        s = self.settings
        aligner = Bertalign(
            source_segments,
            target_segments,
            max_align=s.max_align,
            top_k=s.top_k,
            win=s.win,
            skip=s.skip,
            margin=s.margin,
            len_penalty=s.len_penalty,
            is_split=True,
            encoder=self.encoder_override,
        ).align_sents()

        blocks = [
            AlignmentBlock(
                source_ids=[int(x) for x in src],
                target_ids=[int(x) for x in tgt],
                metadata={"origin": "bertalign"},
            )
            for src, tgt in getattr(aligner, "result", [])
        ]
        return AlignmentResult(engine=self.name, blocks=blocks)
