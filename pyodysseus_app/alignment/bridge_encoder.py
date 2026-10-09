from __future__ import annotations

from typing import List
import numpy as np


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _yield_overlaps(lines: List[str], num_overlaps: int):
    clean = [((line or "").strip() or "BLANK_LINE") for line in lines]
    for overlap in range(1, int(num_overlaps) + 1):
        layer = ["PAD"] * min(overlap - 1, len(clean))
        for i in range(len(clean) - overlap + 1):
            layer.append(" ".join(clean[i : i + overlap]))
        for text in layer:
            yield text[:10000]


def _reshape(vectors: np.ndarray, overlaps: List[str], n_sentences: int, num_overlaps: int):
    vectors = np.asarray(vectors, dtype=np.float32)
    dim = vectors.shape[-1]
    sent_vecs = vectors.reshape(int(num_overlaps), int(n_sentences), dim)
    lens = np.asarray([len(x.encode("utf-8")) for x in overlaps], dtype=np.int64)
    len_vecs = lens.reshape(int(num_overlaps), int(n_sentences))
    return sent_vecs, len_vecs


class PublishedBridgeEncoder:
    """Asymmetric encoder for Bertalign.

    Ancient Greek source segments are encoded by a SentenceTransformer already
    projected into the LaBSE space. Target segments are encoded by LaBSE itself.
    Both sides therefore produce directly comparable normalized embeddings.
    """

    def __init__(
        self,
        source_model: str = "MOdysseus/SPhilBERTa-LaBSE-Bridge-Ridge",
        target_model: str = "sentence-transformers/LaBSE",
        device: str = "auto",
        batch_size: int = 64,
        show_bar: bool = False,
    ):
        self.source_model_name = source_model
        self.target_model_name = target_model
        self.device = _resolve_device(device)
        self.batch_size = int(batch_size)
        self.show_bar = bool(show_bar)
        self._source_model = None
        self._target_model = None

    def _load_source(self):
        if self._source_model is None:
            from sentence_transformers import SentenceTransformer
            self._source_model = SentenceTransformer(self.source_model_name, device=self.device)
        return self._source_model

    def _load_target(self):
        if self._target_model is None:
            from sentence_transformers import SentenceTransformer
            self._target_model = SentenceTransformer(self.target_model_name, device=self.device)
        return self._target_model

    def encode_source(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        return np.asarray(
            self._load_source().encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=self.show_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def encode_target(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        return np.asarray(
            self._load_target().encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=self.show_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def transform_source(self, sents: List[str], num_overlaps: int):
        overlaps = list(_yield_overlaps(list(sents), int(num_overlaps)))
        vectors = self.encode_source(overlaps)
        return _reshape(vectors, overlaps, len(sents), int(num_overlaps))

    def transform_target(self, sents: List[str], num_overlaps: int):
        overlaps = list(_yield_overlaps(list(sents), int(num_overlaps)))
        vectors = self.encode_target(overlaps)
        return _reshape(vectors, overlaps, len(sents), int(num_overlaps))

    @property
    def model_name(self) -> str:
        return f"{self.source_model_name} → espace {self.target_model_name}"
