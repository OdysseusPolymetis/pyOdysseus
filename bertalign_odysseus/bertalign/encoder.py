import numpy as np

from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name, device='cuda', batch_size=64, show_bar=False):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.show_bar = show_bar

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        # Batch + GPU
        sent_vecs = self.model.encode(
            overlaps,
            device=self.device,
            batch_size=self.batch_size,
            show_progress_bar=self.show_bar
        )

        embedding_dim = sent_vecs.size // (len(sents) * num_overlaps)
        sent_vecs.resize(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs)
        len_vecs.resize(num_overlaps, len(sents))

        return sent_vecs, len_vecs