import numpy as np
from sentence_transformers import SentenceTransformer
from bertalign.utils import yield_overlaps

class Encoder:
    def __init__(self, model_name_or_path, device='cuda', batch_size=64, show_bar=False):
        self.model_name = model_name_or_path
        self.device = device
        self.batch_size = batch_size
        self.show_bar = show_bar

        self.model = SentenceTransformer(model_name_or_path, device=device)

    def transform(self, sents, num_overlaps):
        overlaps = []
        for line in yield_overlaps(sents, num_overlaps):
            overlaps.append(line)

        sent_vecs = self.model.encode(
            overlaps,
            device=self.device,
            batch_size=self.batch_size,
            show_progress_bar=self.show_bar,
            convert_to_numpy=True,
        )

        embedding_dim = sent_vecs.shape[-1]
        sent_vecs = sent_vecs.reshape(num_overlaps, len(sents), embedding_dim)

        len_vecs = [len(line.encode("utf-8")) for line in overlaps]
        len_vecs = np.array(len_vecs).reshape(num_overlaps, len(sents))

        return sent_vecs, len_vecs
