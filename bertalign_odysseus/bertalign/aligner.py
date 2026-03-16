import numpy as np
from bertalign import get_model
from bertalign.corelib import *
from bertalign.utils import *

class Bertalign:
    def __init__(
        self,
        src,
        tgt,
        max_align=5,
        top_k=3,
        win=5,
        skip=-0.1,
        margin=True,
        len_penalty=True,
        is_split=False,
        model_name_or_path=None,
        device=None,
        batch_size=None,
        show_bar=None,
    ):
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty

        if isinstance(src, (list, tuple)) and isinstance(tgt, (list, tuple)):
            src_sents = [str(s).strip() for s in src if str(s).strip()]
            tgt_sents = [str(s).strip() for s in tgt if str(s).strip()]
            src_lang = detect_lang(clean_text("\n".join(src_sents[:50])))
            tgt_lang = detect_lang(clean_text("\n".join(tgt_sents[:50])))
        else:
            src = clean_text(src)
            tgt = clean_text(tgt)
            src_lang = detect_lang(src)
            tgt_lang = detect_lang(tgt)

            if is_split:
                src_sents = [s for s in src.splitlines() if s.strip()]
                tgt_sents = [s for s in tgt.splitlines() if s.strip()]
            else:
                src_sents = split_sents(src, src_lang)
                tgt_sents = split_sents(tgt, tgt_lang)

        src_num = len(src_sents)
        tgt_num = len(tgt_sents)
        src_lang = LANG.ISO[src_lang]
        tgt_lang = LANG.ISO[tgt_lang]

        print(f"Source language: {src_lang}, Number of sentences: {src_num}")
        print(f"Target language: {tgt_lang}, Number of sentences: {tgt_num}")

        # modèle spécifique à cet alignement si fourni
        if model_name_or_path is not None:
            from bertalign.encoder import Encoder
            self.model = Encoder(
                model_name_or_path=model_name_or_path,
                device=device or "cuda",
                batch_size=batch_size or 64,
                show_bar=show_bar or False,
            )
        else:
            self.model = get_model()

        print(f"Embedding source and target text using {self.model.model_name} ...")
        src_vecs, src_lens = self.model.transform(src_sents, max_align - 1)
        tgt_vecs, tgt_lens = self.model.transform(tgt_sents, max_align - 1)

        char_ratio = np.sum(src_lens[0,]) / np.sum(tgt_lens[0,])

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents
        self.src_num = src_num
        self.tgt_num = tgt_num
        self.src_lens = src_lens
        self.tgt_lens = tgt_lens
        self.char_ratio = char_ratio
        self.src_vecs = src_vecs
        self.tgt_vecs = tgt_vecs

    def align_sents(self):
        print("Performing first-step alignment ...")
        D, I = find_top_k_sents(self.src_vecs[0, :], self.tgt_vecs[0, :], k=self.top_k)
        first_alignment_types = get_alignment_types(2)
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        first_pointers = first_pass_align(
            self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I
        )
        first_alignment = first_back_track(
            self.src_num, self.tgt_num, first_pointers, first_path, first_alignment_types
        )

        print("Performing second-step alignment ...")
        second_alignment_types = get_alignment_types(self.max_align)
        second_w, second_path = find_second_search_path(
            first_alignment, self.win, self.src_num, self.tgt_num
        )
        second_pointers = second_pass_align(
            self.src_vecs,
            self.tgt_vecs,
            self.src_lens,
            self.tgt_lens,
            second_w,
            second_path,
            second_alignment_types,
            self.char_ratio,
            self.skip,
            margin=self.margin,
            len_penalty=self.len_penalty,
        )
        second_alignment = second_back_track(
            self.src_num, self.tgt_num, second_pointers, second_path, second_alignment_types
        )

        print(
            f"Finished! Successfully aligning {self.src_num} {self.src_lang} "
            f"sentences to {self.tgt_num} {self.tgt_lang} sentences\n"
        )
        self.result = second_alignment
        return self

    def print_sents(self):
        for bead in self.result:
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead, lines):
        line = ''
        if len(bead) > 0:
            line = ' '.join(lines[bead[0]:bead[-1] + 1])
        return line
