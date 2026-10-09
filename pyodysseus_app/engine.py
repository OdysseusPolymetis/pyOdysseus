from __future__ import annotations

import importlib
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from pyodysseus_app.alignment import BertAlignEngine, BertAlignSettings


@dataclass
class AlignParams:
    max_align: int = 5
    top_k: int = 3
    win: int = 5
    skip: float = -0.1
    margin: bool = True
    len_penalty: bool = True


_PUNCT_ONLY = re.compile(r"^[\s\W_]+$")
_SAT_CACHE: Dict[Tuple[str, str], object] = {}

# Heuristique volontairement limitée : elle ne cherche pas à analyser toute la syntaxe
# française. Elle détache seulement les incises de discours les plus fréquentes, ce qui
# est utile pour l'alignement de traductions littéraires.
_REPORTING_VERBS = (
    r"dit|répondit|reprit|demanda|ajouta|s['’]écria|déclara|murmura|cria|"
    r"poursuivit|répliqua|observa|interrompit|fit|songea|pensa|continua|"
    r"conclut|objecta|assura|affirma|annonça|rétorqua"
)
_FRENCH_INCISE_RE = re.compile(
    rf"(?:[,;:!?…—–-]|[»”\"])\s*(?P<incise>\b(?:{_REPORTING_VERBS})\b[^,;:.!?…]{{0,90}},)",
    re.IGNORECASE,
)


def text_id_from_name(name: str) -> str:
    return pathlib.Path(name).stem.strip()


def split_sections(text: str, pattern: str = r"^(Chant\s*\d+)") -> List[str]:
    """Split on section markers while excluding the marker itself from section text.

    A marker such as ``Chant1`` or ``Chant 1`` is metadata, not textual content.
    If text follows the marker on the same line, only the matched marker is removed
    and the remainder is kept as the first line of the new section. Text before the
    first detected marker is treated as front matter and ignored. If no marker is
    found at all, the complete text is returned unchanged as a single section.
    """
    text = text.lstrip("\ufeff")
    rx = re.compile(pattern, re.IGNORECASE)
    sections: List[str] = []
    current: List[str] = []
    found_heading = False

    for line in text.splitlines():
        s = line.strip()
        match = rx.match(s)
        if match:
            # A new marker closes the preceding real section. Anything before the
            # first marker is front matter rather than Section 0.
            if found_heading and current:
                section = "\n".join(current).strip()
                if section:
                    sections.append(section)
            current = []
            found_heading = True

            # Preserve content placed on the same line after the marker.
            remainder = s[match.end():].lstrip(" \t:—–-.")
            if remainder:
                current.append(remainder)
            continue

        if found_heading:
            current.append(s)

    if found_heading:
        if current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
        return sections

    return [text.strip()]




def _normalise_section_key(label: str) -> str:
    """Return a stable key for matching the same section across files.

    For headings ending in a number (e.g. Chant1, Chant 01), the numeric value is
    canonicalised so harmless spacing/zero-padding differences do not matter.
    Other headings are normalised conservatively from their visible label.
    """
    label = (label or "").strip()
    m = re.search(r"(\d+)\s*$", label)
    if m:
        return f"num:{int(m.group(1))}"
    compact = re.sub(r"\s+", " ", label).strip().casefold()
    return f"label:{compact}"


def split_sections_named(text: str, pattern: str = r"^(Chant\s*\d+)") -> List[dict]:
    """Split text into labelled sections, keeping the heading only as metadata.

    Unlike :func:`split_sections`, this function preserves the section identifier.
    It is therefore suitable for aligning multiple complete translations safely:
    Chant2 can never silently become 'section 1' merely because Chant1 is missing
    or malformed in one target file.
    """
    text = text.lstrip("\ufeff")
    rx = re.compile(pattern, re.IGNORECASE)
    sections: List[dict] = []
    current: List[str] = []
    current_label: Optional[str] = None
    current_key: Optional[str] = None
    found_heading = False

    def flush():
        nonlocal current
        if current_label is None:
            current = []
            return
        body = "\n".join(current).strip()
        if body:
            sections.append({
                "key": current_key,
                "label": current_label,
                "text": body,
            })
        current = []

    for line in text.splitlines():
        s = line.strip()
        match = rx.match(s)
        if match:
            if found_heading:
                flush()
            found_heading = True
            try:
                label = match.group(1)
            except IndexError:
                label = match.group(0)
            current_label = label.strip()
            current_key = _normalise_section_key(current_label)
            current = []

            remainder = s[match.end():].lstrip(" \t:—–-.")
            if remainder:
                current.append(remainder)
            continue

        if found_heading:
            current.append(s)

    if found_heading:
        flush()
        return sections

    body = (text or "").strip()
    if not body:
        return []
    return [{"key": "num:1", "label": "Section 1", "text": body}]


def section_map(text: str, pattern: str = r"^(Chant\s*\d+)") -> Dict[str, dict]:
    """Return sections keyed by their canonical heading identifier.

    Duplicate headings are rejected because silently overwriting one would make a
    whole-text alignment unsafe.
    """
    items = split_sections_named(text, pattern=pattern)
    out: Dict[str, dict] = {}
    for item in items:
        key = item["key"]
        if key in out:
            raise ValueError(
                f"Marqueur de section dupliqué : {item['label']!r}. "
                "Chaque section doit avoir un identifiant unique."
            )
        out[key] = item
    return out

def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_sat(model_name: str = "sat-3l-sm", device: str = "auto"):
    resolved = _resolve_device(device)
    key = (model_name, resolved)
    if key in _SAT_CACHE:
        return _SAT_CACHE[key]

    from wtpsplit import SaT

    sat = SaT(model_name)
    try:
        if resolved == "cuda":
            sat.half().to("cuda")
        elif resolved == "mps":
            sat.to("mps")
    except Exception:
        # Le modèle reste utilisable sur CPU si le transfert échoue.
        pass

    _SAT_CACHE[key] = sat
    return sat


def _split_french_incises(segment: str) -> List[str]:
    """Detach common French reporting-clause incises such as 'dit alors Télémaque,'."""
    matches = list(_FRENCH_INCISE_RE.finditer(segment))
    if not matches:
        return [segment]

    out: List[str] = []
    cursor = 0
    for match in matches:
        start, end = match.span("incise")
        before = segment[cursor:start].strip()
        incise = match.group("incise").strip()
        if before:
            out.append(before)
        if incise:
            out.append(incise)
        cursor = end
    after = segment[cursor:].strip()
    if after:
        out.append(after)
    return out or [segment]


def _clean_segments(segments: List[str], detach_french_incises: bool = False) -> List[str]:
    cleaned = [s.strip() for s in segments if s and s.strip()]

    if detach_french_incises:
        refined: List[str] = []
        for segment in cleaned:
            refined.extend(_split_french_incises(segment))
        cleaned = refined

    # Ne pas laisser de ponctuation isolée issue de la segmentation. Les incises, elles,
    # sont bien plus longues que 3 caractères et restent donc autonomes.
    merged: List[str] = []
    for s in cleaned:
        if merged and (len(s) <= 3 or _PUNCT_ONLY.match(s)):
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)
    return merged



_TERMINAL_PUNCT_RE = re.compile(r"[.!?…;:··;»”\"]\s*$")


def _merge_short_segments(
    segments: List[str],
    min_length: int = 1,
    max_length: Optional[int] = None,
) -> List[str]:
    """Merge pathological micro-segments created by a permissive SaT threshold.

    The goal is not to enforce a target length. It only prevents isolated words or
    tiny fragments from becoming alignment units. When possible, a short unit is
    merged with a neighbour without exceeding ``max_length``. Punctuation is used
    as a weak cue: if the previous segment already ends a sentence/clause, prefer
    merging forward; if the short segment itself closes a clause, prefer backward.
    """
    min_length = max(1, int(min_length or 1))
    if min_length <= 1:
        return [s for s in segments if s]

    out = [s.strip() for s in segments if s and s.strip()]
    if len(out) <= 1:
        return out

    def can_merge(a: str, b: str) -> bool:
        if max_length is None:
            return True
        return len(a) + 1 + len(b) <= int(max_length)

    i = 0
    while i < len(out):
        cur = out[i]
        if len(cur) >= min_length or len(out) == 1:
            i += 1
            continue

        left_ok = i > 0 and can_merge(out[i - 1], cur)
        right_ok = i + 1 < len(out) and can_merge(cur, out[i + 1])

        if not left_ok and not right_ok:
            # Keep the fragment rather than violating the user's hard ceiling.
            i += 1
            continue

        if left_ok and not right_ok:
            choice = "left"
        elif right_ok and not left_ok:
            choice = "right"
        else:
            prev_terminal = bool(_TERMINAL_PUNCT_RE.search(out[i - 1]))
            cur_terminal = bool(_TERMINAL_PUNCT_RE.search(cur))
            if prev_terminal and not cur_terminal:
                choice = "right"
            elif cur_terminal and not prev_terminal:
                choice = "left"
            else:
                left_len = len(out[i - 1]) + 1 + len(cur)
                right_len = len(cur) + 1 + len(out[i + 1])
                choice = "left" if left_len <= right_len else "right"

        if choice == "left":
            out[i - 1] = (out[i - 1] + " " + cur).strip()
            del out[i]
            i = max(0, i - 1)
        else:
            out[i] = (cur + " " + out[i + 1]).strip()
            del out[i + 1]
            # Re-evaluate the merged unit: it may still be shorter than min_length.

    return out


def split_units_clean(
    text: str,
    method: str = "sat",
    threshold: float = 0.001,
    sat_model: str = "sat-3l-sm",
    device: str = "auto",
    constrained: bool = False,
    hybrid: bool = False,
    min_length: int = 1,
    target_length: Optional[int] = None,
    max_length: Optional[int] = None,
    spread: Optional[int] = None,
    prior_type: str = "gaussian",
    detach_french_incises: bool = False,
) -> List[str]:
    """Segment text for alignment.

    Three SaT behaviours are supported:

    * threshold mode: keep every boundary whose probability exceeds ``threshold``;
    * constrained mode: globally optimise the segmentation under ``max_length``;
    * hybrid mode: first use the permissive threshold segmentation, then re-segment
      only units longer than ``max_length``. This preserves fine boundaries while
      preventing pathological long alignment units.
    """
    text = (text or "").strip()
    if not text:
        return []

    if method == "lines":
        return _clean_segments(
            [line.strip() for line in text.splitlines() if line.strip()],
            detach_french_incises=detach_french_incises,
        )

    if method == "paragraphs":
        return _clean_segments(
            [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()],
            detach_french_incises=detach_french_incises,
        )

    sat = _load_sat(sat_model, device=device)

    if hybrid:
        if max_length is None:
            raise ValueError("Une longueur maximale est nécessaire en mode hybride.")
        # 1) Preserve the permissive SaT segmentation used successfully by the
        #    original pyOdysseus workflow.
        base = list(sat.split(text, threshold=float(threshold), strip_whitespace=True))
        base = _clean_segments(base, detach_french_incises=detach_french_incises)
        # A permissive threshold can isolate a single word. Merge those pathological
        # micro-segments before asking Viterbi to enforce the upper bound.
        base = _merge_short_segments(base, min_length=min_length, max_length=max_length)

        # 2) Only long units are sent through wtpsplit's constrained Viterbi
        #    decoder. A uniform prior avoids encouraging all segments toward an
        #    arbitrary target length: we merely impose a ceiling and let SaT choose
        #    the most plausible internal boundaries.
        refined: List[str] = []
        for segment in base:
            if len(segment) <= int(max_length):
                refined.append(segment)
                continue
            forced = sat.split(
                segment,
                min_length=max(1, int(min_length)),
                max_length=int(max_length),
                prior_type="uniform",
                algorithm="viterbi",
                strip_whitespace=True,
            )
            refined.extend(_clean_segments(list(forced), detach_french_incises=False))
        refined = _clean_segments(refined, detach_french_incises=False)
        return _merge_short_segments(refined, min_length=min_length, max_length=max_length)

    if constrained:
        if max_length is None:
            raise ValueError("Une longueur maximale est nécessaire en mode contraint.")
        if min_length < 1:
            raise ValueError("La longueur minimale doit être au moins 1.")
        if min_length > max_length:
            raise ValueError("La longueur minimale ne peut pas dépasser la longueur maximale.")
        if target_length is not None and target_length > max_length:
            raise ValueError("La longueur cible ne peut pas dépasser la longueur maximale.")

        prior_kwargs = {}
        if target_length is not None:
            prior_kwargs["target_length"] = int(target_length)
        if spread is not None:
            prior_kwargs["spread"] = int(spread)

        segs = sat.split(
            text,
            min_length=int(min_length),
            max_length=int(max_length),
            prior_type=prior_type,
            prior_kwargs=prior_kwargs or None,
            algorithm="viterbi",
            strip_whitespace=True,
        )
    else:
        segs = sat.split(text, threshold=float(threshold), strip_whitespace=True)

    cleaned = _clean_segments(list(segs), detach_french_incises=detach_french_incises)
    return _merge_short_segments(
        cleaned,
        min_length=min_length,
        max_length=max_length if constrained else None,
    )


def segmentation_stats(segments: List[str]) -> dict:
    lengths = [len(s) for s in segments]
    if not lengths:
        return {"count": 0, "min": 0, "median": 0, "mean": 0.0, "max": 0}
    ordered = sorted(lengths)
    n = len(ordered)
    if n % 2:
        median = ordered[n // 2]
    else:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return {
        "count": len(lengths),
        "min": min(lengths),
        "median": median,
        "mean": sum(lengths) / len(lengths),
        "max": max(lengths),
    }


def _import_local_bertalign(repo_root: pathlib.Path):
    pkg_root = repo_root / "bertalign_odysseus"
    if not pkg_root.exists():
        raise FileNotFoundError(
            "Dossier 'bertalign_odysseus' introuvable. Place l'application à la racine "
            "du dépôt pyOdysseus, à côté de bertalign_odysseus/."
        )
    pkg_root_str = str(pkg_root.resolve())
    if pkg_root_str not in sys.path:
        sys.path.insert(0, pkg_root_str)

    import bertalign

    # Do not reload here: reloading bertalign executes its module-level set_model()
    # and silently resets a user-selected local/HuggingFace model back to LaBSE.
    return bertalign


def configure_bertalign_model(
    repo_root: pathlib.Path,
    model_name_or_path: str = "LaBSE",
    device: str = "auto",
    batch_size: int = 64,
):
    bertalign = _import_local_bertalign(repo_root)
    resolved = _resolve_device(device)
    bertalign.set_model(
        model_name_or_path=model_name_or_path or "LaBSE",
        device=resolved,
        batch_size=batch_size,
        show_bar=False,
    )
    return bertalign, resolved


def get_configured_bertalign_encoder(repo_root: pathlib.Path):
    """Return the already configured shared Bertalign Encoder.

    LocalAlign uses the same SentenceTransformer instance, avoiding a second copy
    of a large local model in memory.
    """
    bertalign = _import_local_bertalign(repo_root)
    return bertalign.get_model()


def beads_to_cells(beads, nb_src: int):
    """Convert source-bearing Bertalign beads to pivot rows.

    IMPORTANT: target-only beads (0→N) are deliberately *not* attached to the
    nearest source row. They are represented separately by ``beads_to_insertions``.
    This prevents a run of insertions from looking like an absurd 1→28 alignment.
    """
    cells = [None] * nb_src

    def attach(anchor, tgts):
        if anchor is None:
            return
        if cells[anchor] in (None, 0):
            cells[anchor] = {"rowspan": 1, "idxs": []}
        seen = set(cells[anchor]["idxs"])
        for t in tgts:
            t = int(t)
            if t not in seen:
                cells[anchor]["idxs"].append(t)
                seen.add(t)

    for sr, tr in beads:
        sr = [int(x) for x in sr]
        tr = [int(x) for x in tr]
        if not sr:
            continue

        top = sr[0]
        bot = sr[-1]
        rowspan = max(1, bot - top + 1)
        if cells[top] in (None, 0):
            cells[top] = {"rowspan": rowspan, "idxs": []}
        else:
            cells[top]["rowspan"] = max(cells[top]["rowspan"], rowspan)
        attach(top, tr)
        for r in range(top + 1, min(nb_src, top + rowspan)):
            cells[r] = 0

    return cells


def beads_to_insertions(beads, nb_src: int):
    """Keep every target-only Bertalign bead in its true gap between source units.

    Keys are source gaps: 0 = before source 0, 1 = between source 0 and 1,
    ``nb_src`` = after the last source unit. Values are lists of target-index groups,
    one group per original Bertalign bead.
    """
    insertions: Dict[str, List[List[int]]] = {}
    src_cursor = 0
    for sr, tr in beads:
        sr = [int(x) for x in sr]
        tr = [int(x) for x in tr]
        if not sr and tr:
            key = str(max(0, min(nb_src, src_cursor)))
            insertions.setdefault(key, []).append(tr)
            continue
        if sr:
            src_cursor = min(nb_src, sr[-1] + 1)
    return insertions


def align_section(
    repo_root: pathlib.Path,
    pivot_segments: List[str],
    targets_segments: Dict[str, List[str]],
    params: AlignParams,
    progress_cb: Optional[Callable[[str, int, int], None]] = None,
    encoder_override=None,
) -> Tuple[
    Dict[str, List[object]],
    Dict[str, List[Tuple[List[int], List[int]]]],
    Dict[str, Dict[str, List[List[int]]]],
]:
    """Align one section through the common pyOdysseus engine abstraction.

    The current UI still consumes the legacy bead/cell representation, so the
    canonical AlignmentResult is converted back at this boundary. Later editors
    can consume AlignmentBlock directly without knowing anything about Bertalign.
    """
    bertalign = _import_local_bertalign(repo_root)
    engine = BertAlignEngine(
        bertalign,
        BertAlignSettings(
            max_align=params.max_align,
            top_k=params.top_k,
            win=params.win,
            skip=params.skip,
            margin=params.margin,
            len_penalty=params.len_penalty,
        ),
        encoder_override=encoder_override,
    )

    cells_by_target: Dict[str, List[object]] = {}
    beads_by_target: Dict[str, List[Tuple[List[int], List[int]]]] = {}
    insertions_by_target: Dict[str, Dict[str, List[List[int]]]] = {}
    items = [(tid, segs) for tid, segs in targets_segments.items() if segs]

    for pos, (tid, tgt_sents) in enumerate(items, start=1):
        if progress_cb:
            progress_cb(tid, pos, len(items))
        result = engine.align(pivot_segments, tgt_sents)
        beads = result.legacy_beads()
        beads_by_target[tid] = beads
        cells_by_target[tid] = beads_to_cells(beads, nb_src=len(pivot_segments))
        insertions_by_target[tid] = beads_to_insertions(beads, nb_src=len(pivot_segments))

    return cells_by_target, beads_by_target, insertions_by_target
