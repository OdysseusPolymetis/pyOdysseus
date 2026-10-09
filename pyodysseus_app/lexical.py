from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import closing
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version as package_version
from itertools import islice
import html
import json
import os
import sqlite3
import math
import threading
import time
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


@dataclass(frozen=True)
class _LexicalToken:
    """Expose les attributs spaCy déjà utilisés dans ce module."""
    text: str
    whitespace_: str = ""
    lemma_: str = ""
    pos_: str = ""
    is_alpha: bool = False
    is_stop: bool = False


def _adapt_stanza_doc(text: str, document) -> List[_LexicalToken]:
    """Conserve rigoureusement le texte/espacement pour les rendus HTML.

    Les annotations Stanza sont au niveau du Word ; les bornes textuelles
    sont celles des Token. En français les rares MWT (du, au, des...) sont
    généralement des mots outils. Pour un MWT, on privilégie son premier
    Word lexical, sinon le premier Word, sans inventer de texte de surface.
    """
    from spacy.lang.fr.stop_words import STOP_WORDS

    stanza_tokens = [tok for sentence in document.sentences for tok in sentence.tokens]
    if not stanza_tokens:
        return [_LexicalToken(text)] if text else []

    result: List[_LexicalToken] = []
    cursor = 0
    for tok in stanza_tokens:
        start, end = tok.start_char, tok.end_char
        if start is None or end is None or not (cursor <= start < end <= len(text)):
            raise ValueError("Stanza a renvoyé des positions de tokens invalides.")

        # Recopie telle quelle des caractères entre deux tokens (espaces,
        # retours ligne, ponctuation non tokenisée, etc.).
        if start > cursor:
            if result:
                result[-1] = replace(result[-1], whitespace_=result[-1].whitespace_ + text[cursor:start])
            else:
                result.append(_LexicalToken(text=text[cursor:start]))

        words = list(tok.words)
        lexical = next(
            (w for w in words if (w.upos or "") in CONTENT_POS and (w.lemma or "").strip()),
            None,
        )
        word = lexical or (words[0] if words else None)
        surface = text[start:end]
        result.append(_LexicalToken(
            text=surface,
            lemma_=(word.lemma or "") if word else "",
            pos_=(word.upos or "") if word else "",
            is_alpha=surface.isalpha(),
            is_stop=surface.lower() in STOP_WORDS,
        ))
        cursor = end

    if cursor < len(text):
        result[-1] = replace(result[-1], whitespace_=result[-1].whitespace_ + text[cursor:])
    # Vérifie que le passage analysé et affiché n'a pas été modifié.
    if "".join(tok.text + tok.whitespace_ for tok in result) != text:
        raise AssertionError("Échec de la reconstruction du texte de Stanza.")
    return result


class _StanzaFrenchPipeline:
    """Compatibilité minimale avec nlp(text) et nlp.pipe(...) de spaCy."""

    def __init__(self, pipeline):
        self._pipeline = pipeline

    def __call__(self, text: str) -> List[_LexicalToken]:
        text = text or ""
        return _adapt_stanza_doc(text, self._pipeline(text)) if text else []

    def pipe(self, texts: Iterable[str], batch_size: int = 64):
        # Utilise l'inférence par lots de Stanza, sans perdre l'ordre ni les
        # segments vides (ceux-ci ne sont pas soumis au modèle).
        batch_size = max(1, int(batch_size))
        iterator = iter(texts)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            nonempty = [(i, x) for i, x in enumerate(batch) if x]
            parsed = self._pipeline.bulk_process([x for _, x in nonempty]) if nonempty else []
            if len(parsed) != len(nonempty):
                raise RuntimeError("Stanza n'a pas conservé le nombre de documents du lot.")
            docs = [[] for _ in batch]
            for (i, text), doc in zip(nonempty, parsed):
                docs[i] = _adapt_stanza_doc(text, doc)
            yield from docs


@lru_cache(maxsize=1)
def load_french_pipeline():
    """Charge une seule fois Stanza FR default_accurate pour l'analyse lexicale.

    Utilise les ressources déjà présentes et ne télécharge que celles
    manquantes (voir README_V3_8.md). Aucun repli silencieux vers spaCy.
    """
    try:
        import stanza
    except ImportError as exc:
        raise RuntimeError(
            "Stanza manque dans l'environnement Python de l'application. "
            "Installe les dépendances de requirements-app.txt."
        ) from exc
    try:
        from stanza.pipeline.core import DownloadMethod
        pipeline = stanza.Pipeline(
            lang="fr",
            package="default_accurate",
            processors="tokenize,mwt,pos,lemma",
            use_gpu=False,
            download_method=DownloadMethod.REUSE_RESOURCES,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Impossible de charger Stanza FR default_accurate. "
            "Dans le venv de l'application, exécute : "
            "python -c \"import stanza; stanza.download('fr', "
            "processors='tokenize,mwt,pos,lemma', package='default_accurate')\""
        ) from exc
    return _StanzaFrenchPipeline(pipeline)


# Cache à deux étages : LRU en RAM et index SQLite PERSISTANT sur disque.
# L'identité est le texte EXACT du segment + la version des modèles / règles.
# Changer le texte d'un segment invalide de fait son annotation.
_SEGMENT_CACHE_MAXSIZE = 10000
_SEGMENT_CACHE: OrderedDict[str, Tuple[_LexicalToken, ...]] = OrderedDict()
_SEGMENT_CACHE_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def lexical_index_path() -> Path:
    """Emplacement indépendant des sauvegardes de projet et du dépôt Git."""
    base = os.environ.get("PYODYSSEUS_LEXICAL_CACHE_DIR")
    folder = Path(base).expanduser() if base else Path.home() / ".cache" / "pyodysseus"
    return folder / "stanza_french_lexical.sqlite3"


@lru_cache(maxsize=1)
def _annotation_signature() -> str:
    # Si les modèles sont mis à jour, augmenter 'format=2' ci-dessous.
    # Inclure aussi la version de spaCy car sa liste de mots outils est conservée.
    def ver(name: str) -> str:
        try:
            return package_version(name)
        except PackageNotFoundError:
            return "absent"
    return (
        f"format=1;lang=fr;package=default_accurate;"
        f"processors=tokenize,mwt,pos,lemma;stanza={ver('stanza')};"
        f"stopwords_spacy={ver('spacy')}"
    )


def _disk_key(text: str) -> str:
    raw = (_annotation_signature() + "\0" + text).encode("utf-8")
    return sha256(raw).hexdigest()


def _connect_index() -> sqlite3.Connection:
    path = lexical_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(path), timeout=30)
    db.execute("""CREATE TABLE IF NOT EXISTS annotations (
        cache_key TEXT PRIMARY KEY,
        payload BLOB NOT NULL
    )""")
    return db


def _encode_tokens(tokens: Tuple[_LexicalToken, ...]) -> bytes:
    rows = [[x.text, x.whitespace_, x.lemma_, x.pos_, x.is_alpha, x.is_stop] for x in tokens]
    return zlib.compress(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _decode_tokens(payload: bytes) -> Tuple[_LexicalToken, ...]:
    rows = json.loads(zlib.decompress(payload).decode("utf-8"))
    return tuple(_LexicalToken(*row) for row in rows)


def _read_index(db: sqlite3.Connection, keys: List[str]) -> dict:
    """Lectures groupées pour ne pas multiplier les requêtes SQL par segment."""
    found = {}
    for start in range(0, len(keys), 400):
        block = keys[start:start + 400]
        marks = ",".join("?" for _ in block)
        for key, payload in db.execute(
            f"SELECT cache_key, payload FROM annotations WHERE cache_key IN ({marks})", block
        ):
            found[key] = _decode_tokens(payload)
    return found


def _annotate_cached(texts: Iterable[str]) -> List[Tuple[_LexicalToken, ...]]:
    """Réutilise les annotations Stanza déjà calculées, y compris après redémarrage.

    Seuls les nouveaux textes sont soumis à Stanza ; les résultats sont
    enregistrés progressivement dans SQLite par petits lots de huit.
    """
    requested = list(texts)
    if not requested:
        return []

    with _SEGMENT_CACHE_LOCK:
        needed = list(dict.fromkeys(text for text in requested if text not in _SEGMENT_CACHE))
        if needed:
            with closing(_connect_index()) as db:
                key_for_text = {text: _disk_key(text) for text in needed}
                stored = _read_index(db, list(key_for_text.values()))
                disk_hits = 0
                to_annotate = []
                for text in needed:
                    doc = stored.get(key_for_text[text])
                    if doc is None:
                        to_annotate.append(text)
                    else:
                        _SEGMENT_CACHE[text] = doc
                        disk_hits += 1

                if to_annotate:
                    pipeline = load_french_pipeline()
                    start = time.perf_counter()
                    pending = []
                    # On persiste après chaque petit lot : une interruption ne
                    # fait pas perdre tout le chant déjà lemmatisé.
                    for i, (text, doc) in enumerate(
                        zip(to_annotate, pipeline.pipe(to_annotate, batch_size=8)), start=1
                    ):
                        frozen = tuple(doc)
                        _SEGMENT_CACHE[text] = frozen
                        pending.append((key_for_text[text], _encode_tokens(frozen)))
                        if len(pending) >= 8:
                            db.executemany(
                                "INSERT OR REPLACE INTO annotations (cache_key, payload) VALUES (?, ?)",
                                pending,
                            )
                            db.commit()
                            pending.clear()
                        if i % 16 == 0:
                            print(
                                f"[pyOdysseus lexical] {i}/{len(to_annotate)} segments nouveaux…",
                                flush=True,
                            )
                    if pending:
                        db.executemany(
                            "INSERT OR REPLACE INTO annotations (cache_key, payload) VALUES (?, ?)",
                            pending,
                        )
                        db.commit()
                    if any(text not in _SEGMENT_CACHE for text in to_annotate):
                        raise RuntimeError("Stanza n'a pas renvoyé tous les segments attendus.")
                    elapsed = time.perf_counter() - start
                    print(
                        f"[pyOdysseus lexical] {len(to_annotate)} nouveaux segments "
                        f"annotés en {elapsed:.1f}s, sauvegardés dans SQLite; "
                        f"{disk_hits} retrouvés sur disque.",
                        flush=True,
                    )
                elif disk_hits:
                    print(
                        f"[pyOdysseus lexical] {disk_hits} segments retrouvés "
                        f"dans l'index SQLite (0 appel à Stanza).",
                        flush=True,
                    )

        docs = []
        for text in requested:
            doc = _SEGMENT_CACHE[text]
            _SEGMENT_CACHE.move_to_end(text)
            docs.append(doc)
        while len(_SEGMENT_CACHE) > _SEGMENT_CACHE_MAXSIZE:
            _SEGMENT_CACHE.popitem(last=False)
        return docs


def _section_docs(section: dict, target_id: str) -> List[Tuple[_LexicalToken, ...]]:
    return _annotate_cached(section.get("target_segments", {}).get(target_id, []))


def _tokens_for_indices(
    section: dict, target_id: str, target_indices: Iterable[int],
) -> List[_LexicalToken]:
    """Réassemble les annotations déjà connues, sans nouvel appel Stanza."""
    segments = section.get("target_segments", {}).get(target_id, [])
    indices = [int(i) for i in target_indices if 0 <= int(i) < len(segments)]
    if not indices:
        return []
    docs = _annotate_cached(segments[i] for i in indices)
    joined: List[_LexicalToken] = []
    for i, doc in enumerate(docs):
        if i:
            # Le texte de référence joint les segments avec un espace.
            joined.append(_LexicalToken(text="", whitespace_=" "))
        joined.extend(doc)
    return joined


def _segment_text(section: dict, target_id: str, idx: int) -> str:
    segs = section.get("target_segments", {}).get(target_id, [])
    return segs[idx] if 0 <= idx < len(segs) else ""


def aligned_text_for_source(section: dict, target_id: str, source_index: int) -> Tuple[str, List[int]]:
    for bead in section.get("beads_by_target", {}).get(target_id, []):
        if not isinstance(bead, (list, tuple)) or len(bead) != 2:
            continue
        src = [int(x) for x in bead[0]]
        tgt = [int(x) for x in bead[1]]
        if int(source_index) in src:
            return " ".join(_segment_text(section, target_id, i) for i in tgt).strip(), tgt
    return "", []


def aligned_text_for_source_window(
    section: dict,
    target_id: str,
    source_index: int,
    radius: int = 1,
) -> Tuple[str, List[int], List[int]]:
    """Return target text aligned to source n-radius..n+radius.

    A target bead is included once even when it covers several source segments
    inside the window. Target segment indices are also de-duplicated while
    preserving textual order.
    """
    pivot_len = len(section.get("pivot_segments", []))
    if pivot_len <= 0:
        return "", [], []

    center = int(source_index)
    r = max(0, int(radius))
    lo = max(0, center - r)
    hi = min(pivot_len - 1, center + r)
    source_window = list(range(lo, hi + 1))
    source_set = set(source_window)

    target_indices: List[int] = []
    seen_target = set()
    for bead in section.get("beads_by_target", {}).get(target_id, []):
        if not isinstance(bead, (list, tuple)) or len(bead) != 2:
            continue
        src = [int(x) for x in bead[0]]
        tgt = [int(x) for x in bead[1]]
        if source_set.intersection(src):
            for idx in tgt:
                if idx not in seen_target:
                    seen_target.add(idx)
                    target_indices.append(idx)

    target_indices.sort()
    text = " ".join(_segment_text(section, target_id, i) for i in target_indices).strip()
    return text, target_indices, source_window


def _content_token(token, include_propn: bool = True) -> bool:
    allowed = set(CONTENT_POS)
    if not include_propn:
        allowed.discard("PROPN")
    return bool(
        token.is_alpha
        and not token.is_stop
        and token.pos_ in allowed
        and token.lemma_
        and token.lemma_.strip()
    )


def lemma_counter(text: str, include_propn: bool = True) -> Counter:
    doc = _annotate_cached([text or ""])[0]
    return Counter(
        token.lemma_.lower().strip()
        for token in doc
        if _content_token(token, include_propn=include_propn)
    )


def compare_aligned_passage(
    section: dict,
    target_ids: List[str],
    source_index: int,
    include_propn: bool = True,
    almost_ratio: float = 0.8,
    context_radius: int = 1,
) -> dict:
    docs = {}
    lemma_sets: Dict[str, set] = {}
    texts = {}
    target_indices = {}

    context_source_indices: List[int] = []
    for tid in target_ids:
        text, idxs, source_window = aligned_text_for_source_window(
            section, tid, source_index, radius=context_radius
        )
        if not context_source_indices:
            context_source_indices = source_window
        texts[tid] = text
        target_indices[tid] = idxs
        # Même annotation qu'à l'échelle du chant : aucune relémmatisation.
        doc = _tokens_for_indices(section, tid, idxs)
        docs[tid] = doc
        lemma_sets[tid] = {
            token.lemma_.lower().strip()
            for token in doc
            if _content_token(token, include_propn=include_propn)
        }

    support = Counter()
    for lemmas in lemma_sets.values():
        support.update(lemmas)

    n = max(1, len(target_ids))
    almost_min = max(2, int(math.ceil(float(almost_ratio) * n)))

    def category(lemma: str) -> str:
        count = support.get(lemma, 0)
        if count == n and n > 1:
            return "common"
        if count >= almost_min and count < n:
            return "almost"
        if count == 1:
            return "unique"
        if count > 1:
            return "shared"
        return "other"

    rendered = {}
    stats = []
    for tid, doc in docs.items():
        chunks = []
        content_count = 0
        unique_count = 0
        common_count = 0
        almost_count = 0
        for token in doc:
            raw = html.escape(token.text)
            ws = html.escape(token.whitespace_)
            if _content_token(token, include_propn=include_propn):
                lemma = token.lemma_.lower().strip()
                cat = category(lemma)
                content_count += 1
                unique_count += int(cat == "unique")
                common_count += int(cat == "common")
                almost_count += int(cat == "almost")
                title = html.escape(f"{lemma} · {support.get(lemma, 0)}/{n} traductions")
                chunks.append(f'<span class="lemma-{cat}" title="{title}">{raw}</span>{ws}')
            else:
                chunks.append(raw + ws)
        rendered[tid] = "".join(chunks)
        stats.append(
            {
                "traduction": tid,
                "lemmes lexicaux": content_count,
                "uniques": unique_count,
                "presque tous": almost_count,
                "communs": common_count,
                "% uniques": round(100.0 * unique_count / content_count, 1) if content_count else 0.0,
            }
        )

    support_rows = [
        {
            "lemme": lemma,
            "présence": count,
            "sur": len(target_ids),
            "catégorie": category(lemma),
            "traductions": ", ".join(tid for tid in target_ids if lemma in lemma_sets[tid]),
        }
        for lemma, count in support.most_common()
    ]

    return {
        "texts": texts,
        "target_indices": target_indices,
        "context_source_indices": context_source_indices,
        "context_radius": max(0, int(context_radius)),
        "rendered": rendered,
        "stats": stats,
        "support": support_rows,
    }


def section_frequencies(section: dict, target_ids: List[str], include_propn: bool = True, top_n: int = 50):
    rows = []
    for tid in target_ids:
        counts = Counter()
        for doc in _section_docs(section, tid):
            counts.update(tok.lemma_.lower().strip() for tok in doc
                          if _content_token(tok, include_propn=include_propn))
        total = sum(counts.values()) or 1
        for rank, (lemma, count) in enumerate(counts.most_common(int(top_n)), start=1):
            rows.append(
                {
                    "traduction": tid,
                    "rang": rank,
                    "lemme": lemma,
                    "fréquence": count,
                    "% des mots lexicaux": round(100.0 * count / total, 2),
                }
            )
    return rows



def rare_shared_analysis(
    section: dict,
    target_ids: List[str],
    include_propn: bool = True,
    support_sizes: Tuple[int, ...] = (2, 3),
    max_total_occurrences: int | None = 12,
) -> dict:
    """Analyse les lemmes rares partagés par exactement 2 ou 3 traductions.

    La rareté combine deux idées simples et interprétables :
    - faible dispersion : le lemme n'apparaît que chez exactement 2 ou 3 traducteurs ;
    - faible fréquence globale optionnelle dans la section.

    Le poids d'un lemme augmente quand il est partagé par moins de traducteurs et
    quand il est peu fréquent dans l'ensemble de la section.
    """
    tids = list(dict.fromkeys(target_ids))
    n = len(tids)
    if n < 2:
        return {"groups": [], "pairs": [], "profiles": {}, "lemmas": []}

    counters: Dict[str, Counter] = {}
    lemma_sets: Dict[str, set] = {}
    for tid in tids:
        counts = Counter()
        for doc in _section_docs(section, tid):
            counts.update(tok.lemma_.lower().strip() for tok in doc
                          if _content_token(tok, include_propn=include_propn))
        counters[tid] = counts
        lemma_sets[tid] = set(counts)

    support = Counter()
    total_tf = Counter()
    for tid in tids:
        support.update(lemma_sets[tid])
        total_tf.update(counters[tid])

    support_sizes_set = {int(x) for x in support_sizes if int(x) >= 2}

    def is_candidate(lemma: str) -> bool:
        df = int(support.get(lemma, 0))
        if df not in support_sizes_set:
            return False
        if max_total_occurrences is not None and int(total_tf.get(lemma, 0)) > int(max_total_occurrences):
            return False
        return True

    def rarity_weight(lemma: str) -> float:
        df = max(1, int(support.get(lemma, 1)))
        tf = max(1, int(total_tf.get(lemma, 1)))
        # IDF traducteurs × pénalité douce de fréquence globale.
        return float(math.log((n + 1.0) / df) / math.log(tf + 1.5))

    lemma_rows = []
    groups: Dict[Tuple[str, ...], dict] = {}
    for lemma in sorted(total_tf):
        if not is_candidate(lemma):
            continue
        users = tuple(tid for tid in tids if lemma in lemma_sets[tid])
        weight = rarity_weight(lemma)
        row = {
            "lemme": lemma,
            "traductions": users,
            "nombre de traductions": len(users),
            "occurrences totales": int(total_tf[lemma]),
            "poids de rareté": round(weight, 4),
        }
        lemma_rows.append(row)
        g = groups.setdefault(users, {
            "traductions": users,
            "taille": len(users),
            "lemmes": [],
            "score": 0.0,
            "occurrences": 0,
        })
        g["lemmes"].append(lemma)
        g["score"] += weight
        g["occurrences"] += int(total_tf[lemma])

    group_rows = []
    for users, g in groups.items():
        group_rows.append({
            "groupe": " + ".join(users),
            "taille": g["taille"],
            "lemmes rares partagés": len(g["lemmes"]),
            "score rare": round(float(g["score"]), 4),
            "occurrences totales": int(g["occurrences"]),
            "lemmes": ", ".join(g["lemmes"]),
        })
    group_rows.sort(key=lambda r: (r["score rare"], r["lemmes rares partagés"]), reverse=True)

    # Analyse par paire. Un lemme partagé par 3 traducteurs compte aussi dans chacune
    # des trois paires si l'utilisateur a demandé de conserver les supports de taille 3.
    pair_rows = []
    for i, a in enumerate(tids):
        for b in tids[i + 1:]:
            shared = [
                lemma for lemma in total_tf
                if is_candidate(lemma) and lemma in lemma_sets[a] and lemma in lemma_sets[b]
            ]
            if not shared:
                continue
            shared_weight = sum(rarity_weight(x) for x in shared)

            rare_a = [lemma for lemma in lemma_sets[a] if is_candidate(lemma)]
            rare_b = [lemma for lemma in lemma_sets[b] if is_candidate(lemma)]
            mass_a = sum(rarity_weight(x) for x in rare_a) or 1.0
            mass_b = sum(rarity_weight(x) for x in rare_b) or 1.0

            pair_rows.append({
                "traducteur A": a,
                "traducteur B": b,
                "lemmes rares partagés": len(shared),
                "score rare partagé": round(float(shared_weight), 4),
                "reprise A→B (%)": round(100.0 * shared_weight / mass_a, 1),
                "reprise B→A (%)": round(100.0 * shared_weight / mass_b, 1),
                "lemmes": ", ".join(sorted(shared)),
            })
    pair_rows.sort(key=lambda r: (r["score rare partagé"], r["lemmes rares partagés"]), reverse=True)

    profiles = {
        tid: {
            "lemmes lexicaux distincts": len(lemma_sets[tid]),
            "lemmes rares candidats": sum(1 for x in lemma_sets[tid] if is_candidate(x)),
        }
        for tid in tids
    }

    return {
        "groups": group_rows,
        "pairs": pair_rows,
        "profiles": profiles,
        "lemmas": lemma_rows,
    }



def _ordered_lemma_chain(
    lemmas_a: List[str], lemmas_b: List[str],
    rare_weights: Dict[str, float] | None = None,
    min_matches: int = 7,
) -> dict:
    """Cherche des reprises ordonnées indépendamment de la rareté.

    Smith-Waterman à poids identiques pour TOUS les lemmes lexicaux,
    avec pénalités pour les insertions, omissions et substitutions.
    Les mots rares ne jouent aucun rôle dans l'alignement ni dans le
    score primaire ; ils sont seulement comptés pour un bonus ultérieur.
    Les occurrences répétées sont alignées dans leur ordre, non via un set.
    """
    empty = {"coherence": 0.0, "matched": 0, "rare_anchors": 0,
             "longest_run": 0, "chain": [], "positions_a": [],
             "positions_b": [], "score_sequence": 0.0, "local_density": 0.0,
             "coverage": 0.0}
    min_matches = max(3, int(min_matches))
    if min(len(lemmas_a), len(lemmas_b)) < min_matches:
        return empty
    # Préfiltrage peu coûteux : éviter l'alignement dynamique lorsque presque
    # aucun lemme distinct n'est commun (notamment les très longues fenêtres).
    if len(set(lemmas_a).intersection(lemmas_b)) < 4:
        return empty

    n, m = len(lemmas_a), len(lemmas_b)
    prev = [0.0] * (m + 1)
    traceback = [bytearray(m + 1) for _ in range(n + 1)]
    best_score, best_i, best_j = 0.0, 0, 0
    for i, a in enumerate(lemmas_a, 1):
        row = [0.0] * (m + 1)
        for j, b in enumerate(lemmas_b, 1):
            diagonal = prev[j - 1] + (1.0 if a == b else -0.9)
            up = prev[j] - 0.7
            left = row[j - 1] - 0.7
            score = max(0.0, diagonal, up, left)
            row[j] = score
            if score > 0:
                traceback[i][j] = (1 if score == diagonal
                                   else 2 if score == up else 3)
            if score > best_score:
                best_score, best_i, best_j = score, i, j
        prev = row

    matches = []
    i, j = best_i, best_j
    while i > 0 and j > 0 and traceback[i][j]:
        direction = traceback[i][j]
        if direction == 1:
            if lemmas_a[i - 1] == lemmas_b[j - 1]:
                matches.append((i - 1, j - 1, lemmas_a[i - 1]))
            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    if len(matches) < min_matches or len({x[2] for x in matches}) < 4:
        return empty

    longest, run = 1, 1
    for previous, current in zip(matches, matches[1:]):
        if current[0] == previous[0] + 1 and current[1] == previous[1] + 1:
            run += 1
        else:
            run = 1
        longest = max(longest, run)

    matched = len(matches)
    # Couverture équilibrée des deux fenêtres : une longue suite commune
    # ne doit pas masquer une traduction qui ajoute beaucoup de matière.
    coverage_a = matched / n
    coverage_b = matched / m
    coverage = 2 * coverage_a * coverage_b / (coverage_a + coverage_b)
    span_a = matches[-1][0] - matches[0][0] + 1
    span_b = matches[-1][1] - matches[0][1] + 1
    local_density = matched / max(span_a, span_b)
    # Les correspondances compactes, conservant l'ordre, dominent.
    coherence = (0.50 * coverage
                 + 0.30 * local_density
                 + 0.20 * min(1.0, longest / 5.0))
    # Evite qu'un fragment identique de 7 mots écrase 20-30 lemmes proches.
    length_factor = 0.45 + 0.55 * min(1.0, matched / 14.0)
    score = 100.0 * coherence * length_factor
    weights = rare_weights or {}
    rare_anchors = sum(lemma in weights for _, _, lemma in matches)
    return {
        "coherence": round(coherence, 4),
        "matched": matched,
        "rare_anchors": rare_anchors,
        "longest_run": longest,
        "chain": [lemma for _, _, lemma in matches],
        "positions_a": [i for i, _, _ in matches],
        "positions_b": [j for _, j, _ in matches],
        "score_sequence": round(score, 3),
        "local_density": round(local_density, 4),
        "coverage": round(coverage, 4),
    }


def rare_shared_passages(
    section: dict,
    reference_target_ids: List[str],
    focus_target_ids: List[str],
    include_propn: bool = True,
    support_sizes: Tuple[int, ...] = (2, 3),
    max_total_occurrences: int | None = 12,
    context_radius: int = 1,
    min_shared_lemmas: int = 2,  # Ancien paramètre conservé, désormais non bloquant.
    top_n: int = 12,
    rare_result: dict | None = None,
    sequence_bonus: float = 0.75,  # Ancien paramètre conservé pour compatibilité.
    min_ordered_lemmas: int = 7,
    rarity_bonus: float = 0.25,
) -> List[dict]:
    """Classement des passages par chaînes lexicales ordonnées.

    Nouveauté V3.8.6 : aucune condition de présence de lemmes rares.
    La rareté ne fait qu'ajouter un bonus borné pour des ancrages rares
    QUI APPARTIENNENT à la chaîne. Les annotations proviennent de SQLite.
    `min_shared_lemmas` et `sequence_bonus` sont conservés pour les appels
    existants, mais ne filtrent plus les candidats.
    """
    refs = list(dict.fromkeys(reference_target_ids))
    focus = list(dict.fromkeys(focus_target_ids))
    if len(focus) != 2:
        return []

    if rare_result is None:
        rare_result = rare_shared_analysis(
            section, refs, include_propn=include_propn,
            support_sizes=support_sizes,
            max_total_occurrences=max_total_occurrences,
        )

    focus_set = set(focus)
    candidate_weights: Dict[str, float] = {
        str(row["lemme"]): float(row.get("poids de rareté", 0.0))
        for row in rare_result.get("lemmas", [])
        if focus_set.issubset(set(row.get("traductions", ())))
    }

    segment_sequences: Dict[str, List[List[str]]] = {}
    segment_counts: Dict[str, List[int]] = {}
    for tid in focus:
        sequences = [
            [tok.lemma_.lower().strip() for tok in doc
             if _content_token(tok, include_propn=include_propn)]
            for doc in _section_docs(section, tid)
        ]
        segment_sequences[tid] = sequences
        segment_counts[tid] = [len(seq) for seq in sequences]

    pivot_segments = list(section.get("pivot_segments", []))
    radius = max(0, int(context_radius))
    min_ordered_lemmas = max(3, int(min_ordered_lemmas))
    # Le bonus maximal représente une fraction du score de continuité.
    rarity_bonus = max(0.0, min(0.5, float(rarity_bonus)))
    windows: List[dict] = []
    chain_cache: dict = {}

    for center in range(len(pivot_segments)):
        lo = max(0, center - radius)
        hi = min(len(pivot_segments) - 1, center + radius)
        texts: Dict[str, str] = {}
        target_indices: Dict[str, List[int]] = {}
        sequences = []

        for tid in focus:
            text, idxs, _ = aligned_text_for_source_window(
                section, tid, center, radius=radius,
            )
            texts[tid] = text
            target_indices[tid] = idxs
            seq = [lemma for idx in idxs
                   if 0 <= idx < len(segment_sequences[tid])
                   for lemma in segment_sequences[tid][idx]]
            sequences.append(seq)

        seq_a, seq_b = sequences
        if min(len(seq_a), len(seq_b)) < min_ordered_lemmas:
            continue
        key = (tuple(seq_a), tuple(seq_b), min_ordered_lemmas)
        if key not in chain_cache:
            # La détection primaire reste identique, quelle que soit la
            # configuration de la rareté ; seules les métadonnées diffèrent.
            chain_cache[key] = _ordered_lemma_chain(
                seq_a, seq_b, rare_weights=None,
                min_matches=min_ordered_lemmas,
            )
        chain = chain_cache[key]
        if not chain["matched"]:
            continue

        common_local = set(seq_a).intersection(seq_b)
        shared_rare = sorted(x for x in common_local if x in candidate_weights)
        rare_score = sum(candidate_weights[x] for x in shared_rare)
        mean_tokens = (len(seq_a) + len(seq_b)) / 2.0
        density = 100.0 * rare_score / mean_tokens if mean_tokens else 0.0
        rare_in_chain = [x for x in chain["chain"] if x in candidate_weights]
        # Bonus local, lié à la même chaîne, et non à des raretés éparpillées
        # ailleurs dans le chant ou dans le passage.
        distinct_rare_anchors = len(set(rare_in_chain))
        bonus_factor = rarity_bonus * min(1.0, distinct_rare_anchors / 3.0)
        primary = chain["score_sequence"]
        final_score = primary * (1.0 + bonus_factor)
        windows.append({
            "centre": center,
            "source_start": lo,
            "source_end": hi,
            "segments pivot": str(lo + 1) if lo == hi else f"{lo + 1}–{hi + 1}",
            "lemmes rares partagés": len(shared_rare),
            "score rare local": round(float(rare_score), 4),
            "densité rare / 100 mots": round(density, 3),
            "score séquence": round(primary, 3),
            "bonus rareté (%)": round(100.0 * bonus_factor, 1),
            "score final": round(final_score, 3),
            "cohérence séquentielle": chain["coherence"],
            "couverture séquentielle": chain["coverage"],
            "densité de chaîne": chain["local_density"],
            "lemmes ordonnés": chain["matched"],
            "ancrages rares ordonnés": len(rare_in_chain),
            "plus longue suite exacte": chain["longest_run"],
            "chaîne de lemmes": chain["chain"],
            "positions de chaîne": {focus[0]: chain["positions_a"],
                                   focus[1]: chain["positions_b"]},
            "mots lexicaux moyens": round(mean_tokens, 1),
            "lemmes": ", ".join(shared_rare),
            "pivot": " ".join(pivot_segments[lo:hi + 1]).strip(),
            "texts": texts,
            "target_indices": target_indices,
        })

    windows.sort(key=lambda r: (
        r["score final"], r["score séquence"], r["lemmes ordonnés"],
    ), reverse=True)
    selected: List[dict] = []
    for row in windows:
        lo, hi = int(row["source_start"]), int(row["source_end"])
        overlaps = any(
            not (hi < int(prev["source_start"]) or lo > int(prev["source_end"]))
            for prev in selected
        )
        if overlaps:
            continue
        selected.append(row)
        if len(selected) >= int(top_n):
            break
    return selected


def highlight_lemmas_html(
    text: str,
    lemmas: Iterable[str],
    include_propn: bool = True,
    *,
    section: dict | None = None,
    target_id: str | None = None,
    target_indices: Iterable[int] | None = None,
    matched_content_positions: Iterable[int] | None = None,
) -> str:
    """HTML sûr ; réutilise les annotations si les segments sont fournis."""
    wanted = {str(x).lower().strip() for x in lemmas if str(x).strip()}
    if section is not None and target_id is not None and target_indices is not None:
        doc = _tokens_for_indices(section, target_id, target_indices)
    else:
        doc = _annotate_cached([text or ""])[0]
    chunks = []
    matched = set(int(i) for i in matched_content_positions or [])
    content_idx = -1
    for token in doc:
        raw = html.escape(token.text)
        ws = html.escape(token.whitespace_)
        lemma = token.lemma_.lower().strip() if token.lemma_ else ""
        is_content = _content_token(token, include_propn=include_propn)
        if is_content:
            content_idx += 1
        if is_content and content_idx in matched:
            raw = f'<span class="lemma-sequence-match" title="Lemme dans la chaîne ordonnée">{raw}</span>'
        if is_content and lemma in wanted:
            chunks.append(f'<mark title="lemme : {html.escape(lemma)}">{raw}</mark>{ws}')
        else:
            chunks.append(raw + ws)
    return "".join(chunks)
