from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import List

from pyodysseus_app.engine import beads_to_cells, beads_to_insertions


class AlignmentEditError(ValueError):
    pass


def _clean_beads(beads):
    out = []
    for bead in beads:
        if not isinstance(bead, (list, tuple)) or len(bead) != 2:
            continue
        src = [int(x) for x in bead[0]]
        tgt = [int(x) for x in bead[1]]
        if src or tgt:
            out.append([src, tgt])
    return out


def _commit(section: dict, target_id: str, beads, action: str):
    beads = _clean_beads(beads)
    section.setdefault("beads_by_target", {})[target_id] = beads
    nb_src = len(section.get("pivot_segments", []))
    section.setdefault("cells_by_target", {})[target_id] = beads_to_cells(beads, nb_src=nb_src)
    section.setdefault("insertions_by_target", {})[target_id] = beads_to_insertions(beads, nb_src=nb_src)
    section.setdefault("edit_log", []).append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target_id,
            "action": action,
        }
    )


def _axis_index(axis: str) -> int:
    if axis == "source":
        return 0
    if axis == "target":
        return 1
    raise AlignmentEditError("Axe inconnu.")


def take_from_previous(section: dict, target_id: str, block_index: int, axis: str):
    beads = deepcopy(section["beads_by_target"][target_id])
    if block_index <= 0 or block_index >= len(beads):
        raise AlignmentEditError("Il n'y a pas de bloc précédent disponible.")
    a = _axis_index(axis)
    prev, cur = beads[block_index - 1], beads[block_index]
    if not prev[a]:
        raise AlignmentEditError(f"Le bloc précédent n'a aucun segment {axis} à transférer.")
    item = prev[a].pop(-1)
    cur[a].insert(0, item)
    _commit(section, target_id, beads, f"{axis}: prendre le dernier segment du bloc précédent")


def take_from_next(section: dict, target_id: str, block_index: int, axis: str):
    beads = deepcopy(section["beads_by_target"][target_id])
    if block_index < 0 or block_index >= len(beads) - 1:
        raise AlignmentEditError("Il n'y a pas de bloc suivant disponible.")
    a = _axis_index(axis)
    cur, nxt = beads[block_index], beads[block_index + 1]
    if not nxt[a]:
        raise AlignmentEditError(f"Le bloc suivant n'a aucun segment {axis} à transférer.")
    item = nxt[a].pop(0)
    cur[a].append(item)
    _commit(section, target_id, beads, f"{axis}: prendre le premier segment du bloc suivant")


def give_to_previous(section: dict, target_id: str, block_index: int, axis: str):
    beads = deepcopy(section["beads_by_target"][target_id])
    if block_index <= 0 or block_index >= len(beads):
        raise AlignmentEditError("Il n'y a pas de bloc précédent disponible.")
    a = _axis_index(axis)
    prev, cur = beads[block_index - 1], beads[block_index]
    if not cur[a]:
        raise AlignmentEditError(f"Le bloc courant n'a aucun segment {axis} à transférer.")
    item = cur[a].pop(0)
    prev[a].append(item)
    _commit(section, target_id, beads, f"{axis}: transférer le premier segment au bloc précédent")


def give_to_next(section: dict, target_id: str, block_index: int, axis: str):
    beads = deepcopy(section["beads_by_target"][target_id])
    if block_index < 0 or block_index >= len(beads) - 1:
        raise AlignmentEditError("Il n'y a pas de bloc suivant disponible.")
    a = _axis_index(axis)
    cur, nxt = beads[block_index], beads[block_index + 1]
    if not cur[a]:
        raise AlignmentEditError(f"Le bloc courant n'a aucun segment {axis} à transférer.")
    item = cur[a].pop(-1)
    nxt[a].insert(0, item)
    _commit(section, target_id, beads, f"{axis}: transférer le dernier segment au bloc suivant")


def merge_with_previous(section: dict, target_id: str, block_index: int):
    beads = deepcopy(section["beads_by_target"][target_id])
    if block_index <= 0 or block_index >= len(beads):
        raise AlignmentEditError("Il n'y a pas de bloc précédent à fusionner.")
    prev = beads[block_index - 1]
    cur = beads[block_index]
    prev[0].extend(cur[0])
    prev[1].extend(cur[1])
    del beads[block_index]
    _commit(section, target_id, beads, "fusion avec le bloc précédent")


def merge_with_next(section: dict, target_id: str, block_index: int):
    beads = deepcopy(section["beads_by_target"][target_id])
    if block_index < 0 or block_index >= len(beads) - 1:
        raise AlignmentEditError("Il n'y a pas de bloc suivant à fusionner.")
    cur = beads[block_index]
    nxt = beads[block_index + 1]
    cur[0].extend(nxt[0])
    cur[1].extend(nxt[1])
    del beads[block_index + 1]
    _commit(section, target_id, beads, "fusion avec le bloc suivant")


def split_block(
    section: dict,
    target_id: str,
    block_index: int,
    source_left_count: int,
    target_left_count: int,
):
    beads = deepcopy(section["beads_by_target"][target_id])
    if not (0 <= block_index < len(beads)):
        raise AlignmentEditError("Bloc introuvable.")
    src, tgt = beads[block_index]
    source_left_count = int(source_left_count)
    target_left_count = int(target_left_count)
    if not (0 <= source_left_count <= len(src)) or not (0 <= target_left_count <= len(tgt)):
        raise AlignmentEditError("Position de scission invalide.")
    left = [src[:source_left_count], tgt[:target_left_count]]
    right = [src[source_left_count:], tgt[target_left_count:]]
    if not (left[0] or left[1]) or not (right[0] or right[1]):
        raise AlignmentEditError("La scission doit créer deux blocs non vides.")
    beads[block_index : block_index + 1] = [left, right]
    _commit(section, target_id, beads, "scission manuelle du bloc")


def edit_segment_text(section: dict, target_id: str, axis: str, segment_index: int, text: str):
    segment_index = int(segment_index)
    if axis == "source":
        segs = section.get("pivot_segments", [])
    elif axis == "target":
        segs = section.get("target_segments", {}).get(target_id, [])
    else:
        raise AlignmentEditError("Axe inconnu.")
    if not (0 <= segment_index < len(segs)):
        raise AlignmentEditError("Segment introuvable.")
    segs[segment_index] = str(text).strip()
    section.setdefault("edit_log", []).append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target_id,
            "action": f"édition texte {axis} {segment_index + 1}",
        }
    )
