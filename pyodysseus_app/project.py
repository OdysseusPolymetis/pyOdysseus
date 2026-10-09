from __future__ import annotations

import csv
import html
import io
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, List, Optional


PROJECT_VERSION = 2


def new_project(name: str, pivot_id: str, pivot_text: str, targets: Dict[str, str]) -> dict:
    return {
        "format": "pyodysseus-project",
        "version": PROJECT_VERSION,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pivot": {"id": pivot_id, "text": pivot_text},
        "targets": {tid: {"id": tid, "text": txt} for tid, txt in targets.items()},
        "settings": {},
        "sections": [],
    }


def dumps_project(project: dict) -> bytes:
    p = deepcopy(project)
    p["updated_at"] = datetime.now(timezone.utc).isoformat()
    return json.dumps(p, ensure_ascii=False, indent=2).encode("utf-8")


def loads_project(data: bytes) -> dict:
    p = json.loads(data.decode("utf-8"))
    if p.get("format") != "pyodysseus-project":
        raise ValueError("Ce fichier n'est pas un projet pyOdysseus reconnu.")
    if int(p.get("version", 0)) > PROJECT_VERSION:
        raise ValueError("Ce projet a été créé avec une version plus récente de l'application.")
    return p


def cell_text(section: dict, target_id: str, row_index: int) -> str:
    cells = section.get("cells_by_target", {}).get(target_id, [])
    segs = section.get("target_segments", {}).get(target_id, [])
    if row_index >= len(cells):
        return ""
    cell = cells[row_index]
    if cell == 0:
        return ""
    if not isinstance(cell, dict):
        return ""
    idxs = cell.get("idxs", [])
    override = cell.get("override")
    if override is not None:
        return override
    return "\n".join(segs[i] for i in idxs if 0 <= i < len(segs))


def _insertion_text(section: dict, target_id: str, gap: int, bead_pos: int) -> str:
    insertions = section.get("insertions_by_target", {}).get(target_id, {})
    groups = insertions.get(str(gap), insertions.get(gap, []))
    if bead_pos >= len(groups):
        return ""
    segs = section.get("target_segments", {}).get(target_id, [])
    idxs = groups[bead_pos]
    return "\n".join(segs[i] for i in idxs if 0 <= i < len(segs))


def synoptic_rows(section: dict, target_ids: Optional[List[str]] = None) -> List[dict]:
    target_ids = target_ids or list(section.get("target_segments", {}).keys())
    pivot = section.get("pivot_segments", [])
    rows = []
    overrides = section.get("pivot_overrides", {})
    pivot_id = section.get("pivot_id", "Pivot")

    # A source gap can contain several distinct 0→N Bertalign beads. We render one
    # source-empty row per bead position instead of concatenating all of them onto
    # the next pivot segment.
    for gap in range(len(pivot) + 1):
        counts = []
        for tid in target_ids:
            ins = section.get("insertions_by_target", {}).get(tid, {})
            groups = ins.get(str(gap), ins.get(gap, []))
            counts.append(len(groups))
        max_insertions = max(counts, default=0)
        for bead_pos in range(max_insertions):
            row = {"#": f"↳ {gap + 1}", pivot_id: ""}
            for tid in target_ids:
                row[tid] = _insertion_text(section, tid, gap, bead_pos)
            rows.append(row)

        if gap < len(pivot):
            src = pivot[gap]
            shown_src = overrides.get(str(gap), src)
            row = {"#": gap + 1, pivot_id: shown_src}
            for tid in target_ids:
                row[tid] = cell_text(section, tid, gap)
            rows.append(row)
    return rows



def alignment_blocks(section: dict, target_id: str) -> List[dict]:
    """Return Bertalign beads as explicit review blocks.

    Indices remain zero-based internally, while ``source_number`` /
    ``target_number`` are one-based for display. The original bead order is
    preserved, including 0→N and N→0 gaps.
    """
    beads = section.get("beads_by_target", {}).get(target_id, [])
    pivot = section.get("pivot_segments", [])
    targets = section.get("target_segments", {}).get(target_id, [])
    pivot_overrides = section.get("pivot_overrides", {})
    cells = section.get("cells_by_target", {}).get(target_id, [])

    blocks: List[dict] = []
    for pos, bead in enumerate(beads):
        if not isinstance(bead, (list, tuple)) or len(bead) != 2:
            continue
        src_raw, tgt_raw = bead
        src_idxs = [int(i) for i in src_raw]
        tgt_idxs = [int(i) for i in tgt_raw]

        source_items = []
        for i in src_idxs:
            if 0 <= i < len(pivot):
                source_items.append(
                    {
                        "index": i,
                        "number": i + 1,
                        "text": pivot_overrides.get(str(i), pivot[i]),
                    }
                )

        target_items = []
        for i in tgt_idxs:
            if 0 <= i < len(targets):
                target_items.append({"index": i, "number": i + 1, "text": targets[i]})

        # Preserve the legacy manual text override when it belongs exactly to this
        # source-bearing bead. This keeps older projects readable in the new view.
        target_override = None
        if src_idxs:
            anchor = src_idxs[0]
            if 0 <= anchor < len(cells) and isinstance(cells[anchor], dict):
                cell = cells[anchor]
                cell_idxs = [int(i) for i in cell.get("idxs", [])]
                if cell_idxs == tgt_idxs and cell.get("override") is not None:
                    target_override = str(cell.get("override"))

        m = len(src_idxs)
        n = len(tgt_idxs)
        if m == 0 and n > 0:
            kind = "target_only"
        elif m > 0 and n == 0:
            kind = "source_only"
        elif m == 1 and n == 1:
            kind = "one_to_one"
        else:
            kind = "complex"

        blocks.append(
            {
                "position": pos,
                "type": f"{m}→{n}",
                "kind": kind,
                "source_indices": src_idxs,
                "target_indices": tgt_idxs,
                "source": source_items,
                "target": target_items,
                "target_override": target_override,
            }
        )
    return blocks


def segment_range_label(indices: List[int], prefix: str = "") -> str:
    """Human-readable one-based label for a contiguous Bertalign index list."""
    if not indices:
        return "—"
    nums = [int(i) + 1 for i in indices]
    if len(nums) == 1:
        core = str(nums[0])
    elif nums == list(range(nums[0], nums[-1] + 1)):
        core = f"{nums[0]}–{nums[-1]}"
    else:
        core = ", ".join(str(n) for n in nums)
    return f"{prefix}{core}" if prefix else core

def export_csv(project: dict, section_index: int, target_ids: Optional[List[str]] = None) -> bytes:
    section = project["sections"][section_index]
    rows = synoptic_rows(section, target_ids=target_ids)
    out = io.StringIO()
    if not rows:
        return b""
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8-sig")


def export_html(project: dict, section_index: int, target_ids: Optional[List[str]] = None) -> bytes:
    section = project["sections"][section_index]
    title = f"{project.get('name','pyOdysseus')} — section {section_index + 1}"
    table = synoptic_table_html(section, target_ids=target_ids)
    doc = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:1.5rem;}}
.synoptic-wrap{{overflow:auto;max-height:90vh;border:1px solid #ddd;border-radius:10px;}}
.synoptic-table{{border-collapse:collapse;width:max-content;min-width:100%;table-layout:fixed;}}
.synoptic-table th,.synoptic-table td{{border:1px solid #ddd;padding:.65rem;vertical-align:top;min-width:260px;max-width:460px;white-space:normal;overflow-wrap:anywhere;}}
.synoptic-table th{{position:sticky;top:0;background:#fafafa;z-index:2;}}
.synoptic-table th:first-child,.synoptic-table td.num{{min-width:55px;max-width:55px;text-align:right;}}
.segno{{display:inline-block;font-size:.75rem;color:#777;margin-right:.4rem;}}
.target-segment + .target-segment{{margin-top:.5rem;padding-top:.5rem;border-top:1px dashed #ddd;}}
.orphans{{margin-top:1rem;}}
</style></head><body><h1>{html.escape(title)}</h1>{table}</body></html>"""
    return doc.encode("utf-8")


def synoptic_table_html(section: dict, target_ids: Optional[List[str]] = None) -> str:
    """Render a true synoptic table with HTML rowspans for N→M blocks."""
    target_ids = target_ids or list(section.get("target_segments", {}).keys())
    pivot = section.get("pivot_segments", [])
    pivot_id = section.get("pivot_id", "Pivot")
    cells_by_target = section.get("cells_by_target", {})

    headers = ["#", pivot_id] + list(target_ids)
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []

    for i, source_text in enumerate(pivot):
        row = [
            f'<td class="num">{i+1}</td>',
            f'<td class="source"><span class="segno">{i+1}</span>{html.escape(str(source_text))}</td>',
        ]
        for tid in target_ids:
            cells = cells_by_target.get(tid, [])
            cell = cells[i] if i < len(cells) else None
            if cell == 0:
                # The preceding cell owns this table slot through rowspan.
                continue
            if isinstance(cell, dict):
                rowspan = max(1, int(cell.get("rowspan", 1)))
                idxs = [int(x) for x in cell.get("idxs", [])]
                segs = section.get("target_segments", {}).get(tid, [])
                pieces = []
                for j in idxs:
                    if 0 <= j < len(segs):
                        pieces.append(
                            f'<div class="target-segment"><span class="segno">{j+1}</span>{html.escape(str(segs[j]))}</div>'
                        )
                content = "".join(pieces) if pieces else '<span class="empty">—</span>'
                row.append(f'<td class="target" rowspan="{rowspan}">{content}</td>')
            else:
                row.append('<td class="target"><span class="empty">—</span></td>')
        body.append("<tr>" + "".join(row) + "</tr>")

    orphan_groups = []
    for tid in target_ids:
        for gap, groups in section.get("insertions_by_target", {}).get(tid, {}).items():
            for idxs in groups:
                segs = section.get("target_segments", {}).get(tid, [])
                text = " ".join(segs[int(j)] for j in idxs if 0 <= int(j) < len(segs))
                nums = ", ".join(str(int(j) + 1) for j in idxs)
                orphan_groups.append((tid, int(gap), nums, text))

    orphan_html = ""
    if orphan_groups:
        items = "".join(
            f'<li><strong>{html.escape(tid)}</strong> · gap source {gap} · C {html.escape(nums)} — {html.escape(text)}</li>'
            for tid, gap, nums, text in orphan_groups
        )
        orphan_html = (
            f'<details class="orphans"><summary>Segments cibles sans source ({len(orphan_groups)})</summary>'
            f'<ul>{items}</ul></details>'
        )

    return (
        '<div class="synoptic-wrap"><table class="synoptic-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
        + orphan_html
    )


def aligned_target_text_for_source(section: dict, target_id: str, source_index: int) -> str:
    """Return the complete target bead covering one source segment."""
    for bead in section.get("beads_by_target", {}).get(target_id, []):
        if not isinstance(bead, (list, tuple)) or len(bead) != 2:
            continue
        src = [int(x) for x in bead[0]]
        tgt = [int(x) for x in bead[1]]
        if int(source_index) in src:
            segs = section.get("target_segments", {}).get(target_id, [])
            return "\n".join(segs[i] for i in tgt if 0 <= i < len(segs))
    return ""
