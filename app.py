from __future__ import annotations

import pathlib
import re
import time
from typing import Dict, List

import pandas as pd
import streamlit as st

from pyodysseus_app.engine import (
    AlignParams,
    align_section,
    configure_bertalign_model,
    segmentation_stats,
    split_sections,
    split_sections_named,
    section_map,
    split_units_clean,
    text_id_from_name,
)
from pyodysseus_app.alignment.bridge_encoder import PublishedBridgeEncoder
from pyodysseus_app.editor import (
    AlignmentEditError,
    edit_segment_text,
    give_to_next,
    give_to_previous,
    merge_with_next,
    merge_with_previous,
    split_block,
    take_from_next,
    take_from_previous,
)
from pyodysseus_app.lexical import compare_aligned_passage, rare_shared_analysis, rare_shared_passages, highlight_lemmas_html
from pyodysseus_app.project import (
    alignment_blocks,
    dumps_project,
    export_csv,
    export_html,
    loads_project,
    new_project,
    segment_range_label,
    synoptic_table_html,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent

st.set_page_config(
    page_title="pyOdysseus",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top:1.35rem; padding-bottom:3rem; max-width:1900px;}
.pyod-muted {opacity:.72; font-size:.92rem;}
textarea {line-height:1.45 !important;}
.synoptic-wrap {overflow:auto; max-height:72vh; border:1px solid rgba(128,128,128,.28); border-radius:12px;}
.synoptic-table {border-collapse:collapse; min-width:100%; width:max-content; table-layout:fixed; font-size:.94rem;}
.synoptic-table th, .synoptic-table td {border:1px solid rgba(128,128,128,.25); padding:.62rem .72rem; vertical-align:top; min-width:250px; max-width:430px; white-space:normal; overflow-wrap:anywhere;}
.synoptic-table th {position:sticky; top:0; background:var(--background-color,#fff); z-index:5; font-weight:700;}
.synoptic-table th:first-child,.synoptic-table td.num {min-width:48px; max-width:48px; width:48px; text-align:right; opacity:.65;}
.synoptic-table td.source {min-width:300px;}
.synoptic-table td.target[rowspan] {vertical-align:middle;}
.target-segment + .target-segment {margin-top:.55rem; padding-top:.55rem; border-top:1px dashed rgba(128,128,128,.25);}
.segno {display:inline-block; font-size:.72rem; opacity:.58; margin-right:.4rem; vertical-align:top;}
.empty {opacity:.4;}
.orphans {margin:.75rem 0 1rem; padding:.5rem .75rem; border:1px solid rgba(128,128,128,.2); border-radius:9px;}
.lemma-common {background:rgba(66,133,244,.18); border-bottom:2px solid rgba(66,133,244,.65);}
.lemma-almost {background:rgba(52,168,83,.18); border-bottom:2px solid rgba(52,168,83,.62);}
.lemma-shared {background:rgba(128,128,128,.12); border-bottom:1px dotted rgba(128,128,128,.8);}
.lemma-unique {background:rgba(251,188,4,.24); border-bottom:2px solid rgba(234,134,0,.72);}
.lex-card {border:1px solid rgba(128,128,128,.22); border-radius:12px; padding:.85rem 1rem; margin:.5rem 0 1rem; line-height:1.75;}
.lemma-sequence-match {text-decoration:underline; text-decoration-color:#5966c9; text-decoration-thickness:2px; text-underline-offset:3px;}
</style>
""",
    unsafe_allow_html=True,
)

if "project" not in st.session_state:
    st.session_state.project = None


def _read_uploaded_text(uploaded) -> str:
    return uploaded.getvalue().decode("utf-8", errors="replace")


def _sanitize_id(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip())
    return value or "texte"


def _checkpoint_path(project: dict) -> pathlib.Path:
    folder = REPO_ROOT / ".pyodysseus_projects"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{_sanitize_id(project.get('name', 'projet'))}.pyodysseus.json"


def _write_checkpoint(project: dict) -> pathlib.Path:
    """Persist the complete current project on the local machine.

    Streamlit is running locally, so this is a real filesystem checkpoint rather
    than a browser download. It is intentionally written after every completed
    section and after manual edits.
    """
    path = _checkpoint_path(project)
    path.write_bytes(dumps_project(project))
    return path


def _select_index(options: List[str], value: str, fallback: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return fallback


def _split_with_profile(text: str, profile: dict, settings: dict) -> List[str]:
    return split_units_clean(
        text,
        method=profile.get("method", "sat"),
        threshold=float(profile.get("threshold", 0.001)),
        sat_model=settings.get("sat_model", "sat-3l-sm"),
        device=settings.get("device", "auto"),
        constrained=bool(profile.get("constrained", False)),
        hybrid=bool(profile.get("hybrid", True)),
        min_length=int(profile.get("min_length", 1)),
        target_length=int(profile["target_length"]) if profile.get("target_length") else None,
        max_length=int(profile["max_length"]) if profile.get("max_length") else None,
        spread=int(profile["spread"]) if profile.get("spread") else None,
        prior_type=profile.get("prior_type", "gaussian"),
        detach_french_incises=bool(profile.get("detach_french_incises", False)),
    )


def _segmentation_controls(prefix: str, defaults: dict, *, target: bool = False) -> dict:
    method_options = ["SaT / wtpsplit", "Une ligne = un segment", "Paragraphes"]
    label_to_method = {
        "SaT / wtpsplit": "sat",
        "Une ligne = un segment": "lines",
        "Paragraphes": "paragraphs",
    }
    method_to_label = {v: k for k, v in label_to_method.items()}
    method_label = st.selectbox(
        "Méthode",
        method_options,
        index=_select_index(method_options, method_to_label.get(defaults.get("method", "sat"), "SaT / wtpsplit")),
        key=f"{prefix}_method",
    )
    method = label_to_method[method_label]
    profile = {
        "method": method,
        "threshold": float(defaults.get("threshold", 0.001)),
        "constrained": bool(defaults.get("constrained", False)),
        "hybrid": bool(defaults.get("hybrid", True)),
        "min_length": int(defaults.get("min_length", 12 if not target else 18)),
        "target_length": int(defaults.get("target_length", 90 if target else 110)),
        "max_length": int(defaults.get("max_length", 80 if target else 80)),
        "spread": int(defaults.get("spread", 35 if target else 45)),
        "prior_type": defaults.get("prior_type", "gaussian"),
        "detach_french_incises": bool(defaults.get("detach_french_incises", target)),
    }

    if method == "sat":
        mode_options = ["Seuil fin + plafond (recommandé)", "Seuil classique", "Longueur contrainte globale"]
        if profile["hybrid"]:
            mode_i = 0
        elif profile["constrained"]:
            mode_i = 2
        else:
            mode_i = 1
        mode = st.radio("Mode SaT", mode_options, index=mode_i, horizontal=True, key=f"{prefix}_mode")
        profile["hybrid"] = mode == mode_options[0]
        profile["constrained"] = mode == mode_options[2]

        if profile["hybrid"]:
            c1, c2, c3 = st.columns(3)
            profile["threshold"] = float(c1.number_input("Seuil SaT", 0.000001, 1.0, profile["threshold"], format="%.6f", key=f"{prefix}_threshold"))
            profile["min_length"] = int(c2.number_input("Longueur minimale", 1, 500, profile["min_length"], key=f"{prefix}_min"))
            profile["max_length"] = int(c3.number_input("Longueur maximale", 20, 2000, profile["max_length"], step=5, key=f"{prefix}_max"))
            st.caption("Le seuil conserve les coupures fines ; le minimum évite les micro-segments ; le plafond ne redécoupe que les unités trop longues.")
        elif profile["constrained"]:
            c1, c2 = st.columns(2)
            profile["target_length"] = int(c1.number_input("Longueur cible", 20, 1000, profile["target_length"], step=10, key=f"{prefix}_target"))
            profile["max_length"] = int(c2.number_input("Longueur maximale", 30, 2000, profile["max_length"], step=10, key=f"{prefix}_cmax"))
            profile["min_length"] = int(st.number_input("Longueur minimale", 1, 500, profile["min_length"], key=f"{prefix}_cmin"))
        else:
            profile["threshold"] = float(st.number_input("Seuil SaT", 0.000001, 1.0, profile["threshold"], format="%.6f", key=f"{prefix}_classic"))

    if target:
        profile["detach_french_incises"] = st.checkbox(
            "Détacher les incises françaises de dialogue",
            value=profile["detach_french_incises"],
            key=f"{prefix}_incises",
        )
    else:
        profile["detach_french_incises"] = False
    return profile


def _build_sections(
    project: dict,
    settings: dict,
    params: AlignParams,
    encoder_override=None,
    section_indices: List[int] | None = None,
):
    pivot_id = project["pivot"]["id"]
    pivot_items = split_sections_named(project["pivot"]["text"], pattern=settings["section_pattern"])
    pivot_keys = [item["key"] for item in pivot_items]
    target_maps = {
        tid: section_map(item["text"], pattern=settings["section_pattern"])
        for tid, item in project["targets"].items()
    }

    if section_indices is None:
        section_indices = list(range(len(pivot_items)))
    section_indices = sorted({int(i) for i in section_indices if 0 <= int(i) < len(pivot_items)})
    if not section_indices:
        return project.get("sections", [])

    total_pairs = sum(
        1
        for sec_idx in section_indices
        for pieces in target_maps.values()
        if pivot_keys[sec_idx] in pieces
    )
    progress = st.progress(0.0, text="Préparation…")
    status = st.empty()
    completed = 0

    existing = {int(sec.get("index", pos)): sec for pos, sec in enumerate(project.get("sections", []))}

    for run_pos, sec_idx in enumerate(section_indices, start=1):
        pivot_item = pivot_items[sec_idx]
        section_key = pivot_item["key"]
        section_label = pivot_item["label"]
        pivot_segments = _split_with_profile(
            pivot_item["text"], settings["pivot_segmentation"], settings
        )

        target_segments = {}
        missing_targets = []
        for tid, pieces in target_maps.items():
            item = pieces.get(section_key)
            if item is None:
                missing_targets.append(tid)
                continue
            target_segments[tid] = _split_with_profile(
                item["text"], settings["target_segmentation"], settings
            )

        if missing_targets:
            st.warning(
                f"{section_label} : section absente chez "
                + ", ".join(missing_targets)
                + ". Ces cibles ne seront pas décalées vers la section suivante."
            )

        section_pair_base = completed

        def progress_cb(tid: str, pos: int, count: int):
            status.info(
                f"{section_label} ({run_pos}/{len(section_indices)}) — **{tid}** ({pos}/{count})"
            )
            if total_pairs:
                progress.progress(min(.99, (section_pair_base + pos - 1) / total_pairs))

        cells, beads, insertions = align_section(
            REPO_ROOT,
            pivot_segments=pivot_segments,
            targets_segments=target_segments,
            params=params,
            progress_cb=progress_cb,
            encoder_override=encoder_override,
        )
        completed += len(target_segments)
        existing[sec_idx] = {
            "index": sec_idx,
            "section_key": section_key,
            "label": section_label,
            "pivot_id": pivot_id,
            "pivot_segments": pivot_segments,
            "target_segments": target_segments,
            "cells_by_target": cells,
            "beads_by_target": beads,
            "insertions_by_target": insertions,
            "edit_log": [],
        }
        project["sections"] = [existing[i] for i in sorted(existing)]
        checkpoint = _write_checkpoint(project)
        status.success(f"{section_label} terminé · checkpoint : {checkpoint.name}")
        progress.progress(min(1.0, completed / max(1, total_pairs)))

    progress.progress(1.0, text="Alignement terminé")
    status.success(
        f"{len(section_indices)} section(s) traitée(s). Checkpoint local mis à jour après chaque section."
    )
    return project["sections"]


def page_project():
    st.header("Projet")
    st.caption("Un texte pivot, plusieurs traductions cibles, un projet sauvegardable.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Nouveau projet")
        name = st.text_input("Nom du projet", value="Nouveau projet")
        pivot = st.file_uploader("Texte pivot (.txt)", type=["txt"], accept_multiple_files=False)
        targets = st.file_uploader("Textes cibles (.txt)", type=["txt"], accept_multiple_files=True)
        if st.button("Créer le projet", type="primary", use_container_width=True):
            if pivot is None or not targets:
                st.error("Ajoute un pivot et au moins une cible.")
            else:
                pivot_id = _sanitize_id(text_id_from_name(pivot.name))
                target_map = {_sanitize_id(text_id_from_name(f.name)): _read_uploaded_text(f) for f in targets}
                ids = [pivot_id] + list(target_map)
                if len(ids) != len(set(ids)):
                    st.error("Deux fichiers ont le même nom sans extension.")
                else:
                    st.session_state.project = new_project(name.strip() or "Nouveau projet", pivot_id, _read_uploaded_text(pivot), target_map)
                    st.success(f"Projet créé : 1 pivot + {len(target_map)} cible(s).")
    with c2:
        st.subheader("Ouvrir un projet")
        saved = st.file_uploader("Projet pyOdysseus (.json)", type=["json"], key="project_loader")
        if saved is not None and st.button("Ouvrir ce projet", use_container_width=True):
            try:
                st.session_state.project = loads_project(saved.getvalue())
                st.success("Projet chargé.")
            except Exception as exc:
                st.error(str(exc))

    project = st.session_state.project
    if project:
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Pivot", project["pivot"]["id"])
        m2.metric("Cibles", len(project["targets"]))
        m3.metric("Sections alignées", len(project.get("sections", [])))
        st.write("**Cibles :** " + ", ".join(project["targets"]))
        st.download_button("💾 Sauvegarder le projet", dumps_project(project), file_name=f"{_sanitize_id(project.get('name','projet'))}.pyodysseus.json", mime="application/json")


def page_alignment():
    st.header("Alignement")
    project = st.session_state.project
    if not project:
        st.info("Crée ou ouvre d'abord un projet.")
        return
    saved = project.get("settings", {})
    st.caption(f"Pivot : {project['pivot']['id']} · {len(project['targets'])} cible(s)")

    left, right = st.columns([1.08, .92])
    with left:
        st.subheader("Segmentation")
        section_pattern = st.text_input("Regex de début de section", value=saved.get("section_pattern", r"^(Chant\s*\d+)"))
        sat_model = st.text_input("Modèle SaT", value=saved.get("sat_model", "sat-3l-sm"))
        pivot_defaults = saved.get("pivot_segmentation", {"method":"sat","threshold":.001,"hybrid":True,"min_length":12,"max_length":80})
        target_defaults = saved.get("target_segmentation", {"method":"sat","threshold":.001,"hybrid":True,"min_length":18,"max_length":80,"detach_french_incises":True})
        t1, t2 = st.tabs(["Pivot", "Cibles"])
        with t1:
            pivot_profile = _segmentation_controls("pivot_seg", pivot_defaults, target=False)
        with t2:
            target_profile = _segmentation_controls("target_seg", target_defaults, target=True)

    with right:
        st.subheader("Embeddings")
        model_modes = ["Pont grec ancien → LaBSE (recommandé)", "SentenceTransformer symétrique"]
        default_mode = saved.get("embedding_mode", "bridge")
        mode_index = 0 if default_mode == "bridge" else 1
        model_mode = st.radio("Mode", model_modes, index=mode_index)
        device_options = ["auto", "cpu", "cuda", "mps"]
        device = st.selectbox("Calcul", device_options, index=_select_index(device_options, saved.get("device", "auto")))
        batch_size = int(st.number_input("Batch size", 1, 512, int(saved.get("batch_size", 64)), step=8))

        if model_mode == model_modes[0]:
            embedding_mode = "bridge"
            st.success("Pivot grec : **MOdysseus/SPhilBERTa-LaBSE-Bridge-Ridge**\n\nCibles : **sentence-transformers/LaBSE**")
            with st.expander("Modèles du pont"):
                bridge_source = st.text_input("Encodeur grec", value=saved.get("bridge_source", "MOdysseus/SPhilBERTa-LaBSE-Bridge-Ridge"))
                bridge_target = st.text_input("Encodeur cible", value=saved.get("bridge_target", "sentence-transformers/LaBSE"))
        else:
            embedding_mode = "symmetric"
            bridge_source = ""
            bridge_target = ""
            symmetric_model = st.text_input("SentenceTransformer", value=saved.get("symmetric_model", "sentence-transformers/LaBSE"))
            st.caption("À utiliser notamment pour français↔français ou pour comparer un autre modèle.")

        old_ba = saved.get("bertalign", {})
        with st.expander("Paramètres Bertalign"):
            max_align = int(st.number_input("max_align", 2, 10, int(old_ba.get("max_align", 5))))
            top_k = int(st.number_input("top_k", 1, 20, int(old_ba.get("top_k", 3))))
            win = int(st.number_input("win", 1, 30, int(old_ba.get("win", 5))))
            skip = float(st.number_input("skip", value=float(old_ba.get("skip", -.1)), step=.05, format="%.2f"))
            margin = st.checkbox("margin", value=bool(old_ba.get("margin", True)))
            len_penalty = st.checkbox("len_penalty", value=bool(old_ba.get("len_penalty", True)))

    current_settings = {
        "section_pattern": section_pattern,
        "sat_model": sat_model,
        "pivot_segmentation": pivot_profile,
        "target_segmentation": target_profile,
        "embedding_mode": embedding_mode,
        "bridge_source": bridge_source,
        "bridge_target": bridge_target,
        "symmetric_model": symmetric_model if embedding_mode == "symmetric" else "",
        "device": device,
        "batch_size": batch_size,
    }

    st.divider()
    st.subheader("Aperçu de la segmentation")
    preview_ids = [project["pivot"]["id"]] + list(project["targets"])
    pc1, pc2 = st.columns([1.2, .8])
    preview_id = pc1.selectbox("Texte", preview_ids, key="seg_preview_text")
    preview_text = project["pivot"]["text"] if preview_id == project["pivot"]["id"] else project["targets"][preview_id]["text"]
    preview_profile = pivot_profile if preview_id == project["pivot"]["id"] else target_profile
    try:
        preview_sections = split_sections_named(preview_text, pattern=section_pattern)
    except Exception:
        preview_sections = [{"key":"num:1", "label":"Section 1", "text":preview_text}]
    preview_sec = int(pc2.number_input("Section", 1, max(1, len(preview_sections)), 1, key=f"preview_section_{preview_id}")) - 1
    if st.button("🔎 Calculer l'aperçu", use_container_width=True):
        try:
            segs = _split_with_profile(preview_sections[preview_sec]["text"], preview_profile, current_settings)
            st.session_state.segmentation_preview = (preview_id, preview_sec, segs)
        except Exception as exc:
            st.exception(exc)
    preview = st.session_state.get("segmentation_preview")
    if preview and preview[:2] == (preview_id, preview_sec):
        segs = preview[2]
        stats = segmentation_stats(segs)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Segments", stats["count"]); m2.metric("Médiane", f"{stats['median']:.0f}"); m3.metric("Moyenne", f"{stats['mean']:.1f}"); m4.metric("Max", stats["max"])
        st.dataframe(pd.DataFrame([{"#":i+1,"car.":len(s),"segment":s} for i,s in enumerate(segs)]), hide_index=True, use_container_width=True, height=420)

    st.divider()
    st.subheader("Sections à aligner")
    try:
        pivot_all_sections = split_sections_named(project["pivot"]["text"], pattern=section_pattern)
        pivot_keys = [item["key"] for item in pivot_all_sections]
        target_section_maps = {
            tid: section_map(item["text"], pattern=section_pattern)
            for tid, item in project["targets"].items()
        }
    except Exception as exc:
        st.error(f"Impossible de découper les sections : {exc}")
        return

    aligned_indices = {int(sec.get("index", pos)) for pos, sec in enumerate(project.get("sections", []))}
    section_options = list(range(len(pivot_all_sections)))

    with st.expander("Contrôle des sections détectées"):
        pivot_labels = [item["label"] for item in pivot_all_sections]
        rows = [{
            "texte": project["pivot"]["id"],
            "rôle": "pivot",
            "sections": len(pivot_all_sections),
            "marqueurs": ", ".join(pivot_labels),
            "manquantes vs pivot": "—",
        }]
        for tid, secmap in target_section_maps.items():
            labels = [item["label"] for item in secmap.values()]
            missing = [
                pivot_all_sections[i]["label"]
                for i, key in enumerate(pivot_keys)
                if key not in secmap
            ]
            rows.append({
                "texte": tid,
                "rôle": "cible",
                "sections": len(secmap),
                "marqueurs": ", ".join(labels),
                "manquantes vs pivot": ", ".join(missing) if missing else "—",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        problematic = {
            tid: [pivot_all_sections[i]["label"] for i, key in enumerate(pivot_keys) if key not in secmap]
            for tid, secmap in target_section_maps.items()
            if any(key not in secmap for key in pivot_keys)
        }
        if problematic:
            st.warning(
                "Certaines sections du pivot ne sont pas présentes dans toutes les cibles. "
                "Elles seront laissées absentes : pyOdysseus ne décale plus jamais Chant2 vers Chant1."
            )

    missing_indices = [i for i in section_options if i not in aligned_indices]
    default_sections = missing_indices if missing_indices else []
    selected_sections = st.multiselect(
        "Sections à aligner / réaligner",
        options=section_options,
        default=default_sections,
        format_func=lambda i: pivot_all_sections[i]["label"] + (" ✓ déjà alignée" if i in aligned_indices else ""),
        help="Les sections déjà alignées qui ne sont pas sélectionnées sont conservées.",
    )
    if aligned_indices:
        st.caption(
            "Sections déjà disponibles : "
            + ", ".join(pivot_all_sections[i]["label"] if i < len(pivot_all_sections) else str(i+1) for i in sorted(aligned_indices))
            + ". Seules les sections sélectionnées seront remplacées."
        )
    st.caption(f"Checkpoint local automatique : `{_checkpoint_path(project)}`")

    if st.button("▶ Aligner les sections sélectionnées", type="primary", use_container_width=True, disabled=not selected_sections):
        try:
            params = AlignParams(max_align=max_align, top_k=top_k, win=win, skip=skip, margin=margin, len_penalty=len_penalty)
            t0 = time.perf_counter()
            if embedding_mode == "bridge":
                encoder = PublishedBridgeEncoder(bridge_source, bridge_target, device=device, batch_size=batch_size)
                actual_device = encoder.device
            else:
                encoder = None
                _, actual_device = configure_bertalign_model(REPO_ROOT, symmetric_model, device=device, batch_size=batch_size)
            st.info(f"Calcul : **{actual_device}**")
            project["settings"] = current_settings | {"bertalign": params.__dict__}
            project["sections"] = _build_sections(
                project, current_settings, params, encoder_override=encoder, section_indices=selected_sections
            )
            st.session_state.project = project
            checkpoint = _write_checkpoint(project)
            st.success(
                f"Alignement terminé en {time.perf_counter()-t0:.1f} s. "
                f"Projet sauvegardé localement dans {checkpoint}."
            )
        except Exception as exc:
            st.exception(exc)


def _show_block(section: dict, target_id: str, block: dict, compact: bool = False):
    src_label = segment_range_label(block["source_indices"], "S ")
    tgt_label = segment_range_label(block["target_indices"], "C ")
    with st.container(border=True):
        st.markdown(f"**{block['type']} · {src_label} · {tgt_label}**")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("SOURCE")
            if block["source"]:
                for item in block["source"]:
                    st.markdown(f"**{item['number']}.** {item['text']}")
            else:
                st.markdown("—")
        with c2:
            st.caption(f"CIBLE · {target_id}")
            if block["target"]:
                for item in block["target"]:
                    st.markdown(f"**{item['number']}.** {item['text']}")
            else:
                st.markdown("—")


def _run_edit(fn, *args):
    try:
        fn(*args)
        if st.session_state.project:
            _write_checkpoint(st.session_state.project)
        st.rerun()
    except AlignmentEditError as exc:
        st.error(str(exc))


def _correction_editor(project: dict, section: dict, sec_idx: int, available_targets: List[str]):
    target_id = st.selectbox("Version cible", available_targets, key=f"edit_target_{sec_idx}")
    blocks = alignment_blocks(section, target_id)
    if not blocks:
        st.info("Aucun bloc.")
        return
    block_no = int(st.number_input("Bloc", 1, len(blocks), 1, key=f"edit_block_{sec_idx}_{target_id}"))
    i = block_no - 1
    st.caption("Les boutons déplacent réellement la frontière entre les beads ; les segments ne sont jamais dupliqués.")

    if i > 0:
        with st.expander("Bloc précédent"):
            _show_block(section, target_id, blocks[i-1], compact=True)
    _show_block(section, target_id, blocks[i])
    if i < len(blocks)-1:
        with st.expander("Bloc suivant"):
            _show_block(section, target_id, blocks[i+1], compact=True)

    st.subheader("Déplacer les frontières")
    for axis, title in [("source", "Source"), ("target", "Cible")]:
        st.markdown(f"**{title}**")
        b1,b2,b3,b4 = st.columns(4)
        if b1.button("＋ précédent", key=f"{axis}_take_prev_{sec_idx}_{target_id}_{i}", use_container_width=True):
            _run_edit(take_from_previous, section, target_id, i, axis)
        if b2.button("− début → précédent", key=f"{axis}_give_prev_{sec_idx}_{target_id}_{i}", use_container_width=True):
            _run_edit(give_to_previous, section, target_id, i, axis)
        if b3.button("− fin → suivant", key=f"{axis}_give_next_{sec_idx}_{target_id}_{i}", use_container_width=True):
            _run_edit(give_to_next, section, target_id, i, axis)
        if b4.button("＋ suivant", key=f"{axis}_take_next_{sec_idx}_{target_id}_{i}", use_container_width=True):
            _run_edit(take_from_next, section, target_id, i, axis)

    st.markdown("**Blocs**")
    m1,m2 = st.columns(2)
    if m1.button("Fusionner avec précédent", key=f"merge_prev_{sec_idx}_{target_id}_{i}", use_container_width=True):
        _run_edit(merge_with_previous, section, target_id, i)
    if m2.button("Fusionner avec suivant", key=f"merge_next_{sec_idx}_{target_id}_{i}", use_container_width=True):
        _run_edit(merge_with_next, section, target_id, i)

    block = blocks[i]
    if len(block["source_indices"]) + len(block["target_indices"]) >= 3:
        with st.expander("Scinder ce bloc"):
            sc1, sc2 = st.columns(2)
            src_left = int(sc1.number_input("Segments source dans le bloc gauche", 0, len(block["source_indices"]), max(0, len(block["source_indices"])//2), key=f"split_s_{sec_idx}_{target_id}_{i}"))
            tgt_left = int(sc2.number_input("Segments cible dans le bloc gauche", 0, len(block["target_indices"]), max(0, len(block["target_indices"])//2), key=f"split_t_{sec_idx}_{target_id}_{i}"))
            if st.button("Scinder", key=f"split_go_{sec_idx}_{target_id}_{i}"):
                _run_edit(split_block, section, target_id, i, src_left, tgt_left)

    with st.expander("Modifier le texte d'un segment"):
        side = st.radio("Côté", ["source", "target"], horizontal=True, key=f"text_side_{sec_idx}_{target_id}_{i}")
        ids = block["source_indices"] if side == "source" else block["target_indices"]
        if ids:
            shown = [x+1 for x in ids]
            selected_num = st.selectbox("Segment", shown, key=f"text_seg_{sec_idx}_{target_id}_{i}_{side}")
            idx = int(selected_num)-1
            current = section["pivot_segments"][idx] if side == "source" else section["target_segments"][target_id][idx]
            edited = st.text_area("Texte", value=current, height=120, key=f"text_edit_{sec_idx}_{target_id}_{i}_{side}_{idx}")
            if st.button("Enregistrer le texte", key=f"save_text_{sec_idx}_{target_id}_{i}_{side}_{idx}"):
                edit_segment_text(section, target_id, side, idx, edited)
                _write_checkpoint(project)
                st.rerun()
        else:
            st.info("Aucun segment sur ce côté du bloc.")


@st.cache_data(max_entries=20, show_spinner=False)
def _cached_rare_analysis(section: dict, target_ids: tuple, include_propn: bool,
                          support_sizes: tuple, max_total_occurrences: int):
    """Résultat global indépendant de la paire affichée ; invalidation par contenu."""
    return rare_shared_analysis(
        section, list(target_ids), include_propn=include_propn,
        support_sizes=support_sizes, max_total_occurrences=max_total_occurrences,
    )


@st.cache_data(max_entries=60, show_spinner=False)
def _cached_rare_passages(section: dict, reference_ids: tuple, focus_ids: tuple,
                          include_propn: bool, support_sizes: tuple, max_total_occurrences: int,
                          context_radius: int, min_ordered_lemmas: int, top_n: int,
                          rare_result: dict, rarity_bonus: float = 0.25):
    """La paire est classée sans présélection par vocabulaire rare."""
    return rare_shared_passages(
        section, reference_target_ids=list(reference_ids), focus_target_ids=list(focus_ids),
        include_propn=include_propn, support_sizes=support_sizes,
        max_total_occurrences=max_total_occurrences, context_radius=context_radius,
        min_ordered_lemmas=min_ordered_lemmas, top_n=top_n,
        rare_result=rare_result, rarity_bonus=rarity_bonus,
    )


def _lexical_tab(section: dict, sec_idx: int, available_targets: List[str]):
    st.caption("Analyse française : mots outils exclus ; noms, verbes, adjectifs, adverbes et éventuellement noms propres sont lemmatisés.")
    chosen = st.multiselect("Traductions comparées", available_targets, default=available_targets, key=f"lex_targets_{sec_idx}")
    if len(chosen) < 2:
        st.info("Sélectionne au moins deux traductions.")
        return
    include_propn = st.checkbox("Inclure les noms propres", value=True, key=f"lex_propn_{sec_idx}")
    almost_ratio = st.slider("Seuil « presque tous »", .5, 1.0, .8, .05, key=f"lex_ratio_{sec_idx}")
    context_radius = int(st.slider(
        "Contexte lexical autour du segment pivot",
        min_value=0, max_value=2, value=1, step=1,
        help="1 = calcul sur n−1, n et n+1. Les blocs m→n couvrant plusieurs segments ne sont comptés qu'une fois.",
        key=f"lex_context_{sec_idx}",
    ))
    source_num = int(st.number_input("Segment pivot", 1, max(1,len(section["pivot_segments"])), 1, key=f"lex_source_{sec_idx}"))
    source_idx = source_num - 1

    lo = max(0, source_idx - context_radius)
    hi = min(len(section["pivot_segments"]) - 1, source_idx + context_radius)
    st.markdown("**Fenêtre pivot utilisée pour le calcul lexical**")
    for idx in range(lo, hi + 1):
        marker = "**→**" if idx == source_idx else "&nbsp;&nbsp;"
        st.markdown(f"{marker} **{idx+1}.** {section['pivot_segments'][idx]}")

    try:
        with st.spinner("Préparation des annotations lexicales du passage…"):
            result = compare_aligned_passage(
                section, chosen, source_idx, include_propn=include_propn,
                almost_ratio=almost_ratio, context_radius=context_radius,
            )
    except Exception as exc:
        st.error("Erreur lors de la comparaison lexicale du passage :")
        st.exception(exc)
        return

    if context_radius == 1:
        st.caption("Fréquences locales calculées sur n−1, n et n+1 ; un même bloc aligné n’est compté qu’une fois.")
    elif context_radius > 1:
        st.caption(f"Fréquences locales calculées sur ±{context_radius} segments autour de n ; un même bloc aligné n’est compté qu’une fois.")
    else:
        st.caption("Fréquences locales calculées sur le seul segment n.")
    st.markdown("<span class='lemma-common'>tous</span> &nbsp; <span class='lemma-almost'>presque tous</span> &nbsp; <span class='lemma-shared'>partagé</span> &nbsp; <span class='lemma-unique'>unique</span>", unsafe_allow_html=True)
    for tid in chosen:
        idxs = result["target_indices"].get(tid, [])
        label = ", ".join(str(i+1) for i in idxs) or "—"
        st.markdown(f"**{tid} · C {label}**")
        rendered = result["rendered"].get(tid, "") or "<em>aucun texte aligné</em>"
        st.markdown(f'<div class="lex-card">{rendered}</div>', unsafe_allow_html=True)

    st.dataframe(pd.DataFrame(result["stats"]), hide_index=True, use_container_width=True)
    with st.expander("Table des lemmes du passage"):
        df = pd.DataFrame(result["support"])
        if not df.empty:
            mapping = {"common":"tous", "almost":"presque tous", "shared":"partagé", "unique":"unique", "other":"autre"}
            df["catégorie"] = df["catégorie"].map(mapping).fillna(df["catégorie"])
        st.dataframe(df, hide_index=True, use_container_width=True, height=420)

    st.divider()
    st.subheader("Proximité textuelle entre traducteurs")
    st.caption(
        "La continuité de longues chaînes de lemmes dans le même ordre est le critère principal. "
        "Les lemmes rares qui apparaissent DANS ces chaînes apportent un bonus, "
        "mais leur présence n'est jamais obligatoire. L'indice ne prouve pas un emprunt."
    )
    rc1, rc2 = st.columns(2)
    support_mode = rc1.multiselect(
        "Rareté : nombre de traducteurs partageant le lemme",
        options=[2, 3], default=[2, 3], key=f"rare_support_{sec_idx}",
        help="Détermine seulement quels lemmes peuvent fournir un bonus de rareté."
    )
    max_occ = int(rc2.number_input(
        "Rareté : occurrences totales max. dans le chant",
        min_value=1, max_value=200, value=12, step=1,
        key=f"rare_max_occ_{sec_idx}",
        help="N'affecte pas la découverte des chaînes, uniquement leur bonus éventuel."
    ))
    try:
        with st.spinner("Préparation des fréquences et de l'index lexical du chant…"):
            rare = _cached_rare_analysis(
                section, tuple(chosen), include_propn,
                tuple(support_mode), max_occ,
            )
    except Exception as exc:
        st.error("Erreur lors du calcul des annotations lexicales :")
        st.exception(exc)
        return

    pair_df = pd.DataFrame(rare["pairs"])
    with st.expander("Diagnostic complémentaire : vocabulaire rare partagé sur le chant"):
        st.caption(
            "Les totaux sur un chant ne sont pas des indices suffisants d'emprunt. "
            "Ils servent à documenter le bonus de rareté local."
        )
        if pair_df.empty:
            st.info("Aucun lemme rare partagé avec ces critères ; la recherche de chaînes reste disponible.")
        else:
            st.dataframe(pair_df, hide_index=True, use_container_width=True, height=300)
        group_df = pd.DataFrame(rare["groups"])
        if not group_df.empty:
            st.markdown("**Groupes exacts de traducteurs**")
            st.dataframe(group_df, hide_index=True, use_container_width=True, height=240)

    # Toutes les paires sont proposées, même si elles n'ont aucun lemme rare.
    pairs = [(a, b) for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    pair_labels = [f"{a} ↔ {b}" for a, b in pairs]
    selected_pair_label = st.selectbox(
        "Comparer deux traductions", pair_labels, key=f"rare_pair_{sec_idx}",
    )
    focus = pairs[pair_labels.index(selected_pair_label)]

    pc1, pc2, pc3 = st.columns(3)
    passage_radius = int(pc1.slider(
        "Fenêtre autour du pivot", 0, 4, 1, 1,
        key=f"rare_passage_radius_{sec_idx}",
        help="1 = n−1, n, n+1 ; les fenêtres se fondent sur les segments alignés."
    ))
    min_ordered = int(pc2.number_input(
        "Minimum de lemmes en ordre", 4, 50, 7, 1,
        key=f"lex_chain_min_{sec_idx}",
        help="Évite de remonter les ressemblances trop courtes (7 par défaut)."
    ))
    top_passages = int(pc3.number_input(
        "Nombre de passages", 3, 50, 12, 1,
        key=f"rare_passage_top_{sec_idx}",
    ))
    use_rarity = st.checkbox(
        "Ajouter un bonus de rareté aux chaînes lexicales",
        value=True, key=f"lex_rarity_bonus_{sec_idx}",
        help="Bonus de 0 à 25 % de la note de continuité, attribué uniquement "
             "aux lemmes rares présents dans la chaîne détectée."
    )
    st.caption(
        "Score principal : continuité, couverture des deux passages et longueur de la chaîne "
        "(0–100). Bonus de rareté : 0 à 25 %, sans condition minimale de rareté. "
        "Les pondérations sont exploratoires et non probabilistes."
    )
    try:
        with st.spinner("Recherche des chaînes lexicales proches (sans re-lemmatisation)…"):
            concentrated = _cached_rare_passages(
                section, tuple(chosen), focus, include_propn,
                tuple(support_mode), max_occ, passage_radius, min_ordered,
                top_passages, rare, 0.25 if use_rarity else 0.0,
            )
    except Exception as exc:
        st.error("Erreur lors de la recherche de reprises lexicales :")
        st.exception(exc)
        return
    if not concentrated:
        st.info("Aucune chaîne ne satisfait ces critères. Essaie une fenêtre plus large ou un seuil plus faible.")
        return

    passage_table = pd.DataFrame([
        {
            "segments pivot": r["segments pivot"],
            "lemmes en ordre": r["lemmes ordonnés"],
            "couverture": r["couverture séquentielle"],
            "plus longue suite": r["plus longue suite exacte"],
            "score séquence": r["score séquence"],
            "raretés dans la chaîne": r["ancrages rares ordonnés"],
            "bonus rareté (%)": r["bonus rareté (%)"],
            "score final": r["score final"],
        }
        for r in concentrated
    ])
    st.dataframe(passage_table, hide_index=True, use_container_width=True, height=360)
    passage_labels = [
        f"Pivot {r['segments pivot']} · {r['lemmes ordonnés']} lemmes alignés · score {r['score final']}"
        for r in concentrated
    ]
    selected_passage_label = st.selectbox(
        "Examiner un passage proche", passage_labels,
        key=f"rare_passage_select_{sec_idx}",
    )
    passage = concentrated[passage_labels.index(selected_passage_label)]
    st.markdown(
        f"**Chaîne ordonnée :** {passage['lemmes ordonnés']} correspondances "
        f"· plus longue suite contiguë : {passage['plus longue suite exacte']} "
        f"· score de continuité : {passage['score séquence']} "
        f"· bonus rareté : {passage['bonus rareté (%)']} %"
    )
    if passage["chaîne de lemmes"]:
        st.caption(" → ".join(passage["chaîne de lemmes"]))
        st.caption("Les mots de la chaîne sont soulignés en bleu ; les raretés sont surlignées séparément.")
    st.markdown(f"**Pivot grec · segments {passage['segments pivot']}**")
    st.info(passage["pivot"] or "—")
    rare_lemmas_here = [x.strip() for x in passage["lemmes"].split(",") if x.strip()]
    for tid in focus:
        idxs = passage["target_indices"].get(tid, [])
        idx_label = ", ".join(str(i + 1) for i in idxs) or "—"
        st.markdown(f"**{tid} · segments {idx_label}**")
        txt = passage["texts"].get(tid, "") or ""
        rendered = highlight_lemmas_html(
            txt, rare_lemmas_here, include_propn=include_propn,
            section=section, target_id=tid, target_indices=idxs,
            matched_content_positions=passage.get("positions de chaîne", {}).get(tid, []),
        ) if txt else "—"
        st.markdown(f'<div class="lex-card">{rendered}</div>', unsafe_allow_html=True)


def page_review():
    st.header("Relecture, correction et analyse")

    with st.expander("Ouvrir directement un projet / des alignements sauvegardés"):
        review_saved = st.file_uploader(
            "Projet pyOdysseus (.json)", type=["json"], key="review_project_loader"
        )
        if review_saved is not None and st.button("Charger dans Relecture & analyse", use_container_width=True):
            try:
                st.session_state.project = loads_project(review_saved.getvalue())
                st.success("Projet et alignements chargés.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

        current = st.session_state.project
        if current:
            checkpoint = _checkpoint_path(current)
            st.caption(f"Checkpoint local : `{checkpoint}`")
            if checkpoint.exists() and st.button("Recharger le checkpoint local", use_container_width=True):
                try:
                    st.session_state.project = loads_project(checkpoint.read_bytes())
                    st.success("Checkpoint local rechargé.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    project = st.session_state.project
    if not project or not project.get("sections"):
        st.info("Aucun alignement disponible. Charge un projet contenant des sections alignées ci-dessus.")
        return
    labels = [s.get("label", f"Section {i+1}") for i,s in enumerate(project["sections"])]
    sec_label = st.selectbox("Section", labels)
    sec_idx = labels.index(sec_label)
    section = project["sections"][sec_idx]
    targets = list(section.get("target_segments", {}))
    if not targets:
        return

    tab_syn, tab_edit, tab_lex, tab_exp = st.tabs(["Vue synoptique", "Corriger les alignements", "Analyse lexicale", "Exports"])

    with tab_syn:
        chosen = st.multiselect("Colonnes affichées", targets, default=targets, key=f"syn_targets_{sec_idx}")
        if chosen:
            st.markdown(synoptic_table_html(section, chosen), unsafe_allow_html=True)
        else:
            st.info("Sélectionne au moins une traduction.")

    with tab_edit:
        _correction_editor(project, section, sec_idx, targets)

    with tab_lex:
        _lexical_tab(section, sec_idx, targets)

    with tab_exp:
        chosen = st.multiselect("Cibles à exporter", targets, default=targets, key=f"export_targets_{sec_idx}")
        e1,e2,e3 = st.columns(3)
        e1.download_button("CSV synoptique", export_csv(project, sec_idx, chosen), file_name=f"{_sanitize_id(project.get('name','projet'))}_section_{sec_idx+1}.csv", mime="text/csv", use_container_width=True)
        e2.download_button("HTML synoptique", export_html(project, sec_idx, chosen), file_name=f"{_sanitize_id(project.get('name','projet'))}_section_{sec_idx+1}.html", mime="text/html", use_container_width=True)
        e3.download_button("Projet complet JSON", dumps_project(project), file_name=f"{_sanitize_id(project.get('name','projet'))}.pyodysseus.json", mime="application/json", use_container_width=True)
        if section.get("edit_log"):
            with st.expander("Historique des corrections"):
                st.dataframe(pd.DataFrame(section["edit_log"]), hide_index=True, use_container_width=True)


with st.sidebar:
    st.title("🧭 pyOdysseus")
    st.caption("Alignement multiversions · Bertalign + SPhilBERTa→LaBSE")
    page = st.radio("Navigation", ["Projet", "Alignement", "Relecture & analyse"])
    st.divider()
    if st.session_state.project:
        p = st.session_state.project
        st.markdown(f"**{p.get('name','Projet')}**")
        st.caption(f"Pivot : {p['pivot']['id']}")
        st.caption(f"{len(p['targets'])} cible(s) · {len(p.get('sections',[]))} section(s)")
    else:
        st.caption("Aucun projet ouvert")

if page == "Projet":
    page_project()
elif page == "Alignement":
    page_alignment()
else:
    page_review()
