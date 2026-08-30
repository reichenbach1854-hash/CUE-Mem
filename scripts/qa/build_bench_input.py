"""Build benchmark input files for the base, category, and audio-caption runs.

This module is the single entry point for the three former input builders.
The default mode creates the regular ``base`` inputs.  The other modes can be
selected with ``--mode`` (or as a positional mode)::

    python -m scripts.qa.build_bench_input --mode base
    python -m scripts.qa.build_bench_input --mode category --categories brief,medium
    python -m scripts.qa.build_bench_input --mode audio-caption --models qwen_audio

The old mode-specific options are kept as compatible aliases.  All paths are
resolved through :mod:`scripts.qa.config`; no credential or network setting is
needed by this data-preparation script.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Make ``python scripts/qa/build_bench_input.py`` work from the repository
# root as well as the recommended ``python -m scripts.qa.build_bench_input``.
if __package__ in {None, ""}:
    _project_root = Path(__file__).resolve().parents[2]
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from scripts.common.io import load_json_or_jsonl, load_record_list
from scripts.qa.config import (
    BENCHMARK_DATA_ROOT,
    BENCHMARK_ROOT,
    BENCHMARK_RUN_ROOT,
    profile_path,
    qa_path,
)

# ---------------------------------------------------------------------------
# Paths and mode-specific source files
# ---------------------------------------------------------------------------

COMMON_QA_PATHS: dict[str, Path] = {
    "pref_img": qa_path("qa_pref_image_mcq.json"),
    "rec_img": qa_path("qa_rec_image_mcq.json"),
    "entity_img": qa_path("qa_entity_image_mcq.json"),
    "pref_text": qa_path("qa_preference_mcq.json"),
    "rec_text": qa_path("qa_recommendation_mcq.json"),
    "entity_text": qa_path("qa_entity_mcq.json"),
    "refusal_text": qa_path("qa_refusal_mcq.json"),
    "adversarial_text": qa_path("qa_adversarial_llm_mcq.json"),
}

AUDIO_QA_PATHS: dict[str, Path] = {
    "pref_img": qa_path("qa_pref_image_mcq_000_002.json"),
    "rec_img": qa_path("qa_rec_image_mcq_000_002.json"),
    "entity_img": qa_path("qa_entity_image_mcq_000_002.json"),
    "pref_text": qa_path("qa_preference_mcq_000_002.json"),
    "rec_text": qa_path("qa_recommendation_mcq_000_002.json"),
    "entity_text": qa_path("qa_entity_mcq_000_002.json"),
    "refusal_text": qa_path("qa_refusal_mcq_000_002.json"),
    "adversarial_text": qa_path("qa_adversarial_mcq_000_002.json"),
    "audio_context": qa_path("qa_audio_context_mcq_000_002.json"),
}

IMAGE_QA_KEYS = ("pref_img", "rec_img", "entity_img")
IMAGE_CAPTION_KEYS = {
    "pref_img": "pref_img_captions.json",
    "rec_img": "rec_img_captions.json",
    "entity_img": "entity_img_captions.json",
}
COMMON_IMAGE_CAPTION_PATHS = {
    key: qa_path(filename) for key, filename in IMAGE_CAPTION_KEYS.items()
}

BASE_PROFILE_PATH = profile_path("profiles_with_anchors.jsonl")
BASE_HISTORY_FILE = BENCHMARK_RUN_ROOT / "history_000_019.json"
BASE_FORMATTED_DATA = qa_path("qa_formatted_data_000_019.json")
BASE_OUTPUT_DIR = BENCHMARK_DATA_ROOT / "dialog" / "base"
ITEM_EXPLICITNESS_REPORT = qa_path("item_entity_explicitness_report.csv")

CATEGORY_CAPTION_BASE = qa_path("qwen3.5-9b")
CATEGORY_HISTORY_BASE = BENCHMARK_RUN_ROOT
CATEGORY_OUTPUT_BASE = BENCHMARK_DATA_ROOT / "dialog"
CATEGORY_FORMATTED_DATA = qa_path("qa_formatted_data_000_019.json")
CATEGORY_PROFILE_PATH = profile_path("profiles_with_anchors.jsonl")
CROSS_COMPARE_DIR = (
    BENCHMARK_ROOT / "result_question_only" / "base" / "cross_compare"
)
ALL_CATEGORIES = ("brief", "medium", "detailed")

AUDIO_PROFILE_PATH = profile_path("profiles_000_002_with_anchors.jsonl")
AUDIO_HISTORY_FILE = BENCHMARK_RUN_ROOT / "detailed_history_dialogue.json"
AUDIO_FORMATTED_DATA = qa_path("qa_formatted_data_000_002.json")
AUDIO_MEDIA_CAPTIONS_FILE = qa_path(
    "qa_formatted_data_000_002_with_media_captions.json"
)
AUDIO_OUTPUT_BASE = BENCHMARK_DATA_ROOT / "dialog" / "audio_caption"
AUDIO_DEFAULT_NUM_PROFILES = 3

AUDIO_MODELS: dict[str, tuple[Path, str]] = {
    "qwen3_asr_1.7b": (
        qa_path(
            "qwen3_asr_1.7b/"
            "qa_formatted_data_000_002_with_audio_captions_qwen3_asr.json"
        ),
        "audio_caption_qwen3_asr",
    ),
    "qwen_audio": (
        qa_path(
            "qwen_audio/"
            "qa_formatted_data_000_002_with_audio_captions_qwen_audio.json"
        ),
        "audio_caption_qwen_audio",
    ),
    "qwen2_audio_7b": (
        qa_path(
            "qwen2_audio_7b/"
            "qa_formatted_data_000_002_with_audio_captions_qwen2_7b.json"
        ),
        "audio_caption_qwen2_7b",
    ),
    "moss_audio_8b": (
        qa_path(
            "moss_audio_8b/"
            "qa_formatted_data_000_002_with_audio_captions_moss_audio_8b.json"
        ),
        "audio_caption_moss_audio_8b",
    ),
}

MODE_ALIASES = {
    "base": "base",
    "category": "category",
    "categories": "category",
    "audio": "audio-caption",
    "audio_caption": "audio-caption",
    "audio-caption": "audio-caption",
}


# ---------------------------------------------------------------------------
# Generic loading and output helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    """Load a JSON or JSONL file using the repository's shared loader."""

    return load_json_or_jsonl(path)


def load_profiles(path: Path) -> list[dict]:
    return load_record_list(path)


def as_record_list(value: Any, source: Path | str) -> list[dict]:
    """Normalize a JSON object/list to a list of object records."""

    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise TypeError(f"{source} must contain a JSON object, list, or JSONL records")
    records = []
    for item in value:
        if not isinstance(item, dict):
            raise TypeError(f"{source} contains a non-object record")
        records.append(item)
    return records


def load_qa_sources(paths: dict[str, Path]) -> dict[str, list[dict]]:
    """Load source QA files, treating absent optional files as empty."""

    return {
        key: load_record_list(path) if path.exists() else []
        for key, path in paths.items()
    }


def split_sessions_by_pid(
    all_sessions: list, session_counts: list[int]
) -> list[list]:
    """Split the flat history list into consecutive per-profile groups."""

    groups = []
    offset = 0
    for count in session_counts:
        groups.append(all_sessions[offset : offset + count])
        offset += count
    return groups


def session_counts_by_profile(
    formatted_profiles: list[dict],
    num_profiles: int,
    *,
    unique_session_ids: bool = False,
) -> list[int]:
    counts = []
    for profile in formatted_profiles[:num_profiles]:
        events = profile.get("events", []) or []
        if unique_session_ids:
            counts.append(
                len(
                    {
                        event.get("session_id", "")
                        for event in events
                        if event.get("session_id")
                    }
                )
            )
        else:
            counts.append(len(events))
    return counts


def resolve_num_profiles(
    profiles: list[dict],
    formatted_profiles: list[dict],
    requested: int | None,
    *,
    default: int | None = None,
) -> int:
    count = requested if requested is not None else (
        default if default is not None else len(formatted_profiles)
    )
    if count < 1:
        raise ValueError("--num-profiles must be a positive integer")
    return min(count, len(profiles), len(formatted_profiles))


def build_character_profile(profile: dict, p_id: int) -> dict[str, str]:
    basic = profile.get("Basic", {})
    return {
        "name": basic.get("name", f"user_{p_id}"),
        "description": f"Multi-modal memory benchmark user profile (p_id={p_id})",
    }


def build_profile_output(
    profile: dict,
    p_id: int,
    sessions: list,
    qas: list[dict],
) -> dict:
    return {
        "character_profile": build_character_profile(profile, p_id),
        "multi_session_dialogues": sessions,
        "human-annotated QAs": qas,
    }


def write_profile_output(
    output: dict,
    output_path: Path,
    *,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"      (dry-run) would write: {output_path}")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      Output -> {output_path}")


def group_qa_by_pid(qas: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for qa in qas:
        grouped.setdefault(qa.get("p_id", 0), []).append(qa)
    return grouped


def qa_counts(qas: list[dict]) -> Counter:
    return Counter(str(qa.get("point", "")) for qa in qas)


def print_qa_counts(qas: list[dict], *, indent: str = "  ") -> None:
    for point, count in sorted(qa_counts(qas).items()):
        print(f"{indent}{point:20s}: {count}")


def print_source_counts(sources: dict[str, list[dict]], *, indent: str = "  ") -> None:
    for key, records in sources.items():
        print(f"{indent}{key:20s}: {len(records)}")


def parse_csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


# ---------------------------------------------------------------------------
# MCQ conversion and entity explicitness
# ---------------------------------------------------------------------------

MCQ_OPTION_LABELS = ("A", "B", "C", "D")


def format_mcq_question(q: str, options: dict | None) -> str:
    """Combine a question and its options into the benchmark text format."""

    lines = [str(q)]
    options = options or {}
    for label in MCQ_OPTION_LABELS:
        if label in options:
            lines.append(f"{label}. {options[label]}")
    lines.append("请在 A/B/C/D 中选择最符合的选项。")
    return "\n".join(lines)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _preference_sources(pref: dict) -> list[str]:
    return [
        str(source)
        for source in _as_list(pref.get("evidence_sources", pref.get("sources", [])))
    ]


def _preference_anchors(pref: dict) -> list[str]:
    anchors = []
    for key in ("entity_anchors", "entity_anchor"):
        for anchor in _as_list(pref.get(key)):
            text = str(anchor).strip()
            if text:
                anchors.append(text)
    return list(dict.fromkeys(anchors))


def _category_id(category: str, index: int) -> str:
    return f"{category}-{index}"


MANUAL_ITEM_ANCHOR_ALIASES: dict[tuple[int, str], str] = {
    (2, "泛黄破损边缘的集点卡"): "泛黄破损边缘的李记面馆纸质集点卡",
    (2, "贴在储物柜上的路线图"): "贴在储物柜上的手绘云吞面馆路线图",
    (3, "黑色把手磨亮的电话机"): "黑色把手磨亮的老式桌面电话机",
}


def build_item_explicitness_index(
    profiles: list[dict],
) -> dict[tuple[int, str], dict]:
    """Map each item anchor to its explicit/implicit source type."""

    index: dict[tuple[int, str], dict] = {}
    for p_id, profile in enumerate(profiles):
        for category, preferences in profile.items():
            if category == "Basic" or not isinstance(preferences, list):
                continue
            for item_index, preference in enumerate(preferences):
                if not isinstance(preference, dict):
                    continue
                anchors = _preference_anchors(preference)
                if not anchors:
                    continue
                expression_type = str(
                    preference.get("expression_type", "")
                ).strip().lower()
                if expression_type not in {"explicit", "implicit"}:
                    expression_type = "unknown"
                reference = {
                    "category": _category_id(category, item_index),
                    "subcategory": preference.get("subcategory", ""),
                    "expression_type": expression_type,
                    "sources": _preference_sources(preference),
                }
                for anchor in anchors:
                    info = index.setdefault(
                        (p_id, anchor),
                        {
                            "p_id": p_id,
                            "anchor": anchor,
                            "explicit_count": 0,
                            "implicit_count": 0,
                            "unknown_count": 0,
                            "source_refs": [],
                        },
                    )
                    info[f"{expression_type}_count"] += 1
                    info["source_refs"].append(reference)

    for info in index.values():
        has_explicit = info["explicit_count"] > 0
        has_implicit = info["implicit_count"] > 0
        if has_explicit and has_implicit:
            info["item_explicitness"] = "mixed"
        elif has_explicit:
            info["item_explicitness"] = "explicit"
        elif has_implicit:
            info["item_explicitness"] = "implicit"
        else:
            info["item_explicitness"] = "unknown"
    return index


def _entity_display_name(item: dict) -> str:
    for key in ("entity_name", "entity_description", "description"):
        value = str(item.get(key, "")).strip()
        if value:
            return value
    return ""


def _entity_anchor_lookup_name(item: dict) -> str:
    display_name = _entity_display_name(item)
    if item.get("entity_type") != "Items":
        return display_name
    try:
        p_id = int(item.get("p_id", 0))
    except (TypeError, ValueError):
        p_id = 0
    return MANUAL_ITEM_ANCHOR_ALIASES.get((p_id, display_name), display_name)


def entity_explicitness_for_item(
    item: dict,
    item_index: dict[tuple[int, str], dict],
) -> tuple[str, list[dict]]:
    entity_type = item.get("entity_type", "")
    if entity_type in {"Relationship", "Pets"}:
        return "explicit", []
    if entity_type != "Items":
        return "unknown", []
    try:
        p_id = int(item.get("p_id", 0))
    except (TypeError, ValueError):
        p_id = 0
    info = item_index.get((p_id, _entity_anchor_lookup_name(item)))
    if not info:
        return "unknown", []
    return info.get("item_explicitness", "unknown"), info.get("source_refs", [])


def save_item_explicitness_report(
    profiles: list[dict],
    item_index: dict[tuple[int, str], dict],
    output_path: Path,
    max_profiles: int | None = None,
) -> None:
    """Write the audit CSV used to inspect Items entity provenance."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (p_id, anchor), info in sorted(
        item_index.items(), key=lambda entry: (entry[0][0], entry[0][1])
    ):
        if max_profiles is not None and p_id >= max_profiles:
            continue
        profile_name = (
            profiles[p_id].get("Basic", {}).get("name", f"user_{p_id}")
            if p_id < len(profiles)
            else f"user_{p_id}"
        )
        references = info.get("source_refs", [])
        rows.append(
            {
                "p_id": p_id,
                "profile_name": profile_name,
                "anchor": anchor,
                "item_explicitness": info.get("item_explicitness", "unknown"),
                "explicit_occurrence_count": info.get("explicit_count", 0),
                "implicit_occurrence_count": info.get("implicit_count", 0),
                "unknown_occurrence_count": info.get("unknown_count", 0),
                "source_categories": "; ".join(
                    ref.get("category", "") for ref in references
                ),
                "source_subcategories": "; ".join(
                    str(ref.get("subcategory", "")) for ref in references
                ),
                "source_expression_types": "; ".join(
                    ref.get("expression_type", "") for ref in references
                ),
                "source_refs_json": json.dumps(references, ensure_ascii=False),
            }
        )

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "p_id",
                "profile_name",
                "anchor",
                "item_explicitness",
                "explicit_occurrence_count",
                "implicit_occurrence_count",
                "unknown_occurrence_count",
                "source_categories",
                "source_subcategories",
                "source_expression_types",
                "source_refs_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _first_session_id(item: dict) -> str:
    value = item.get("matched_session_ids")
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return str(value) if value else ""


def convert_preference(items: list[dict]) -> list[dict]:
    return [
        {
            "qa_id": item.get("qa_id", ""),
            "question": format_mcq_question(item["Q"], item.get("options")),
            "answer": item.get("A", ""),
            "point": "",
            "qa_type": item.get("expression_type", ""),
            "session_id": _first_session_id(item),
            "clue": item.get("memory clue", []),
            "p_id": item.get("p_id", 0),
            "category": item.get("category", ""),
            "subcategory": item.get(
                "main_target_subcategory", item.get("subcategory", "")
            ),
        }
        for item in items
    ]


def convert_refusal(items: list[dict]) -> list[dict]:
    return [
        {
            "qa_id": item.get("qa_id", ""),
            "question": format_mcq_question(item["Q"], item.get("options")),
            "answer": item.get("A", ""),
            "point": "",
            "qa_type": item.get("refusal_type", ""),
            "session_id": _first_session_id(item),
            "clue": item.get("memory clue", []),
            "p_id": item.get("p_id", 0),
            "category": item.get("category", ""),
            "rationale": item.get("rationale", ""),
        }
        for item in items
    ]


def convert_adversarial(items: list[dict]) -> list[dict]:
    return [
        {
            "qa_id": item.get("qa_id", ""),
            "question": format_mcq_question(item["Q"], item.get("options")),
            "answer": item.get("A", ""),
            "point": "",
            "qa_type": item.get("expression_type", ""),
            "session_id": _first_session_id(item),
            "clue": item.get("memory clue", []),
            "p_id": item.get("p_id", 0),
            "category": item.get("category", ""),
            "subcategory": item.get("subcategory", ""),
            "rationale": item.get("rationale", ""),
            "adversarial_answer": item.get("adversarial_answer", ""),
            "adversarial_type": item.get("adversarial_type", ""),
        }
        for item in items
    ]


def convert_recommendation(items: list[dict]) -> list[dict]:
    return [
        {
            "qa_id": item.get("qa_id", ""),
            "question": format_mcq_question(item["Q"], item.get("options")),
            "answer": item.get("answer", item.get("A", "")),
            "point": "",
            "qa_type": item.get("expression_type", ""),
            "session_id": _first_session_id(item),
            "clue": item.get("memory clue", []),
            "p_id": item.get("p_id", 0),
            "category": item.get("category", ""),
            "subcategory": item.get(
                "main_target_subcategory", item.get("subcategory", "")
            ),
        }
        for item in items
    ]


def convert_audio_context(items: list[dict]) -> list[dict]:
    return [
        {
            "qa_id": item.get("qa_id", ""),
            "question": format_mcq_question(item["Q"], item.get("options")),
            "answer": item.get("A", item.get("answer", "")),
            "point": "",
            "qa_type": item.get("question_type", ""),
            "session_id": _first_session_id(item),
            "clue": item.get("memory clue", []),
            "p_id": item.get("p_id", 0),
            "category": item.get("category", ""),
            "subcategory": item.get("subcategory", ""),
            "question_audio": item.get("question_audio", ""),
            "question_audio_description": item.get(
                "question_audio_description", ""
            ),
        }
        for item in items
    ]


def convert_entity(
    items: list[dict],
    item_index: dict[tuple[int, str], dict] | None = None,
) -> list[dict]:
    item_index = item_index or {}
    converted = []
    for item in items:
        entity_type = item.get("entity_type", "")
        entity_name = _entity_display_name(item)
        explicitness, source_refs = entity_explicitness_for_item(item, item_index)
        converted.append(
            {
                "qa_id": item.get("qa_id", ""),
                "question": format_mcq_question(item["Q"], item.get("options")),
                "answer": item.get("A", ""),
                "point": "",
                "qa_type": entity_type,
                "session_id": _first_session_id(item),
                "clue": item.get("memory clue", []),
                "p_id": item.get("p_id", 0),
                "entity_name": entity_name,
                "entity_anchor_lookup_name": _entity_anchor_lookup_name(item),
                "dimension": item.get("dimension", ""),
                "entity_explicitness": explicitness,
                "entity_source_refs": source_refs,
            }
        )
    return converted


def set_point(qa_list: list[dict], point_name: str) -> list[dict]:
    for qa in qa_list:
        qa["point"] = point_name
    return qa_list


def convert_qa_sources(
    sources: dict[str, list[dict]],
    *,
    item_index: dict[tuple[int, str], dict] | None = None,
    include_audio_context: bool = False,
) -> list[dict]:
    """Convert all source QA types to the common benchmark schema."""

    converted: list[dict] = []
    converters: list[tuple[str, str, Callable]] = [
        ("pref_img", "pref_img", convert_preference),
        ("rec_img", "rec_img", convert_recommendation),
        ("entity_img", "entity_img", convert_entity),
        ("pref_text", "pref_text", convert_preference),
        ("rec_text", "rec_text", convert_recommendation),
        ("entity_text", "entity_text", convert_entity),
        ("refusal_text", "refusal_text", convert_refusal),
        ("adversarial_text", "adversarial_text", convert_adversarial),
    ]
    if include_audio_context:
        converters.append(("audio_context", "audio_context", convert_audio_context))

    for source_key, point, converter in converters:
        records = sources.get(source_key, [])
        if converter is convert_entity:
            records_out = converter(records, item_index)
        else:
            records_out = converter(records)
        converted.extend(set_point(records_out, point))
    return converted


# ---------------------------------------------------------------------------
# Shared image option/caption injection
# ---------------------------------------------------------------------------

def _build_lookup(records: list[dict]) -> dict[str, dict]:
    return {record["qa_id"]: record for record in records if record.get("qa_id")}


def _build_caption_lookup(records: list[dict]) -> dict[str, Any]:
    return {
        record["qa_id"]: record.get("option_captions", {})
        for record in records
        if record.get("qa_id")
    }


def _load_if_exists(path: Path) -> list[dict]:
    return load_record_list(path) if path.exists() else []


def build_image_lookup_bundle(
    image_qa_paths: dict[str, Path],
    caption_paths: dict[str, Path],
) -> dict[str, dict]:
    """Load image QA/options once for use by any builder mode."""

    qa_lookups = {
        key: _build_lookup(_load_if_exists(image_qa_paths[key]))
        for key in IMAGE_QA_KEYS
    }
    caption_lookups = {
        key: _build_caption_lookup(_load_if_exists(caption_paths[key]))
        for key in IMAGE_QA_KEYS
    }
    return {"qa": qa_lookups, "captions": caption_lookups}


def image_lookup_counts(bundle: dict[str, dict]) -> dict[str, int]:
    counts = {
        f"{key}_qa": len(bundle["qa"][key]) for key in IMAGE_QA_KEYS
    }
    counts.update(
        {f"{key}_captions": len(bundle["captions"][key]) for key in IMAGE_QA_KEYS}
    )
    return counts


def inject_image_options(
    qa_list: list[dict],
    bundle: dict[str, dict],
) -> dict[str, int]:
    """Inject option images/descriptions and option captions into image QA."""

    qa_lookups = bundle["qa"]
    caption_lookups = bundle["captions"]
    point_to_key = {
        "pref_img": "pref_img",
        "rec_img": "rec_img",
        "entity_img": "entity_img",
    }
    injected = 0
    caption_injected = 0

    for qa in qa_list:
        qid = qa.get("qa_id", "")
        point = str(qa.get("point", "")).lower()
        lookup_key = point_to_key.get(point)
        if lookup_key is None and point.endswith("_text"):
            continue
        if lookup_key is None:
            lookup_key = next(
                (key for key in IMAGE_QA_KEYS if qid in qa_lookups[key]),
                None,
            )
        if lookup_key is None or qid not in qa_lookups[lookup_key]:
            continue

        source = qa_lookups[lookup_key][qid]
        if source.get("option_images"):
            qa["option_images"] = source["option_images"]
        if source.get("question_image_descriptions"):
            qa["question_image_descriptions"] = source[
                "question_image_descriptions"
            ]
        if qid in caption_lookups[lookup_key]:
            qa["option_captions"] = caption_lookups[lookup_key][qid]
            caption_injected += 1
        injected += 1

    return {"injected": injected, "caption_injected": caption_injected}


# ---------------------------------------------------------------------------
# Retention filtering for category mode
# ---------------------------------------------------------------------------

ADVERSARIAL_POINT = "adversarial_text"


def normalize_category(category: str) -> str:
    aliases = {
        "preference_same_category": "pref_img",
        "recommendation_same_category": "rec_img",
        "entity": "entity_img",
    }
    return aliases.get(category, category)


def profile_name_from_record(record: dict) -> str:
    profile_name = str(record.get("profile_name") or "").strip()
    if profile_name:
        return profile_name
    source_name = str(
        record.get("source_name") or record.get("source_file") or ""
    ).strip()
    if source_name.startswith("history_with_qa_p"):
        return source_name.split("_results", 1)[0].split(".json", 1)[0]
    compare_key = str(record.get("compare_key") or "")
    return compare_key.split("::", 1)[0] if "::" in compare_key else ""


def make_retention_key(
    profile_name: str,
    category: str,
    qa_id: str,
) -> tuple[str, str, str]:
    return profile_name, normalize_category(category or ""), qa_id or ""


def load_retained_keys(
    cross_compare_dir: Path,
    max_correct: int,
) -> tuple[set[tuple[str, str, str]], Counter]:
    retained: set[tuple[str, str, str]] = set()
    stats = Counter()
    for correct_count in range(max_correct + 1):
        path = cross_compare_dir / f"correct_{correct_count}.json"
        if not path.exists():
            print(f"  WARN: missing retention bucket: {path}")
            continue
        records = as_record_list(load_json(path), path)
        stats[f"correct_{correct_count}"] = len(records)
        for record in records:
            key = make_retention_key(
                profile_name_from_record(record),
                str(record.get("category") or record.get("point") or ""),
                str(record.get("qa_id") or ""),
            )
            if all(key):
                retained.add(key)
            else:
                stats["bad_retention_records"] += 1
    return retained, stats


def filter_qas_by_retention(
    qa_list: list[dict],
    retained_keys: set[tuple[str, str, str]],
) -> tuple[list[dict], Counter]:
    filtered = []
    stats = Counter()
    for qa in qa_list:
        p_id = qa.get("p_id", 0)
        profile_name = f"history_with_qa_p{p_id}"
        point = str(qa.get("point") or qa.get("category") or "")
        key = make_retention_key(profile_name, point, str(qa.get("qa_id") or ""))
        if point == ADVERSARIAL_POINT:
            filtered.append(qa)
            stats["kept_adversarial"] += 1
            stats[f"kept_point::{point}"] += 1
        elif key in retained_keys:
            filtered.append(qa)
            stats["kept_retention"] += 1
            stats[f"kept_point::{point}"] += 1
        else:
            stats["dropped"] += 1
            stats[f"dropped_point::{point}"] += 1
    stats["input_total"] = len(qa_list)
    stats["kept_total"] = len(filtered)
    return filtered, stats


# ---------------------------------------------------------------------------
# Base and category modes
# ---------------------------------------------------------------------------

def run_base(args: argparse.Namespace) -> None:
    print("Loading source files …")
    profiles = load_profiles(BASE_PROFILE_PATH)
    history = as_record_list(load_json(BASE_HISTORY_FILE), BASE_HISTORY_FILE)
    formatted_profiles = as_record_list(
        load_json(BASE_FORMATTED_DATA), BASE_FORMATTED_DATA
    )
    num_profiles = resolve_num_profiles(
        profiles, formatted_profiles, args.num_profiles
    )

    item_index = build_item_explicitness_index(profiles)
    if not args.dry_run:
        save_item_explicitness_report(
            profiles,
            item_index,
            args.item_explicitness_report,
            max_profiles=num_profiles,
        )

    sources = load_qa_sources(COMMON_QA_PATHS)
    print(f"  profiles           : {len(profiles)}")
    print(f"  formatted profiles : {len(formatted_profiles)}")
    print(f"  output profiles    : {num_profiles}")
    print(f"  history sessions   : {len(history)}")
    print_source_counts(sources)
    print(f"  item report        : {args.item_explicitness_report}")

    all_qa = convert_qa_sources(sources, item_index=item_index)
    print(f"\nTotal QA items after conversion: {len(all_qa)}")
    print_qa_counts(all_qa)

    image_bundle = build_image_lookup_bundle(
        {key: COMMON_QA_PATHS[key] for key in IMAGE_QA_KEYS},
        COMMON_IMAGE_CAPTION_PATHS,
    )
    print("\nInjecting image options & captions …")
    print(f"  loaded image sources: {image_lookup_counts(image_bundle)}")
    inject_stats = inject_image_options(all_qa, image_bundle)
    print(f"  injected option_images : {inject_stats['injected']}")
    print(f"  injected captions      : {inject_stats['caption_injected']}")

    session_counts = session_counts_by_profile(formatted_profiles, num_profiles)
    print(f"\nSession counts per profile: {session_counts} (sum={sum(session_counts)})")
    if sum(session_counts) != len(history):
        print(
            f"  WARN: history sessions ({len(history)}) != formatted event count "
            f"({sum(session_counts)}). Session splitting uses formatted counts."
        )
    session_groups = split_sessions_by_pid(history, session_counts)
    qa_by_pid = group_qa_by_pid(all_qa)
    output_dir = args.output_dir or BASE_OUTPUT_DIR

    for p_id in range(num_profiles):
        profile = profiles[p_id]
        sessions = session_groups[p_id] if p_id < len(session_groups) else []
        pid_qas = qa_by_pid.get(p_id, [])
        output_path = output_dir / f"history_with_qa_p{p_id}.json"
        print(f"\n[p_id={p_id}] {build_character_profile(profile, p_id)['name']}")
        print(f"  sessions : {len(sessions)}")
        print(
            f"  QA total : {len(pid_qas)} "
            f"({sum(bool(qa.get('option_images')) for qa in pid_qas)} with images, "
            f"{sum(bool(qa.get('option_captions')) for qa in pid_qas)} with captions)"
        )
        print_qa_counts(pid_qas, indent="    ")
        write_profile_output(
            build_profile_output(profile, p_id, sessions, pid_qas),
            output_path,
            dry_run=args.dry_run,
        )

    action = "would be written to" if args.dry_run else "written to"
    print(f"\nDone. {num_profiles} base files {action} {output_dir}")


def run_category(args: argparse.Namespace) -> None:
    categories = parse_csv_values(args.categories)
    invalid = [category for category in categories if category not in ALL_CATEGORIES]
    if invalid:
        raise ValueError(
            f"unknown category {invalid[0]!r}; valid categories: {list(ALL_CATEGORIES)}"
        )
    if args.max_correct < 0:
        raise ValueError("--max-correct must be non-negative")

    print("Loading shared source files …")
    profiles = load_profiles(CATEGORY_PROFILE_PATH)
    sources = load_qa_sources(COMMON_QA_PATHS)
    formatted_profiles = as_record_list(
        load_json(CATEGORY_FORMATTED_DATA), CATEGORY_FORMATTED_DATA
    )
    num_profiles = resolve_num_profiles(
        profiles, formatted_profiles, args.num_profiles
    )
    item_index = build_item_explicitness_index(profiles)
    retained_keys, retention_stats = load_retained_keys(
        args.cross_compare_dir, args.max_correct
    )
    print(f"  retention dir      : {args.cross_compare_dir}")
    print(f"  keep correct range : correct_0..correct_{args.max_correct}")
    print(f"  retained QA keys   : {len(retained_keys)}")
    for key, value in sorted(retention_stats.items()):
        print(f"    {key:20s}: {value}")
    print(f"  profiles           : {len(profiles)}")
    print(f"  output profiles    : {num_profiles}")
    print_source_counts(sources)

    base_qa = convert_qa_sources(sources, item_index=item_index)
    print(f"\nTotal QA items after conversion: {len(base_qa)}")
    print_qa_counts(base_qa)
    base_qa, filter_stats = filter_qas_by_retention(base_qa, retained_keys)
    print(f"\nQA items after retention filtering: {len(base_qa)}")
    print(f"  kept_retention     : {filter_stats['kept_retention']}")
    print(f"  kept_adversarial   : {filter_stats['kept_adversarial']}")
    print(f"  dropped            : {filter_stats['dropped']}")
    print_qa_counts(base_qa)

    session_counts = session_counts_by_profile(formatted_profiles, num_profiles)
    output_root = args.output_root or args.output_dir or CATEGORY_OUTPUT_BASE
    image_qa_paths = {key: COMMON_QA_PATHS[key] for key in IMAGE_QA_KEYS}

    for category in categories:
        print(f"\n{'=' * 60}\nCategory: {category}\n{'=' * 60}")
        history_file = CATEGORY_HISTORY_BASE / f"{category}_history_dialogue.json"
        if not history_file.exists():
            print(f"  WARN: {history_file} not found, skipping")
            continue
        history = as_record_list(load_json(history_file), history_file)
        print(f"  history sessions: {len(history)}")
        session_groups = split_sessions_by_pid(history, session_counts)
        all_qa = copy.deepcopy(base_qa)
        caption_paths = {
            key: CATEGORY_CAPTION_BASE / category / filename
            for key, filename in IMAGE_CAPTION_KEYS.items()
        }
        image_bundle = build_image_lookup_bundle(image_qa_paths, caption_paths)
        print(f"  loaded image sources: {image_lookup_counts(image_bundle)}")
        inject_stats = inject_image_options(all_qa, image_bundle)
        print(f"  injected option_images : {inject_stats['injected']}")
        print(f"  injected captions      : {inject_stats['caption_injected']}")

        qa_by_pid = group_qa_by_pid(all_qa)
        output_dir = output_root / category
        for p_id in range(num_profiles):
            profile = profiles[p_id]
            sessions = session_groups[p_id] if p_id < len(session_groups) else []
            pid_qas = qa_by_pid.get(p_id, [])
            output_path = output_dir / f"history_with_qa_p{p_id}.json"
            print(
                f"\n  [p_id={p_id}] "
                f"{build_character_profile(profile, p_id)['name']}"
            )
            print(f"    sessions : {len(sessions)}")
            print(
                f"    QA total : {len(pid_qas)} "
                f"({sum(bool(qa.get('option_images')) for qa in pid_qas)} with images, "
                f"{sum(bool(qa.get('option_captions')) for qa in pid_qas)} with captions)"
            )
            print_qa_counts(pid_qas, indent="      ")
            write_profile_output(
                build_profile_output(profile, p_id, sessions, pid_qas),
                output_path,
                dry_run=args.dry_run,
            )
    print("\nDone.")


# ---------------------------------------------------------------------------
# Audio-caption mode
# ---------------------------------------------------------------------------

def build_image_caption_map(
    media_data: list[dict],
) -> dict[tuple[int, str], dict[str, str]]:
    """Build ``(p_id, session_id) -> image_id -> caption`` mapping."""

    caption_map: dict[tuple[int, str], dict[str, str]] = {}
    for profile in media_data:
        p_id = profile.get("p_id", -1)
        for event in profile.get("events", []) or []:
            session_id = event.get("session_id", "")
            for turn in event.get("dialog_list", []) or []:
                for key, value in turn.items():
                    if not key.endswith(".png"):
                        continue
                    if not isinstance(value, str) or not value.strip():
                        continue
                    if value.endswith(".png") or "/" in value or "\\" in value:
                        continue
                    image_id = key[:-4]
                    caption_map.setdefault((p_id, session_id), {})[image_id] = value.strip()
    return caption_map


def build_audio_caption_map(
    audio_formatted_data: list[dict],
    caption_field: str,
) -> dict[tuple[int, str], dict[str, str]]:
    """Build a voice-id caption mapping from one audio model's data."""

    path_to_voice: dict[str, tuple[int, str, str]] = {}
    for profile in audio_formatted_data:
        p_id = profile.get("p_id", -1)
        for event in profile.get("events", []) or []:
            session_id = event.get("session_id", "")
            for turn in event.get("dialog_list", []) or []:
                for key, value in turn.items():
                    if key.endswith(".wav") and isinstance(value, str) and value.strip():
                        path_to_voice[value.strip()] = (p_id, session_id, key[:-4])

    path_to_caption: dict[str, str] = {}
    for profile in audio_formatted_data:
        for event in profile.get("events", []) or []:
            for turn in event.get("dialog", []) or []:
                audio_path = str(turn.get("audio_path") or "").strip()
                caption = str(turn.get(caption_field) or "").strip()
                if audio_path and caption:
                    path_to_caption[audio_path] = caption

    caption_map: dict[tuple[int, str], dict[str, str]] = {}
    for audio_path, (p_id, session_id, voice_id) in path_to_voice.items():
        caption = path_to_caption.get(audio_path)
        if caption:
            caption_map.setdefault((p_id, session_id), {})[voice_id] = caption
    return caption_map


def build_background_audio_map(
    audio_formatted_data: list[dict],
) -> dict[tuple[int, str], str]:
    background_map: dict[tuple[int, str], str] = {}
    for profile in audio_formatted_data:
        p_id = profile.get("p_id", -1)
        for event in profile.get("events", []) or []:
            session_id = event.get("session_id", "")
            background = str(event.get("background_audio_info") or "").strip()
            if session_id and background and background.lower() != "none":
                background_map[(p_id, session_id)] = background
    return background_map


def patch_history_audio_captions(
    history: list,
    voice_caption_map: dict[tuple[int, str], dict[str, str]],
    background_caption_map: dict[tuple[int, str], str],
    p_id: int,
) -> tuple[list, int, int]:
    patched = copy.deepcopy(history)
    replaced_voice = 0
    replaced_background = 0
    for session in patched:
        session_id = session.get("session_id", "")
        key = (p_id, session_id)
        session_captions = voice_caption_map.get(key, {})
        if key in background_caption_map:
            session["background_audio_info"] = background_caption_map[key]
            replaced_background += 1
        for turn in session.get("dialogues", []) or []:
            voice_ids = turn.get("voice_id", []) or []
            if not voice_ids:
                continue
            original = turn.get("voice_caption", []) or []
            new_captions = []
            changed = False
            for index, voice_id in enumerate(voice_ids):
                if voice_id in session_captions:
                    new_captions.append(session_captions[voice_id])
                    changed = True
                else:
                    new_captions.append(
                        original[index] if index < len(original) else ""
                    )
            if changed:
                turn["voice_caption"] = new_captions
                turn["user_voice_message_caption"] = (
                    new_captions[0] if new_captions else ""
                )
                replaced_voice += 1
    return patched, replaced_voice, replaced_background


def patch_history_image_captions(
    history: list,
    image_caption_map: dict[tuple[int, str], dict[str, str]],
    p_id: int,
) -> tuple[list, int]:
    patched = copy.deepcopy(history)
    replaced_image = 0
    for session in patched:
        session_id = session.get("session_id", "")
        session_captions = image_caption_map.get((p_id, session_id), {})
        for turn in session.get("dialogues", []) or []:
            image_ids = turn.get("image_id", []) or []
            if not image_ids:
                continue
            original = turn.get("image_caption", []) or []
            new_captions = []
            changed = False
            for index, image_id in enumerate(image_ids):
                if image_id in session_captions:
                    new_captions.append(session_captions[image_id])
                    changed = True
                else:
                    new_captions.append(
                        original[index] if index < len(original) else ""
                    )
            if changed:
                turn["image_caption"] = new_captions
                replaced_image += 1
    return patched, replaced_image


def run_audio_caption(args: argparse.Namespace) -> None:
    models = parse_csv_values(args.models)
    invalid = [model for model in models if model not in AUDIO_MODELS]
    if invalid:
        raise ValueError(
            f"unknown audio model {invalid[0]!r}; valid models: {list(AUDIO_MODELS)}"
        )

    print("Loading shared source files …")
    profiles = load_profiles(AUDIO_PROFILE_PATH)
    formatted_profiles = as_record_list(
        load_json(AUDIO_FORMATTED_DATA), AUDIO_FORMATTED_DATA
    )
    num_profiles = resolve_num_profiles(
        profiles,
        formatted_profiles,
        args.num_profiles,
        default=AUDIO_DEFAULT_NUM_PROFILES,
    )
    sources = load_qa_sources(AUDIO_QA_PATHS)
    print(f"  profiles           : {len(profiles)}")
    print(f"  formatted profiles : {len(formatted_profiles)}")
    print(f"  output profiles    : {num_profiles}")
    print_source_counts(sources)

    session_counts = session_counts_by_profile(
        formatted_profiles, num_profiles, unique_session_ids=True
    )
    print(f"  session counts     : {session_counts} (sum={sum(session_counts)})")

    image_caption_map: dict[tuple[int, str], dict[str, str]] = {}
    if AUDIO_MEDIA_CAPTIONS_FILE.exists():
        media_data = as_record_list(
            load_json(AUDIO_MEDIA_CAPTIONS_FILE), AUDIO_MEDIA_CAPTIONS_FILE
        )
        image_caption_map = build_image_caption_map(media_data)
        print(
            f"  image captions     : {sum(len(value) for value in image_caption_map.values())} "
            f"items across {len(image_caption_map)} (p_id,session_id) keys"
        )
    else:
        print(f"  WARN: {AUDIO_MEDIA_CAPTIONS_FILE} not found; image captions unchanged")

    if not AUDIO_HISTORY_FILE.exists():
        raise FileNotFoundError(AUDIO_HISTORY_FILE)
    history = as_record_list(load_json(AUDIO_HISTORY_FILE), AUDIO_HISTORY_FILE)
    print(f"  history sessions   : {len(history)}")
    session_groups = split_sessions_by_pid(history, session_counts)

    all_qa_base = convert_qa_sources(
        sources,
        include_audio_context=True,
    )
    image_bundle = build_image_lookup_bundle(
        {key: AUDIO_QA_PATHS[key] for key in IMAGE_QA_KEYS},
        COMMON_IMAGE_CAPTION_PATHS,
    )
    image_inject_stats = inject_image_options(all_qa_base, image_bundle)
    print(f"\nTotal QA items after conversion: {len(all_qa_base)}")
    print_qa_counts(all_qa_base)
    print(f"  loaded image sources: {image_lookup_counts(image_bundle)}")
    print(f"  injected image fields: {image_inject_stats}")

    output_root = args.output_root or args.output_dir or AUDIO_OUTPUT_BASE
    for model_name in models:
        audio_data_path, caption_field = AUDIO_MODELS[model_name]
        print(f"\n{'=' * 60}")
        print(f"Audio model: {model_name} (caption_field={caption_field})")
        print(f"  data: {audio_data_path}")
        print(f"{'=' * 60}")
        if not audio_data_path.exists():
            print(f"  WARN: audio caption file not found, skipping: {audio_data_path}")
            continue

        audio_data = as_record_list(load_json(audio_data_path), audio_data_path)
        voice_map = build_audio_caption_map(audio_data, caption_field)
        background_map = build_background_audio_map(audio_data)
        print(
            f"  voice captions loaded: {sum(len(v) for v in voice_map.values())} "
            f"items across {len(voice_map)} (p_id,session_id) keys"
        )
        print(f"  background captions : {len(background_map)} keys")

        all_qa = copy.deepcopy(all_qa_base)
        context_path = qa_path(model_name, "context_audio_captions.json")
        context_injected = 0
        if context_path.exists():
            context_records = load_record_list(context_path)
            context_by_id = {
                record["qa_id"]: record.get("question_audio_caption", "")
                for record in context_records
                if record.get("qa_id")
            }
            for qa in all_qa:
                if qa.get("point") != "audio_context":
                    continue
                caption = context_by_id.get(qa.get("qa_id", ""), "")
                if caption:
                    qa["question_audio_caption"] = caption
                    context_injected += 1
            print(f"  audio context captions: {context_injected} from {context_path}")
        else:
            print(f"  WARN: {context_path} not found; audio context captions unchanged")

        qa_by_pid = group_qa_by_pid(all_qa)
        output_dir = output_root / model_name
        total_voice = total_background = total_image = 0
        for p_id in range(num_profiles):
            raw_sessions = session_groups[p_id] if p_id < len(session_groups) else []
            image_sessions, image_count = patch_history_image_captions(
                raw_sessions, image_caption_map, p_id
            )
            patched_sessions, voice_count, background_count = patch_history_audio_captions(
                image_sessions, voice_map, background_map, p_id
            )
            total_voice += voice_count
            total_background += background_count
            total_image += image_count
            profile = profiles[p_id]
            pid_qas = qa_by_pid.get(p_id, [])
            output_path = output_dir / f"history_with_qa_p{p_id}.json"
            print(
                f"\n    [p_id={p_id}] "
                f"{build_character_profile(profile, p_id)['name']}"
            )
            print(f"      sessions         : {len(patched_sessions)}")
            print(f"      voice turns patch : {voice_count}")
            print(f"      image turns patch : {image_count}")
            print(f"      background patch  : {background_count}")
            print(f"      QA total          : {len(pid_qas)}")
            print_qa_counts(pid_qas, indent="        ")
            write_profile_output(
                build_profile_output(profile, p_id, patched_sessions, pid_qas),
                output_path,
                dry_run=args.dry_run,
            )
        print(
            f"  Total patches: voice={total_voice}, "
            f"background={total_background}, image={total_image}"
        )
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build benchmark inputs. Modes: base, category, audio-caption "
            "(default: base)."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Optional positional mode: base, category, or audio-caption.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(set(MODE_ALIASES.values())),
        default=None,
        help="Builder mode (default: base).",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        dest="dry_run",
        action="store_true",
        help="Print planned outputs without writing generated files.",
    )
    parser.add_argument(
        "--num-profiles",
        "--num_profiles",
        dest="num_profiles",
        type=int,
        default=None,
        help="Limit the number of profile outputs.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Override the exact base output directory, or the root in other modes.",
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        type=Path,
        default=None,
        help="Override the category/audio output root directory.",
    )
    parser.add_argument(
        "--item-explicitness-report",
        "--item_explicitness_report",
        dest="item_explicitness_report",
        type=Path,
        default=ITEM_EXPLICITNESS_REPORT,
        help="Base-mode Items provenance CSV output path.",
    )
    parser.add_argument(
        "--categories",
        "--category",
        dest="categories",
        default=",".join(ALL_CATEGORIES),
        help="Category-mode granularities, comma-separated (default: all).",
    )
    parser.add_argument(
        "--cross-compare-dir",
        "--cross_compare_dir",
        dest="cross_compare_dir",
        type=Path,
        default=CROSS_COMPARE_DIR,
        help="Directory containing category-mode correct_*.json retention buckets.",
    )
    parser.add_argument(
        "--max-correct",
        "--max_correct",
        dest="max_correct",
        type=int,
        default=5,
        help="Keep correct_0..correct_N in category mode (default: 5).",
    )
    parser.add_argument(
        "--models",
        "--audio-models",
        "--audio_models",
        dest="models",
        default=",".join(AUDIO_MODELS),
        help="Audio-caption models, comma-separated (default: all).",
    )
    return parser


def normalize_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or "base").strip().lower()
    try:
        return MODE_ALIASES[mode]
    except KeyError as error:
        valid = ", ".join(sorted(MODE_ALIASES))
        raise ValueError(
            f"unknown mode {raw_mode!r}; valid modes/aliases: {valid}"
        ) from error


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command and args.mode and normalize_mode(args.command) != args.mode:
        parser.error("positional mode and --mode disagree")
    try:
        mode = normalize_mode(args.mode or args.command)
        if mode == "base":
            run_base(args)
        elif mode == "category":
            run_category(args)
        else:
            run_audio_caption(args)
    except (FileNotFoundError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
