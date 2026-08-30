#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recheck QA memory clues with an OpenAI-compatible LLM.

For each QA, this script:
1. Loads the current memory clue list.
2. Locally removes duplicate text clues when the same turn also has a voice clue
   such as D03:00 + D03-001.wav.
3. Builds compact evidence summaries only for text clues:
   - text clue Dxx:xx -> same turn user/assistant text.
   Image/audio clues are preserved automatically and are not sent to the LLM.
4. Calls an LLM to keep only text clues that are causally useful for answering the QA.
5. Writes rechecked QA files and a detailed report without overwriting sources by default.

Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.common.io import load_json_or_jsonl as load_records, write_json as write_json_file
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.common.paths import resolve_path
from scripts.qa.config import qa_path

FORMATTED_DATA_PATH = qa_path("qa_formatted_data_000_019.json")
DEFAULT_OUTPUT_DIR = qa_path("rechecked_memory_clues")

# Credentials and optional endpoint are supplied only at runtime.
DEFAULT_LLM_MODEL = env_value("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")
DEFAULT_LLM_BASE_URL = env_value("CUE_MEM_LLM_BASE_URL")
DEFAULT_LLM_API_KEY = env_value("CUE_MEM_LLM_API_KEY")

QA_FILES: Dict[str, Path] = {
    "pref_img": qa_path("qa_pref_image_mcq.json"),
    "rec_img": qa_path("qa_rec_image_mcq.json"),
    "entity_img": qa_path("qa_entity_image_mcq.json"),
    "pref_text": qa_path("qa_preference_mcq.json"),
    "rec_text": qa_path("qa_recommendation_mcq.json"),
    "entity_text": qa_path("qa_entity_mcq.json"),
}


ROUND_RE = re.compile(r"^D(\d{2}):(\d{2})$")
AUDIO_ID_RE = re.compile(r"^D(\d{2})-(\d{3})\.(?:wav|mp3|m4a|flac|ogg|aac)$", re.IGNORECASE)
IMAGE_RE = re.compile(r"\.(?:png|jpg|jpeg|webp)$", re.IGNORECASE)
AUDIO_RE = re.compile(r"\.(?:wav|mp3|m4a|flac|ogg|aac)$", re.IGNORECASE)


def load_json(path: Path) -> Any:
    return load_records(path)


def write_json(path: Path, data: Any) -> None:
    write_json_file(path, data)


def memory_clues(qa: Mapping[str, Any]) -> List[str]:
    raw = qa.get("memory clue")
    if raw is None:
        raw = qa.get("clue")
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def set_memory_clues(qa: Dict[str, Any], clues: Sequence[str]) -> None:
    if "memory clue" in qa or "clue" not in qa:
        qa["memory clue"] = list(clues)
    else:
        qa["clue"] = list(clues)


def answer_letter(qa: Mapping[str, Any]) -> str:
    return str(qa.get("A") or qa.get("answer") or qa.get("original_answer") or "").strip()


def answer_text(qa: Mapping[str, Any]) -> str:
    letter = answer_letter(qa)
    options = qa.get("options")
    if isinstance(options, Mapping) and letter in options:
        return str(options[letter])
    descriptions = qa.get("question_image_descriptions")
    if isinstance(descriptions, Mapping) and letter in descriptions:
        return str(descriptions[letter])
    return ""


def clue_kind(clue: str) -> str:
    if ROUND_RE.match(clue):
        return "text"
    if IMAGE_RE.search(clue):
        return "image"
    if AUDIO_RE.search(clue):
        return "audio"
    return "other"


def audio_id_to_round(audio_id: str) -> Optional[str]:
    match = AUDIO_ID_RE.match(str(audio_id).strip())
    if not match:
        return None
    session_id, one_based = match.groups()
    return f"D{session_id}:{int(one_based) - 1:02d}"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_text(text: str) -> str:
    text = strip_code_fence(text)
    if not text:
        return text
    start_positions = [idx for idx in [text.find("{"), text.find("[")] if idx >= 0]
    if not start_positions:
        return text
    start = min(start_positions)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    if end >= start:
        return text[start : end + 1]
    return text


def parse_llm_json(text: str) -> Any:
    return json.loads(extract_json_text(text))


@dataclass
class Evidence:
    clue: str
    modality: str
    content: str
    round_id: str = ""
    task_id: str = ""


class EvidenceIndex:
    def __init__(self, formatted_path: Path) -> None:
        self.rounds: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self.media_to_round: Dict[Tuple[int, str], str] = {}
        self.media_to_event: Dict[Tuple[int, str], Dict[str, Any]] = {}
        self._load(formatted_path)

    def _load(self, formatted_path: Path) -> None:
        profiles = load_json(formatted_path)
        if not isinstance(profiles, list):
            raise ValueError(f"formatted data must be a list: {formatted_path}")
        for fallback_pid, profile in enumerate(profiles):
            if not isinstance(profile, Mapping):
                continue
            p_id = profile.get("p_id", fallback_pid)
            if not isinstance(p_id, int):
                continue
            for event in profile.get("events", []) or []:
                if not isinstance(event, Mapping):
                    continue
                session_id = str(event.get("session_id") or "")
                task_id = str(event.get("task_id") or "")
                group_id = event.get("group_id", "")
                image_desc = str(event.get("user_shared_image_description") or "none")
                scene = str(event.get("scene_description") or "")
                dialog_user_turns = [
                    turn for turn in (event.get("dialog") or [])
                    if isinstance(turn, Mapping) and turn.get("role") == "user"
                ]
                for user_turn_idx, turn in enumerate(event.get("dialog_list", []) or []):
                    if not isinstance(turn, Mapping):
                        continue
                    round_id = str(turn.get("round") or "").strip()
                    if not round_id:
                        continue
                    dialog_turn = dialog_user_turns[user_turn_idx] if user_turn_idx < len(dialog_user_turns) else {}
                    turn_bg_audio = str(dialog_turn.get("background_audio") or "").strip()
                    turn_record = {
                        "p_id": p_id,
                        "session_id": session_id,
                        "task_id": task_id,
                        "group_id": group_id,
                        "round": round_id,
                        "scene_description": scene,
                        "user_shared_image_description": image_desc,
                        "background_audio": turn_bg_audio,
                        "user": str(turn.get("user") or ""),
                        "assistant": str(turn.get("assistant") or ""),
                    }
                    self.rounds[(p_id, round_id)] = turn_record
                    for key, value in turn.items():
                        if not isinstance(key, str):
                            continue
                        key = key.strip()
                        if IMAGE_RE.search(key) or AUDIO_RE.search(key):
                            self.media_to_round[(p_id, key)] = round_id
                            self.media_to_event[(p_id, key)] = turn_record

    def duplicate_audio_text_rounds(self, p_id: int, clues: Sequence[str]) -> List[str]:
        clue_set = set(clues)
        audio_rounds = set()
        for clue in clues:
            if clue_kind(clue) != "audio":
                continue
            round_id = self.media_to_round.get((p_id, clue)) or audio_id_to_round(clue)
            if round_id:
                audio_rounds.add(round_id)
        return [clue for clue in clues if clue_kind(clue) == "text" and clue in audio_rounds and clue in clue_set]

    def remove_duplicate_audio_text(self, p_id: int, clues: Sequence[str]) -> Tuple[List[str], List[str]]:
        duplicates = set(self.duplicate_audio_text_rounds(p_id, clues))
        cleaned = [clue for clue in clues if clue not in duplicates]
        return cleaned, [clue for clue in clues if clue in duplicates]

    def evidence_for_clue(self, p_id: int, clue: str) -> Evidence:
        kind = clue_kind(clue)
        if kind == "text":
            turn = self.rounds.get((p_id, clue), {})
            content = (
                f"User text: {turn.get('user', '')}\n"
                f"Assistant text: {turn.get('assistant', '')}"
            ).strip()
            return Evidence(clue=clue, modality="text", content=content, round_id=clue, task_id=str(turn.get("task_id", "")))

        if kind == "image":
            turn = self.media_to_event.get((p_id, clue), {})
            image_desc = str(turn.get("user_shared_image_description") or "")
            if not image_desc or image_desc.lower() == "none":
                image_desc = "(no image description found)"
            round_id = self.media_to_round.get((p_id, clue), "")
            return Evidence(
                clue=clue,
                modality="image",
                content=f"user_shared_image_description: {image_desc}",
                round_id=round_id,
                task_id=str(turn.get("task_id", "")),
            )

        if kind == "audio":
            turn = self.media_to_event.get((p_id, clue), {})
            round_id = self.media_to_round.get((p_id, clue), "") or audio_id_to_round(clue) or ""
            if not turn and round_id:
                turn = self.rounds.get((p_id, round_id), {})
            content = (
                f"Same-turn user text: {turn.get('user', '')}\n"
                f"Same-turn background_audio field: {turn.get('background_audio', '')}"
            ).strip()
            return Evidence(clue=clue, modality="audio", content=content, round_id=round_id, task_id=str(turn.get("task_id", "")))

        return Evidence(clue=clue, modality="other", content="(unrecognized clue format)")


def build_prompt(qa: Mapping[str, Any], qa_file_key: str, evidences: Sequence[Evidence]) -> str:
    options = qa.get("options")
    if not isinstance(options, Mapping):
        options = qa.get("question_image_descriptions")
    prompt_payload = {
        "qa_id": qa.get("qa_id", ""),
        "qa_file_type": qa_file_key,
        "p_id": qa.get("p_id", ""),
        "question": qa.get("Q") or qa.get("question") or "",
        "options_or_image_option_descriptions": options if isinstance(options, Mapping) else {},
        "correct_answer_letter": answer_letter(qa),
        "correct_answer_text": answer_text(qa),
        "qa_metadata": {
            key: qa.get(key)
            for key in [
                "category",
                "subcategory",
                "preference",
                "expression_type",
                "evidence_sources",
                "entity_type",
                "entity_name",
                "entity_relation",
                "dimension",
                "adversarial_type",
                "trap_reason",
                "reasonable_answer",
            ]
            if key in qa
        },
        "candidate_memory_clues": [
            {
                "clue": ev.clue,
                "modality": ev.modality,
                "round_id": ev.round_id,
                "task_id": ev.task_id,
                "evidence_content": ev.content,
            }
            for ev in evidences
        ],
    }
    return (
        "You are rechecking memory clues for a multimodal memory benchmark QA.\n"
        "Decide which candidate TEXT clues are truly useful for answering the QA correctly.\n"
        "Image and audio clues are handled separately and are always preserved; do not discuss or infer from missing image/audio clues here.\n\n"
        "Keep a clue only if it has strong causal value for the correct answer: it directly supports the preference, entity fact, or recommendation basis.\n"
        "Drop clues that are merely same-session context, weakly related, generic, redundant without adding answer-relevant evidence, or about a different preference/entity.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\n"
        '  "keep_clues": ["exact clue id from candidates"],\n'
        '  "drop_clues": [{"clue": "exact clue id", "reason": "short reason"}],\n'
        '  "notes": "optional short note"\n'
        "}\n"
        "Every candidate clue must appear in exactly one of keep_clues or drop_clues. Do not invent clue IDs.\n\n"
        "INPUT:\n"
        f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
    )


def call_llm(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    timeout: int,
    reasoning_effort: Optional[str],
    max_retries: int,
    verbose: bool = False,
) -> Tuple[str, int, int]:
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a strict JSON-only evaluator."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "timeout": timeout,
            }
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**kwargs)
            message = message_content_to_text(response.choices[0].message.content)
            usage = getattr(response, "usage", None)
            prompt_tokens = usage_value(usage, "prompt_tokens")
            completion_tokens = usage_value(usage, "completion_tokens")
            if not message.strip():
                raise ValueError("empty LLM response")
            return message, prompt_tokens, completion_tokens
        except TypeError:
            # Some OpenAI-compatible endpoints reject reasoning_effort.
            if reasoning_effort and "reasoning_effort" in kwargs:
                if verbose:
                    print("reasoning_effort rejected; retrying without it")
                reasoning_effort = None
                continue
            raise
        except Exception as exc:  # pragma: no cover - network behavior
            last_error = exc
            wait = min(60.0, 1.5 * (2 ** (attempt - 1)))
            print(f"  API attempt {attempt}/{max_retries} failed: {exc}; wait {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


def normalize_decision(payload: Any, candidates: Sequence[str]) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    candidate_set = set(candidates)
    errors: List[str] = []
    if not isinstance(payload, Mapping):
        return [], [], ["LLM response is not an object"]
    keep_raw = payload.get("keep_clues", [])
    drop_raw = payload.get("drop_clues", [])
    keep = [str(item).strip() for item in keep_raw if str(item).strip()] if isinstance(keep_raw, list) else []
    drops: List[Dict[str, str]] = []
    if isinstance(drop_raw, list):
        for item in drop_raw:
            if isinstance(item, Mapping):
                clue = str(item.get("clue", "")).strip()
                reason = str(item.get("reason", "")).strip()
            else:
                clue = str(item).strip()
                reason = ""
            if clue:
                drops.append({"clue": clue, "reason": reason})
    else:
        errors.append("drop_clues is not a list")

    drop_ids = [item["clue"] for item in drops]
    for clue in keep + drop_ids:
        if clue not in candidate_set:
            errors.append(f"unknown clue returned: {clue}")
    duplicated = set(keep) & set(drop_ids)
    if duplicated:
        errors.append(f"clues appear in both keep and drop: {sorted(duplicated)}")
    missing = [clue for clue in candidates if clue not in set(keep) and clue not in set(drop_ids)]
    if missing:
        errors.append(f"missing decisions for clues: {missing}")
        for clue in missing:
            keep.append(clue)
    keep = [clue for clue in candidates if clue in set(keep)]
    drops = [item for item in drops if item["clue"] in candidate_set and item["clue"] not in set(keep)]
    return keep, drops, errors


@dataclass
class TaskResult:
    task_key: str
    qa_file_key: str
    qa_id: str
    p_id: Any
    keep_clues: List[str]
    drop_clues: List[Dict[str, str]]
    duplicate_text_removed: List[str]
    auto_kept_non_text: List[str]
    errors: List[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0


def process_one(
    qa: Mapping[str, Any],
    qa_file_key: str,
    evidence_index: EvidenceIndex,
    client: Any,
    args: argparse.Namespace,
) -> TaskResult:
    p_id = qa.get("p_id")
    if not isinstance(p_id, int):
        try:
            p_id = int(p_id)
        except Exception:
            p_id = -1
    qa_id = str(qa.get("qa_id") or "")
    task_key = f"{qa_file_key}::{p_id}::{qa_id}"
    original_clues = memory_clues(qa)
    cleaned_clues, duplicate_removed = evidence_index.remove_duplicate_audio_text(p_id, original_clues)
    text_clues = [clue for clue in cleaned_clues if clue_kind(clue) == "text"]
    auto_kept_non_text = [clue for clue in cleaned_clues if clue_kind(clue) != "text"]
    evidences = [evidence_index.evidence_for_clue(p_id, clue) for clue in text_clues]

    if not cleaned_clues:
        return TaskResult(task_key, qa_file_key, qa_id, p_id, [], [], duplicate_removed, [], [])

    if not text_clues:
        return TaskResult(task_key, qa_file_key, qa_id, p_id, cleaned_clues, [], duplicate_removed, auto_kept_non_text, [])

    prompt = build_prompt(qa, qa_file_key, evidences)
    if args.dry_run:
        print("\n" + "=" * 80)
        print(task_key)
        print(prompt[: args.dry_run_chars])
        return TaskResult(task_key, qa_file_key, qa_id, p_id, cleaned_clues, [], duplicate_removed, auto_kept_non_text, ["dry_run"])

    raw, pt, ct = call_llm(
        client=client,
        model=args.model,
        prompt=prompt,
        temperature=args.temperature,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )
    try:
        payload = parse_llm_json(raw)
        keep_text, drops, errors = normalize_decision(payload, text_clues)
    except Exception as exc:
        keep_text, drops, errors = text_clues, [], [f"parse/normalize failed: {exc}", raw[:500]]

    keep_text_set = set(keep_text)
    auto_keep_set = set(auto_kept_non_text)
    keep = [clue for clue in cleaned_clues if clue in keep_text_set or clue in auto_keep_set]
    return TaskResult(task_key, qa_file_key, qa_id, p_id, keep, drops, duplicate_removed, auto_kept_non_text, errors, pt, ct)


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        if isinstance(data, Mapping):
            checkpoint: Dict[str, Dict[str, Any]] = {}
            for k, v in data.items():
                if not isinstance(v, Mapping):
                    continue
                if "dry_run" in v.get("errors", []):
                    continue
                # Older checkpoints may contain LLM decisions for image/audio clues.
                # Reprocess them so the current text-only policy is applied.
                if "auto_kept_non_text" not in v:
                    continue
                checkpoint[str(k)] = dict(v)
            return checkpoint
    except Exception as exc:
        print(f"WARN: cannot load checkpoint {path}: {exc}")
    return {}


def save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    write_json(path, checkpoint)


def apply_results_to_qa_files(
    qa_data_by_key: Mapping[str, List[Dict[str, Any]]],
    results: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
) -> None:
    for qa_file_key, qa_items in qa_data_by_key.items():
        output_items = deepcopy(qa_items)
        for idx, qa in enumerate(output_items):
            p_id = qa.get("p_id")
            qa_id = str(qa.get("qa_id") or "")
            task_key = f"{qa_file_key}::{p_id}::{qa_id}"
            result = results.get(task_key)
            if not result:
                continue
            set_memory_clues(qa, result.get("keep_clues", memory_clues(qa)))
            qa["memory_clue_recheck"] = {
                "policy": "llm_recheck_text_only_keep_image_audio",
                "original_count": len(memory_clues(qa_items[idx])),
                "kept_count": len(result.get("keep_clues", [])),
                "dropped_count": len(result.get("drop_clues", [])),
                "duplicate_audio_text_removed": result.get("duplicate_text_removed", []),
                "auto_kept_non_text": result.get("auto_kept_non_text", []),
                "drop_clues": result.get("drop_clues", []),
                "errors": result.get("errors", []),
            }
        src_name = QA_FILES[qa_file_key].name
        write_json(output_dir / src_name, output_items)


def write_report_csv(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_key",
        "qa_file_key",
        "p_id",
        "qa_id",
        "kept_count",
        "dropped_count",
        "duplicate_text_removed_count",
        "auto_kept_non_text_count",
        "drop_clues",
        "duplicate_text_removed",
        "auto_kept_non_text",
        "errors",
        "prompt_tokens",
        "completion_tokens",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for task_key, result in sorted(results.items()):
            writer.writerow({
                "task_key": task_key,
                "qa_file_key": result.get("qa_file_key", ""),
                "p_id": result.get("p_id", ""),
                "qa_id": result.get("qa_id", ""),
                "kept_count": len(result.get("keep_clues", [])),
                "dropped_count": len(result.get("drop_clues", [])),
                "duplicate_text_removed_count": len(result.get("duplicate_text_removed", [])),
                "auto_kept_non_text_count": len(result.get("auto_kept_non_text", [])),
                "drop_clues": json.dumps(result.get("drop_clues", []), ensure_ascii=False),
                "duplicate_text_removed": json.dumps(result.get("duplicate_text_removed", []), ensure_ascii=False),
                "auto_kept_non_text": json.dumps(result.get("auto_kept_non_text", []), ensure_ascii=False),
                "errors": json.dumps(result.get("errors", []), ensure_ascii=False),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            })


def parse_id_filter(raw_values: Optional[List[str]]) -> Optional[set[int]]:
    if not raw_values:
        return None
    out: set[int] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                out.add(int(part))
    return out


def parse_type_filter(raw_values: Optional[List[str]]) -> Optional[set[str]]:
    if not raw_values:
        return None
    out: set[str] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                out.add(part)
    return out


def result_to_dict(result: TaskResult) -> Dict[str, Any]:
    return {
        "task_key": result.task_key,
        "qa_file_key": result.qa_file_key,
        "qa_id": result.qa_id,
        "p_id": result.p_id,
        "keep_clues": result.keep_clues,
        "drop_clues": result.drop_clues,
        "duplicate_text_removed": result.duplicate_text_removed,
        "auto_kept_non_text": result.auto_kept_non_text,
        "errors": result.errors,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调用 LLM recheck QA memory clue，并输出清洗后的 QA 文件")
    parser.add_argument("--formatted_data", type=Path, default=FORMATTED_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", "--llm-model", dest="model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--url", "--base-url", "--base_url", dest="base_url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--api-key", "--api_key", dest="api_key", default=DEFAULT_LLM_API_KEY)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning_effort", default="high")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0, help="只处理前 N 条任务，0 表示全部")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--dry_run_chars", type=int, default=6000)
    parser.add_argument("--only_profile_ids", nargs="*", default=None)
    parser.add_argument("--only_types", nargs="*", default=None, help=f"可选：{','.join(QA_FILES)}")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.formatted_data = resolve_path(args.formatted_data)
    args.output_dir = resolve_path(args.output_dir)
    only_pids = parse_id_filter(args.only_profile_ids)
    only_types = parse_type_filter(args.only_types)
    if only_types:
        unknown = only_types - set(QA_FILES)
        if unknown:
            raise ValueError(f"unknown QA file types: {sorted(unknown)}")

    qa_data_by_key: Dict[str, List[Dict[str, Any]]] = {}
    tasks: List[Tuple[str, Dict[str, Any]]] = []
    for qa_file_key, path in QA_FILES.items():
        if only_types and qa_file_key not in only_types:
            continue
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"QA file top-level must be a list: {path}")
        qa_data_by_key[qa_file_key] = data
        for qa in data:
            if not isinstance(qa, dict):
                continue
            p_id = qa.get("p_id")
            if only_pids is not None and p_id not in only_pids:
                continue
            tasks.append((qa_file_key, qa))

    if args.shuffle:
        random.Random(args.seed).shuffle(tasks)
    if args.sample and args.sample > 0:
        tasks = tasks[: args.sample]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "memory_clue_recheck_checkpoint.json"
    report_path = args.output_dir / "memory_clue_recheck_report.csv"
    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    pending = []
    for qa_file_key, qa in tasks:
        task_key = f"{qa_file_key}::{qa.get('p_id')}::{qa.get('qa_id') or ''}"
        if args.resume and task_key in checkpoint:
            continue
        pending.append((qa_file_key, qa))

    print(f"Loaded QA tasks={len(tasks)}; pending={len(pending)}; output_dir={args.output_dir}")

    evidence_index = EvidenceIndex(args.formatted_data)
    if args.dry_run:
        for qa_file_key, qa in pending:
            process_one(qa, qa_file_key, evidence_index, None, args)
        print("\nDry run only: no checkpoint, report, or rechecked QA files were written.")
        return

    client = openai_client(
        api_key=args.api_key,
        base_url=args.base_url,
        api_key_env="CUE_MEM_LLM_API_KEY",
        base_url_env="CUE_MEM_LLM_BASE_URL",
    )

    total_pt = total_ct = 0
    if args.workers <= 1:
        for qa_file_key, qa in pending:
            result = process_one(qa, qa_file_key, evidence_index, client, args)
            checkpoint[result.task_key] = result_to_dict(result)
            total_pt += result.prompt_tokens
            total_ct += result.completion_tokens
            print(
                f"[{len(checkpoint)}/{len(tasks)}] {result.task_key} "
                f"keep={len(result.keep_clues)} drop={len(result.drop_clues)} "
                f"dup_text={len(result.duplicate_text_removed)} errors={len(result.errors)}"
            )
            save_checkpoint(checkpoint_path, checkpoint)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(process_one, qa, qa_file_key, evidence_index, client, args): (qa_file_key, qa)
                for qa_file_key, qa in pending
            }
            for future in as_completed(future_map):
                result = future.result()
                checkpoint[result.task_key] = result_to_dict(result)
                total_pt += result.prompt_tokens
                total_ct += result.completion_tokens
                print(
                    f"[{len(checkpoint)}/{len(tasks)}] {result.task_key} "
                    f"keep={len(result.keep_clues)} drop={len(result.drop_clues)} "
                    f"dup_text={len(result.duplicate_text_removed)} errors={len(result.errors)}"
                )
                save_checkpoint(checkpoint_path, checkpoint)

    apply_results_to_qa_files(qa_data_by_key, checkpoint, args.output_dir)
    write_report_csv(report_path, checkpoint)
    print(f"\nDone. results={len(checkpoint)} prompt_tokens={total_pt} completion_tokens={total_ct}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Report CSV: {report_path}")
    print(f"Rechecked QA files: {args.output_dir}")


if __name__ == "__main__":
    main()
