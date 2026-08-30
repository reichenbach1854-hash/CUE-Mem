#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check and repair dialogue implicit-preference leakage with an LLM.

This script reads qa/qa_formatted_data_000_019.json by default, validates each
event dialogue against only the implicit preferences used by that event, and
repairs leaking dialogues by asking the LLM to rewrite only message content.

Important behavior:
- The validator receives only:
  current dialogue role/content + current event implicit_preferences.
- It does NOT receive all profile implicit preferences.
- Repair preserves role order, number of messages, and non-content fields such
  as background_audio, audio_path, audio_source.
- For formatted QA data, repair writes back both event.dialog and
  event.dialog_list. The repaired task_id list is written to profile/regen_list.txt
  so the corresponding TTS/audio can be regenerated.
- Source files are not overwritten by default. Use --in_place to overwrite.

Python 3.10+.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from scripts.common.llm import env_value, openai_client
from scripts.common.paths import project_path, resolve_path

try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover
    def repair_json(text: str) -> str:
        return text

INPUT_PATH = project_path("qa", "qa_formatted_data_000_019.json")
OUTPUT_PATH = project_path("qa", "qa_formatted_data_000_019_dialogue_rechecked.json")
REPORT_PATH = project_path("event", "formatted_dialogue_implicit_leakage_recheck_report.csv")
CHECKPOINT_PATH = project_path("event", "formatted_dialogue_implicit_leakage_recheck_checkpoint.json")
REPAIR_LOG_PATH = project_path("event", "formatted_dialogue_implicit_leakage_repair_log.csv")
REGEN_LIST_PATH = project_path("profile", "regen_list.txt")
DEFAULT_MODEL = os.getenv("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")


VALIDATOR_PROMPT = """你是一个严格的对话隐式偏好泄露检查器。

请判断下面这段“用户与AI助手之间的对话”是否泄露了【本次事件涉及的隐式偏好】。

注意：
1. 只检查下面列出的本次事件 implicit_preferences，不要扩展到其他人物偏好。
2. 只检查 dialogue 中 role/content 的文字内容。
3. 不检查 background_audio、audio_path、image_turn_indices 等非对话字段。
4. 如果 user 或 assistant 直接提到、复述、改写、暗示、解释了隐式偏好的内容、动作、物品、环境线索、声音线索、entity anchor，都算泄露。
5. 如果只是泛泛的自然表达，无法对应到本次隐式偏好，不要误判。
6. 如果 dialogue 只围绕显式偏好展开，且没有泄露隐式偏好，判为通过。

[当前 dialogue]
{dialogue_text}

[本次事件涉及的 implicit_preferences]
{implicit_prefs_text}

只输出 JSON，不要输出 markdown，不要输出解释：
{{
  "is_valid": true,
  "leakage_points": [
    {{
      "message_index": 0,
      "role": "user",
      "leaked_text": "泄露隐式偏好的原句或短语",
      "implicit_category": "对应隐式偏好 category",
      "reason": "为什么这句话泄露了该隐式偏好"
    }}
  ],
  "summary": "一句话总结"
}}
"""


REPAIR_PROMPT = """你是一个对话重写器。请修复下面 dialogue 中的隐式偏好泄露。

任务：
1. 根据[泄露检查结果]，重写 dialogue 的 content，移除所有隐式偏好泄露。
2. 只保留和显式偏好、recommended_main_scene、scene_description 自然相关的内容。
3. 不能提及、暗示、解释任何[本次事件涉及的 implicit_preferences]。
4. 如果场景中涉及人物或宠物，可以自然谈论该人物/宠物与用户的关系、职业/身份或基本特征；但不能把隐式偏好嫁接给这个人物/宠物。
5. 对话角色必须仍然只有 user 和 assistant。assistant 是 AI 助手，不能扮演亲友、宠物或其他人物。
6. 必须保持原 dialogue 的消息数量、role 顺序完全不变。
7. 每条消息不超过 100 个中文字符。
8. 只重写 content；不要输出 background_audio、audio_path 等字段。

[显式偏好]
{explicit_prefs_text}

[recommended_main_scene]
{recommended_main_scene}

[scene_description]
{scene_description}

[本次事件涉及的 implicit_preferences（禁止泄露）]
{implicit_prefs_text}

[原 dialogue]
{dialogue_text}

[泄露检查结果]
{leakage_text}

[上一轮修复失败原因]
{retry_feedback}

严格输出 JSON list，不要 markdown，不要解释：
[
  {{"role": "user", "content": "重写后的用户消息"}},
  {{"role": "assistant", "content": "重写后的助手消息"}}
]
"""


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    records: List[Dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except Exception as exc:
            raise ValueError(f"{path} line {line_no} is not valid JSON/JSONL: {exc}") from exc
        if isinstance(item, dict):
            records.append(item)
    return records


def is_formatted_profile_data(data: Any) -> bool:
    return isinstance(data, list) and any(
        isinstance(item, Mapping) and isinstance(item.get("events"), list)
        for item in data
    )


def load_input_records(path: Path) -> Tuple[Any, List[Dict[str, Any]], str]:
    """Load either formatted profile data or legacy event records."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], [], "empty"

    try:
        data = json.loads(text)
    except Exception:
        records = load_json_or_jsonl(path)
        return records, records, "event_records"

    if is_formatted_profile_data(data):
        records: List[Dict[str, Any]] = []
        for profile_index, profile in enumerate(data):
            if not isinstance(profile, Mapping):
                continue
            p_id = profile.get("p_id", profile_index)
            profile_name = profile.get("profile_name") or profile.get("name") or ""
            for event_index, event in enumerate(profile.get("events") or []):
                if not isinstance(event, Mapping):
                    continue
                record = dict(event)
                record["p_id"] = p_id
                record["profile_name"] = profile_name
                record["_profile_index"] = profile_index
                record["_event_index"] = event_index
                record["_source_format"] = "formatted"
                records.append(record)
        return data, records, "formatted"

    if isinstance(data, list):
        records = [x for x in data if isinstance(x, dict)]
        return records, records, "event_records"
    if isinstance(data, dict):
        return [data], [data], "event_records"
    return data, [], "unknown"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_text(text: str) -> str:
    text = strip_fences(text)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return text
    start = min(starts)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    end = text.rfind(closer)
    if end >= start:
        return text[start:end + 1]
    return text[start:]


def parse_llm_json(text: str) -> Any:
    cleaned = extract_json_text(text)
    try:
        return json.loads(cleaned)
    except Exception:
        return json.loads(repair_json(cleaned))


def record_event(record: Mapping[str, Any]) -> Mapping[str, Any]:
    event = record.get("event")
    if isinstance(event, Mapping):
        return event
    return record


def dialogue_messages(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    event = record_event(record)
    dialog = event.get("dialog") if isinstance(event, Mapping) else None
    if not isinstance(dialog, list):
        return []
    return [m for m in dialog if isinstance(m, dict)]


def dialogue_to_text(dialog: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for i, msg in enumerate(dialog):
        role = str(msg.get("role", "") or "")
        content = str(msg.get("content", "") or "").strip()
        lines.append(f"{i}. {role}: {content}")
    return "\n".join(lines)


def prefs_to_text(prefs: Sequence[Mapping[str, Any]], include_anchors: bool = True) -> str:
    if not prefs:
        return "（无）"
    lines: List[str] = []
    for pref in prefs:
        if not isinstance(pref, Mapping):
            continue
        cat = str(pref.get("category", "") or "")
        subcat = str(pref.get("subcategory", "") or "")
        content = str(pref.get("content", "") or pref.get("preference", "") or "")
        sources = ", ".join(str(s) for s in (pref.get("sources") or pref.get("evidence_sources") or []))
        lines.append(f"- [{cat}] {subcat}（sources: {sources}）")
        if content:
            lines.append(f"  content: {content}")
        rationale = pref.get("rationale") or pref.get("analysis") or []
        if isinstance(rationale, list):
            for item in rationale[:4]:
                if str(item).strip():
                    lines.append(f"  evidence: {str(item).strip()}")
        elif isinstance(rationale, str) and rationale.strip():
            lines.append(f"  evidence: {rationale.strip()}")
        anchors = pref.get("entity_anchors", pref.get("entity_anchor"))
        if include_anchors and anchors:
            if isinstance(anchors, list):
                anchor_text = "、".join(str(a) for a in anchors if str(a).strip())
            else:
                anchor_text = str(anchors)
            if anchor_text:
                lines.append(f"  entity_anchors: {anchor_text}")
    return "\n".join(lines) if lines else "（无）"


def client_call(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
    timeout_sleep: float = 1.5,
) -> Tuple[str, int, int]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            raw = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            pt = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            ct = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            if raw:
                return raw, pt, ct
            last_exc = RuntimeError("empty LLM response")
        except Exception as exc:
            last_exc = exc
        sleep = timeout_sleep * (2 ** (attempt - 1))
        time.sleep(min(sleep, 20))
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_exc}")


def validate_dialogue(
    client: Any,
    model: str,
    dialog: Sequence[Mapping[str, Any]],
    implicit_prefs: Sequence[Mapping[str, Any]],
    max_retries: int,
) -> Tuple[bool, List[Dict[str, Any]], str, int, int, str]:
    if not dialog:
        return True, [], "no dialogue", 0, 0, ""
    if not implicit_prefs:
        return True, [], "no implicit preferences", 0, 0, ""

    prompt = VALIDATOR_PROMPT.format(
        dialogue_text=dialogue_to_text(dialog),
        implicit_prefs_text=prefs_to_text(implicit_prefs),
    )
    raw, pt, ct = client_call(client, model, prompt, temperature=0.1, max_retries=max_retries)
    obj = parse_llm_json(raw)
    if not isinstance(obj, Mapping):
        raise ValueError("validator response is not a JSON object")
    is_valid = bool(obj.get("is_valid"))
    points = obj.get("leakage_points") or []
    if not isinstance(points, list):
        points = []
    normalized = [p for p in points if isinstance(p, dict)]
    summary = str(obj.get("summary") or "")
    return is_valid, normalized, summary, pt, ct, raw


def normalize_repaired_dialog(
    repaired: Any,
    original_dialog: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    if not isinstance(repaired, list):
        return [], ["repair response is not a list"]
    if len(repaired) != len(original_dialog):
        errors.append(f"message count changed: {len(original_dialog)} -> {len(repaired)}")

    merged: List[Dict[str, Any]] = []
    n = min(len(repaired), len(original_dialog))
    for i in range(n):
        src = original_dialog[i]
        item = repaired[i]
        if not isinstance(item, Mapping):
            errors.append(f"message {i} is not an object")
            item = {}
        expected_role = str(src.get("role", "") or "")
        role = str(item.get("role", "") or "")
        if role != expected_role:
            errors.append(f"message {i} role changed: {expected_role!r} -> {role!r}")
        content = str(item.get("content", "") or "").strip()
        if not content:
            errors.append(f"message {i} content is empty")
            content = str(src.get("content", "") or "")
        if len(content) > 110:
            errors.append(f"message {i} content too long: {len(content)} chars")
        new_msg = dict(src)
        new_msg["role"] = expected_role
        new_msg["content"] = content
        merged.append(new_msg)

    if len(original_dialog) > n:
        for i in range(n, len(original_dialog)):
            merged.append(dict(original_dialog[i]))

    return merged, errors


def repair_dialogue(
    client: Any,
    model: str,
    record: Mapping[str, Any],
    current_dialog: Sequence[Mapping[str, Any]],
    leakage_points: Sequence[Mapping[str, Any]],
    retry_feedback: str,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], List[str], int, int, str]:
    event = record_event(record)
    explicit_prefs = record.get("explicit_preferences") or event.get("explicit_preferences") or []
    implicit_prefs = record.get("implicit_preferences") or event.get("implicit_preferences") or []
    prompt = REPAIR_PROMPT.format(
        explicit_prefs_text=prefs_to_text(explicit_prefs, include_anchors=True),
        recommended_main_scene=str(record.get("recommended_main_scene") or event.get("recommended_main_scene") or ""),
        scene_description=str(event.get("scene_description") or "") if isinstance(event, Mapping) else "",
        implicit_prefs_text=prefs_to_text(implicit_prefs, include_anchors=True),
        dialogue_text=dialogue_to_text(current_dialog),
        leakage_text=json.dumps(list(leakage_points), ensure_ascii=False, indent=2),
        retry_feedback=retry_feedback or "无",
    )
    raw, pt, ct = client_call(client, model, prompt, temperature=0.4, max_retries=max_retries)
    repaired_obj = parse_llm_json(raw)
    repaired_dialog, errors = normalize_repaired_dialog(repaired_obj, current_dialog)
    return repaired_dialog, errors, pt, ct, raw


def task_key(record: Mapping[str, Any]) -> str:
    return str(record.get("task_id") or f"p{record.get('p_id')}_g{record.get('group_id')}")


def process_record(
    record: Mapping[str, Any],
    client: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    key = task_key(record)
    p_id = record.get("p_id")
    group_id = record.get("group_id")
    implicit_prefs = record.get("implicit_preferences") or []
    dialog = dialogue_messages(record)
    total_pt = 0
    total_ct = 0

    result: Dict[str, Any] = {
        "task_id": key,
        "p_id": p_id,
        "group_id": group_id,
        "initial_status": "",
        "final_status": "",
        "repair_attempts": 0,
        "leakage_points": [],
        "messages": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "original_dialog": dialog,
        "repaired_dialog": None,
    }

    if not dialog:
        result["initial_status"] = "skip_no_dialogue"
        result["final_status"] = "skip_no_dialogue"
        return result
    if not implicit_prefs:
        result["initial_status"] = "valid_no_implicit"
        result["final_status"] = "valid_no_implicit"
        return result

    is_valid, leaks, summary, pt, ct, _ = validate_dialogue(
        client, args.model, dialog, implicit_prefs, args.max_retries
    )
    total_pt += pt
    total_ct += ct
    result["initial_status"] = "valid" if is_valid else "leaked"
    result["leakage_points"] = leaks
    result["messages"].append(summary)

    if is_valid:
        result["final_status"] = "valid"
        result["prompt_tokens"] = total_pt
        result["completion_tokens"] = total_ct
        return result

    current_dialog = [dict(m) for m in dialog]
    retry_feedback = "无"
    last_leaks = leaks

    for attempt in range(1, args.max_repair_retries + 1):
        result["repair_attempts"] = attempt
        repaired_dialog, repair_errors, rpt, rct, _ = repair_dialogue(
            client,
            args.model,
            record,
            current_dialog,
            last_leaks,
            retry_feedback,
            args.max_retries,
        )
        total_pt += rpt
        total_ct += rct
        if repair_errors:
            retry_feedback = "上一轮修复格式不合格：\n" + "\n".join(f"- {e}" for e in repair_errors[:12])
            result["messages"].extend(repair_errors)
            continue

        ok, new_leaks, new_summary, vpt, vct, _ = validate_dialogue(
            client, args.model, repaired_dialog, implicit_prefs, args.max_retries
        )
        total_pt += vpt
        total_ct += vct
        result["messages"].append(f"repair attempt {attempt}: {new_summary}")
        if ok:
            result["final_status"] = "repaired"
            result["repaired_dialog"] = repaired_dialog
            result["prompt_tokens"] = total_pt
            result["completion_tokens"] = total_ct
            return result

        last_leaks = new_leaks
        current_dialog = repaired_dialog
        retry_feedback = (
            "上一轮修复后仍泄露隐式偏好，请继续重写这些泄露点：\n"
            + json.dumps(new_leaks, ensure_ascii=False, indent=2)
        )

    result["final_status"] = "repair_failed"
    result["repaired_dialog"] = current_dialog if current_dialog != dialog else None
    result["prompt_tokens"] = total_pt
    result["completion_tokens"] = total_ct
    return result


def load_checkpoint(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return {str(k): v for k, v in obj.items() if isinstance(v, dict)}
        if isinstance(obj, list):
            return {str(x.get("task_id")): x for x in obj if isinstance(x, dict) and x.get("task_id")}
    except Exception:
        return {}
    return {}


def save_checkpoint(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_dialog_list_from_dialog(event: Dict[str, Any]) -> None:
    """Keep formatted event.dialog_list aligned with repaired event.dialog."""
    dialog = event.get("dialog")
    dialog_list = event.get("dialog_list")
    if not isinstance(dialog, list) or not isinstance(dialog_list, list):
        return

    turn_index = -1
    for msg in dialog:
        if not isinstance(msg, Mapping):
            continue
        role = str(msg.get("role") or "")
        content = str(msg.get("content") or "")
        if role == "user":
            turn_index += 1
            if 0 <= turn_index < len(dialog_list) and isinstance(dialog_list[turn_index], dict):
                dialog_list[turn_index]["user"] = content
        elif role == "assistant":
            target_index = turn_index if turn_index >= 0 else 0
            if 0 <= target_index < len(dialog_list) and isinstance(dialog_list[target_index], dict):
                dialog_list[target_index]["assistant"] = content


def apply_recheck_meta(item: Dict[str, Any], result: Mapping[str, Any]) -> None:
    item["dialogue_implicit_leakage_recheck"] = {
        "final_status": result.get("final_status"),
        "repair_attempts": result.get("repair_attempts", 0),
        "initial_leakage_points": result.get("leakage_points", []),
        "messages": result.get("messages", []),
    }


def apply_repairs_to_event_records(records: List[Dict[str, Any]], results: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = deepcopy(records)
    for item in out:
        key = task_key(item)
        result = results.get(key)
        if not result:
            continue
        repaired = result.get("repaired_dialog")
        if repaired and result.get("final_status") == "repaired":
            event = item.get("event")
            if isinstance(event, dict):
                event["dialog"] = repaired
                sync_dialog_list_from_dialog(event)
                apply_recheck_meta(item, result)
        elif result.get("final_status"):
            apply_recheck_meta(item, result)
    return out


def apply_repairs_to_formatted_data(
    data: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out = deepcopy(data)
    for record in records:
        profile_index = record.get("_profile_index")
        event_index = record.get("_event_index")
        if not isinstance(profile_index, int) or not isinstance(event_index, int):
            continue
        try:
            event = out[profile_index]["events"][event_index]
        except Exception:
            continue
        if not isinstance(event, dict):
            continue
        result = results.get(task_key(record))
        if not result:
            continue
        repaired = result.get("repaired_dialog")
        if repaired and result.get("final_status") == "repaired":
            event["dialog"] = repaired
            sync_dialog_list_from_dialog(event)
        if result.get("final_status"):
            apply_recheck_meta(event, result)
    return out


def apply_repairs(
    input_data: Any,
    records: List[Dict[str, Any]],
    results: Mapping[str, Mapping[str, Any]],
    input_format: str,
) -> Any:
    if input_format == "formatted" and isinstance(input_data, list):
        return apply_repairs_to_formatted_data(input_data, records, results)
    return apply_repairs_to_event_records(records, results)


def write_report(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "p_id",
        "group_id",
        "initial_status",
        "final_status",
        "repair_attempts",
        "num_leakage_points",
        "leakage_points",
        "messages",
        "prompt_tokens",
        "completion_tokens",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results.values(), key=lambda x: (x.get("p_id", 9999), str(x.get("task_id", "")))):
            leaks = result.get("leakage_points") or []
            writer.writerow({
                "task_id": result.get("task_id", ""),
                "p_id": result.get("p_id", ""),
                "group_id": result.get("group_id", ""),
                "initial_status": result.get("initial_status", ""),
                "final_status": result.get("final_status", ""),
                "repair_attempts": result.get("repair_attempts", 0),
                "num_leakage_points": len(leaks) if isinstance(leaks, list) else 0,
                "leakage_points": json.dumps(leaks, ensure_ascii=False),
                "messages": json.dumps(result.get("messages", []), ensure_ascii=False),
                "prompt_tokens": result.get("prompt_tokens", 0),
                "completion_tokens": result.get("completion_tokens", 0),
            })


def write_repair_log(path: Path, results: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id",
        "p_id",
        "group_id",
        "initial_status",
        "final_status",
        "repair_attempts",
        "leakage_points",
        "original_dialogue_text",
        "repaired_dialogue_text",
        "original_dialog_json",
        "repaired_dialog_json",
        "messages",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results.values(), key=lambda x: (x.get("p_id", 9999), str(x.get("task_id", "")))):
            if result.get("initial_status") != "leaked" and result.get("final_status") not in {"repaired", "repair_failed"}:
                continue
            original_dialog = result.get("original_dialog") or []
            repaired_dialog = result.get("repaired_dialog") or []
            writer.writerow({
                "task_id": result.get("task_id", ""),
                "p_id": result.get("p_id", ""),
                "group_id": result.get("group_id", ""),
                "initial_status": result.get("initial_status", ""),
                "final_status": result.get("final_status", ""),
                "repair_attempts": result.get("repair_attempts", 0),
                "leakage_points": json.dumps(result.get("leakage_points", []), ensure_ascii=False),
                "original_dialogue_text": dialogue_to_text(original_dialog),
                "repaired_dialogue_text": dialogue_to_text(repaired_dialog),
                "original_dialog_json": json.dumps(original_dialog, ensure_ascii=False),
                "repaired_dialog_json": json.dumps(repaired_dialog, ensure_ascii=False),
                "messages": json.dumps(result.get("messages", []), ensure_ascii=False),
            })


def write_regen_list(path: Path, results: Mapping[str, Mapping[str, Any]]) -> List[str]:
    task_ids = sorted(
        str(result.get("task_id"))
        for result in results.values()
        if result.get("final_status") == "repaired" and result.get("task_id")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(task_ids) + ("\n" if task_ids else ""), encoding="utf-8")
    return task_ids


def parse_id_list(values: Optional[List[str]]) -> Optional[set]:
    if values is None:
        return None
    ids = set()
    for raw in values:
        for part in str(raw).replace(",", " ").split():
            if part.strip():
                try:
                    ids.add(int(part))
                except ValueError:
                    ids.add(part.strip())
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM check and repair dialogue implicit leakage.")
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    parser.add_argument("--repair_log", default=str(REPAIR_LOG_PATH))
    parser.add_argument("--regen_list", default=str(REGEN_LIST_PATH))
    parser.add_argument("--in_place", action="store_true", help="覆盖 --input 文件；默认写到 --output。")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base_url", default=None, help="runtime endpoint override")
    parser.add_argument("--api_key", default=None, help="runtime API key override")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--max_repair_retries", type=int, default=3)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--include_no_dialogue", action="store_true", help="默认跳过没有 event.dialog 的记录；加此参数才处理空 dialogue 记录。")
    parser.add_argument("--only_profile_ids", nargs="*", default=None)
    parser.add_argument("--task_ids", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = input_path if args.in_place else resolve_path(args.output)
    report_path = resolve_path(args.report)
    checkpoint_path = resolve_path(args.checkpoint)
    repair_log_path = resolve_path(args.repair_log)
    regen_list_path = resolve_path(args.regen_list)

    if args.workers <= 0:
        raise ValueError("--workers must be > 0")
    if not args.api_key and not env_value("CUE_MEM_LLM_API_KEY") and not args.dry_run:
        raise ValueError("--api_key or CUE_MEM_LLM_API_KEY is required")

    input_data, records, input_format = load_input_records(input_path)
    only_pids = parse_id_list(args.only_profile_ids)
    task_ids = parse_id_list(args.task_ids)

    tasks: List[Dict[str, Any]] = []
    skipped_no_dialogue = 0
    for record in records:
        if only_pids is not None and record.get("p_id") not in only_pids:
            continue
        if task_ids is not None and task_key(record) not in task_ids:
            continue
        if not args.include_no_dialogue and not dialogue_messages(record):
            skipped_no_dialogue += 1
            continue
        tasks.append(record)

    if args.shuffle:
        random.Random(args.seed).shuffle(tasks)
    if args.sample and args.sample > 0:
        tasks = tasks[:args.sample]

    checkpoint = load_checkpoint(checkpoint_path) if args.resume else {}
    pending = [r for r in tasks if task_key(r) not in checkpoint]

    print(
        f"Loaded records={len(records)} selected={len(tasks)} pending={len(pending)} format={input_format}"
        + (f" skipped_no_dialogue={skipped_no_dialogue}" if skipped_no_dialogue else "")
    )
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    if args.dry_run:
        for record in pending[: max(1, args.sample or 1)]:
            dialog = dialogue_messages(record)
            prompt = VALIDATOR_PROMPT.format(
                dialogue_text=dialogue_to_text(dialog),
                implicit_prefs_text=prefs_to_text(record.get("implicit_preferences") or []),
            )
            print("\n" + "=" * 80)
            print(task_key(record))
            print(prompt[:4000])
        print("Dry run only; no API call and no output written.")
        return

    client = openai_client(api_key=args.api_key, base_url=args.base_url)

    results: Dict[str, Dict[str, Any]] = dict(checkpoint)
    total_pt = total_ct = 0

    if args.workers == 1:
        for record in pending:
            key = task_key(record)
            try:
                result = process_record(record, client, args)
            except Exception as exc:
                result = {
                    "task_id": key,
                    "p_id": record.get("p_id"),
                    "group_id": record.get("group_id"),
                    "initial_status": "error",
                    "final_status": "error",
                    "repair_attempts": 0,
                    "leakage_points": [],
                    "messages": [f"{type(exc).__name__}: {exc}"],
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "repaired_dialog": None,
                }
            results[key] = result
            total_pt += int(result.get("prompt_tokens", 0) or 0)
            total_ct += int(result.get("completion_tokens", 0) or 0)
            print(f"[{len(results)}/{len(tasks)}] {key} {result.get('initial_status')} -> {result.get('final_status')} repairs={result.get('repair_attempts')}")
            save_checkpoint(checkpoint_path, results)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_record, record, client, args): record for record in pending}
            for future in as_completed(futures):
                record = futures[future]
                key = task_key(record)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "task_id": key,
                        "p_id": record.get("p_id"),
                        "group_id": record.get("group_id"),
                        "initial_status": "error",
                        "final_status": "error",
                        "repair_attempts": 0,
                        "leakage_points": [],
                        "messages": [f"{type(exc).__name__}: {exc}"],
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "repaired_dialog": None,
                    }
                results[key] = result
                total_pt += int(result.get("prompt_tokens", 0) or 0)
                total_ct += int(result.get("completion_tokens", 0) or 0)
                print(f"[{len(results)}/{len(tasks)}] {key} {result.get('initial_status')} -> {result.get('final_status')} repairs={result.get('repair_attempts')}")
                save_checkpoint(checkpoint_path, results)

    output_data = apply_repairs(input_data, records, results, input_format)
    write_json(output_path, output_data)
    write_report(report_path, results)
    write_repair_log(repair_log_path, results)
    regen_task_ids = write_regen_list(regen_list_path, results)

    status_counts: Dict[str, int] = {}
    for result in results.values():
        status = str(result.get("final_status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print("\nDone.")
    print(f"Final status counts: {status_counts}")
    print(f"Token usage this run: prompt={total_pt}, completion={total_ct}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")
    print(f"Repair log: {repair_log_path}")
    print(f"Regen list: {regen_list_path} ({len(regen_task_ids)} task_id)")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
