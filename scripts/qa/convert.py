"""
Convert current MemGallery-style historical dialogue data into the new
voice-caption-based input format.

Input format example:
[
  {
    "p_id": 0,
    "profile_str": "...",
    "events": [
      {
        "session_id": "D00",
        "scene_description": "01/14/2025;家中客厅;...",
        "dialog_list": [
          {
            "round": "D00:00",
            "user": "...",
            "assistant": "...",
            "D00-001.png": "图片caption",
            "D00-001.wav": "音频caption 或 音频路径"
          }
        ]
      }
    ]
  }
]

Output format example:
[
  {
    "p_id": 0,
    "profile_str": "...",
    "sessions": [
      {
        "session_id": "D00",
        "date": "01/14/2025",
        "dialogues": [
          {
            "round": "D00:00",
            "user_voice_message_caption": "...",
            "assistant": "...",
            "input_voice_message": ["../voice/D00/D00-001.wav"],
            "voice_caption": ["..."],
            "voice_id": ["D00-001"],
            "input_image": ["../image/D00/D00-001.png"],
            "image_caption": ["..."],
            "image_id": ["D00-001"]
          }
        ]
      }
    ]
  }
]

转换模式：
  - single（默认）：用 --input/--output 转换单个 JSON/JSONL 文件。
  - category：用 --categories 批量转换 brief/medium/detailed 文件，并从
    原始文件合并音频 caption；输出到 --output_dir。

示例：
  python scripts/qa/convert.py --input input.json --output output.json
  python scripts/qa/convert.py --mode category --categories brief,medium,detailed
"""

import argparse
import copy
import difflib
import os
import re
from pathlib import Path
from typing import Any

from scripts.common.io import load_json_or_jsonl, write_json
from scripts.common.paths import resolve_path
from scripts.qa.config import BENCHMARK_RUN_ROOT, qa_path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VOICE_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}

DEFAULT_ORIGINAL_FILE = qa_path("qa_formatted_data_000_019_with_media_captions.json")
DEFAULT_CATEGORY_DIR = qa_path("qwen3.5-9b")
DEFAULT_CATEGORIES = ("brief", "medium", "detailed")


def normalize_text_for_match(text: str) -> str:
    """
    去除标点、空白，用于文本相似度比较。
    """
    return re.sub(r"""[\s，。！？、；："'「」【】\(\)（）,\.!?]""", "", str(text or ""))


def build_dialog_user_turns(dialog_entries: list[dict]) -> list[dict]:
    """
    从 event["dialog"] 中提取所有 user 轮，保留真实媒体路径。
    返回列表顺序与原始 dialog 中 user 轮出现顺序一致。

    每项结构：
      {
        "content"    : str,          # user 发言文本
        "image_path" : str or None,  # 真实图片路径
        "audio_path" : str or None,  # 真实音频路径
      }
    """
    result = []
    for entry in (dialog_entries or []):
        if not isinstance(entry, dict):
            continue
        if entry.get("role") == "user":
            result.append({
                "content":    entry.get("content", ""),
                "image_path": entry.get("image_path"),
                "audio_path": entry.get("audio_path"),
            })
    return result


def find_matching_dialog_turn(
    user_text: str,
    dialog_user_turns: list[dict],
    threshold: float = 0.60,
) -> dict | None:
    """
    在 dialog_user_turns 中找到与 user_text 最相似的轮次。

    策略：
    1. 先尝试精确匹配（normalize 后）；
    2. 再用 difflib.SequenceMatcher 模糊匹配，取最高相似度 >= threshold 的结果。

    返回匹配到的 dict，或 None。
    """
    if not user_text or not dialog_user_turns:
        return None

    norm_query = normalize_text_for_match(user_text)

    best: dict | None = None
    best_ratio: float = 0.0

    for turn in dialog_user_turns:
        norm_cand = normalize_text_for_match(turn["content"])

        # 精确匹配优先
        if norm_query == norm_cand:
            return turn

        ratio = difflib.SequenceMatcher(None, norm_query, norm_cand).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = turn

    if best_ratio >= threshold:
        return best
    return None


def is_probably_path(value: Any) -> bool:
    """
    判断一个字符串是否更像文件路径，而不是 caption。
    例如:
      event/voice_mixed_000_002/0-0-35-0/xxx.wav
      ../voice/D12/D12-001.wav
    """
    if not isinstance(value, str):
        return False

    s = value.strip()
    lower = s.lower()

    if any(lower.endswith(ext) for ext in VOICE_EXTS | IMAGE_EXTS):
        return True

    # 含中文字符的一定是 caption，不是路径
    if re.search(r'[一-鿿]', s):
        return False

    # 包含路径分隔符，且不像自然语言 caption
    return bool("/" in s or "\\" in s)


def media_id_from_filename(filename: str) -> str:
    """
    D12-001.wav -> D12-001
    D12-001.png -> D12-001
    """
    return Path(filename).stem


def make_prefixed_path(prefix: str, session_id: str, filename: str) -> str:
    """
    prefix="../voice", session_id="D12", filename="D12-001.wav"
    -> "../voice/D12/D12-001.wav"
    """
    return str(Path(prefix) / session_id / filename).replace("\\", "/")


def normalize_existing_path(path: str) -> str:
    """
    保留已有路径，但统一 Windows 反斜杠为正斜杠。
    """
    return path.replace("\\", "/")


def resolve_media_path(path: str, base_dir: str | None) -> str:
    """
    当 base_dir 不为 None 时，将相对路径解析为绝对路径并统一使用正斜杠。
    已经是绝对路径则直接规范化后返回。
    """
    if not base_dir:
        return path
    if os.path.isabs(path):
        return path.replace("\\", "/")
    return os.path.abspath(os.path.join(base_dir, path)).replace("\\", "/")


def extract_date_from_scene_description(scene_description: str) -> str:
    """
    从 scene_description 中提取日期。
    例如:
      "01/14/2025;家中客厅;周六上午..."
    返回:
      "01/14/2025"
    """
    if not isinstance(scene_description, str) or not scene_description.strip():
        return ""

    first_part = scene_description.split(";", 1)[0].strip()

    # 简单判断是否像日期；不强制转换格式，避免误伤。
    if re.match(r"^\d{1,4}[/-]\d{1,2}[/-]\d{1,4}$", first_part):
        return first_part

    return first_part


def split_media_fields(dialog: dict[str, Any]) -> tuple[list[tuple[str, Any]], list[tuple[str, Any]]]:
    """
    从一轮 dialog 中找出动态媒体字段：
      D00-001.png: 图片caption
      D00-001.wav: 音频caption 或 音频路径

    返回:
      image_items: [(filename, value), ...]
      voice_items: [(filename, value), ...]
    """
    image_items = []
    voice_items = []

    for key, value in dialog.items():
        if not isinstance(key, str):
            continue

        suffix = Path(key).suffix.lower()

        if suffix in IMAGE_EXTS:
            image_items.append((key, value))
        elif suffix in VOICE_EXTS:
            voice_items.append((key, value))

    return image_items, voice_items


def convert_dialog(
    dialog: dict[str, Any],
    session_id: str,
    image_prefix: str,
    voice_prefix: str,
    fallback_voice_caption_from_user: bool = True,
    keep_text_user_when_no_voice: bool = True,
    add_caption_source: bool = False,
    matched_dialog_turn: dict | None = None,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """
    转换单轮 dialogue。

    核心规则：
    1. 如果存在 voice，则输出 user_voice_message_caption，不输出 user。
    2. 如果不存在 voice，默认保留 user。
    3. 图片字段转换为 input_image / image_caption / image_id。
    4. 音频字段转换为 input_voice_message / voice_caption / voice_id。

    matched_dialog_turn: 通过用户文本匹配到的 event["dialog"] 对应轮次，
        包含真实的 image_path / audio_path。优先用这里的路径代替构造路径。
    base_dir: 不为 None 时，所有媒体路径均解析为绝对路径。
    """

    new_dialog: dict[str, Any] = {}

    round_id = dialog.get("round", "")
    assistant_text = dialog.get("assistant", "")
    user_text = dialog.get("user", "")

    image_items, voice_items = split_media_fields(dialog)

    new_dialog["round"] = round_id

    # 从匹配到的真实 dialog 轮取路径（如果有）
    real_image_path: str | None = (
        normalize_existing_path(matched_dialog_turn["image_path"])
        if matched_dialog_turn and matched_dialog_turn.get("image_path")
        else None
    )
    real_audio_path: str | None = (
        normalize_existing_path(matched_dialog_turn["audio_path"])
        if matched_dialog_turn and matched_dialog_turn.get("audio_path")
        else None
    )

    # -------------------------
    # 1. 处理图片字段
    # -------------------------
    input_images: list[str] = []
    image_captions: list[str] = []
    image_ids: list[str] = []

    for filename, value in image_items:
        image_id = media_id_from_filename(filename)

        if real_image_path:
            # 优先使用从 dialog["image_path"] 匹配到的真实路径
            image_path = real_image_path
            # value 是 caption（若 value 本身是路径则丢弃）
            image_caption = "" if is_probably_path(value) else (str(value) if value is not None else "")
        elif is_probably_path(value):
            image_path = normalize_existing_path(str(value))
            image_caption = ""
        else:
            image_path = make_prefixed_path(image_prefix, session_id, filename)
            image_caption = str(value) if value is not None else ""

        input_images.append(resolve_media_path(image_path, base_dir))
        image_captions.append(image_caption)
        image_ids.append(image_id)

    if input_images:
        new_dialog["input_image"] = input_images
        new_dialog["image_caption"] = image_captions
        new_dialog["image_id"] = image_ids

    # -------------------------
    # 2. 处理音频字段
    # -------------------------
    input_voices: list[str] = []
    voice_captions: list[str] = []
    voice_ids: list[str] = []
    voice_caption_sources: list[str] = []

    for filename, value in voice_items:
        voice_id = media_id_from_filename(filename)

        if real_audio_path:
            # 优先使用从 dialog["audio_path"] 匹配到的真实路径
            voice_path = real_audio_path
            if is_probably_path(value):
                # value 也是路径而非 caption，用 user 文本 fallback
                voice_caption = str(user_text) if fallback_voice_caption_from_user and user_text else ""
                caption_source = "fallback_user_text"
            else:
                voice_caption = str(value) if value is not None else ""
                caption_source = "media_caption"
        elif is_probably_path(value):
            # value 是真实音频路径，不是 caption。
            voice_path = normalize_existing_path(str(value))

            if fallback_voice_caption_from_user and user_text:
                # 这种情况常见于你当前数据：
                # "D00-004.wav": "event/voice_mixed_...wav"
                # 没有 caption，只能用 user 文本作为用户语音转写的 fallback。
                voice_caption = str(user_text)
                caption_source = "fallback_user_text"
            else:
                voice_caption = ""
                caption_source = "missing"
        else:
            # value 是音频 caption。
            voice_path = make_prefixed_path(voice_prefix, session_id, filename)
            voice_caption = str(value) if value is not None else ""
            caption_source = "media_caption"

        input_voices.append(resolve_media_path(voice_path, base_dir))
        voice_captions.append(voice_caption)
        voice_ids.append(voice_id)
        voice_caption_sources.append(caption_source)

    if input_voices:
        new_dialog["input_voice_message"] = input_voices
        new_dialog["voice_caption"] = voice_captions
        new_dialog["voice_id"] = voice_ids

        if add_caption_source:
            new_dialog["voice_caption_source"] = voice_caption_sources

        # 核心：有语音时，不再输出 user，而是输出 user_voice_message_caption。
        # 一轮多个 voice 时，用换行合并。
        valid_voice_caps = [cap for cap in voice_captions if cap]

        if valid_voice_caps:
            user_voice_message_caption = "\n".join(valid_voice_caps)
        elif fallback_voice_caption_from_user and user_text:
            user_voice_message_caption = str(user_text)
        else:
            user_voice_message_caption = ""

        new_dialog["user_voice_message_caption"] = user_voice_message_caption

    else:
        # 没有 voice 时，保留原本 user 文本。
        if keep_text_user_when_no_voice and user_text:
            new_dialog["user"] = user_text

    # -------------------------
    # 3. assistant
    # -------------------------
    if assistant_text:
        new_dialog["assistant"] = assistant_text

    return new_dialog


def convert_event(
    event: dict[str, Any],
    image_prefix: str,
    voice_prefix: str,
    fallback_voice_caption_from_user: bool,
    keep_text_user_when_no_voice: bool,
    add_caption_source: bool,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """
    转换一个 event/session。
    兼容 event["dialog_list"] 和 event["dialogues"]。

    当 event 中包含 "dialog" 字段时，从中提取真实 image_path / audio_path，
    通过用户消息文本匹配 dialog_list 中对应的轮次，以纠正媒体路径。
    base_dir: 不为 None 时，所有媒体路径均解析为绝对路径。
    """

    session_id = event.get("session_id", "")
    scene_description = event.get("scene_description", "")
    date = event.get("date") or extract_date_from_scene_description(scene_description)

    old_dialogues = event.get("dialogues")
    if old_dialogues is None:
        old_dialogues = event.get("dialog_list", [])

    # 构建真实路径索引：从 event["dialog"] 中提取所有 user 轮
    dialog_user_turns = build_dialog_user_turns(event.get("dialog", []))

    new_dialogues = []

    for dialog in old_dialogues:
        if not isinstance(dialog, dict):
            continue

        # 根据 user 文本找到对应的真实 dialog 轮次（含 image_path / audio_path）
        user_text = dialog.get("user", "")
        matched = find_matching_dialog_turn(user_text, dialog_user_turns)

        new_dialog = convert_dialog(
            dialog=dialog,
            session_id=session_id,
            image_prefix=image_prefix,
            voice_prefix=voice_prefix,
            fallback_voice_caption_from_user=fallback_voice_caption_from_user,
            keep_text_user_when_no_voice=keep_text_user_when_no_voice,
            add_caption_source=add_caption_source,
            matched_dialog_turn=matched,
            base_dir=base_dir,
        )
        new_dialogues.append(new_dialog)

    converted = {
        "session_id": session_id,
        "date": date,
        "dialogues": new_dialogues,
    }

    # 保留一些有用元信息，方便后续追踪。
    for key in [
        "task_id",
        "group_id",
        "scene_description",
        "user_shared_image_description",
        "background_audio_info",
        "human_speech_content",
        "explicit_preferences",
        "implicit_preferences",
    ]:
        if key in event:
            converted[key] = event[key]

    return converted


def convert_dataset(
    data: Any,
    image_prefix: str,
    voice_prefix: str,
    fallback_voice_caption_from_user: bool,
    keep_text_user_when_no_voice: bool,
    add_caption_source: bool,
    output_mode: str,
    base_dir: str | None = None,
) -> Any:
    """
    支持两种输出模式：

    1. flat_sessions:
       [
         {"session_id": "D00", "date": "...", "dialogues": [...]},
         {"session_id": "D01", "date": "...", "dialogues": [...]}
       ]

    2. grouped_by_profile:
       [
         {
           "p_id": 0,
           "profile_str": "...",
           "sessions": [...]
         }
       ]
    """

    if not isinstance(data, list):
        raise TypeError("Input JSON top-level should be a list.")

    flat_sessions = []
    grouped_profiles = []

    for person in data:
        if not isinstance(person, dict):
            continue

        p_id = person.get("p_id")
        profile_str = person.get("profile_str", "")
        events = person.get("events", [])

        converted_sessions = []

        for event in events:
            if not isinstance(event, dict):
                continue

            converted_event = convert_event(
                event=event,
                image_prefix=image_prefix,
                voice_prefix=voice_prefix,
                fallback_voice_caption_from_user=fallback_voice_caption_from_user,
                keep_text_user_when_no_voice=keep_text_user_when_no_voice,
                add_caption_source=add_caption_source,
                base_dir=base_dir,
            )

            # 给 flat 输出也带上 profile 追踪信息。
            converted_event["_p_id"] = p_id
            converted_event["_profile_str"] = profile_str

            converted_sessions.append(converted_event)
            flat_sessions.append(converted_event)

        grouped_profiles.append(
            {
                "p_id": p_id,
                "profile_str": profile_str,
                "sessions": converted_sessions,
            }
        )

    if output_mode == "flat_sessions":
        return flat_sessions

    if output_mode == "grouped_by_profile":
        return grouped_profiles

    raise ValueError(f"Unsupported output_mode: {output_mode}")


def merge_audio_captions(category_data: list, original_data: list) -> list:
    """Copy audio captions from the original dataset into category data.

    Category files contain image captions generated at a particular granularity,
    while their ``.wav`` values may only be paths. Both datasets share the same
    profile/event/dialog ordering, so audio fields can be merged positionally
    without changing category-specific image fields.
    """

    merged = copy.deepcopy(category_data)
    if not isinstance(merged, list) or not isinstance(original_data, list):
        return merged

    for category_person, original_person in zip(merged, original_data):
        if not isinstance(category_person, dict) or not isinstance(original_person, dict):
            continue

        for category_event, original_event in zip(
            category_person.get("events", []), original_person.get("events", [])
        ):
            if not isinstance(category_event, dict) or not isinstance(original_event, dict):
                continue

            for category_dialog, original_dialog in zip(
                category_event.get("dialog_list", []), original_event.get("dialog_list", [])
            ):
                if not isinstance(category_dialog, dict) or not isinstance(original_dialog, dict):
                    continue

                for key, value in original_dialog.items():
                    if isinstance(key, str) and Path(key).suffix.lower() in VOICE_EXTS:
                        category_dialog[key] = value

    return merged


def count_caption_fields(data: Any, extensions: set[str]) -> int:
    """Count non-path caption values stored under media-like keys."""

    count = 0
    for person in data if isinstance(data, list) else []:
        if not isinstance(person, dict):
            continue
        for event in person.get("events", []):
            if not isinstance(event, dict):
                continue
            for dialog in event.get("dialog_list", []):
                if not isinstance(dialog, dict):
                    continue
                for key, value in dialog.items():
                    if (
                        isinstance(key, str)
                        and Path(key).suffix.lower() in extensions
                        and not is_probably_path(value)
                    ):
                        count += 1
    return count


def resolve_base_dir(raw_base_dir: str | None, reference_path: Path) -> str | None:
    """Resolve ``--base_dir``, including the historical ``auto`` behavior."""

    if not raw_base_dir:
        return None
    if raw_base_dir.lower() == "auto":
        return str(reference_path.resolve().parent.parent)
    return str(resolve_path(raw_base_dir).resolve())


def run_single(args: argparse.Namespace) -> None:
    """Convert one input file, preserving the original ``convert.py`` CLI."""

    if not args.input or not args.output:
        raise ValueError("single mode requires both --input and --output")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    base_dir = resolve_base_dir(args.base_dir, input_path)
    data = load_json_or_jsonl(input_path)

    converted = convert_dataset(
        data=data,
        image_prefix=args.image_prefix,
        voice_prefix=args.voice_prefix,
        fallback_voice_caption_from_user=not args.no_fallback_voice_caption_from_user,
        keep_text_user_when_no_voice=not args.drop_text_user_when_no_voice,
        add_caption_source=args.add_caption_source,
        output_mode=args.output_mode,
        base_dir=base_dir,
    )

    write_json(output_path, converted)
    print(f"Converted data saved to: {output_path}")
    print(f"Output mode: {args.output_mode}")
    print(f"Top-level items: {len(converted)}")


def run_by_category(args: argparse.Namespace) -> None:
    """Convert category-specific caption files, preserving the old workflow."""

    categories = [category.strip() for category in args.categories.split(",") if category.strip()]
    unknown = [category for category in categories if category not in DEFAULT_CATEGORIES]
    if unknown:
        valid = ", ".join(DEFAULT_CATEGORIES)
        raise ValueError(f"unknown category '{unknown[0]}'; valid categories: {valid}")

    original_path = resolve_path(args.original or DEFAULT_ORIGINAL_FILE)
    category_dir = resolve_path(args.category_dir or DEFAULT_CATEGORY_DIR)
    output_dir = resolve_path(args.output_dir or BENCHMARK_RUN_ROOT)
    base_dir = resolve_base_dir(args.base_dir, original_path)

    print(f"Loading original file: {original_path}")
    original_data = load_json_or_jsonl(original_path)
    print(f"  {len(original_data)} profiles loaded")
    print(f"  audio captions in original: {count_caption_fields(original_data, VOICE_EXTS)}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for category in categories:
        print(f"\n{'=' * 60}")
        print(f"Processing category: {category}")
        print(f"{'=' * 60}")

        category_file = category_dir / category / "qa_formatted_data_with_media_captions.json"
        if not category_file.exists():
            print(f"  WARN: {category_file} not found, skipping")
            continue

        category_data = load_json_or_jsonl(category_file)
        print(f"  image captions in {category}: {count_caption_fields(category_data, IMAGE_EXTS)}")
        print("  Merging audio captions from original ...")
        merged_data = merge_audio_captions(category_data, original_data)
        print("  Converting to history_dialogue format ...")

        converted = convert_dataset(
            data=merged_data,
            image_prefix=args.image_prefix,
            voice_prefix=args.voice_prefix,
            fallback_voice_caption_from_user=not args.no_fallback_voice_caption_from_user,
            keep_text_user_when_no_voice=not args.drop_text_user_when_no_voice,
            add_caption_source=args.add_caption_source,
            output_mode=args.output_mode,
            base_dir=base_dir,
        )

        dialogues = [
            dialogue
            for session in converted
            for dialogue in session.get("dialogues", [])
        ]
        image_dialogues = [dialogue for dialogue in dialogues if dialogue.get("input_image")]
        voice_dialogues = [dialogue for dialogue in dialogues if dialogue.get("input_voice_message")]
        image_captions = [
            caption
            for dialogue in dialogues
            for caption in dialogue.get("image_caption", [])
            if caption
        ]
        voice_captions = [
            caption
            for dialogue in dialogues
            for caption in dialogue.get("voice_caption", [])
            if caption
        ]
        print(f"  top-level items: {len(converted)}")
        print(
            f"  dialogues with images: {len(image_dialogues)} "
            f"(with caption: {len(image_captions)})"
        )
        print(
            f"  dialogues with voice:  {len(voice_dialogues)} "
            f"(with caption: {len(voice_captions)})"
        )

        output_path = output_dir / f"{category}_history_dialogue.json"
        if args.dry_run:
            print(f"  (dry-run) would write to: {output_path}")
        else:
            write_json(output_path, converted)
            print(f"  Output -> {output_path}")

    print("\nDone.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MemGallery-style QA data. Use single mode for one input "
            "file, or category mode for brief/medium/detailed caption datasets."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["single", "category"],
        default="single",
        help="Conversion workflow (default: single).",
    )
    parser.add_argument("--input", help="Single mode: input JSON/JSONL path.")
    parser.add_argument("--output", help="Single mode: output JSON path.")
    parser.add_argument(
        "--image_prefix",
        default="../image",
        help="Prefix used when constructing image paths from dynamic image keys.",
    )
    parser.add_argument(
        "--voice_prefix",
        default="../voice",
        help="Prefix used when constructing voice paths from dynamic voice keys.",
    )
    parser.add_argument(
        "--output_mode",
        choices=["flat_sessions", "grouped_by_profile"],
        default="flat_sessions",
        help="Single mode output structure (default: flat_sessions).",
    )
    parser.add_argument(
        "--no_fallback_voice_caption_from_user",
        action="store_true",
        help=(
            "If a .wav value is a path rather than a caption, do not use the "
            "original user text as fallback user_voice_message_caption."
        ),
    )
    parser.add_argument(
        "--drop_text_user_when_no_voice",
        action="store_true",
        help="If a turn has no voice message, do not keep the original user text.",
    )
    parser.add_argument(
        "--add_caption_source",
        action="store_true",
        help="Add voice_caption_source fields for debugging.",
    )
    parser.add_argument(
        "--base_dir",
        default=None,
        help=(
            "Resolve relative media paths against this directory; use 'auto' "
            "for the historical input-parent behavior."
        ),
    )

    # Options retained from the former category-specific workflow.
    parser.add_argument(
        "--categories",
        default=",".join(DEFAULT_CATEGORIES),
        help="Category mode: comma-separated categories (default: brief,medium,detailed).",
    )
    parser.add_argument(
        "--original",
        default=None,
        help=f"Category mode: audio-caption source (default: {DEFAULT_ORIGINAL_FILE}).",
    )
    parser.add_argument(
        "--category_dir",
        default=None,
        help=f"Category mode: root containing category subdirectories (default: {DEFAULT_CATEGORY_DIR}).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=f"Category mode: output directory (default: {BENCHMARK_RUN_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Category mode: print statistics without writing output files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "single":
        if not args.input or not args.output:
            raise SystemExit("single mode requires --input and --output")
        if args.dry_run:
            raise SystemExit("--dry-run is only valid with --mode category")
        run_single(args)
        return

    if args.input or args.output:
        raise SystemExit("--input/--output are only valid with --mode single")
    if args.output_mode != "flat_sessions":
        raise SystemExit("category mode always writes the historical flat_sessions format")
    run_by_category(args)


if __name__ == "__main__":
    main()
