"""加载 history_with_qa_p{N}.json，解析为结构化数据。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config


def resolve_media_path(raw_path: str, base_dir: Path | None = None) -> Path | None:
    """将数据文件中的相对路径解析为实际绝对路径。

    典型输入:
      'event/voice_mixed_000_002/0-0-35-0/0-35-0_turn0.wav'
      'event/images/pid_0000_task_0-12-0.png'
      'qa\\pref_images\\0-FoodAndDrink-0_C.png'
    """
    if not raw_path:
        return None
    if base_dir is None:
        base_dir = config.DATA_DIR
    base_dir = Path(base_dir).expanduser().resolve()

    raw_path = raw_path.replace("\\", "/")
    raw_path = raw_path.removeprefix("./")

    # event/voice_mixed_000_002/... → data/voice_mixed_000_002/...
    if raw_path.startswith("event/voice_mixed_000_002/"):
        candidate = base_dir / raw_path.removeprefix("event/")
        if candidate.exists():
            return candidate.resolve()

    # event/images/pid_... → data/event_image/pid_...
    if raw_path.startswith("event/images/"):
        candidate = base_dir / "event_image" / raw_path.removeprefix("event/images/")
        if candidate.exists():
            return candidate.resolve()

    # qa/pref_images/... → data/qa_image/pref_images/...
    if raw_path.startswith("qa/"):
        candidate = base_dir / "qa_image" / raw_path.removeprefix("qa/")
        if candidate.exists():
            return candidate.resolve()

    # 直接在 base_dir 下查找
    candidate = base_dir / raw_path
    if candidate.exists():
        return candidate.resolve()

    # 原始路径本身
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p.resolve()

    return None


def _parse_turn(turn: dict[str, Any], session_id: str) -> dict[str, Any]:
    """将原始 dialogue round 解析为标准化的 turn dict。"""
    voice_captions = turn.get("voice_caption") or []
    voice_paths = turn.get("input_voice_message") or []
    voice_ids = turn.get("voice_id") or []
    image_captions = turn.get("image_caption") or []
    image_paths = turn.get("input_image") or []
    image_ids = turn.get("image_id") or []

    voice_caption_str = turn.get("user_voice_message_caption") or ""
    if not voice_caption_str and voice_captions:
        voice_caption_str = voice_captions[0]

    image_caption_str = image_captions[0] if image_captions else ""

    user_text = turn.get("user") or ""
    assistant_text = turn.get("assistant") or ""

    return {
        "turn_id": turn.get("round", ""),
        "session_id": session_id,
        "user_text": user_text,
        "assistant": assistant_text,
        "voice_caption": voice_caption_str or None,
        "voice_paths": voice_paths,
        "voice_ids": voice_ids,
        "image_caption": image_caption_str or None,
        "image_paths": image_paths,
        "image_ids": image_ids,
        "background_audio": turn.get("background_audio") or None,
    }


def load_profile(json_path: Path) -> dict[str, Any]:
    """加载一个 profile 的完整数据。

    返回:
        {
            'profile': dict,
            'sessions': list[dict],   每个 session 含 turns
            'qas': list[dict],        QA 列表
            'p_id': int,
        }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    profile = raw.get("character_profile", {})
    raw_sessions = raw.get("multi_session_dialogues", [])
    qas = raw.get("human-annotated QAs", [])

    sessions = []
    for sess in raw_sessions:
        session_id = sess.get("session_id", "")
        turns = []
        for dlg in sess.get("dialogues", []):
            turns.append(_parse_turn(dlg, session_id))
        sessions.append({
            "session_id": session_id,
            "date": sess.get("date", ""),
            "turns": turns,
            "scene_description": sess.get("scene_description", ""),
            "explicit_preferences": sess.get("explicit_preferences", []),
            "implicit_preferences": sess.get("implicit_preferences", []),
            "task_id": sess.get("task_id", ""),
        })

    # 从文件名推断 p_id
    stem = json_path.stem
    p_id = int(stem.split("p")[-1]) if "p" in stem else 0

    return {
        "profile": profile,
        "sessions": sessions,
        "qas": qas,
        "p_id": p_id,
    }


def flatten_turns(sessions: list[dict]) -> list[dict]:
    """将所有 session 的 turn 展平为一个列表。"""
    return [turn for sess in sessions for turn in sess["turns"]]


def build_turn_registry(sessions: list[dict]) -> dict[str, dict]:
    """turn_id → turn_data 的快速查找表。"""
    registry: dict[str, dict] = {}
    for sess in sessions:
        for turn in sess["turns"]:
            registry[turn["turn_id"]] = turn
    return registry


def build_voice_id_to_turn_id(sessions: list[dict]) -> dict[str, str]:
    """voice_id → turn_id 的映射。

    用于将 clue 中的 'D15-001.wav' 映射回 turn。
    """
    mapping: dict[str, str] = {}
    for sess in sessions:
        for turn in sess["turns"]:
            for vid in turn["voice_ids"]:
                mapping[vid] = turn["turn_id"]
    return mapping


def build_image_id_to_turn_id(sessions: list[dict]) -> dict[str, str]:
    """image_id → turn_id 的映射。"""
    mapping: dict[str, str] = {}
    for sess in sessions:
        for turn in sess["turns"]:
            for iid in turn["image_ids"]:
                mapping[iid] = turn["turn_id"]
    return mapping


def clue_to_turn_ids(clue_list: list[str], sessions: list[dict]) -> set[str]:
    """将 QA 的 clue 列表转换为 turn_id 集合。

    clue 格式混合了两种:
      'D15:00'       → 直接是 turn_id (round)
      'D15-001.wav'  → voice_id + .wav 后缀
      'D01-001.png'  → image_id + .png 后缀 (如果有的话)
    """
    turn_reg = build_turn_registry(sessions)
    vid_map = build_voice_id_to_turn_id(sessions)
    iid_map = build_image_id_to_turn_id(sessions)

    result: set[str] = set()
    for clue in clue_list:
        if clue in turn_reg:
            result.add(clue)
        elif clue.endswith(".wav"):
            vid = clue[:-4]
            if vid in vid_map:
                result.add(vid_map[vid])
        elif clue.endswith(".png"):
            iid = clue[:-4]
            if iid in iid_map:
                result.add(iid_map[iid])
    return result


def build_text_for_turn(turn: dict) -> str:
    """将一个 turn 的所有文本信息拼接为一段文本。

    用于 Text-Index 编码。
    包含: voice_caption (或 user_text) + assistant + image_caption
    """
    parts = []

    if turn["voice_caption"]:
        parts.append(f"用户语音: {turn['voice_caption']}")
    elif turn["user_text"]:
        parts.append(f"用户: {turn['user_text']}")

    if turn["assistant"]:
        parts.append(f"助手: {turn['assistant']}")

    if turn["image_caption"]:
        parts.append(f"图像: {turn['image_caption']}")

    return "\n".join(parts)


def build_text_for_turn_use(turn: dict) -> str:
    """将一个 turn 格式化为 Text-Use 的记忆文本。

    与 build_text_for_turn 类似，但格式更适合作为 LLM memory 输入。
    """
    parts = []

    if turn["voice_caption"]:
        parts.append(f"[语音消息] {turn['voice_caption']}")
    elif turn["user_text"]:
        parts.append(f"用户: {turn['user_text']}")

    if turn["assistant"]:
        parts.append(f"助手: {turn['assistant']}")

    if turn["image_caption"]:
        parts.append(f"[用户分享图片] {turn['image_caption']}")

    return "\n".join(parts)
