"""将检索到的 memory turns 格式化为 LLM 输入 (Text-Use / MM-Use)。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from .data_loader import build_text_for_turn_use, resolve_media_path


def _encode_file_base64(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        suffix.lstrip("."), "image/png"
    )


# ──────────────── Text-Use ────────────────

def format_memory_text(retrieved: list[dict]) -> str:
    """Text-Use: 将检索到的 turns 格式化为纯文本 memory。

    返回类似:
        [Memory Start]
        --- Memory 1 (Session D15, 03/20/2025) ---
        [语音消息] 人声：...
        助手: ...
        --- Memory 2 ...
        [Memory End]
    """
    if not retrieved:
        return "[Memory Start]\n（无相关记忆）\n[Memory End]"

    parts = ["[Memory Start]"]
    for i, item in enumerate(retrieved, 1):
        turn = item["turn_data"]
        sid = turn["session_id"]
        header = f"--- Memory {i} (Session {sid}) ---"
        body = build_text_for_turn_use(turn)
        parts.append(header)
        parts.append(body)
    parts.append("[Memory End]")
    return "\n".join(parts)


def format_question_text(qa: dict) -> str:
    """Text-Use: 格式化 QA 问题为纯文本。

    对图片类 QA，将 option_captions 附加到 question 后面。
    """
    question = qa.get("question", "")
    point = qa.get("point", "")

    if point.endswith("_img"):
        captions = qa.get("option_captions") or qa.get("question_image_descriptions")
        if captions and isinstance(captions, dict):
            question += "\n选项图片描述："
            for letter in sorted(captions):
                question += f"\n{letter}. {captions[letter]}"

    return question


# ──────────────── MM-Use ────────────────

def format_memory_multimodal(retrieved: list[dict],
                              max_audio: int = 3,
                              data_dir: Path | None = None) -> dict[str, Any]:
    """MM-Use: 将检索到的 turns 格式化为多模态 content blocks。

    返回: {
        'content_blocks': list[dict],   OpenAI-style content blocks
        'audio_inputs': list[dict],     独立的音频输入 (path + turn 信息)
    }

    max_audio: 最多传入的原始音频文件数量，超出的改用 voice_caption。
    """
    if not retrieved:
        return {
            "content_blocks": [{"type": "text", "text": "[Memory Start]\n（无相关记忆）\n[Memory End]"}],
            "audio_inputs": [],
        }

    blocks: list[dict] = [{"type": "text", "text": "[Memory Start]"}]
    audio_inputs: list[dict] = []
    audio_count = 0

    for i, item in enumerate(retrieved, 1):
        turn = item["turn_data"]
        sid = turn["session_id"]

        header = f"\n--- Memory {i} (Session {sid}) ---\n"

        # 文本部分
        text_parts = [header]
        if turn["voice_caption"]:
            # 人声内容作为文本
            vc = turn["voice_caption"]
            human_part = vc.split("\n")[0] if "\n" in vc else vc
            text_parts.append(f"用户: {human_part}")
        elif turn["user_text"]:
            text_parts.append(f"用户: {turn['user_text']}")
        if turn["assistant"]:
            text_parts.append(f"助手: {turn['assistant']}")

        blocks.append({"type": "text", "text": "\n".join(text_parts)})

        # 原始图像
        if turn["image_paths"]:
            img_path = resolve_media_path(turn["image_paths"][0], data_dir)
            if img_path and img_path.exists():
                b64 = _encode_file_base64(img_path)
                mt = _image_media_type(img_path)
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mt};base64,{b64}"},
                })
            elif turn["image_caption"]:
                blocks.append({"type": "text", "text": f"[图像描述] {turn['image_caption']}"})

        # 原始音频（限制数量）
        if turn["voice_paths"] and audio_count < max_audio:
            aud_path = resolve_media_path(turn["voice_paths"][0], data_dir)
            if aud_path and aud_path.exists():
                audio_inputs.append({
                    "path": str(aud_path),
                    "turn_id": turn["turn_id"],
                    "session_id": sid,
                })
                b64 = _encode_file_base64(aud_path)
                blocks.append({
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{b64}"},
                })
                audio_count += 1
            elif turn["voice_caption"]:
                blocks.append({"type": "text", "text": f"[语音描述] {turn['voice_caption']}"})
        elif turn["voice_paths"] and turn["voice_caption"]:
            blocks.append({"type": "text", "text": f"[语音描述] {turn['voice_caption']}"})

    blocks.append({"type": "text", "text": "\n[Memory End]"})

    return {"content_blocks": blocks, "audio_inputs": audio_inputs}


def format_question_multimodal(
    qa: dict,
    data_dir: Path | None = None,
) -> list[dict]:
    """MM-Use: 格式化 QA 问题为多模态 content blocks。

    对图片类 QA，直接传入 4 张选项图片。
    """
    question = qa.get("question", "")
    point = qa.get("point", "")
    blocks: list[dict] = [{"type": "text", "text": question}]

    if point.endswith("_img"):
        option_images = qa.get("option_images", {})
        for letter in sorted(option_images):
            img_raw = option_images[letter]
            img_path = resolve_media_path(img_raw, data_dir)
            if img_path and img_path.exists():
                b64 = _encode_file_base64(img_path)
                mt = _image_media_type(img_path)
                blocks.append({"type": "text", "text": f"\n选项 {letter}:"})
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mt};base64,{b64}"},
                })
            else:
                # fallback to caption
                captions = qa.get("option_captions") or qa.get("question_image_descriptions") or {}
                cap = captions.get(letter, "")
                if cap:
                    blocks.append({"type": "text", "text": f"\n选项 {letter}: {cap}"})

    return blocks
