"""
根据每个 task 的 image_turn_indices / tts_turn_indices，把生成好的
图片和语音文件路径注入到 dialog 中对应轮次。

输入文件实际是 JSON 数组（虽然后缀 .jsonl）。
对每个 task 的 event.dialog[i]：
  - 若 i in image_turn_indices: 写入 image_path
  - 若 i in tts_turn_indices  : 写入 audio_path
        优先  event/voice_mixed/{p_id}-{task_id}/{task_id}_turn{i}.wav
        回退  event/voice_message/{task_id}_turn{i}.wav

注意：tts_turn_indices 现在从 background_audio_manifest（MANIFEST_PATH）中读取，
      dialogue 文件中的 tts_turn_indices 字段已不再使用。

路径一律保存为「相对项目根」的 POSIX 风格相对路径（与 inspect_images.py 兼容）。

用法：
    python event/attach_assets_to_dialogue.py
    python event/attach_assets_to_dialogue.py --in-place
    python event/attach_assets_to_dialogue.py --input  event/v1_dialogue_000_001_test.jsonl ^
                                              --output event/v1_dialogue_000_001_test_with_assets.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scripts.common.io import load_json_or_jsonl
from scripts.common.paths import project_path, project_relative, resolve_path

DEFAULT_INPUT = project_path("event", "dialogue_005_019_with_anchors.jsonl")
DEFAULT_OUTPUT = project_path("event", "dialogue_005_019_with_assets.jsonl")
MANIFEST_PATH = project_path("event", "background_audio_manifest_kling.json")

IMAGES_DIR = project_path("event", "images")
VOICE_MIXED_DIR = project_path("event", "voice_mixed")
VOICE_MESSAGE_DIR = project_path("event", "voice_message")

IMAGE_FIELD = "image_path"
AUDIO_FIELD = "audio_path"
AUDIO_SOURCE_FIELD = "audio_source"  # "mixed" | "voice"


def _safe_tid(task_id: str) -> str:
    """与 inspect_images.py / gen_images_from_descriptions.py 一致的 task_id 规范化。"""
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(task_id))


def _rel_posix(p: Path) -> str:
    return project_relative(p)


def image_path_for(p_id: int, task_id: str) -> Path:
    return IMAGES_DIR / f"pid_{p_id:04d}_task_{_safe_tid(task_id)}.png"


def audio_path_for(p_id: int, task_id: str, turn: int) -> tuple[Path | None, str | None]:
    """返回 (实际存在的相对路径, 来源标签)；都不存在时返回 (None, None)。"""
    mixed = VOICE_MIXED_DIR / f"{p_id}-{task_id}" / f"{task_id}_turn{turn}.wav"
    if mixed.exists():
        return mixed, "mixed"
    voice = VOICE_MESSAGE_DIR / f"{task_id}_turn{turn}.wav"
    if voice.exists():
        return voice, "voice"
    return None, None


def attach_to_task(task: dict[str, Any], stats: dict[str, int], warnings: list[str],
                   manifest_tts: dict[str, set[int]]) -> None:
    p_id = task.get("p_id")
    task_id = task.get("task_id")
    event = task.get("event") or {}
    dialog = event.get("dialog")

    if p_id is None or task_id is None or not isinstance(dialog, list):
        warnings.append(f"[skip] task 缺失 p_id/task_id/dialog: p_id={p_id}, task_id={task_id}")
        stats["task_skipped"] += 1
        return

    img_idx = set(event.get("image_turn_indices") or [])
    # tts_turn_indices 从 manifest 中读取（dialogue 文件里的字段已废弃）
    tts_idx = manifest_tts.get(str(task_id), set())
    n = len(dialog)

    img_rel: str | None = None
    if img_idx:
        img_path = image_path_for(p_id, task_id)
        if img_path.exists():
            img_rel = _rel_posix(img_path)
        else:
            warnings.append(
                f"[image-missing] p_id={p_id} task_id={task_id}: 期望 {img_path}"
            )

    for i in img_idx:
        if not (0 <= i < n):
            warnings.append(
                f"[image-oob] p_id={p_id} task_id={task_id} turn={i} 超出 dialog 长度 {n}"
            )
            stats["image_oob"] += 1
            continue
        if not isinstance(dialog[i], dict):
            warnings.append(
                f"[image-bad-turn] p_id={p_id} task_id={task_id} turn={i} 不是 dict"
            )
            stats["turn_bad"] += 1
            continue
        if img_rel is not None:
            dialog[i][IMAGE_FIELD] = img_rel
            stats["image_attached"] += 1
        else:
            stats["image_missing"] += 1

    for i in tts_idx:
        if not (0 <= i < n):
            warnings.append(
                f"[audio-oob] p_id={p_id} task_id={task_id} turn={i} 超出 dialog 长度 {n}"
            )
            stats["audio_oob"] += 1
            continue
        if not isinstance(dialog[i], dict):
            warnings.append(
                f"[audio-bad-turn] p_id={p_id} task_id={task_id} turn={i} 不是 dict"
            )
            stats["turn_bad"] += 1
            continue
        wav, source = audio_path_for(p_id, task_id, i)
        if wav is None:
            warnings.append(
                f"[audio-missing] p_id={p_id} task_id={task_id} turn={i}: "
                f"voice_mixed 与 voice_message 均不存在"
            )
            stats["audio_missing"] += 1
            continue
        dialog[i][AUDIO_FIELD] = _rel_posix(wav)
        dialog[i][AUDIO_SOURCE_FIELD] = source
        if source == "mixed":
            stats["audio_attached_mixed"] += 1
        else:
            stats["audio_attached_voice"] += 1

    stats["task_processed"] += 1


def main() -> None:
    global MANIFEST_PATH, IMAGES_DIR, VOICE_MIXED_DIR, VOICE_MESSAGE_DIR
    ap = argparse.ArgumentParser(description="把图片/语音路径注入对话轮次。")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="输入 JSON 文件")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT), help="输出 JSON 文件")
    ap.add_argument("--manifest", default=str(MANIFEST_PATH), help="音频 manifest 文件")
    ap.add_argument("--images-dir", default=str(IMAGES_DIR))
    ap.add_argument("--voice-mixed-dir", default=str(VOICE_MIXED_DIR))
    ap.add_argument("--voice-message-dir", default=str(VOICE_MESSAGE_DIR))
    ap.add_argument("--in-place", action="store_true", help="直接覆盖输入文件（忽略 --output）")
    ap.add_argument("--verbose", action="store_true", help="打印所有 warning 明细")
    args = ap.parse_args()

    in_path = resolve_path(args.input)
    out_path = in_path if args.in_place else resolve_path(args.output)
    MANIFEST_PATH = resolve_path(args.manifest)
    IMAGES_DIR = resolve_path(args.images_dir)
    VOICE_MIXED_DIR = resolve_path(args.voice_mixed_dir)
    VOICE_MESSAGE_DIR = resolve_path(args.voice_message_dir)

    if not in_path.exists():
        raise SystemExit(f"输入文件不存在: {in_path}")

    data = load_json_or_jsonl(in_path)
    if not isinstance(data, list):
        raise SystemExit(f"期望文件根是数组，实际是 {type(data).__name__}: {in_path}")

    # ── 加载 manifest，构建 task_id → tts_turn 集合 ──────────────────────────
    manifest_tts: dict[str, set[int]] = {}
    if MANIFEST_PATH.exists():
        for entry in json.loads(MANIFEST_PATH.read_text(encoding="utf-8")):
            tid = str(entry.get("task_id", ""))
            if not tid:
                continue
            turns: set[int] = set()
            for bg in (entry.get("background_audio_path") or []):
                for idx in (bg.get("tts_turn_indices") or []):
                    turns.add(int(idx))
            if turns:
                manifest_tts[tid] = turns
    else:
        print(f"[warn] manifest 文件不存在，音频将全部跳过: {MANIFEST_PATH}")

    stats = {
        "task_processed": 0,
        "task_skipped": 0,
        "image_attached": 0,
        "image_missing": 0,
        "image_oob": 0,
        "audio_attached_mixed": 0,
        "audio_attached_voice": 0,
        "audio_missing": 0,
        "audio_oob": 0,
        "turn_bad": 0,
    }
    warnings: list[str] = []

    for task in data:
        if isinstance(task, dict):
            attach_to_task(task, stats, warnings, manifest_tts)
        else:
            warnings.append(f"[skip] 顶层元素不是 dict: {type(task).__name__}")
            stats["task_skipped"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    print(f"输入:  {in_path}")
    print(f"输出:  {out_path}")
    print(f"任务:  处理 {stats['task_processed']} | 跳过 {stats['task_skipped']}")
    print(
        f"图片:  注入 {stats['image_attached']} | 缺文件 {stats['image_missing']} | 越界 {stats['image_oob']}"
    )
    print(
        f"音频:  注入 mixed={stats['audio_attached_mixed']} voice={stats['audio_attached_voice']}"
        f" | 缺文件 {stats['audio_missing']} | 越界 {stats['audio_oob']}"
    )
    if stats["turn_bad"]:
        print(f"非法 turn: {stats['turn_bad']}")

    if warnings:
        print(f"\n共 {len(warnings)} 条警告。", end=" ")
        if args.verbose:
            print("详情：")
            for w in warnings:
                print("  -", w)
        else:
            print("仅展示前 10 条（加 --verbose 查看全部）：")
            for w in warnings[:10]:
                print("  -", w)


if __name__ == "__main__":
    main()
