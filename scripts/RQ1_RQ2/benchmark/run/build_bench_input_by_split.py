"""
Build benchmark input for split voice/background-audio captions.

The output keeps QA items from data/dialog/base unchanged, and only patches
multi_session_dialogues:
  - voice_caption = speech caption + background audio caption
  - image_caption = image captions from qa_formatted_data_000_002_with_media_captions.json

Usage:
    cd <repository-root>
    python build_bench_input_by_split.py --dry-run
    python build_bench_input_by_split.py
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.paths import DATA_ROOT, DIALOG_ROOT, QA_ROOT


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = DATA_ROOT.parent
PROJECT_ROOT = BENCHMARK_ROOT.parent

DEFAULT_BASE_DIR = DIALOG_ROOT / "base"
DEFAULT_OUTPUT_DIR = (
    DIALOG_ROOT / "audio_caption" / "voice_bgm_split"
)
DEFAULT_ASR_DATA = (
    QA_ROOT
    / "qwen3_asr_1.7b"
    / "qa_formatted_data_000_002_with_audio_captions_qwen3_asr.json"
)
DEFAULT_BGM_CAPTIONS = (
    QA_ROOT
    / "qwen3_asr_1.7b"
    / "background_audio_captions_gemini-3.1-pro.json"
)
DEFAULT_MEDIA_CAPTIONS = QA_ROOT / "qa_formatted_data_000_002_with_media_captions.json"

NUM_PROFILES = 3


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_bgm_captions(raw) -> dict[str, dict]:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for name, value in raw.items():
        if isinstance(value, str):
            out[name] = {"caption": value}
        elif isinstance(value, dict):
            out[name] = value
    return out


def build_image_caption_map(media_data: list) -> dict[tuple[int, str], dict[str, str]]:
    """Return {(p_id, session_id): {image_id: caption}}."""
    cap_map: dict[tuple[int, str], dict[str, str]] = {}
    for profile in media_data:
        pid = profile.get("p_id", -1)
        for event in profile.get("events", []) or []:
            sid = event.get("session_id", "")
            if not sid:
                continue
            for turn in event.get("dialog_list", []) or []:
                for key, value in turn.items():
                    if not isinstance(key, str) or not key.endswith(".png"):
                        continue
                    if not isinstance(value, str) or not value.strip():
                        continue
                    # In the media-caption file, .png values should be captions, not paths.
                    if value.endswith(".png") or "/" in value or "\\" in value:
                        continue
                    image_id = key[:-4]
                    cap_map.setdefault((pid, sid), {})[image_id] = value.strip()
    return cap_map


def build_voice_path_index(asr_data: list) -> dict[str, tuple[int, str, str]]:
    """Return {audio_path: (p_id, session_id, voice_id)} from dialog_list .wav keys."""
    path_index: dict[str, tuple[int, str, str]] = {}
    for profile in asr_data:
        pid = profile.get("p_id", -1)
        for event in profile.get("events", []) or []:
            sid = event.get("session_id", "")
            for turn in event.get("dialog_list", []) or []:
                for key, value in turn.items():
                    if not isinstance(key, str) or not key.endswith(".wav"):
                        continue
                    if not isinstance(value, str) or not value.strip():
                        continue
                    voice_id = key[:-4]
                    path_index[value.strip()] = (pid, sid, voice_id)
    return path_index


def build_voice_caption_map(
    asr_data: list,
    caption_field: str,
) -> dict[tuple[int, str], dict[str, dict]]:
    """Return {(p_id, session_id): {voice_id: {speech_caption, bg_name}}}."""
    path_index = build_voice_path_index(asr_data)
    cap_map: dict[tuple[int, str], dict[str, dict]] = {}

    for profile in asr_data:
        for event in profile.get("events", []) or []:
            for turn in event.get("dialog", []) or []:
                audio_path = (turn.get("audio_path") or "").strip()
                caption = (turn.get(caption_field) or "").strip()
                bg_name = (turn.get("background_audio") or "").strip()
                if not audio_path or audio_path not in path_index:
                    continue
                pid, sid, voice_id = path_index[audio_path]
                cap_map.setdefault((pid, sid), {})[voice_id] = {
                    "speech_caption": caption,
                    "background_audio": bg_name,
                    "audio_path": audio_path,
                }

    return cap_map


def compose_voice_caption(speech_caption: str, bg_name: str, bg_caption: str) -> str:
    speech = (speech_caption or "").strip()
    bg_name = (bg_name or "").strip()
    bg_caption = (bg_caption or "").strip()

    speech_line = f"人声：{speech}" if speech else "人声：无可识别人声。"
    if bg_caption:
        bg_line = f"背景音：{bg_caption}"
    elif bg_name and bg_name.lower() != "none":
        bg_line = f"背景音：{bg_name}"
    else:
        bg_line = "背景音：无明确背景音。"
    return f"{speech_line}\n{bg_line}"


def patch_history(
    history: dict,
    *,
    p_id: int,
    voice_map: dict[tuple[int, str], dict[str, dict]],
    bgm_captions: dict[str, dict],
    image_map: dict[tuple[int, str], dict[str, str]],
) -> tuple[dict, Counter]:
    patched = copy.deepcopy(history)
    stats = Counter()

    for session in patched.get("multi_session_dialogues", []) or []:
        sid = session.get("session_id", "")
        session_key = (p_id, sid)
        session_voice = voice_map.get(session_key, {})
        session_images = image_map.get(session_key, {})

        for turn in session.get("dialogues", []) or []:
            voice_ids = turn.get("voice_id", []) or []
            if voice_ids:
                stats["voice_turns"] += 1
                original_caps = turn.get("voice_caption", []) or []
                new_caps = []
                any_voice_patch = False
                for idx, voice_id in enumerate(voice_ids):
                    meta = session_voice.get(voice_id)
                    if meta:
                        bg_name = (meta.get("background_audio") or "").strip()
                        bg_cap = (bgm_captions.get(bg_name, {}).get("caption") or "").strip()
                        new_caps.append(
                            compose_voice_caption(
                                meta.get("speech_caption", ""),
                                bg_name,
                                bg_cap,
                            )
                        )
                        if bg_name and bg_name.lower() != "none":
                            turn["background_audio"] = bg_name
                        any_voice_patch = True
                        stats["voice_patched"] += 1
                        if bg_cap:
                            stats["bgm_caption_used"] += 1
                        elif bg_name and bg_name.lower() != "none":
                            stats["bgm_name_fallback"] += 1
                    else:
                        new_caps.append(original_caps[idx] if idx < len(original_caps) else "")
                        stats["voice_missing"] += 1
                if any_voice_patch:
                    turn["voice_caption"] = new_caps
                    turn["user_voice_message_caption"] = new_caps[0] if new_caps else ""

            image_ids = turn.get("image_id", []) or []
            if image_ids:
                stats["image_turns"] += 1
                original_img_caps = turn.get("image_caption", []) or []
                new_img_caps = []
                any_image_patch = False
                for idx, image_id in enumerate(image_ids):
                    cap = session_images.get(image_id)
                    if cap:
                        new_img_caps.append(cap)
                        any_image_patch = True
                        stats["image_patched"] += 1
                    else:
                        new_img_caps.append(
                            original_img_caps[idx] if idx < len(original_img_caps) else ""
                        )
                        stats["image_missing"] += 1
                if any_image_patch:
                    turn["image_caption"] = new_img_caps

    return patched, stats


def load_base_histories(base_dir: Path) -> dict[int, dict]:
    histories: dict[int, dict] = {}
    for p_id in range(NUM_PROFILES):
        path = base_dir / f"history_with_qa_p{p_id}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        histories[p_id] = load_json(path)
    return histories


def qa_count_summary(history: dict) -> Counter:
    return Counter(qa.get("point", "unknown") for qa in history.get("human-annotated QAs", []))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build benchmark input with split voice/background-audio captions."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--asr-data", type=Path, default=DEFAULT_ASR_DATA)
    parser.add_argument("--bgm-captions", type=Path, default=DEFAULT_BGM_CAPTIONS)
    parser.add_argument("--media-captions", type=Path, default=DEFAULT_MEDIA_CAPTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--caption-field", default="audio_caption_qwen3_asr")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("Loading source files...")
    base_histories = load_base_histories(args.base_dir)
    asr_data = load_json(args.asr_data)
    media_data = load_json(args.media_captions)

    if args.bgm_captions.exists():
        bgm_captions = normalize_bgm_captions(load_json(args.bgm_captions))
    else:
        print(f"WARN: background caption file not found: {args.bgm_captions}")
        print("      The script will fall back to raw background_audio names.")
        bgm_captions = {}

    voice_map = build_voice_caption_map(asr_data, args.caption_field)
    image_map = build_image_caption_map(media_data)

    print(f"Base histories      : {len(base_histories)} profiles")
    print(f"Voice caption keys  : {sum(len(v) for v in voice_map.values())} turns")
    print(f"Image caption keys  : {sum(len(v) for v in image_map.values())} turns")
    print(f"BGM captions        : {len(bgm_captions)} names")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    total_stats = Counter()
    for p_id in range(NUM_PROFILES):
        patched, stats = patch_history(
            base_histories[p_id],
            p_id=p_id,
            voice_map=voice_map,
            bgm_captions=bgm_captions,
            image_map=image_map,
        )
        total_stats.update(stats)

        out_path = args.output_dir / f"history_with_qa_p{p_id}.json"
        qas = patched.get("human-annotated QAs", [])
        point_counts = qa_count_summary(patched)

        print(f"\n[p_id={p_id}]")
        print(f"  sessions             : {len(patched.get('multi_session_dialogues', []))}")
        print(f"  QA total             : {len(qas)}")
        for point, count in sorted(point_counts.items()):
            print(f"    {point:18s}: {count}")
        print(f"  voice turns          : {stats['voice_turns']}")
        print(f"  voice captions patched: {stats['voice_patched']}")
        print(f"  bgm captions used    : {stats['bgm_caption_used']}")
        print(f"  bgm name fallback    : {stats['bgm_name_fallback']}")
        print(f"  image turns          : {stats['image_turns']}")
        print(f"  image captions patched: {stats['image_patched']}")

        if args.dry_run:
            print(f"  dry-run output       : {out_path}")
        else:
            write_json(out_path, patched)
            print(f"  output               : {out_path}")

    print("\nTotal patch stats:")
    for key in sorted(total_stats):
        print(f"  {key:22s}: {total_stats[key]}")
    if not args.dry_run:
        print(f"\nDone. Files written to {args.output_dir}")


if __name__ == "__main__":
    main()
