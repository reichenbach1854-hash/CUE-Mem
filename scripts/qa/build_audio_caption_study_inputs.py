"""
Build benchmark inputs for the audio-caption quality study.

This script keeps the QA set and session structure from the current benchmark
base directory, then creates parallel datasets that differ only in voice/audio
captions:

  asr          : speech-only caption from qwen3-asr-1.7b.
  hint         : current mixed-audio caption from data/dialog/base.
  asr_bg_split : speech-only caption + separately generated background caption.

Image captions are patched from medium_history_dialogue.json for all modes so
the image side remains fixed at medium quality across the study.

Usage:
    cd <repository-root>
    python -m scripts.qa.build_audio_caption_study_inputs --dry-run
    python -m scripts.qa.build_audio_caption_study_inputs --modes asr hint asr_bg_split --num-profiles 5
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.RQ1_RQ2.benchmark.paths import DIALOG_ROOT, QA_ROOT

DEFAULT_BASE_DIR = DIALOG_ROOT / "base"
DEFAULT_OUTPUT_ROOT = DIALOG_ROOT / "audio_caption"
DEFAULT_ASR_DATA = (
    QA_ROOT
    / "qwen3_asr_1.7b"
    / "qa_formatted_data_with_audio_captions_qwen3_asr.json"
)
DEFAULT_BGM_CAPTIONS = (
    QA_ROOT
    / "qwen3_asr_1.7b"
    / "background_audio_captions_gemini-3.1-pro.json"
)
DEFAULT_MEDIA_CAPTIONS = QA_ROOT / "qa_formatted_data_000_002_with_media_captions.json"

VALID_MODES = ("asr", "hint", "asr_bg_split")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_profile_ids(text: str | None) -> set[int] | None:
    if not text:
        return None
    out: set[int] = set()
    for part in re.split(r"[,\s]+", text.strip()):
        if part:
            out.add(int(part))
    return out


def normalize_bgm_captions(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name, value in raw.items():
        if isinstance(value, str):
            out[str(name)] = {"caption": value}
        elif isinstance(value, dict):
            out[str(name)] = value
    return out


def discover_base_histories(
    base_dir: Path,
    *,
    num_profiles: int | None,
    only_profile_ids: set[int] | None,
) -> dict[int, dict[str, Any]]:
    histories: dict[int, dict[str, Any]] = {}
    pattern = re.compile(r"history_with_qa_p(\d+)\.json$")

    for path in sorted(base_dir.glob("history_with_qa_p*.json")):
        match = pattern.match(path.name)
        if not match:
            continue
        p_id = int(match.group(1))
        if only_profile_ids is not None and p_id not in only_profile_ids:
            continue
        if num_profiles is not None and p_id >= num_profiles:
            continue
        histories[p_id] = load_json(path)

    if not histories:
        raise FileNotFoundError(f"No history_with_qa_p*.json files found in {base_dir}")
    return histories


def build_image_caption_map(media_data: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, str]]:
    """Return {(p_id, session_id): {image_id: caption}} from media-caption data.

    Supports both:
    - flat history_dialogue sessions, e.g. medium_history_dialogue.json;
    - old formatted profile/events/dialog_list files.
    """
    cap_map: dict[tuple[int, str], dict[str, str]] = {}
    if media_data and isinstance(media_data[0], dict) and "dialogues" in media_data[0]:
        for session in media_data:
            p_id = session.get("_p_id")
            if p_id is None:
                continue
            p_id = int(p_id)
            session_id = str(session.get("session_id") or "")
            if not session_id:
                continue
            for turn in session.get("dialogues", []) or []:
                if not isinstance(turn, dict):
                    continue
                image_ids = turn.get("image_id", []) or []
                image_caps = turn.get("image_caption", []) or []
                if isinstance(image_ids, str):
                    image_ids = [image_ids]
                if isinstance(image_caps, str):
                    image_caps = [image_caps]
                for idx, image_id in enumerate(image_ids):
                    caption = image_caps[idx] if idx < len(image_caps) else ""
                    if isinstance(caption, str) and caption.strip():
                        cap_map.setdefault((p_id, session_id), {})[str(image_id)] = caption.strip()
        return cap_map

    for profile in media_data:
        p_id = int(profile.get("p_id", -1))
        for event in profile.get("events", []) or []:
            session_id = str(event.get("session_id") or "")
            if not session_id:
                continue
            for turn in event.get("dialog_list", []) or []:
                if not isinstance(turn, dict):
                    continue
                for key, value in turn.items():
                    if not isinstance(key, str) or not key.endswith(".png"):
                        continue
                    if not isinstance(value, str) or not value.strip():
                        continue
                    # Caption files store captions in .png slots; raw paths are not useful here.
                    if value.endswith(".png") or "/" in value or "\\" in value:
                        continue
                    cap_map.setdefault((p_id, session_id), {})[key[:-4]] = value.strip()
    return cap_map


def build_voice_path_index(asr_data: list[dict[str, Any]]) -> dict[str, tuple[int, str, str]]:
    """Return {audio_path: (p_id, session_id, voice_id)} from dialog_list .wav slots."""
    path_index: dict[str, tuple[int, str, str]] = {}
    for profile in asr_data:
        p_id = int(profile.get("p_id", -1))
        for event in profile.get("events", []) or []:
            session_id = str(event.get("session_id") or "")
            for turn in event.get("dialog_list", []) or []:
                if not isinstance(turn, dict):
                    continue
                for key, value in turn.items():
                    if not isinstance(key, str) or not key.endswith(".wav"):
                        continue
                    if not isinstance(value, str) or not value.strip():
                        continue
                    path_index[value.strip()] = (p_id, session_id, key[:-4])
    return path_index


def build_voice_caption_map(
    asr_data: list[dict[str, Any]],
    caption_field: str,
) -> dict[tuple[int, str], dict[str, dict[str, str]]]:
    """Return {(p_id, session_id): {voice_id: {speech_caption, background_audio}}}."""
    path_index = build_voice_path_index(asr_data)
    cap_map: dict[tuple[int, str], dict[str, dict[str, str]]] = {}

    for profile in asr_data:
        for event in profile.get("events", []) or []:
            for turn in event.get("dialog", []) or []:
                if not isinstance(turn, dict):
                    continue
                audio_path = str(turn.get("audio_path") or "").strip()
                if not audio_path or audio_path not in path_index:
                    continue
                p_id, session_id, voice_id = path_index[audio_path]
                cap_map.setdefault((p_id, session_id), {})[voice_id] = {
                    "speech_caption": str(turn.get(caption_field) or "").strip(),
                    "background_audio": str(turn.get("background_audio") or "").strip(),
                    "audio_path": audio_path,
                }
    return cap_map


def compose_split_caption(speech_caption: str, bg_name: str, bg_caption: str) -> str:
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


def patch_history_for_mode(
    base_history: dict[str, Any],
    *,
    p_id: int,
    mode: str,
    asr_data_source: Path,
    bgm_caption_source: Path,
    media_caption_source: Path,
    voice_map: dict[tuple[int, str], dict[str, dict[str, str]]],
    bgm_captions: dict[str, dict[str, Any]],
    image_map: dict[tuple[int, str], dict[str, str]],
) -> tuple[dict[str, Any], Counter]:
    patched = copy.deepcopy(base_history)
    stats: Counter = Counter()

    patched.setdefault("_audio_caption_study", {})
    patched["_audio_caption_study"] = {
        "mode": mode,
        "asr_data_source": str(asr_data_source),
        "background_audio_caption_source": str(bgm_caption_source),
        "image_caption_source": str(media_caption_source),
    }

    for session in patched.get("multi_session_dialogues", []) or []:
        session_id = str(session.get("session_id") or "")
        session_key = (p_id, session_id)
        session_voice = voice_map.get(session_key, {})
        session_images = image_map.get(session_key, {})

        for turn in session.get("dialogues", []) or []:
            if not isinstance(turn, dict):
                continue

            voice_ids = turn.get("voice_id", []) or []
            if voice_ids:
                stats["voice_turns"] += 1
                original_caps = turn.get("voice_caption", []) or []
                new_caps: list[str] = []
                patched_voice = False

                for idx, voice_id in enumerate(voice_ids):
                    meta = session_voice.get(voice_id)
                    if mode == "hint":
                        new_caps.append(
                            original_caps[idx] if idx < len(original_caps) else ""
                        )
                        continue

                    if not meta:
                        new_caps.append(
                            original_caps[idx] if idx < len(original_caps) else ""
                        )
                        stats["voice_missing"] += 1
                        continue

                    speech_caption = meta.get("speech_caption", "")
                    bg_name = meta.get("background_audio", "")

                    if mode == "asr":
                        new_caps.append(speech_caption)
                        stats["asr_caption_used"] += 1
                    elif mode == "asr_bg_split":
                        bg_caption = str(
                            bgm_captions.get(bg_name, {}).get("caption") or ""
                        ).strip()
                        new_caps.append(compose_split_caption(speech_caption, bg_name, bg_caption))
                        if bg_caption:
                            stats["bgm_caption_used"] += 1
                        elif bg_name and bg_name.lower() != "none":
                            stats["bgm_name_fallback"] += 1
                        else:
                            stats["bgm_none"] += 1
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")

                    if bg_name and bg_name.lower() != "none":
                        turn["background_audio"] = bg_name
                    patched_voice = True
                    stats["voice_patched"] += 1

                if patched_voice or mode == "hint":
                    turn["voice_caption"] = new_caps
                    turn["user_voice_message_caption"] = new_caps[0] if new_caps else ""

            image_ids = turn.get("image_id", []) or []
            if image_ids:
                stats["image_turns"] += 1
                original_img_caps = turn.get("image_caption", []) or []
                new_img_caps: list[str] = []
                patched_image = False

                for idx, image_id in enumerate(image_ids):
                    caption = session_images.get(image_id)
                    if caption:
                        new_img_caps.append(caption)
                        patched_image = True
                        stats["image_patched"] += 1
                    else:
                        new_img_caps.append(
                            original_img_caps[idx] if idx < len(original_img_caps) else ""
                        )
                        stats["image_missing"] += 1

                if patched_image:
                    turn["image_caption"] = new_img_caps

    return patched, stats


def qa_count_summary(history: dict[str, Any]) -> Counter:
    return Counter(
        qa.get("point", "unknown")
        for qa in history.get("human-annotated QAs", []) or []
        if isinstance(qa, dict)
    )


def build_one_mode(
    *,
    mode: str,
    base_histories: dict[int, dict[str, Any]],
    output_root: Path,
    asr_data_source: Path,
    bgm_caption_source: Path,
    media_caption_source: Path,
    voice_map: dict[tuple[int, str], dict[str, dict[str, str]]],
    bgm_captions: dict[str, dict[str, Any]],
    image_map: dict[tuple[int, str], dict[str, str]],
    dry_run: bool,
) -> Counter:
    output_dir = output_root / mode
    total_stats: Counter = Counter()

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Building mode: {mode} ===")
    print(f"Output dir: {output_dir}")

    for p_id, base_history in sorted(base_histories.items()):
        patched, stats = patch_history_for_mode(
            base_history,
            p_id=p_id,
            mode=mode,
            asr_data_source=asr_data_source,
            bgm_caption_source=bgm_caption_source,
            media_caption_source=media_caption_source,
            voice_map=voice_map,
            bgm_captions=bgm_captions,
            image_map=image_map,
        )
        total_stats.update(stats)

        qas = patched.get("human-annotated QAs", []) or []
        point_counts = qa_count_summary(patched)
        out_path = output_dir / f"history_with_qa_p{p_id}.json"

        print(f"\n[p_id={p_id}]")
        print(f"  sessions       : {len(patched.get('multi_session_dialogues', []) or [])}")
        print(f"  QA total       : {len(qas)}")
        for point, count in sorted(point_counts.items()):
            print(f"    {point:18s}: {count}")
        print(f"  voice patched  : {stats['voice_patched']}")
        print(f"  image patched  : {stats['image_patched']}")
        if mode == "asr_bg_split":
            print(f"  bgm captions   : {stats['bgm_caption_used']}")
            print(f"  bgm fallback   : {stats['bgm_name_fallback']}")

        if dry_run:
            print(f"  dry-run output : {out_path}")
        else:
            write_json(out_path, patched)
            print(f"  output         : {out_path}")

    return total_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build asr/hint/asr_bg_split benchmark inputs for audio caption study."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--asr-data", type=Path, default=DEFAULT_ASR_DATA)
    parser.add_argument("--bgm-captions", type=Path, default=DEFAULT_BGM_CAPTIONS)
    parser.add_argument(
        "--media-captions",
        type=Path,
        default=DEFAULT_MEDIA_CAPTIONS,
        help="Image-caption source. Defaults to medium_history_dialogue.json so all audio-caption modes use medium image captions.",
    )
    parser.add_argument("--caption-field", default="audio_caption_qwen3_asr")
    parser.add_argument("--modes", nargs="+", default=list(VALID_MODES), choices=VALID_MODES)
    parser.add_argument("--num-profiles", type=int, default=None)
    parser.add_argument("--only-profile-ids", default=None, help="Comma/space separated p_id list.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    only_profile_ids = parse_profile_ids(args.only_profile_ids)

    print("Loading source files...")
    base_histories = discover_base_histories(
        args.base_dir,
        num_profiles=args.num_profiles,
        only_profile_ids=only_profile_ids,
    )
    asr_data = load_json(args.asr_data)
    media_data = load_json(args.media_captions)

    if args.bgm_captions.exists():
        bgm_captions = normalize_bgm_captions(load_json(args.bgm_captions))
    else:
        print(f"WARN: background caption file not found: {args.bgm_captions}")
        print("      asr_bg_split will fall back to raw background_audio names.")
        bgm_captions = {}

    voice_map = build_voice_caption_map(asr_data, args.caption_field)
    image_map = build_image_caption_map(media_data)

    print(f"Base dir          : {args.base_dir}")
    print(f"Output root       : {args.output_root}")
    print(f"Selected profiles : {sorted(base_histories)}")
    print(f"ASR data          : {args.asr_data}")
    print(f"ASR caption field : {args.caption_field}")
    print(f"BGM captions      : {args.bgm_captions} ({len(bgm_captions)} names)")
    print(f"Media captions    : {args.media_captions}")
    print(f"Voice caption keys: {sum(len(v) for v in voice_map.values())}")
    print(f"Image caption keys: {sum(len(v) for v in image_map.values())}")

    all_stats: dict[str, Counter] = {}
    for mode in args.modes:
        all_stats[mode] = build_one_mode(
            mode=mode,
            base_histories=base_histories,
            output_root=args.output_root,
            asr_data_source=args.asr_data,
            bgm_caption_source=args.bgm_captions,
            media_caption_source=args.media_captions,
            voice_map=voice_map,
            bgm_captions=bgm_captions,
            image_map=image_map,
            dry_run=args.dry_run,
        )

    print("\n=== Summary ===")
    for mode, stats in all_stats.items():
        print(f"[{mode}]")
        for key in sorted(stats):
            print(f"  {key:18s}: {stats[key]}")

    if args.dry_run:
        print("\nDry run only; no files written.")
    else:
        print(f"\nDone. Files written under {args.output_root}")


if __name__ == "__main__":
    main()
