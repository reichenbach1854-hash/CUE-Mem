"""Mix generated background audio with generated user speech."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.common.paths import project_path, resolve_path


DEFAULT_MANIFEST = project_path("event", "background_audio_manifest_kling.json")
DEFAULT_VOICE_MESSAGE_DIR = project_path("event", "voice_message_005_019")
DEFAULT_OUTPUT_DIR = project_path("event", "voice_mixed")
DEFAULT_TARGET_SNR_DB = 15


def configure_ffmpeg() -> None:
    """Add a Windows winget ffmpeg installation to PATH when present."""

    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return
    root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    for package in sorted(root.glob("Gyan.FFmpeg*"), reverse=True):
        candidates = sorted(package.glob("*/bin"), reverse=True)
        if candidates and (candidates[0] / "ffmpeg.exe").exists():
            os.environ["PATH"] = str(candidates[0]) + os.pathsep + os.environ.get("PATH", "")
            return


def mix_voice_with_bgm(
    voice_path: Path,
    bgm_path: Path,
    output_path: Path,
    target_snr_db: int = DEFAULT_TARGET_SNR_DB,
) -> None:
    configure_ffmpeg()
    from pydub import AudioSegment

    voice = AudioSegment.from_file(voice_path)
    bgm = AudioSegment.from_file(bgm_path)
    bgm = bgm + ((voice.dBFS - target_snr_db) - bgm.dBFS)
    bgm = bgm[: len(voice)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    voice.overlay(bgm, position=0).export(output_path, format="wav")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--voice-message-dir", default=str(DEFAULT_VOICE_MESSAGE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-snr-db", type=int, default=DEFAULT_TARGET_SNR_DB)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest_path = resolve_path(args.manifest)
    voice_message_dir = resolve_path(args.voice_message_dir)
    output_dir = resolve_path(args.output_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)

    total = success = 0
    missing_voice: list[str] = []
    failed: list[tuple[str, str]] = []
    for entry in manifest:
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        p_id = entry.get("p_id")
        task_id = entry.get("task_id")
        bg_list = entry.get("background_audio_path") or []
        if not bg_list:
            continue
        out_dir = output_dir / f"{p_id}-{task_id}"
        for background in bg_list:
            bgm_path = resolve_path(background.get("path"))
            if not bgm_path.exists():
                print(f"[Skip] background audio missing: {bgm_path}")
                continue
            for turn_idx in background.get("tts_turn_indices") or []:
                total += 1
                voice_path = voice_message_dir / f"{task_id}_turn{turn_idx}.wav"
                output_path = out_dir / f"{task_id}_turn{turn_idx}.wav"
                if not voice_path.exists():
                    missing_voice.append(str(voice_path))
                    continue
                try:
                    mix_voice_with_bgm(voice_path, bgm_path, output_path, args.target_snr_db)
                    success += 1
                except Exception as exc:  # pragma: no cover - ffmpeg/runtime dependent
                    failed.append((str(voice_path), str(exc)))

    print(f"完成: total={total} success={success} missing_voice={len(missing_voice)} failed={len(failed)}")
    if missing_voice:
        print(f"缺失语音示例: {missing_voice[:5]}")
    if failed:
        print(f"失败示例: {failed[:5]}")


if __name__ == "__main__":
    main()
