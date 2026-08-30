"""Generate one designed voice per profile.

The TTS model is loaded lazily so importing or displaying ``--help`` does not
require CUDA or the optional TTS packages.
"""

from __future__ import annotations

import argparse

from scripts.common.io import load_json_or_jsonl
from scripts.common.paths import project_path, resolve_path


DEFAULT_PROFILE_PATH = project_path("profile", "profiles_with_anchors.jsonl")
DEFAULT_OUTPUT_DIR = project_path("profile", "voice_style")
DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def profile_prompt(profile: dict) -> tuple[str, str]:
    basic = profile.get("Basic") or {}
    instructions = (
        f"性别：{basic.get('gender', '未提供')}。"
        f"年龄：{basic.get('age', '未提供')}。"
        f"声音特征：{basic.get('voice_timbre', '未提供')}"
    )
    text = f"大家好。我的名字是{basic.get('name', '未命名')}。很高兴认识你们！"
    return instructions, text


def parse_id_input(value: str, max_count: int) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        return list(range(max_count))
    if "-" in value:
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        if not (0 <= start <= end < max_count):
            raise ValueError(f"范围超出边界: {start}-{end}，有效范围是 0-{max_count - 1}")
        return list(range(start, end + 1))
    p_id = int(value)
    if not 0 <= p_id < max_count:
        raise ValueError(f"人物编号超出范围: {p_id}，有效范围是 0-{max_count - 1}")
    return [p_id]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "p_id",
        nargs="?",
        help="profile index: a number, a range such as 0-5, or all; omitted asks interactively",
    )
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILE_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profile_path = resolve_path(args.profiles)
    output_dir = resolve_path(args.output_dir)
    profiles = load_json_or_jsonl(profile_path)
    if not isinstance(profiles, list):
        raise ValueError(f"profiles file must contain a list: {profile_path}")
    profiles = [item for item in profiles if isinstance(item, dict)]
    if not profiles:
        raise ValueError(f"no profiles found: {profile_path}")

    raw_id = args.p_id
    if raw_id is None:
        raw_id = input(f"请输入要生成的人物编号 [0-{len(profiles) - 1}]，支持范围或 all: ").strip()
    p_ids = parse_id_input(raw_id, len(profiles))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("install torch, soundfile, and qwen_tts to generate voices") from exc

    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    for p_id in p_ids:
        instructions, text = profile_prompt(profiles[p_id])
        name = str((profiles[p_id].get("Basic") or {}).get("name", p_id)).replace(" ", "-")
        output_path = output_dir / f"{p_id}_{name}_voice_design.wav"
        wavs, sample_rate = model.generate_voice_design(
            text=text,
            language="Chinese",
            instruct=instructions,
        )
        sf.write(output_path, wavs[0], sample_rate)
        print(f"[{p_id}] saved {output_path}")


if __name__ == "__main__":
    main()
