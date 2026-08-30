"""
Generate captions for background audio clips with Gemini.

This script reads background_audio names from the split-ASR formatted data,
resolves each name to an mp3 path via event/background_audio_manifest_*.json,
and writes an incremental caption cache.

Usage:
    python -m scripts.qa.generate_background_audio_captions --dry-run
    python -m scripts.qa.generate_background_audio_captions --sample 5 --workers 2
    python -m scripts.qa.generate_background_audio_captions --workers 4
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from scripts.common.llm import message_content_to_text, openai_client
from scripts.RQ1_RQ2.benchmark.paths import EVENT_ROOT, QA_ROOT, REPOSITORY_ROOT
from scripts.RQ1_RQ2.benchmark.security import redact_runtime_text

PROJECT_ROOT = REPOSITORY_ROOT

DEFAULT_INPUT = QA_ROOT / "qwen3_asr_1.7b" / "qa_formatted_data_with_audio_captions_qwen3_asr.json"
DEFAULT_MANIFEST = EVENT_ROOT / "background_audio_manifest_kling_000_019.json"
DEFAULT_OUTPUT = QA_ROOT / "qwen3_asr_1.7b" / "background_audio_captions_gemini-3.1-pro.json"

_AIFAST_AUDIO_MODEL = "gemini-3.1-pro-preview"
_OPENROUTER_AUDIO_MODEL = "google/gemini-3.1-pro-preview"

DEFAULT_AUDIO_MODEL = os.environ.get(
    "CUE_MEM_AUDIO_MODEL", _AIFAST_AUDIO_MODEL
)

CAPTION_PROMPT = (
    "请为这段背景音频生成一条中文描述。"
    "只描述能听到的背景声音，不要猜测画面、人物或对话内容。"
    "仅输出纯文本，不要加标题。"
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def resolve_path(raw_path: str) -> Path:
    p = str(raw_path or "").strip()
    if not p:
        return Path()
    if os.path.isabs(p):
        return Path(p).resolve()
    p = p.replace("\\", "/").lstrip("./")
    return (PROJECT_ROOT / p).resolve()


def audio_format(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in {"wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"}:
        return ext
    return "mp3"


def collect_background_audio_names(formatted_data: list) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for profile in formatted_data:
        for event in profile.get("events", []) or []:
            for turn in event.get("dialog", []) or []:
                name = (turn.get("background_audio") or "").strip()
                if not name or name.lower() == "none" or name in seen:
                    continue
                seen.add(name)
                names.append(name)
    return names


def build_manifest_index(manifest: list) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for entry in manifest:
        for item in entry.get("background_audio_path", []) or []:
            name = (item.get("query") or "").strip()
            raw_path = (item.get("path") or "").strip()
            if not name or not raw_path or name in index:
                continue
            resolved = resolve_path(raw_path)
            index[name] = {
                "audio_path": raw_path,
                "resolved_path": str(resolved),
                "exists": resolved.exists(),
                "manifest_query": name,
            }
    return index


def normalize_audio_name(name: str) -> str:
    text = (name or "").strip().replace(" ", "")
    for suffix in ("的声音", "声音", "声"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text


def add_manifest_aliases(manifest_index: dict[str, dict], bg_names: list[str]) -> None:
    """Add aliases for small naming variants such as 炒菜声 vs 炒菜的声音."""
    norm_to_name = {}
    for name in manifest_index:
        norm = normalize_audio_name(name)
        if norm and norm not in norm_to_name:
            norm_to_name[norm] = name

    for name in bg_names:
        if name in manifest_index:
            continue
        norm = normalize_audio_name(name)
        matched_name = norm_to_name.get(norm)
        if not matched_name:
            continue
        alias_info = dict(manifest_index[matched_name])
        alias_info["alias_of"] = matched_name
        manifest_index[name] = alias_info


def normalize_existing(existing) -> dict:
    if not isinstance(existing, dict):
        return {}
    normalized = {}
    for name, value in existing.items():
        if isinstance(value, str):
            normalized[name] = {"caption": value}
        elif isinstance(value, dict):
            normalized[name] = value
    return normalized


def usage_from_response(api_res) -> tuple[int, int]:
    usage = getattr(api_res, "usage", None)
    if usage is None:
        return 0, 0
    pt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    ct = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
    return int(pt or 0), int(ct or 0)


def resolve_audio_endpoint(
    provider: str,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str, str]:
    """Resolve OpenAI-compatible audio caption endpoint settings."""
    provider_key = (provider or "aifast").strip().lower()
    selected_key = (
        (api_key or "").strip()
        or os.environ.get("CUE_MEM_AUDIO_API_KEY", "").strip()
        or os.environ.get("CUE_MEM_LLM_API_KEY", "").strip()
    )
    selected_base_url = (
        (base_url or "").strip()
        or os.environ.get("CUE_MEM_AUDIO_BASE_URL", "").strip()
        or os.environ.get("CUE_MEM_LLM_BASE_URL", "").strip()
    )

    if provider_key == "aifast":
        selected_model = (model or "").strip() or _AIFAST_AUDIO_MODEL
    elif provider_key == "openrouter":
        selected_model = (model or "").strip() or _OPENROUTER_AUDIO_MODEL
    elif provider_key == "custom":
        selected_model = (model or "").strip() or DEFAULT_AUDIO_MODEL
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return selected_model, selected_base_url, selected_key


def generate_caption(
    name: str,
    audio_path: Path,
    *,
    model: str,
    api_base_url: str,
    api_key: str,
) -> tuple[str, int, int, str]:
    if not audio_path.exists():
        return "", 0, 0, f"missing file: {audio_path}"

    with audio_path.open("rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"背景音名称：{name}\n\n{CAPTION_PROMPT}",
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": b64,
                        "format": audio_format(audio_path),
                    },
                },
            ],
        }
    ]

    client = openai_client(
        api_key=api_key,
        base_url=api_base_url,
    )

    max_attempts = 6
    for attempt in range(max_attempts):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            text = message_content_to_text(res.choices[0].message.content).strip()
            pt, ct = usage_from_response(res)
            return text, pt, ct, ""
        except Exception as exc:  # noqa: BLE001 - retry varied client exceptions
            if attempt == max_attempts - 1:
                return "", 0, 0, redact_runtime_text(exc)
            time.sleep(2 ** attempt)

    return "", 0, 0, "unknown error"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemini captions for background audio clips."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--provider",
        choices=["aifast", "openrouter", "custom"],
        default="aifast",
        help="Audio caption API provider preset. Default: aifast.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Audio model override. Defaults to gemini-3.1-pro-preview for aifast, "
            "google/gemini-3.1-pro-preview for openrouter."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL override.",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Regenerate existing captions")
    parser.add_argument("--dry-run", action="store_true", help="Only print coverage stats")
    args = parser.parse_args()

    args.model, args.base_url, args.api_key = resolve_audio_endpoint(
        args.provider,
        args.model,
        args.base_url,
        args.api_key,
    )

    print(f"[config] provider = {args.provider}")
    print(f"[config] model    = {args.model}")

    formatted_data = load_json(args.input)
    manifest = load_json(args.manifest)

    bg_names = collect_background_audio_names(formatted_data)
    manifest_index = build_manifest_index(manifest)
    add_manifest_aliases(manifest_index, bg_names)

    missing_manifest = [name for name in bg_names if name not in manifest_index]
    missing_files = [
        name for name in bg_names
        if name in manifest_index and not manifest_index[name]["exists"]
    ]

    print(f"Unique background_audio names: {len(bg_names)}")
    print(f"Resolved in manifest        : {len(bg_names) - len(missing_manifest)}")
    print(f"Missing manifest entries    : {len(missing_manifest)}")
    print(f"Missing audio files         : {len(missing_files)}")

    if missing_manifest:
        print("Missing manifest examples:")
        for name in missing_manifest[:20]:
            print(f"  - {name}")
    if missing_files:
        print("Missing file examples:")
        for name in missing_files[:20]:
            print(f"  - {name}: {manifest_index[name]['resolved_path']}")

    captions = {}
    if args.output.exists():
        captions = normalize_existing(load_json(args.output))
        print(f"[resume] loaded existing captions: {len(captions)}")

    candidates = [
        name for name in bg_names
        if name in manifest_index and manifest_index[name]["exists"]
    ]
    if args.sample is not None:
        candidates = candidates[:args.sample]
        print(f"[sample] processing first {len(candidates)} candidates")

    if not args.force:
        todo = [
            name for name in candidates
            if not (captions.get(name, {}).get("caption") or "").strip()
        ]
    else:
        todo = candidates

    print(f"Already captioned           : {len(candidates) - len(todo)}")
    print(f"Remaining                   : {len(todo)}")

    if args.dry_run:
        print("Dry run only; no API calls.")
        return

    if not args.base_url:
        raise RuntimeError(
            "missing audio API endpoint; set CUE_MEM_AUDIO_BASE_URL or "
            "CUE_MEM_LLM_BASE_URL, or pass --base-url"
        )
    if not args.api_key:
        raise RuntimeError(
            "missing audio API key; set CUE_MEM_AUDIO_API_KEY or "
            "CUE_MEM_LLM_API_KEY, or pass --api-key"
        )

    total_pt = 0
    total_ct = 0
    success = 0

    def process_one(name: str):
        info = manifest_index[name]
        caption, pt, ct, err = generate_caption(
            name,
            Path(info["resolved_path"]),
            model=args.model,
            api_base_url=args.base_url,
            api_key=args.api_key,
        )
        return name, caption, pt, ct, err

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, name): name for name in todo}
        for completed, fut in enumerate(tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Background audio captions",
        ), start=1):
            name, caption, pt, ct, err = fut.result()
            total_pt += pt
            total_ct += ct
            info = manifest_index[name]
            if caption:
                success += 1
                captions[name] = {
                    "audio_path": info["audio_path"],
                    "resolved_path": info["resolved_path"],
                    "caption": caption,
                    "provider": args.provider,
                    "model": args.model,
                }
                tqdm.write(f"{name} -> {caption[:160]}")
            else:
                captions[name] = {
                    "audio_path": info["audio_path"],
                    "resolved_path": info["resolved_path"],
                    "caption": "",
                    "provider": args.provider,
                    "model": args.model,
                    "error": err,
                }
                tqdm.write(f"FAIL {name}: {err}")

            if completed % 10 == 0:
                write_json(args.output, captions)

    write_json(args.output, captions)
    print(f"Done: {success}/{len(todo)} generated")
    print(f"Tokens: prompt={total_pt}, completion={total_ct}, total={total_pt + total_ct}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
