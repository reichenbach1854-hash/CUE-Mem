"""
先为 qa_formatted_data 中的会话图生成 Caption（ChatAnywhere / OpenAI 兼容 API），
再为有 audio_path 的用户轮次生成音频 Caption（OpenRouter + input_audio）。

默认：
  - 图像模型：gpt-5.4（ChatAnywhere，chat/completions + image_url）
  - 音频模型：google/gemini-2.5-flash（OpenRouter，chat/completions + input_audio base64）

用法（在项目根目录）::

    python qa/gen_media_captions.py
    python qa/gen_media_captions.py --image-only
    python qa/gen_media_captions.py --audio-only
    python qa/gen_media_captions.py --image-only --only_profile_ids 0
    python qa/gen_media_captions.py --audio-only --force-audio
    python qa/gen_media_captions.py --sample 5    # 仅生成前 5 条音频 Caption（试运行）

图像与音频可分别指定模型 / Base URL / Key（环境变量优先级低于命令行里显式传的模型名）。

输出::

    qa/qa_formatted_data_000_002_with_media_captions.json
    qa/qa_formatted_data_000_002_with_media_captions_image_tokens.json
    qa/qa_formatted_data_000_002_with_media_captions_audio_tokens.json
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client
from scripts.common.paths import resolve_path
from scripts.qa.config import qa_path

DEFAULT_INPUT = qa_path("qa_formatted_data_000_004.json")
DEFAULT_OUTPUT = qa_path("qa_formatted_data_000_004_with_media_captions.json")
DEFAULT_IMG_TOKEN_STATS = qa_path("qa_formatted_data_with_media_captions_image_tokens.json")
DEFAULT_AUDIO_TOKEN_STATS = qa_path("qa_formatted_data_with_media_captions_audio_tokens.json")

# Credentials and optional endpoints are supplied only at runtime.  Separate
# variables allow image and audio caption models to use different providers.
IMAGE_BASE_URL = env_value("CUE_MEM_QA_IMAGE_BASE_URL") or env_value("CUE_MEM_LLM_BASE_URL")
IMAGE_API_KEY = env_value("CUE_MEM_QA_IMAGE_API_KEY") or env_value("CUE_MEM_LLM_API_KEY")
AUDIO_BASE_URL = env_value("CUE_MEM_QA_AUDIO_BASE_URL") or env_value("CUE_MEM_LLM_BASE_URL")
AUDIO_API_KEY = env_value("CUE_MEM_QA_AUDIO_API_KEY") or env_value("CUE_MEM_LLM_API_KEY")

DEFAULT_IMAGE_MODEL = env_value("CUE_MEM_QA_IMAGE_MODEL", "gpt-5.4")
DEFAULT_AUDIO_MODEL = env_value("CUE_MEM_QA_AUDIO_MODEL", "gemini-3.1-pro-preview")


IMG_PROMPT = (
    "为这张图片生成一条简洁的图片说明。"
    "仅描述可见的细节，且仅输出纯文本。"
    "请用中文回答。"
)

AUDIO_PROMPT = (
    "请为这段音频生成中文 caption，必须严格使用下面两行格式：\n"
    "人声：<逐字转录音频中的人声内容；不要概括、不要改写、不要省略；如果没有人声，写“无”>\n"
    "背景音：<描述能听到的背景音/环境音；如果没有明显背景音，写“无明显背景音”>\n"
    "要求：\n"
    "1. 人声部分必须尽可能逐字转录原话，包括口语停顿、语气词和关键信息。\n"
    "2. 背景音部分只描述实际能听到的声音，不要推测看不见的场景。\n"
    "3. 不要输出解释、Markdown、编号、JSON 或额外说明，只输出上述两行纯文本。"
)


def resolve_effective_models(
    image_model_cli: str | None, audio_model_cli: str | None
) -> tuple[str, str]:
    img = (image_model_cli or "").strip() or DEFAULT_IMAGE_MODEL
    aud = (audio_model_cli or "").strip() or DEFAULT_AUDIO_MODEL
    return img, aud


def resolve_media_path(raw: str) -> str:
    if not raw or not str(raw).strip():
        return ""
    return str(resolve_path(str(raw).strip()).resolve())


def _usage_from_response(api_res) -> tuple[int, int]:
    u = getattr(api_res, "usage", None)
    if u is None:
        return 0, 0
    pt = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None)
    ct = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None)
    tt = getattr(u, "total_tokens", None)
    p, c = int(pt or 0), int(ct or 0)
    if tt and p == 0 and c == 0:
        return int(tt), 0
    return p, c


def _audio_clip_format(local_path: str) -> str:
    ext = Path(local_path).suffix.lower().lstrip(".")
    if ext in {"wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"}:
        return ext
    return "wav"


def image_session(
    task_id: str, img_path: str, *, image_model: str, api_base_url: str, api_key: str
) -> tuple[str, str, int, int]:
    local_img = resolve_media_path(img_path)
    if not local_img or not os.path.exists(local_img):
        print(f"[image] missing {task_id}: {img_path} -> {local_img}")
        return task_id, "", 0, 0

    mime, _ = mimetypes.guess_type(local_img)
    if not mime:
        mime = "image/png"

    with open(local_img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    url = f"data:{mime};base64,{b64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": IMG_PROMPT},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]

    client = openai_client(api_key=api_key, base_url=api_base_url)
    for attempt in range(6):
        try:
            api_res = client.chat.completions.create(
                model=image_model,
                messages=messages,
                reasoning_effort="low",
            )
            text = message_content_to_text(api_res.choices[0].message.content).strip()
            pt, ct = _usage_from_response(api_res)
            return task_id, text, pt, ct
        except Exception:
            try:
                api_res = client.chat.completions.create(
                    model=image_model,
                    messages=messages,
                    temperature=0.0,
                )
                text = message_content_to_text(api_res.choices[0].message.content).strip()
                pt, ct = _usage_from_response(api_res)
                return task_id, text, pt, ct
            except Exception as e:
                print(f"[image] {task_id} retry {attempt + 1}: {e}")

    print(f"[image] FAIL {task_id}")
    return task_id, "", 0, 0


def audio_session(
    task_id: str,
    rel_audio_path: str,
    *,
    audio_model: str,
    api_base_url: str,
    api_key: str,
) -> tuple[str, str, int, int]:
    """使用 OpenAI 兼容消息中的 ``input_audio``（base64 + format）调用多模态 API。"""
    local_path = resolve_media_path(rel_audio_path)
    if not local_path or not os.path.exists(local_path):
        print(f"[audio] missing {task_id}: {rel_audio_path} -> {local_path}")
        return task_id, "", 0, 0

    fmt = _audio_clip_format(local_path)
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": AUDIO_PROMPT},
                {
                    "type": "input_audio",
                    "input_audio": {"data": b64, "format": fmt},
                },
            ],
        }
    ]

    referer = env_value("CUE_MEM_HTTP_REFERER")
    client = openai_client(
        api_key=api_key,
        base_url=api_base_url,
        default_headers={"HTTP-Referer": referer} if referer else None,
    )

    for attempt in range(6):
        try:
            api_res = client.chat.completions.create(
                model=audio_model,
                messages=messages,
                temperature=0.0,
            )
            text = (api_res.choices[0].message.content or "").strip()
            pt, ct = _usage_from_response(api_res)
            print(f"[audio] {task_id} | {rel_audio_path}\n  -> {text[:300]}")
            return task_id, text, pt, ct
        except Exception as e:
            wait = 2 ** attempt
            print(f"[audio] {task_id} attempt {attempt + 1} failed: {e}; wait {wait}s")
            time.sleep(wait)

    print(f"[audio] FAIL {task_id}")
    return task_id, "", 0, 0


def _is_media_path(value: str) -> bool:
    """Return True if value is a resolvable file path (not already a caption text)."""
    if not value:
        return False
    resolved = resolve_media_path(value)
    return bool(resolved and os.path.exists(resolved))


def _profile_id(profile: dict, fallback_idx: int) -> int:
    raw = profile.get("p_id", profile.get("id", fallback_idx))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return fallback_idx


def _parse_profile_ids(raw_values: list[str] | None) -> set[int] | None:
    if not raw_values:
        return None
    ids: set[int] = set()
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            if not part.strip():
                continue
            try:
                ids.add(int(part))
            except ValueError as exc:
                raise ValueError(f"invalid profile id: {part!r}") from exc
    return ids


def _patch_media_paths(
    profiles: list,
    fresh_profiles: list,
    *,
    only_profile_ids: set[int] | None = None,
    suffixes: set[str] | None = None,
    force_selected: bool = False,
    force_suffixes: set[str] | None = None,
) -> None:
    """将 fresh_profiles 中 dialog_list 的 .png/.wav 路径值补回 profiles。

    仅对尚未生成 caption 的条目恢复路径；已有 caption（值为文本而非路径）的条目
    保持不变，从而实现增量续跑。

    当 force_selected=True 时，只对目标 profile 的 force_suffixes 媒体类型
    强制恢复原始路径，用于定向重新生成 caption，同时不影响其他媒体类型。
    """
    for p_idx, (profile, fresh_profile) in enumerate(zip(profiles, fresh_profiles)):
        pid = _profile_id(profile if isinstance(profile, dict) else {}, p_idx)
        if only_profile_ids is not None and pid not in only_profile_ids:
            continue
        events = profile.get("events") or []
        fresh_events = fresh_profile.get("events") or []
        for e_idx, (event, fresh_event) in enumerate(zip(events, fresh_events)):
            dlg_list = event.get("dialog_list") or []
            fresh_dlg_list = fresh_event.get("dialog_list") or []
            for dl_idx, (entry, fresh_entry) in enumerate(zip(dlg_list, fresh_dlg_list)):
                for key in list(fresh_entry.keys()):
                    key_suffix = ".png" if key.endswith(".png") else ".wav" if key.endswith(".wav") else ""
                    if not key_suffix:
                        continue
                    if suffixes is not None and key_suffix not in suffixes:
                        continue
                    fresh_val = (fresh_entry.get(key) or "").strip()
                    if not _is_media_path(fresh_val):
                        continue
                    current_val = (entry.get(key) or "").strip()
                    should_force = force_selected and (
                        force_suffixes is None or key_suffix in force_suffixes
                    )
                    if should_force or not current_val or _is_media_path(current_val):
                        entry[key] = fresh_val


def run_image_phase(
    profiles: list,
    max_workers: int,
    *,
    image_model: str,
    api_base_url: str,
    api_key: str,
    checkpoint_path: "Path | None" = None,
    checkpoint_every: int = 20,
    only_profile_ids: set[int] | None = None,
) -> tuple[dict, list[dict]]:
    if not api_key:
        raise SystemExit("[image] API key 为空，请设置 CUE_MEM_QA_IMAGE_API_KEY 或 CUE_MEM_LLM_API_KEY")

    # 从 dialog_list 的 .png key 中直接读取 image_path
    future_to_meta: dict = {}
    stats: list[dict] = []
    total_pt = total_ct = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p_idx, profile in enumerate(profiles):
            pid = _profile_id(profile if isinstance(profile, dict) else {}, p_idx)
            if only_profile_ids is not None and pid not in only_profile_ids:
                continue
            for e_idx, event in enumerate(profile.get("events", []) or []):
                for dl_idx, entry in enumerate(event.get("dialog_list") or []):
                    for key in list(entry.keys()):
                        if not key.endswith(".png"):
                            continue
                        raw = (entry.get(key) or "").strip()
                        if not _is_media_path(raw):
                            continue
                        tid = f"img_{p_idx}_{e_idx}_{dl_idx}"
                        fut = ex.submit(
                            image_session,
                            tid,
                            raw,
                            image_model=image_model,
                            api_base_url=api_base_url,
                            api_key=api_key,
                        )
                        future_to_meta[fut] = (p_idx, e_idx, dl_idx, key, raw)

        completed = 0
        for fut in tqdm(
            concurrent.futures.as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="图像 Caption (API)",
        ):
            _, cap, pt, ct = fut.result()
            p_idx, e_idx, dl_idx, key, ip = future_to_meta[fut]
            total_pt += pt
            total_ct += ct
            stats.append({
                "task_id": f"img_{p_idx}_{e_idx}_{dl_idx}",
                "phase": "image",
                "p_idx": p_idx,
                "event_idx": e_idx,
                "dl_idx": dl_idx,
                "key": key,
                "path": ip,
                "success": bool(cap),
                "prompt_tokens": pt,
                "completion_tokens": ct,
            })
            if cap:
                event = profiles[p_idx]["events"][e_idx]
                event["dialog_list"][dl_idx][key] = cap
                event["user_shared_image_description"] = cap
            completed += 1
            if checkpoint_path and checkpoint_every > 0 and completed % checkpoint_every == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as _f:
                    json.dump(profiles, _f, ensure_ascii=False, indent=4)
                tqdm.write(f"[image] checkpoint saved ({completed}/{len(future_to_meta)}) -> {checkpoint_path}")

    summary = {
        "model": image_model,
        "base_url_configured_at_runtime": bool(api_base_url),
        "requests": len(future_to_meta),
        "successful": sum(1 for s in stats if s["success"]),
        "only_profile_ids": sorted(only_profile_ids) if only_profile_ids is not None else None,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
        "per_request": stats,
    }
    return summary, stats


def run_audio_phase(
    profiles: list,
    max_workers: int,
    *,
    audio_model: str,
    api_base_url: str,
    api_key: str,
    sample: int | None = None,
    checkpoint_path: "Path | None" = None,
    checkpoint_every: int = 50,
    only_profile_ids: set[int] | None = None,
) -> tuple[dict, list[dict]]:
    if not api_key:
        raise SystemExit("[audio] API key 为空，请设置 CUE_MEM_QA_AUDIO_API_KEY 或 CUE_MEM_LLM_API_KEY")

    # 从 dialog_list 的 .wav key 中直接读取 audio_path
    pairs: list[tuple[int, int, int, str, str, str, str]] = []

    for p_idx, profile in enumerate(profiles):
        pid = _profile_id(profile if isinstance(profile, dict) else {}, p_idx)
        if only_profile_ids is not None and pid not in only_profile_ids:
            continue
        for e_idx, event in enumerate(profile.get("events", []) or []):
            for dl_idx, entry in enumerate(event.get("dialog_list") or []):
                for key in list(entry.keys()):
                    if not key.endswith(".wav"):
                        continue
                    raw = (entry.get(key) or "").strip()
                    if not _is_media_path(raw):
                        continue
                    tid = f"a{p_idx}_{e_idx}_dl{dl_idx}"
                    pairs.append((p_idx, e_idx, dl_idx, key, tid, raw))

    eligible_audio = len(pairs)
    if sample is not None:
        pairs = pairs[:sample]

    total_pt = total_ct = 0
    stats: list[dict] = []
    wall0 = time.time()

    future_to_meta: dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for p_idx, e_idx, dl_idx, key, tid, raw in pairs:
            fut = ex.submit(
                audio_session,
                tid,
                raw,
                audio_model=audio_model,
                api_base_url=api_base_url,
                api_key=api_key,
            )
            future_to_meta[fut] = (p_idx, e_idx, dl_idx, key, tid, raw)

        completed = 0
        for fut in tqdm(
            concurrent.futures.as_completed(future_to_meta),
            total=len(future_to_meta),
            desc="音频 Caption (API)",
        ):
            _, cap, pt, ct = fut.result()
            p_idx, e_idx, dl_idx, key, tid, raw = future_to_meta[fut]
            total_pt += pt
            total_ct += ct
            stats.append({
                "task_id": tid,
                "phase": "audio",
                "p_idx": p_idx,
                "event_idx": e_idx,
                "dl_idx": dl_idx,
                "key": key,
                "path": raw,
                "success": bool(cap),
                "prompt_tokens": pt,
                "completion_tokens": ct,
            })
            if cap:
                profiles[p_idx]["events"][e_idx]["dialog_list"][dl_idx][key] = cap
            completed += 1
            if checkpoint_path and checkpoint_every > 0 and completed % checkpoint_every == 0:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as _f:
                    json.dump(profiles, _f, ensure_ascii=False, indent=4)
                tqdm.write(f"[audio] checkpoint saved ({completed}/{len(pairs)}) -> {checkpoint_path}")

    summary = {
        "model": audio_model,
        "base_url_configured_at_runtime": bool(api_base_url),
        "engine": "OpenAI-compatible chat.completions (input_audio)",
        "audio_sample_limit": sample,
        "only_profile_ids": sorted(only_profile_ids) if only_profile_ids is not None else None,
        "eligible_audio_rounds": eligible_audio,
        "requests": len(pairs),
        "successful": sum(1 for s in stats if s["success"]),
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
        "seconds_total_wall": round(time.time() - wall0, 4),
        "per_request": stats,
    }
    return summary, stats


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        description="图像 + 音频 Caption（ChatAnywhere / OpenAI 兼容 API，默认 gpt-5.4 + gemini-3.1-pro-preview）"
    )
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="输入 JSON")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON")
    ap.add_argument(
        "--img-token-stats",
        type=Path,
        default=DEFAULT_IMG_TOKEN_STATS,
        help="图像阶段 token 明细 JSON",
    )
    ap.add_argument(
        "--audio-token-stats",
        type=Path,
        default=DEFAULT_AUDIO_TOKEN_STATS,
        help="音频阶段 token 明细 JSON",
    )
    ap.add_argument("--audio-stats", type=Path, default=None, help="（兼容）同 --audio-token-stats，若指定则覆盖")
    ap.add_argument("--image-workers", type=int, default=8, help="图像 API 并发数")
    ap.add_argument("--audio-workers", type=int, default=4, help="音频 API 并发数")
    ap.add_argument(
        "--image-model",
        type=str,
        default=None,
        help=f"图像模型（默认 {DEFAULT_IMAGE_MODEL!r}）",
    )
    ap.add_argument(
        "--audio-model",
        type=str,
        default=None,
        help=f"音频模型（默认 {DEFAULT_AUDIO_MODEL!r}）",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--image-only", action="store_true", help="只做图像")
    g.add_argument("--audio-only", action="store_true", help="只做音频")
    ap.add_argument(
        "--force-audio",
        action="store_true",
        help=(
            "强制重新生成音频 caption：从 --input 中恢复 .wav 路径并覆盖已有音频 caption。"
            "通常配合 --audio-only 使用，不会恢复或改写已有图片 caption。"
        ),
    )
    ap.add_argument(
        "--only_profile_ids",
        "--only-profile-ids",
        nargs="*",
        default=None,
        metavar="P_ID",
        help=(
            "只处理指定 p_id，支持空格或逗号分隔，例如 "
            "--only_profile_ids 0 或 --only_profile_ids 1,2,3。"
        ),
    )
    ap.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="仅生成前 N 条音频 Caption（按数据中出现的顺序）；默认不限制。仅影响音频阶段。",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.input = resolve_path(args.input)
    args.output = resolve_path(args.output)
    args.img_token_stats = resolve_path(args.img_token_stats)
    args.audio_token_stats = resolve_path(args.audio_token_stats)
    if args.audio_stats is not None:
        args.audio_stats = resolve_path(args.audio_stats)
    in_path = args.input

    try:
        only_profile_ids = _parse_profile_ids(args.only_profile_ids)
    except ValueError as exc:
        print(f"(error) {exc}", file=sys.stderr)
        return 1

    if args.sample is not None and args.sample < 1:
        print("(error) --sample 须为正整数，表示最多生成的音频 Caption 条数", file=sys.stderr)
        return 1
    if args.force_audio and args.image_only:
        print("(error) --force-audio 需要运行音频阶段，不能与 --image-only 同用", file=sys.stderr)
        return 1

    audio_token_path = args.audio_stats or args.audio_token_stats

    if not in_path.exists():
        print(f"(error) missing input: {in_path}", file=sys.stderr)
        return 1

    image_model, audio_model = resolve_effective_models(
        args.image_model, args.audio_model
    )

    print(f"[config] image model={image_model!r}")
    print(f"[config] audio model={audio_model!r}")
    if only_profile_ids is not None:
        print(f"[config] only_profile_ids={sorted(only_profile_ids)}")

    img_checkpoint = args.output.with_name(args.output.stem + "_image_checkpoint.json")
    aud_checkpoint = args.output.with_name(args.output.stem + "_audio_checkpoint.json")

    fresh_profiles = load_records(in_path)

    resume_path = None
    for candidate in [aud_checkpoint, img_checkpoint, args.output]:
        if candidate.exists():
            resume_path = candidate
            break

    if resume_path:
        print(f"[resume] 从已有进度恢复: {resume_path}")
        profiles = load_records(resume_path)
    elif args.force_audio:
        print("[force-audio] 未发现已有输出/检查点，将直接从输入文件生成音频 caption。")
    else:
        profiles = fresh_profiles

    run_image = not args.audio_only
    run_audio = not args.image_only

    patch_suffixes: set[str] = set()
    if run_image:
        patch_suffixes.add(".png")
    if run_audio:
        patch_suffixes.add(".wav")
    if resume_path:
        force_suffixes: set[str] = set()
        if only_profile_ids is not None:
            force_suffixes.update(patch_suffixes)
        if args.force_audio and run_audio:
            force_suffixes.add(".wav")
        _patch_media_paths(
            profiles,
            fresh_profiles,
            only_profile_ids=only_profile_ids,
            suffixes=patch_suffixes or None,
            force_selected=bool(force_suffixes),
            force_suffixes=force_suffixes or None,
        )

    combined_report: dict[str, object] = {
        "input": str(in_path),
        "output": str(args.output),
        "image_model": image_model,
        "audio_model": audio_model,
        "image_base_url_configured_at_runtime": bool(IMAGE_BASE_URL),
        "audio_base_url_configured_at_runtime": bool(AUDIO_BASE_URL),
        "image_phase": None,
        "audio_phase": None,
        "audio_sample": args.sample,
        "only_profile_ids": sorted(only_profile_ids) if only_profile_ids is not None else None,
    }

    if run_image:
        img_summary, _ = run_image_phase(
            profiles,
            max_workers=max(1, args.image_workers),
            image_model=image_model,
            api_base_url=IMAGE_BASE_URL,
            api_key=IMAGE_API_KEY,
            checkpoint_path=img_checkpoint,
            checkpoint_every=20,
            only_profile_ids=only_profile_ids,
        )
        combined_report["image_phase"] = img_summary
        args.img_token_stats.parent.mkdir(parents=True, exist_ok=True)
        with open(args.img_token_stats, "w", encoding="utf-8") as f:
            json.dump(img_summary, f, ensure_ascii=False, indent=2)
        print(
            f"[image] model={image_model} | tokens prompt={img_summary['prompt_tokens']} "
            f"completion={img_summary['completion_tokens']} "
            f"total={img_summary['total_tokens']} -> {args.img_token_stats}"
        )
        # 图像阶段完成后立即落盘中间文件，防止中断丢失
        img_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with open(img_checkpoint, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=4)
        print(f"[image] checkpoint saved -> {img_checkpoint}")

    if run_audio:
        if args.sample is not None:
            print(f"[audio] sample 模式：最多生成 {args.sample} 条音频 Caption")
        aud_summary, _ = run_audio_phase(
            profiles,
            max_workers=max(1, args.audio_workers),
            audio_model=audio_model,
            api_base_url=AUDIO_BASE_URL,
            api_key=AUDIO_API_KEY,
            checkpoint_path=aud_checkpoint,
            checkpoint_every=50,
            sample=args.sample,
            only_profile_ids=only_profile_ids,
        )
        combined_report["audio_phase"] = aud_summary
        audio_token_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audio_token_path, "w", encoding="utf-8") as f:
            json.dump(aud_summary, f, ensure_ascii=False, indent=2)
        aud_req = aud_summary["requests"]
        aud_eligible = aud_summary["eligible_audio_rounds"]
        sample_note = ""
        if aud_summary.get("audio_sample_limit") is not None:
            sample_note = f" | eligible={aud_eligible} (实际调用 {aud_req})"
        print(
            f"[audio] model={audio_model} | tokens prompt={aud_summary['prompt_tokens']} "
            f"completion={aud_summary['completion_tokens']} "
            f"total={aud_summary['total_tokens']} | requests={aud_req}{sample_note} "
            f"-> {audio_token_path}"
        )
        # 音频阶段完成后落盘 checkpoint（含已有图像 caption + 本次音频 caption）
        aud_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        with open(aud_checkpoint, "w", encoding="utf-8") as f:
            json.dump(profiles, f, ensure_ascii=False, indent=4)
        print(f"[audio] checkpoint saved -> {aud_checkpoint}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=4)

    rep_path = args.output.with_name(args.output.stem + "_pipeline_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(combined_report, f, ensure_ascii=False, indent=2)

    print(f"Done -> {args.output}")
    print(f"Report -> {rep_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
