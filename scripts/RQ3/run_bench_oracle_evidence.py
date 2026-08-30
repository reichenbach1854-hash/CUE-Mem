"""Run RQ3 oracle-evidence QA evaluation.

Text oracle mode renders evidence sessions as text. Multimodal oracle mode
feeds raw user/assistant text plus original images/audio from the matched
sessions or clue turns as OpenAI-compatible content blocks.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import (
    ALIYUN_OMNI_API_BASE,
    ALIYUN_OMNI_API_KEY,
    ALIYUN_OMNI_MODEL,
    DATA_DIR,
    OMNI_API_BASE,
    OMNI_API_KEY,
    OMNI_MODEL,
    RESULT_DIR,
    profile_paths,
    redact_sensitive_text,
    required_runtime_value,
    resolve_path,
)
from .omni_client import (
    FORMAT_PROMPTS,
    REASONING_FORMAT,
    SYSTEM_PROMPT,
    _extract_answer,
    _split_answer_reasoning,
)

DEFAULT_ORACLE_VLLM_MODEL = os.getenv("RQ3_ORACLE_VLLM_MODEL", "qwen3.6-omni-30b-a3b")
def warn(message: str) -> None:
    print(f"[ORACLE WARN] {message}", file=sys.stderr, flush=True)


def is_retryable_error_text(text: str) -> bool:
    lowered = text.lower()
    retry_markers = (
        "429",
        "rate limit",
        "rate_limit",
        "too many requests",
        "too_many_requests",
        "request rate increased too quickly",
        "throttl",
        "temporarily unavailable",
        "service unavailable",
        "timeout",
        "timed out",
    )
    return any(marker in lowered for marker in retry_markers)


def message_block_counts(messages: list[dict] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not messages:
        return counts
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            typ = str(block.get("type", "<missing>"))
            counts[typ] = counts.get(typ, 0) + 1
    return counts


def media_ref_counts(media_refs: list[dict] | None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for ref in media_refs or []:
        key = f"{ref.get('source', '')}:{ref.get('modality', '')}:{ref.get('status', '')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def print_oracle_error_context(
    *,
    stage: str,
    p_id: int,
    qid: str,
    qa: dict,
    error: Any,
    messages: list[dict] | None = None,
    meta: dict | None = None,
    redaction_values: tuple[str | None, ...] = (),
) -> None:
    print(
        f"\n[ORACLE QA ERROR] stage={stage} p{p_id} qa_id={qid} "
        f"point={qa.get('point', '')} answer={qa.get('answer', '')}",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"  error={redact_sensitive_text(error, *redaction_values)[:1000]}",
        file=sys.stderr,
        flush=True,
    )
    print(f"  matched_session_ids={qa.get('matched_session_ids', [])}", file=sys.stderr, flush=True)
    clue = qa.get("clue", [])
    if isinstance(clue, list):
        print(f"  clue_count={len(clue)} clue_preview={clue[:10]}", file=sys.stderr, flush=True)
    else:
        print(f"  clue={clue}", file=sys.stderr, flush=True)
    if messages is not None:
        print(f"  message_block_counts={message_block_counts(messages)}", file=sys.stderr, flush=True)
    if meta is not None:
        print(f"  evidence_session_ids={meta.get('evidence_session_ids', [])}", file=sys.stderr, flush=True)
        print(f"  related_history_outline={meta.get('related_history_outline', [])}", file=sys.stderr, flush=True)
        print(f"  media_ref_counts={media_ref_counts(meta.get('related_media_refs', []))}", file=sys.stderr, flush=True)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_raw_session_index(dataset: dict) -> dict[str, dict]:
    """Return raw session_id -> session dict from history_with_qa data."""
    sessions = dataset.get("multi_session_dialogues") or []
    return {
        str(session.get("session_id")): session
        for session in sessions
        if session.get("session_id")
    }


def build_raw_turn_indices(dataset: dict) -> dict[str, dict[str, tuple[dict, dict]]]:
    """Build round/image_id/voice_id registries for raw dialogue turns."""
    by_round: dict[str, tuple[dict, dict]] = {}
    by_image_id: dict[str, tuple[dict, dict]] = {}
    by_voice_id: dict[str, tuple[dict, dict]] = {}

    for session in dataset.get("multi_session_dialogues") or []:
        for turn in session.get("dialogues") or []:
            round_id = str(turn.get("round") or "")
            if round_id:
                by_round[round_id] = (session, turn)
            for image_id in normalize_list(turn.get("image_id")):
                by_image_id[str(image_id)] = (session, turn)
            for voice_id in normalize_list(turn.get("voice_id")):
                by_voice_id[str(voice_id)] = (session, turn)

    return {"round": by_round, "image": by_image_id, "voice": by_voice_id}


def derive_session_ids(qa: dict) -> list[str]:
    """Infer relevant session ids from matched_session_ids/session_id/clue."""
    result: list[str] = []

    def add(session_id: Any) -> None:
        if session_id is None:
            return
        sid = str(session_id).strip()
        if not sid:
            return
        if ":" in sid:
            sid = sid.split(":", 1)[0]
        if "-" in sid and re.match(r"^D\d{2}-", sid):
            sid = sid.split("-", 1)[0]
        if sid not in result:
            result.append(sid)

    for sid in normalize_list(qa.get("matched_session_ids")):
        add(sid)
    for sid in normalize_list(qa.get("session_id")):
        add(sid)
    for clue in normalize_list(qa.get("clue")):
        add(str(clue).removesuffix(".wav").removesuffix(".png"))

    return result


def derive_clue_turn_ids(qa: dict, dataset: dict) -> dict[str, set[str]]:
    """Return session_id -> round ids resolved from QA clue entries."""
    indices = build_raw_turn_indices(dataset)
    grouped: dict[str, set[str]] = defaultdict(set)

    for raw_clue in normalize_list(qa.get("clue")):
        clue = str(raw_clue).strip()
        if not clue:
            continue

        hit: tuple[dict, dict] | None = None
        if clue in indices["round"]:
            hit = indices["round"][clue]
        elif clue.lower().endswith(".wav"):
            hit = indices["voice"].get(clue[:-4])
        elif clue.lower().endswith(".png"):
            hit = indices["image"].get(clue[:-4])

        if hit:
            session, turn = hit
            sid = str(session.get("session_id") or "")
            rid = str(turn.get("round") or "")
            if sid and rid:
                grouped[sid].add(rid)

    return grouped


def resolve_media_path(raw_path: str | Path, media_root: Path) -> Path:
    """Resolve raw dataset media path under RQ3/data."""
    raw = str(raw_path).replace("\\", "/").strip()
    if not raw:
        return media_root / raw

    p = Path(raw)
    if p.is_absolute():
        return p

    if raw.startswith("event/voice_mixed_000_002/"):
        suffix = raw.removeprefix("event/voice_mixed_000_002/")
        return media_root / "voice_mixed_000_002" / Path(suffix)
    if raw.startswith("event/images/"):
        suffix = raw.removeprefix("event/images/")
        return media_root / "event_image" / Path(suffix)
    if raw.startswith("qa/"):
        suffix = raw.removeprefix("qa/")
        return media_root / "qa_image" / Path(suffix)

    return media_root / Path(raw)


def find_media_root(arg_value: str | Path | None, default_root: Path = DATA_DIR) -> Path:
    if arg_value:
        return resolve_path(arg_value)

    candidate = default_root
    if (candidate / "event_image").exists() or (candidate / "qa_image").exists():
        return candidate
    warn(f"No media root candidate fully matched; using configured data directory {candidate}")
    return candidate


def encode_file_data_url(path: Path, default_mime: str) -> tuple[str, str]:
    mime = mimetypes.guess_type(path.name)[0] or default_mime
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}", mime


def audio_format(path: Path, mime: str) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext:
        return "mp3" if ext == "mpeg" else ext
    if "/" in mime:
        return mime.split("/", 1)[1].split(";", 1)[0]
    return "wav"


def first_text(value: Any) -> str:
    items = normalize_list(value)
    return "\n".join(str(x) for x in items if x)


def format_session_text(session: dict) -> str:
    """Text oracle evidence formatter. Keeps textual session annotations."""
    sid = session.get("session_id", "")
    date = session.get("date", "")
    lines = [f"Session {sid}" + (f" ({date})" if date else "")]

    for key, label in [
        ("scene_description", "Scene"),
        ("user_shared_image_description", "Shared image description"),
        ("background_audio_info", "Background audio info"),
        ("human_speech_content", "Human speech content"),
        ("explicit_preferences", "Explicit preferences"),
        ("implicit_preferences", "Implicit preferences"),
    ]:
        value = session.get(key)
        if value:
            lines.append(f"{label}: {value}")

    for turn in session.get("dialogues") or []:
        rid = turn.get("round", "")
        user = turn.get("user")
        if user:
            lines.append(f"[{rid}] User: {user}")
        voice_caption = turn.get("user_voice_message_caption") or first_text(turn.get("voice_caption"))
        if voice_caption:
            lines.append(f"[{rid}] User voice: {voice_caption}")
        image_caption = first_text(turn.get("image_caption"))
        if image_caption:
            lines.append(f"[{rid}] User image: {image_caption}")
        background_audio = turn.get("background_audio")
        if background_audio:
            lines.append(f"[{rid}] Background audio: {background_audio}")
        assistant = turn.get("assistant")
        if assistant:
            lines.append(f"[{rid}] Assistant: {assistant}")

    return "\n".join(lines)


def _media_ref(
    *,
    session_id: str,
    round_id: str,
    modality: str,
    raw_path: str,
    resolved_path: Path,
    source: str,
    status: str = "included",
    note: str = "",
) -> dict:
    ref = {
        "source": source,
        "session_id": session_id,
        "round": round_id,
        "modality": modality,
        "raw_path": raw_path,
        "path": str(resolved_path),
        "status": status,
    }
    if note:
        ref["note"] = note
    return ref


def format_session_multimodal_blocks(
    session: dict,
    media_root: Path,
    max_audio: int | None = None,
    max_images: int | None = None,
    turn_filter: set[str] | None = None,
    counters: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Format one raw session as multimodal content blocks plus media refs."""
    blocks: list[dict] = []
    media_refs: list[dict] = []
    counters = counters if counters is not None else {"audio": 0, "image": 0}
    sid = str(session.get("session_id") or "")
    date = session.get("date", "")

    title = f"\nSession {sid}" + (f" ({date})" if date else "")
    blocks.append({"type": "text", "text": title})

    for turn in session.get("dialogues") or []:
        rid = str(turn.get("round") or "")
        if turn_filter is not None and rid not in turn_filter:
            continue

        user = turn.get("user")
        if user:
            blocks.append({"type": "text", "text": f"[{rid}] User: {user}"})

        for raw_image in normalize_list(turn.get("input_image")):
            raw_image = str(raw_image)
            resolved = resolve_media_path(raw_image, media_root)
            if max_images is not None and counters["image"] >= max_images:
                media_refs.append(_media_ref(
                    session_id=sid,
                    round_id=rid,
                    modality="image",
                    raw_path=raw_image,
                    resolved_path=resolved,
                    source="evidence",
                    status="skipped_limit",
                    note=f"max_images_per_request={max_images}",
                ))
                continue
            if resolved.exists():
                data_url, mime = encode_file_data_url(resolved, "image/png")
                blocks.append({"type": "text", "text": f"[{rid}] User shared image:"})
                blocks.append({"type": "image_url", "image_url": {"url": data_url}})
                counters["image"] += 1
                media_refs.append(_media_ref(
                    session_id=sid,
                    round_id=rid,
                    modality="image",
                    raw_path=raw_image,
                    resolved_path=resolved,
                    source="evidence",
                    note=mime,
                ))
            else:
                caption = first_text(turn.get("image_caption"))
                warn(f"Missing evidence image: {raw_image} -> {resolved}")
                media_refs.append(_media_ref(
                    session_id=sid,
                    round_id=rid,
                    modality="image",
                    raw_path=raw_image,
                    resolved_path=resolved,
                    source="evidence",
                    status="missing",
                    note="fallback_to_caption" if caption else "",
                ))
                if caption:
                    blocks.append({"type": "text", "text": f"[{rid}] User shared image caption fallback: {caption}"})

        for raw_audio in normalize_list(turn.get("input_voice_message")):
            raw_audio = str(raw_audio)
            resolved = resolve_media_path(raw_audio, media_root)
            if max_audio is not None and counters["audio"] >= max_audio:
                media_refs.append(_media_ref(
                    session_id=sid,
                    round_id=rid,
                    modality="audio",
                    raw_path=raw_audio,
                    resolved_path=resolved,
                    source="evidence",
                    status="skipped_limit",
                    note=f"max_audio_per_request={max_audio}",
                ))
                continue
            if resolved.exists():
                data_url, mime = encode_file_data_url(resolved, "audio/wav")
                blocks.append({"type": "text", "text": f"[{rid}] User sent audio:"})
                blocks.append({
                    "type": "input_audio",
                    "input_audio": {"data": data_url, "format": audio_format(resolved, mime)},
                })
                counters["audio"] += 1
                media_refs.append(_media_ref(
                    session_id=sid,
                    round_id=rid,
                    modality="audio",
                    raw_path=raw_audio,
                    resolved_path=resolved,
                    source="evidence",
                    note=mime,
                ))
            else:
                caption = turn.get("user_voice_message_caption") or first_text(turn.get("voice_caption"))
                warn(f"Missing evidence audio: {raw_audio} -> {resolved}")
                media_refs.append(_media_ref(
                    session_id=sid,
                    round_id=rid,
                    modality="audio",
                    raw_path=raw_audio,
                    resolved_path=resolved,
                    source="evidence",
                    status="missing",
                    note="fallback_to_caption" if caption else "",
                ))
                if caption:
                    blocks.append({"type": "text", "text": f"[{rid}] User audio caption fallback: {caption}"})

        assistant = turn.get("assistant")
        if assistant:
            blocks.append({"type": "text", "text": f"[{rid}] Assistant: {assistant}"})

    return blocks, media_refs


def format_question_text(qa: dict) -> str:
    lines = [qa.get("question", "")]
    if str(qa.get("point", "")).endswith("_img"):
        captions = qa.get("option_captions") or qa.get("question_image_descriptions") or {}
        if isinstance(captions, dict):
            for letter in sorted(captions):
                lines.append(f"Option {letter}: {captions[letter]}")
    return "\n".join(line for line in lines if line)


def format_question_multimodal_blocks(
    qa: dict,
    media_root: Path,
    image_option_mode: str = "raw",
) -> tuple[list[dict], list[dict]]:
    """Format current QA. Image QA uses all A/B/C/D option images."""
    del image_option_mode
    blocks: list[dict] = [{"type": "text", "text": qa.get("question", "")}]
    media_refs: list[dict] = []

    if not str(qa.get("point", "")).endswith("_img"):
        return blocks, media_refs

    option_images = qa.get("option_images") or {}
    captions = qa.get("option_captions") or qa.get("question_image_descriptions") or {}
    for letter in sorted(option_images):
        raw_image = str(option_images[letter])
        resolved = resolve_media_path(raw_image, media_root)
        blocks.append({"type": "text", "text": f"Option {letter}:"})
        if resolved.exists():
            data_url, mime = encode_file_data_url(resolved, "image/png")
            blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            media_refs.append({
                "source": "question_option",
                "option": letter,
                "modality": "image",
                "raw_path": raw_image,
                "path": str(resolved),
                "status": "included",
                "note": mime,
            })
        else:
            warn(f"Missing QA option image: option={letter} {raw_image} -> {resolved}")
            caption = captions.get(letter, "") if isinstance(captions, dict) else ""
            media_refs.append({
                "source": "question_option",
                "option": letter,
                "modality": "image",
                "raw_path": raw_image,
                "path": str(resolved),
                "status": "missing",
                "note": "fallback_to_caption" if caption else "",
            })
            if caption:
                blocks.append({"type": "text", "text": f"Option {letter} caption fallback: {caption}"})

    return blocks, media_refs


def related_history_outline(sessions: list[dict], turn_filters: dict[str, set[str]] | None = None) -> list[str]:
    outline: list[str] = []
    for session in sessions:
        sid = str(session.get("session_id") or "")
        turns = session.get("dialogues") or []
        if turn_filters is not None:
            wanted = turn_filters.get(sid, set())
            turns = [turn for turn in turns if str(turn.get("round") or "") in wanted]
        image_count = sum(len(normalize_list(turn.get("input_image"))) for turn in turns)
        audio_count = sum(len(normalize_list(turn.get("input_voice_message"))) for turn in turns)
        outline.append(f"Session {sid}: {len(turns)} turns, {image_count} image, {audio_count} audio")
    return outline


def _format_prompt_for_qa(qa: dict, with_reasoning: bool) -> str:
    prompt = FORMAT_PROMPTS.get(qa.get("point", "pref_text"), "")
    if with_reasoning and prompt:
        prompt = prompt.rstrip() + REASONING_FORMAT
    return prompt


def build_oracle_text_messages(
    qa: dict,
    session_index: dict[str, dict],
    with_reasoning: bool = False,
) -> tuple[list[dict], dict]:
    session_ids = derive_session_ids(qa)
    sessions = [session_index[sid] for sid in session_ids if sid in session_index]
    missing = [sid for sid in session_ids if sid not in session_index]
    for sid in missing:
        warn(f"QA {qa.get('qa_id', '')}: matched session {sid} not found")

    memory = "\n\n".join(format_session_text(session) for session in sessions)
    question = format_question_text(qa)
    format_prompt = _format_prompt_for_qa(qa, with_reasoning)
    user_text = (
        f"{format_prompt}\n\n"
        "The oracle evidence memory contents are as follows:\n"
        f"{memory}\n\n"
        "The current question is as follows:\n"
        f"{question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ], {
        "related_media_refs": [],
        "related_history_outline": related_history_outline(sessions),
        "evidence_session_ids": [str(s.get("session_id")) for s in sessions],
    }


def build_oracle_multimodal_messages(
    qa: dict,
    dataset: dict,
    session_index: dict[str, dict],
    media_root: Path,
    max_audio: int | None = None,
    max_images: int | None = None,
    oracle_scope: str = "session",
    with_reasoning: bool = False,
) -> tuple[list[dict], dict]:
    session_ids = derive_session_ids(qa)
    turn_filters: dict[str, set[str]] | None = None
    if oracle_scope == "clue_turn":
        turn_filters = derive_clue_turn_ids(qa, dataset)
        session_ids = [sid for sid in session_ids if sid in turn_filters]
        for sid in turn_filters:
            if sid not in session_ids:
                session_ids.append(sid)

    sessions = [session_index[sid] for sid in session_ids if sid in session_index]
    for sid in session_ids:
        if sid not in session_index:
            warn(f"QA {qa.get('qa_id', '')}: matched session {sid} not found")

    blocks: list[dict] = []
    format_prompt = _format_prompt_for_qa(qa, with_reasoning)
    if format_prompt:
        blocks.append({"type": "text", "text": format_prompt})
    blocks.append({"type": "text", "text": "The oracle evidence memory contents are as follows:"})

    media_refs: list[dict] = []
    counters = {"audio": 0, "image": 0}
    for session in sessions:
        sid = str(session.get("session_id") or "")
        session_blocks, refs = format_session_multimodal_blocks(
            session,
            media_root,
            max_audio=max_audio,
            max_images=max_images,
            turn_filter=turn_filters.get(sid) if turn_filters is not None else None,
            counters=counters,
        )
        blocks.extend(session_blocks)
        media_refs.extend(refs)

    blocks.append({"type": "text", "text": "The current question is as follows:"})
    question_blocks, question_refs = format_question_multimodal_blocks(qa, media_root)
    blocks.extend(question_blocks)
    media_refs.extend(question_refs)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": blocks},
    ], {
        "related_media_refs": media_refs,
        "related_history_outline": related_history_outline(sessions, turn_filters),
        "evidence_session_ids": [str(s.get("session_id")) for s in sessions],
    }


def build_oracle_messages(
    qa: dict,
    dataset: dict,
    session_index: dict[str, dict],
    media_root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    if args.oracle_input_mode == "text":
        return build_oracle_text_messages(qa, session_index, args.with_reasoning)
    return build_oracle_multimodal_messages(
        qa,
        dataset,
        session_index,
        media_root,
        max_audio=args.max_audio_per_request,
        max_images=args.max_images_per_request,
        oracle_scope=args.oracle_scope,
        with_reasoning=args.with_reasoning,
    )


class OracleLLMClient:
    def __init__(
        self,
        provider: str,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        retries: int,
        retry_base_wait: float,
    ) -> None:
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_base_wait = retry_base_wait
        if provider == "aliyun":
            base_env_names = ("RQ3_ALIYUN_OMNI_API_BASE",)
            key_env_names = ("RQ3_ALIYUN_OMNI_API_KEY", "DASHSCOPE_API_KEY")
        else:
            base_env_names = ("RQ3_OMNI_API_BASE", "CUE_MEM_LLM_BASE_URL")
            key_env_names = ("RQ3_OMNI_API_KEY", "CUE_MEM_LLM_API_KEY")
        self.api_base = required_runtime_value(
            api_base,
            argument_name="evaluation API base",
            env_names=base_env_names,
        )
        self.api_key = required_runtime_value(
            api_key,
            argument_name="evaluation API key",
            env_names=key_env_names,
        )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on user environment
            raise RuntimeError(
                "install the optional `openai` dependency to use an evaluation service"
            ) from exc
        try:
            self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
        except Exception as exc:  # noqa: BLE001
            safe_error = redact_sensitive_text(exc, self.api_key, self.api_base)
            raise RuntimeError(
                f"failed to initialize evaluation client: {safe_error}"
            ) from None

    def answer(self, messages: list[dict], with_reasoning: bool = False) -> dict[str, Any]:
        max_tokens = max(self.max_tokens, 512) if with_reasoning else self.max_tokens
        raw = self._call_with_retries(messages, max_tokens)
        if with_reasoning and raw and not raw.startswith("[ERROR]"):
            choice_part, reasoning, reasoning_sessions = _split_answer_reasoning(raw)
            model_answer = _extract_answer(choice_part)
        else:
            reasoning = ""
            reasoning_sessions = []
            model_answer = _extract_answer(raw)
        return {
            "model_answer": model_answer,
            "raw_response": raw,
            "reasoning": reasoning,
            "reasoning_sessions": reasoning_sessions,
        }

    def _call_with_retries(self, messages: list[dict], max_tokens: int) -> str:
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if self.provider == "aliyun":
                    return self._call_aliyun(messages, max_tokens)
                return self._call_vllm(messages, max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                msg = redact_sensitive_text(exc, self.api_key, self.api_base)
                retryable = is_retryable_error_text(msg)
                if not retryable or attempt >= self.retries:
                    print(
                        f"[ORACLE LLM ERROR] provider={self.provider} model={self.model} "
                        f"block_counts={message_block_counts(messages)}: {msg}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return f"[ERROR] {msg}"
                wait = min(60.0, self.retry_base_wait * (2 ** attempt))
                print(
                    f"[ORACLE LLM RETRY] provider={self.provider} model={self.model} "
                    f"attempt={attempt + 1}/{self.retries} wait={wait:.1f}s error={msg[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        return f"[ERROR] {redact_sensitive_text(last_error, self.api_key, self.api_base)}"

    def _call_vllm(self, messages: list[dict], max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (response.choices[0].message.content or "").strip()

    def _call_aliyun(self, messages: list[dict], max_tokens: int) -> str:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            stream=True,
            modalities=["text"],
        )
        chunks: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                chunks.append(content)
        return "".join(chunks).strip()


def resolve_eval_model(args: argparse.Namespace) -> str:
    if args.llm_name:
        return args.llm_name
    if args.eval_provider == "aliyun":
        return ALIYUN_OMNI_MODEL
    return DEFAULT_ORACLE_VLLM_MODEL or OMNI_MODEL


def resolve_api_base(args: argparse.Namespace) -> str:
    if args.api_base:
        return args.api_base
    if args.eval_provider == "aliyun":
        return ALIYUN_OMNI_API_BASE or ""
    return OMNI_API_BASE or ""


def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key
    if args.eval_provider == "aliyun":
        return ALIYUN_OMNI_API_KEY or ""
    return OMNI_API_KEY or ""


def qa_id(qa: dict, index: int) -> str:
    return str(qa.get("qa_id") or qa.get("question_id") or index)


def qa_uid_for(p_id: int, qa: dict, index: int) -> str:
    """Stable unique QA key. qa_id alone is not unique in RQ3."""
    return f"p{p_id}:{index:04d}:{qa_id(qa, index)}"


def result_path_for_profile(args: argparse.Namespace, model: str, p_id: int) -> Path:
    root = resolve_path(
        args.output_root,
        RESULT_DIR / "oracle_evidence",
    )
    return (
        root
        / args.oracle_input_mode
        / args.eval_provider
        / safe_model_name(model)
        / f"p{p_id}_results.json"
    )


def load_existing_result_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
    except (OSError, TypeError, ValueError) as exc:
        warn(f"Failed to load existing results {path}: {exc}")
        return []
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        return [row for row in data["results"] if isinstance(row, dict)]
    elif isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    else:
        return []


def index_existing_results(rows: list[dict], qas: list[dict], p_id: int) -> dict[str, dict]:
    """Index checkpoint rows by qa_uid, upgrading legacy qa_id-only rows.

    Older oracle files used qa_id as the result dict key. Some pref/rec items
    intentionally share qa_id, so those files lost rows by overwrite. Existing
    rows that remain can still be matched back to a unique raw QA by
    qa_id+point+question and kept; missing duplicates are then rerun.
    """
    raw_lookup: dict[tuple[str, str, str], list[tuple[int, dict]]] = defaultdict(list)
    for idx, qa in enumerate(qas):
        raw_lookup[
            (
                qa_id(qa, idx),
                str(qa.get("point", "")),
                str(qa.get("question", "")),
            )
        ].append((idx, qa))

    indexed: dict[str, dict] = {}
    for row in rows:
        uid = row.get("qa_uid")
        if uid:
            indexed[str(uid)] = row
            continue

        key = (
            str(row.get("qa_id", "")),
            str(row.get("point", "")),
            str(row.get("question", "")),
        )
        matches = raw_lookup.get(key, [])
        if len(matches) == 1:
            idx, qa = matches[0]
            uid = qa_uid_for(p_id, qa, idx)
            row = {**row, "qa_uid": uid, "qa_index": idx, "legacy_qa_id_key": True}
            indexed[uid] = row
        else:
            warn(
                "Could not uniquely map legacy oracle result row "
                f"qa_id={row.get('qa_id', '')} point={row.get('point', '')}; "
                "it will not count as completed for resume."
            )
    return indexed


def load_qa_uid_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    data = load_json(Path(path))
    if isinstance(data, dict):
        values = data.get("missing_qa_uids") or data.get("qa_uids") or data.get("uids") or []
    else:
        values = data
    return {str(value) for value in values}


def get_available_profile_files(history_dir: Path) -> list[Path]:
    """Scan the configured history directory for evaluation JSON files."""
    if not history_dir.exists():
        return []
    files = []
    for path in history_dir.glob("*.json"):
        name = path.name
        if "_results_" in name or "_evaluate_result_" in name:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.name)


def profile_id_from_path(path: Path, fallback: int) -> int:
    m = re.search(r"p(\d+)", path.stem)
    return int(m.group(1)) if m else fallback


def run_profile(profile_path: Path, p_index: int, args: argparse.Namespace, client: OracleLLMClient, media_root: Path) -> None:
    p_id = profile_id_from_path(profile_path, p_index)
    dataset = load_json(profile_path)
    session_index = build_raw_session_index(dataset)
    qas = dataset.get("human-annotated QAs") or dataset.get("human_annotated_QAs") or []
    if args.sample is not None:
        qas = qas[: args.sample]

    model = resolve_eval_model(args)
    out_path = result_path_for_profile(args, model, p_id)
    existing_rows = load_existing_result_rows(out_path) if args.resume else []
    existing = index_existing_results(existing_rows, qas, p_id) if args.resume else {}
    only_uids = load_qa_uid_filter(args.only_qa_uids_file)
    results: dict[str, dict] = dict(existing)
    pending = [
        (i, qa)
        for i, qa in enumerate(qas)
        if qa_uid_for(p_id, qa, i) not in existing
        and (only_uids is None or qa_uid_for(p_id, qa, i) in only_uids)
    ]

    print(f"\nProfile p{p_id}: {profile_path.name}")
    print(f"  Sessions: {len(session_index)}, QAs: {len(qas)}")
    print(f"  Output: {out_path}")
    if args.resume and existing:
        print(f"  Resume: {len(existing)} done, {len(pending)} remaining")
    if only_uids is not None:
        print(f"  QA uid filter: {len(only_uids)} requested, {len(pending)} pending for this profile")
    print(f"  Max workers: {args.max_workers}")

    def evaluate_one(item: tuple[int, dict]) -> tuple[str, dict]:
        idx, qa = item
        qid = qa_id(qa, idx)
        uid = qa_uid_for(p_id, qa, idx)
        messages: list[dict] | None = None
        meta: dict = {
            "evidence_session_ids": [],
            "related_history_outline": [],
            "related_media_refs": [],
        }

        try:
            messages, meta = build_oracle_messages(qa, dataset, session_index, media_root, args)
            llm_result = client.answer(messages, with_reasoning=args.with_reasoning)
            row = {
                "qa_uid": uid,
                "qa_index": idx,
                "qa_id": qid,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "model_answer": llm_result["model_answer"],
                "raw_response": llm_result["raw_response"],
                "correct": llm_result["model_answer"] == qa.get("answer", ""),
                "point": qa.get("point", ""),
                "qa_type": qa.get("qa_type", ""),
                "category": qa.get("category", ""),
                "subcategory": qa.get("subcategory", ""),
                "clue": qa.get("clue", []),
                "matched_session_ids": qa.get("matched_session_ids", []),
                "evidence_session_ids": meta["evidence_session_ids"],
                "related_history_outline": meta["related_history_outline"],
                "related_media_refs": meta["related_media_refs"],
                "oracle_input_mode": args.oracle_input_mode,
                "oracle_scope": args.oracle_scope if args.oracle_input_mode == "multimodal" else "session",
                "eval_provider": args.eval_provider,
                "llm_name": model,
            }
            if args.with_reasoning:
                row["reasoning"] = llm_result.get("reasoning", "")
                row["reasoning_sessions"] = llm_result.get("reasoning_sessions", [])

            if llm_result["raw_response"].startswith("[ERROR]"):
                print_oracle_error_context(
                    stage="llm_error_response",
                    p_id=p_id,
                    qid=qid,
                    qa=qa,
                    error=llm_result["raw_response"],
                    messages=messages,
                    meta=meta,
                    redaction_values=(client.api_key, client.api_base),
                )
        except Exception as exc:
            print_oracle_error_context(
                stage="qa_exception",
                p_id=p_id,
                qid=qid,
                qa=qa,
                error=exc,
                messages=messages,
                meta=meta,
                redaction_values=(client.api_key, client.api_base),
            )
            row = {
                "qa_uid": uid,
                "qa_index": idx,
                "qa_id": qid,
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "model_answer": "",
                "raw_response": (
                    "[ERROR] "
                    f"{redact_sensitive_text(exc, client.api_key, client.api_base)}"
                ),
                "correct": False,
                "point": qa.get("point", ""),
                "qa_type": qa.get("qa_type", ""),
                "category": qa.get("category", ""),
                "subcategory": qa.get("subcategory", ""),
                "clue": qa.get("clue", []),
                "matched_session_ids": qa.get("matched_session_ids", []),
                "evidence_session_ids": meta.get("evidence_session_ids", []),
                "related_history_outline": meta.get("related_history_outline", []),
                "related_media_refs": meta.get("related_media_refs", []),
                "oracle_input_mode": args.oracle_input_mode,
                "oracle_scope": args.oracle_scope if args.oracle_input_mode == "multimodal" else "session",
                "eval_provider": args.eval_provider,
                "llm_name": model,
            }
            if args.fail_fast:
                raise

        return uid, row

    completed_in_run = 0
    if pending:
        worker_count = min(args.max_workers, len(pending))
        print(f"  Running {len(pending)} QA(s) with {worker_count} worker(s)")
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=worker_count)
        futures = {
            executor.submit(evaluate_one, item): item
            for item in pending
        }
        try:
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"oracle p{p_id}",
                ncols=90,
            ):
                uid, row = future.result()
                results[uid] = row
                completed_in_run += 1
                if completed_in_run % args.checkpoint_every == 0:
                    write_profile_results(out_path, results, args, model)
        except Exception:
            for future in futures:
                future.cancel()
            write_profile_results(out_path, results, args, model)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

    write_profile_results(out_path, results, args, model)
    ordered_results = list(results.values())
    correct = sum(1 for row in ordered_results if row.get("correct"))
    total = len(ordered_results)
    print(f"  Saved: {out_path}")
    print(f"  Accuracy: {correct}/{total} = {(correct / total * 100 if total else 0):.1f}%")


def write_profile_results(path: Path, results: dict[str, dict], args: argparse.Namespace, model: str) -> None:
    rows = sorted(
        results.values(),
        key=lambda row: (
            int(row.get("qa_index", 10**9)) if str(row.get("qa_index", "")).isdigit() else 10**9,
            str(row.get("qa_uid") or row.get("qa_id") or ""),
        ),
    )
    correct = sum(1 for row in rows if row.get("correct"))
    payload = {
        "meta": {
            "oracle_input_mode": args.oracle_input_mode,
            "oracle_scope": args.oracle_scope,
            "eval_provider": args.eval_provider,
            "llm_name": model,
            "sample": args.sample,
            "max_workers": args.max_workers,
            "max_audio_per_request": args.max_audio_per_request,
            "max_images_per_request": args.max_images_per_request,
            "accuracy": correct / len(rows) if rows else 0.0,
            "total": len(rows),
        },
        "results": rows,
    }
    save_json_atomic(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="*", type=int, default=[0, 1, 2], help="Profile ids to run.")
    parser.add_argument(
        "--all_datasets",
        "--all-datasets",
        action="store_true",
        help="Run all *.json datasets under RQ3/data/history_dialogue.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Directory containing history_with_qa_pN.json files.",
    )
    parser.add_argument(
        "--profile-files",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit profile JSON files; overrides --history-dir.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Run only first N QAs per profile.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True, help="Resume existing outputs.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save every N completed QA.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum concurrent QA requests within each profile (default: 8).",
    )

    parser.add_argument("--oracle-input-mode", choices=["text", "multimodal"], default="text")
    parser.add_argument("--oracle-scope", choices=["session", "clue_turn"], default="session")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="RQ3 data directory; defaults to the project-relative configuration.",
    )
    parser.add_argument("--media-root", default=None, help="Path to RQ3/data. Auto-detected when omitted.")
    parser.add_argument("--max-audio-per-request", type=int, default=None)
    parser.add_argument("--max-images-per-request", type=int, default=None)

    parser.add_argument("--eval-provider", choices=["vllm", "aliyun"], default="vllm")
    parser.add_argument("--llm_name", "--llm-name", default=None, help="Evaluation model name.")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--with-reasoning", action="store_true")
    parser.add_argument("--llm-retries", type=int, default=4)
    parser.add_argument("--llm-retry-base-wait", type=float, default=2.0)
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately after a QA exception.")
    parser.add_argument(
        "--only-qa-uids-file",
        default=None,
        help="JSON list/file with qa_uid values to run; useful for backfilling missing rows.",
    )
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_path(args.data_dir, DATA_DIR)
    args.history_dir = resolve_path(
        args.history_dir,
        args.data_dir / "history_dialogue",
    )
    args.profile_files = (
        [resolve_path(path) for path in args.profile_files]
        if args.profile_files
        else profile_paths(args.history_dir)
    )
    if args.only_qa_uids_file:
        args.only_qa_uids_file = resolve_path(args.only_qa_uids_file)
    media_root = find_media_root(args.media_root, args.data_dir)
    model = resolve_eval_model(args)
    api_base = resolve_api_base(args)
    api_key = resolve_api_key(args)

    print("RQ3 Oracle Evidence")
    print(f"  Oracle input mode: {args.oracle_input_mode}")
    print(f"  Oracle scope: {args.oracle_scope}")
    print(f"  Eval provider: {args.eval_provider}")
    print(f"  LLM: {model}")
    print(f"  Media root: {media_root}")
    print(f"  Resume: {args.resume}")
    print(f"  Max workers: {args.max_workers}")

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be >= 1")

    client = OracleLLMClient(
        provider=args.eval_provider,
        model=model,
        api_base=api_base,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        retries=args.llm_retries,
        retry_base_wait=args.llm_retry_base_wait,
    )

    if args.all_datasets:
        profile_jobs = list(enumerate(get_available_profile_files(args.history_dir)))
        if not profile_jobs:
            warn(f"No JSON datasets found under {args.history_dir}")
            return
        print(f"  All datasets: {len(profile_jobs)} files")
        for _, path in profile_jobs:
            print(f"    - {path.name}")
    else:
        profile_jobs = []
        for profile_id in args.profiles:
            if profile_id < 0 or profile_id >= len(args.profile_files):
                warn(f"Invalid profile id {profile_id}; skipping")
                continue
            profile_jobs.append((profile_id, args.profile_files[profile_id]))

    for profile_id, profile_path in profile_jobs:
        if not profile_path.exists():
            warn(f"Profile file not found: {profile_path}")
            continue
        run_profile(profile_path, profile_id, args, client, media_root)

    print("\nAll oracle evidence runs completed.")


if __name__ == "__main__":
    main()
