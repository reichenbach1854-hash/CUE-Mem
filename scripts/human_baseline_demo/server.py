from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import time
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODULE_ROOT = Path(__file__).resolve().parents[2]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

try:
    from scripts.common.io import load_json_or_jsonl, write_json
    from scripts.common.paths import PROJECT_ROOT, project_path, resolve_path
except ModuleNotFoundError:
    # Keep a copied script directly executable before it is placed in the
    # complete public checkout.  The normal package path uses scripts.common.
    PROJECT_ROOT = Path(
        os.environ.get("CUE_MEM_PROJECT_ROOT", "").strip() or MODULE_ROOT
    ).expanduser().resolve()

    def project_path(*parts: str | os.PathLike[str]) -> Path:
        return PROJECT_ROOT.joinpath(*parts)

    def resolve_path(
        value: str | os.PathLike[str] | None,
        default: Path | None = None,
    ) -> Path:
        if value is None:
            if default is None:
                raise ValueError("a path or default path is required")
            candidate = default
        else:
            candidate = Path(value)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate

    def load_json_or_jsonl(path: str | os.PathLike[str]) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))

    def write_json(path: str | os.PathLike[str], data: Any, *, indent: int = 2) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(data, ensure_ascii=False, indent=indent),
            encoding="utf-8",
        )


DEFAULT_DATA_RELATIVE = Path("benchmark") / "data" / "dialog" / "base" / "history_with_qa_p0.json"
DEFAULT_OUTPUT_DIR = project_path("human_baseline_demo", "results")
STATIC_DIR = Path(__file__).resolve().parent / "static"

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CUE-MEM Human Baseline</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Human Baseline</h1>
      <p id="profileLine">Loading...</p>
    </div>
    <div class="actions">
      <label>
        Variant
        <select id="variantSelect">
          <option value="full_multimodal">Human (Full Multimodal)</option>
        </select>
      </label>
      <label>
        Tester
        <input id="participant" placeholder="name / id" />
      </label>
      <button id="submitBtn">提交并计算正确率</button>
    </div>
  </header>

  <main>
    <section class="panel">
      <div class="section-head">
        <h2 id="historyTitle">对话历史</h2>
        <input id="sessionFilter" placeholder="过滤 session / task / text" />
      </div>
      <div id="sessions" class="sessions"></div>
    </section>

    <section class="panel">
      <div class="section-head">
        <h2>QA 测试</h2>
        <div class="filters">
          <select id="pointFilter">
            <option value="">全部类型</option>
          </select>
          <select id="qaTypeFilter">
            <option value="">全部显隐式</option>
          </select>
          <span id="progressText"></span>
        </div>
      </div>
      <div id="qas" class="qas"></div>
    </section>
  </main>

  <dialog id="resultDialog">
    <div class="dialog-body">
      <h2>测试结果</h2>
      <div id="resultSummary" class="result-summary"></div>
      <button id="closeDialog">关闭</button>
    </div>
  </dialog>

  <script src="/static/app.js"></script>
</body>
</html>
"""

CHOICES = ("A", "B", "C", "D")
EXCLUDED_QA_KEYS: set[tuple[str, str]] = set()


def is_excluded_qa(qa: dict[str, Any]) -> bool:
    return (str(qa.get("qa_id", "")), str(qa.get("point", ""))) in EXCLUDED_QA_KEYS


def qa_key(qa: dict[str, Any]) -> str:
    return "::".join(
        [
            str(qa.get("qa_id", "")),
            str(qa.get("point", "")),
            str(qa.get("qa_type", "")),
        ]
    )


def default_data_path() -> Path:
    """Find the default p0 file without depending on a local machine path."""

    configured_root = os.environ.get("CUE_MEM_BENCHMARK_ROOT", "").strip()
    if configured_root:
        roots = [resolve_path(configured_root)]
    else:
        roots = [
            project_path("RQ1_RQ2", "benchmark"),
            project_path("benchmark"),
            # Compatibility with the legacy checkout name used by the data.
            project_path("Mem-Gallery-main", "benchmark"),
        ]
    relative_data = Path("data") / "dialog" / "base" / "history_with_qa_p0.json"
    for root in roots:
        candidate = (root / relative_data).resolve()
        if candidate.is_file():
            return candidate
    return project_path(*DEFAULT_DATA_RELATIVE.parts)


def public_path(path: Path) -> str:
    """Expose only a project-relative path (or a filename for external paths)."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return path.name or "configured-path"


def legacy_asset_suffix(value: str) -> str | None:
    """Map old absolute dataset paths to a known project-relative subtree."""

    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    for marker in ("/event/", "/qa/", "/profile/", "/benchmark/"):
        start = lowered.find(marker)
        if start >= 0:
            return normalized[start + 1 :]
    return None


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = load_json_or_jsonl(path)
    except Exception as exc:
        raise ValueError(f"unable to load data file: {path.name}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"data file must contain an object: {path.name}")
    return data


def resolve_media_path(raw_path: str, media_root: Path) -> Path | None:
    value = str(raw_path or "").strip()
    if not value:
        return None
    normalized = value.replace("\\", "/")
    path = Path(normalized)
    root = media_root.expanduser().resolve()
    if not path.is_absolute() and re.match(r"^[A-Za-z]:/", normalized):
        suffix = legacy_asset_suffix(normalized)
        if not suffix:
            return None
        candidate = root / suffix
    elif path.is_absolute():
        candidate = path
    else:
        candidate = root / path
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        suffix = legacy_asset_suffix(normalized)
        if not suffix:
            return None
        try:
            resolved = (root / suffix).resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return None
    return resolved


def public_asset_path(raw_path: str, media_root: Path) -> str:
    """Return a relative asset path without exposing local filesystem paths."""

    value = str(raw_path or "").strip()
    resolved = resolve_media_path(value, media_root)
    if resolved:
        return resolved.relative_to(media_root.expanduser().resolve()).as_posix()
    normalized = value.replace("\\", "/")
    if Path(normalized).is_absolute() or re.match(r"^[A-Za-z]:/", normalized):
        return Path(normalized).name
    return Path(normalized).as_posix() if normalized else ""


def media_url(raw_path: str, media_root: Path) -> str:
    path = resolve_media_path(raw_path, media_root)
    if not path or not path.is_file():
        return ""
    relative = path.relative_to(media_root.expanduser().resolve()).as_posix()
    return "/media?path=" + urllib.parse.quote(relative, safe="")


def split_question_options(question: str) -> dict[str, str]:
    """Best-effort parse of A/B/C/D option text from MCQ question strings."""
    text = str(question or "")
    options: dict[str, str] = {}
    pattern = re.compile(
        r"(?ms)(?:^|\n)\s*([A-D])\s*[\.．、:：]\s*(.*?)(?=(?:\n\s*[A-D]\s*[\.．、:：])|\n\s*请在|\Z)"
    )
    for letter, option_text in pattern.findall(text):
        clean = re.sub(r"\s+", " ", option_text).strip()
        if clean:
            options[letter] = clean
    return options


def question_stem(question: str) -> str:
    text = str(question or "").strip()
    match = re.search(r"(?m)^\s*A\s*[\.．、:：]", text)
    if match:
        return text[: match.start()].strip()
    return text


def normalize_answer(answer: Any) -> str:
    value = str(answer or "").strip().upper()
    if value and value[0] in CHOICES:
        return value[0]
    match = re.search(r"\b([A-D])\b", value)
    return match.group(1) if match else value


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value or "").strip())
    return cleaned.strip("._") or "anonymous"


def normalize_profile_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.fullmatch(r"(?:p|profile_?)?(\d{1,3})", text, flags=re.IGNORECASE)
    if match:
        return f"p{int(match.group(1))}"
    match = re.search(r"history_with_qa_p(\d{1,3})\.json$", text, flags=re.IGNORECASE)
    if match:
        return f"p{int(match.group(1))}"
    return ""


def data_path_for_profile(profile: Any, default_data_path: Path, data_dir: Path) -> Path | None:
    profile_id = normalize_profile_id(profile)
    if not profile_id:
        return default_data_path
    pid = int(profile_id.removeprefix("p"))
    candidate = (data_dir / f"history_with_qa_p{pid}.json").resolve()
    try:
        candidate.relative_to(data_dir.resolve())
    except ValueError:
        return None
    return candidate if candidate.exists() else None


def profile_id_from_data_path(data_path: Path) -> str:
    match = re.search(r"history_with_qa_p(\d+)\.json$", data_path.name, flags=re.IGNORECASE)
    return f"p{int(match.group(1))}" if match else "default"


def _list_get(values: Any, idx: int, default: str = "") -> str:
    if isinstance(values, list) and 0 <= idx < len(values):
        return str(values[idx] or "")
    if isinstance(values, str) and idx == 0:
        return values
    return default


def extract_turn_media(
    turn: dict[str, Any], media_root: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    images: list[dict[str, str]] = []
    audios: list[dict[str, str]] = []

    for idx, (media_id, raw_path) in enumerate(zip(turn.get("image_id") or [], turn.get("input_image") or [])):
        images.append(
            {
                "id": str(media_id),
                "path": public_asset_path(str(raw_path), media_root),
                "url": media_url(str(raw_path), media_root),
                "caption": _list_get(turn.get("image_caption"), idx),
            }
        )

    for idx, (media_id, raw_path) in enumerate(zip(turn.get("voice_id") or [], turn.get("input_voice_message") or [])):
        audios.append(
            {
                "id": str(media_id),
                "path": public_asset_path(str(raw_path), media_root),
                "url": media_url(str(raw_path), media_root),
                "caption": _list_get(turn.get("voice_caption"), idx, str(turn.get("user_voice_message_caption") or "")),
            }
        )

    for key, value in turn.items():
        if not isinstance(key, str):
            continue
        lower = key.lower()
        if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            images.append(
                {
                    "id": key.rsplit(".", 1)[0],
                    "path": public_asset_path(str(value), media_root),
                    "url": media_url(str(value), media_root),
                    "caption": _list_get(turn.get("image_caption"), 0),
                }
            )
        elif lower.endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")):
            audios.append(
                {
                    "id": key.rsplit(".", 1)[0],
                    "path": public_asset_path(str(value), media_root),
                    "url": media_url(str(value), media_root),
                    "caption": str(turn.get("user_voice_message_caption") or _list_get(turn.get("voice_caption"), 0)),
                }
            )

    seen_img: set[str] = set()
    dedup_images: list[dict[str, str]] = []
    for item in images:
        key = item["id"] + "|" + item["path"]
        if key not in seen_img:
            seen_img.add(key)
            dedup_images.append(item)

    seen_audio: set[str] = set()
    dedup_audios: list[dict[str, str]] = []
    for item in audios:
        key = item["id"] + "|" + item["path"]
        if key not in seen_audio:
            seen_audio.add(key)
            dedup_audios.append(item)

    return dedup_images, dedup_audios


def textual_user(turn: dict[str, Any]) -> str:
    user = str(turn.get("user") or "").strip()
    if user:
        return user
    return str(turn.get("user_voice_message_caption") or "").strip()


def build_clue_index(sessions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for session in sessions:
        session_id = str(session.get("session_id") or "")
        for turn in session.get("dialogues", []) or []:
            round_id = str(turn.get("round") or "")
            if round_id:
                index[round_id] = {
                    "id": round_id,
                    "modality": "text",
                    "session_id": session_id,
                    "round": round_id,
                    "text": turn.get("textual_user") or turn.get("user") or "",
                    "assistant": turn.get("assistant") or "",
                }
            for image in turn.get("images", []) or []:
                media_id = str(image.get("id") or "")
                entry = {
                    "id": media_id,
                    "modality": "image",
                    "session_id": session_id,
                    "round": round_id,
                    "path": image.get("path") or "",
                    "url": image.get("url") or "",
                    "caption": image.get("caption") or "",
                }
                for key in {media_id, f"{media_id}.png", f"{media_id}.jpg", f"{media_id}.jpeg", f"{media_id}.webp"}:
                    if key:
                        index[key] = entry
            for audio in turn.get("audios", []) or []:
                media_id = str(audio.get("id") or "")
                entry = {
                    "id": media_id,
                    "modality": "audio",
                    "session_id": session_id,
                    "round": round_id,
                    "path": audio.get("path") or "",
                    "url": audio.get("url") or "",
                    "caption": audio.get("caption") or "",
                }
                for key in {media_id, f"{media_id}.wav", f"{media_id}.mp3", f"{media_id}.m4a", f"{media_id}.aac"}:
                    if key:
                        index[key] = entry
    return index


def build_payload(data_path: Path, media_root: Path) -> dict[str, Any]:
    data = read_json(data_path)
    profile_id = profile_id_from_data_path(data_path)
    sessions: list[dict[str, Any]] = []
    qas: list[dict[str, Any]] = []

    for session in data.get("multi_session_dialogues", []) or []:
        turns = []
        for turn in session.get("dialogues", []) or []:
            images, audios = extract_turn_media(turn, media_root)
            turns.append(
                {
                    "round": turn.get("round", ""),
                    "user": turn.get("user", ""),
                    "textual_user": textual_user(turn),
                    "assistant": turn.get("assistant", ""),
                    "voice_caption": turn.get("voice_caption") or [],
                    "user_voice_message_caption": turn.get("user_voice_message_caption", ""),
                    "image_caption": turn.get("image_caption") or [],
                    "images": images,
                    "audios": audios,
                }
            )
        sessions.append(
            {
                "session_id": session.get("session_id", ""),
                "date": session.get("date", ""),
                "task_id": session.get("task_id", ""),
                "scene_description": session.get("scene_description", ""),
                "user_shared_image_description": session.get("user_shared_image_description", ""),
                "background_audio_info": session.get("background_audio_info", ""),
                "dialogues": turns,
            }
        )

    clue_index = build_clue_index(sessions)

    for qa in data.get("human-annotated QAs", []) or []:
        qa_id = str(qa.get("qa_id", ""))
        if is_excluded_qa(qa):
            continue
        option_images = qa.get("option_images") or {}
        question_options = split_question_options(qa.get("question", ""))
        options = {}
        for choice in CHOICES:
            img_path = str(option_images.get(choice, "") or "")
            options[choice] = {
                "text": question_options.get(choice, ""),
                "image_path": public_asset_path(img_path, media_root),
                "image_url": media_url(img_path, media_root),
                "caption": (qa.get("option_captions") or {}).get(choice, ""),
                "description": (qa.get("question_image_descriptions") or {}).get(choice, ""),
            }
        oracle_clues = []
        for clue in qa.get("clue", []) or []:
            clue_text = str(clue)
            entry = clue_index.get(clue_text)
            if entry:
                oracle_clues.append({"clue": clue_text, **entry})
            else:
                oracle_clues.append({"clue": clue_text, "id": clue_text, "modality": "missing"})
        qas.append(
            {
                "key": qa_key(qa),
                "qa_id": qa_id,
                "question": qa.get("question", ""),
                "question_stem": question_stem(qa.get("question", "")),
                "point": qa.get("point", ""),
                "qa_type": qa.get("qa_type", ""),
                "category": qa.get("category", ""),
                "subcategory": qa.get("subcategory", ""),
                "session_id": qa.get("session_id", ""),
                "clue": qa.get("clue", []) or [],
                "oracle_clues": oracle_clues,
                "options": options,
            }
        )

    return {
        "profile": data.get("character_profile", {}),
        "profile_id": profile_id,
        "data_file": public_path(data_path),
        "variants": [
            {
                "id": "full_multimodal",
                "name": "Human (Full Multimodal)",
                "description": "阅读完整原始多模态历史，自行检索并回答。",
            },
            {
                "id": "full_text",
                "name": "Human (Full Text)",
                "description": "阅读完整 textualized history，自行检索并回答。",
            },
            {
                "id": "oracle_multimodal",
                "name": "Human (Oracle Multimodal)",
                "description": "只阅读标注的原始多模态 supporting clues。",
            },
            {
                "id": "oracle_text",
                "name": "Human (Oracle Text)",
                "description": "只阅读标注的文本 supporting clues。",
            },
        ],
        "sessions": sessions,
        "qas": qas,
    }


class HumanBaselineServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        *,
        default_data_path: Path,
        media_root: Path,
        output_dir: Path,
    ) -> None:
        super().__init__(server_address, request_handler)
        self.default_data_path = default_data_path
        self.data_dir = default_data_path.parent
        self.media_root = media_root
        self.output_dir = output_dir


class HumanBaselineHandler(BaseHTTPRequestHandler):
    server_version = "HumanBaselineDemo/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            return self.send_html(INDEX_HTML)
        if parsed.path == "/api/data":
            query = urllib.parse.parse_qs(parsed.query)
            profile = query.get("profile", [""])[0]
            data_path = data_path_for_profile(
                profile,
                self.server.default_data_path,  # type: ignore[attr-defined]
                self.server.data_dir,  # type: ignore[attr-defined]
            )
            if not data_path:
                return self.send_json({"error": f"profile not found: {profile}"}, HTTPStatus.NOT_FOUND)
            return self.send_json(build_payload(data_path, self.server.media_root))  # type: ignore[attr-defined]
        if parsed.path == "/media":
            query = urllib.parse.parse_qs(parsed.query)
            raw_path = query.get("path", [""])[0]
            return self.serve_media(raw_path)
        if parsed.path.startswith("/static/"):
            return self.serve_static(parsed.path.removeprefix("/static/"))
        return self.send_text("Not found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/submit":
            return self.send_text("Not found", HTTPStatus.NOT_FOUND)

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self.send_json({"error": "Invalid JSON"}, HTTPStatus.BAD_REQUEST)
        if not isinstance(payload, dict):
            return self.send_json({"error": "request must be an object"}, HTTPStatus.BAD_REQUEST)

        answers = payload.get("answers") or {}
        if not isinstance(answers, dict):
            return self.send_json({"error": "answers must be an object"}, HTTPStatus.BAD_REQUEST)

        data_path = data_path_for_profile(
            payload.get("profile"),
            self.server.default_data_path,  # type: ignore[attr-defined]
            self.server.data_dir,  # type: ignore[attr-defined]
        )
        if not data_path:
            return self.send_json({"error": f"profile not found: {payload.get('profile')}"}, HTTPStatus.NOT_FOUND)
        source = read_json(data_path)
        gold_by_id = {
            qa_key(qa): normalize_answer(qa.get("answer"))
            for qa in source.get("human-annotated QAs", []) or []
            if not is_excluded_qa(qa)
        }
        qas_by_id = {
            qa_key(qa): qa
            for qa in source.get("human-annotated QAs", []) or []
            if not is_excluded_qa(qa)
        }

        details = []
        correct = 0
        answered = 0
        for item_key, gold in gold_by_id.items():
            user_answer = normalize_answer(answers.get(item_key))
            is_answered = user_answer in CHOICES
            is_correct = bool(is_answered and user_answer == gold)
            if is_answered:
                answered += 1
            if is_correct:
                correct += 1
            qa = qas_by_id.get(item_key, {})
            details.append(
                {
                    "key": item_key,
                    "qa_id": qa.get("qa_id", ""),
                    "point": qa.get("point", ""),
                    "qa_type": qa.get("qa_type", ""),
                    "category": qa.get("category", ""),
                    "gold": gold,
                    "answer": user_answer,
                    "answered": is_answered,
                    "correct": is_correct,
                }
            )

        total = len(gold_by_id)
        profile_id = profile_id_from_data_path(data_path)
        variant_id = safe_filename(str(payload.get("variant") or "unknown_variant"))
        result = {
            "participant": str(payload.get("participant") or "").strip() or "anonymous",
            "variant": str(payload.get("variant") or ""),
            "profile": profile_id,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_file": public_path(data_path),
            "total": total,
            "answered": answered,
            "correct": correct,
            "accuracy": round(correct / total * 100, 2) if total else 0.0,
            "answered_accuracy": round(correct / answered * 100, 2) if answered else 0.0,
            "details": details,
        }

        filename = f"{int(time.time())}_{safe_filename(result['participant'])}.json"
        out_path = self.server.output_dir / profile_id / variant_id / filename  # type: ignore[attr-defined]
        write_json(out_path, result)
        self.send_json({"result": result, "saved_to": public_path(out_path)})

    def serve_static(self, name: str) -> None:
        target = (STATIC_DIR / urllib.parse.unquote(name).replace("\\", "/")).resolve()
        try:
            target.relative_to(STATIC_DIR.resolve())
        except ValueError:
            return self.send_text("Forbidden", HTTPStatus.FORBIDDEN)
        if not target.exists() or not target.is_file():
            return self.send_text("Not found", HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_media(self, raw_path: str) -> None:
        target = resolve_media_path(raw_path, self.server.media_root)  # type: ignore[attr-defined]
        if not target:
            return self.send_text("Forbidden", HTTPStatus.FORBIDDEN)
        if not target.exists() or not target.is_file():
            return self.send_text("Media not found", HTTPStatus.NOT_FOUND)
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Human baseline web demo for CUE-MEM QA.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="data JSON path, relative to the project root (default: benchmark/.../p0)",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="root for relative image/audio paths (default: project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for submitted result JSON files",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    data_path = resolve_path(args.data, default=default_data_path()).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data file not found: {public_path(data_path)}")

    media_root = resolve_path(args.media_root, default=PROJECT_ROOT).expanduser().resolve()
    output_dir = resolve_path(args.output_dir, default=DEFAULT_OUTPUT_DIR).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    server = HumanBaselineServer(
        (args.host, args.port),
        HumanBaselineHandler,
        default_data_path=data_path,
        media_root=media_root,
        output_dir=output_dir,
    )
    print(f"Serving data file {public_path(data_path)}")
    print(f"Profile siblings are loaded from {public_path(data_path.parent)}")
    print(f"Open local server at {args.host}:{args.port}/")
    print(f"Answers will be saved under {public_path(output_dir)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
