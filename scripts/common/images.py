"""Small runtime-configured image-generation helpers.

The original QA scripts imported a local ``aifast_image_client`` module that
was not part of the project tree.  Keeping the HTTP adapter here makes the
public scripts self-contained while leaving credentials and endpoints outside
the repository.
"""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any, Iterable

import requests

from scripts.common.llm import env_value, required_env


def _aspect_ratio(width: int = 1024, height: int = 1024) -> str:
    if width == height:
        return "1:1"
    ratio = width / max(height, 1)
    candidates = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
    }
    return min(candidates, key=lambda name: abs(candidates[name] - ratio))


def _data_part(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"inlineData": {"mimeType": mime, "data": encoded}}


def _inline_images(value: Any) -> list[str]:
    found: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            inline = node.get("inlineData") or node.get("inline_data")
            if isinstance(inline, dict):
                mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
                data = inline.get("data")
                if mime.startswith("image/") and isinstance(data, str) and data.strip():
                    found.append(data.strip())
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return found


def generate_aifast_image_to_path(
    prompt: str,
    save_path: str | Path,
    *,
    retries: int = 3,
    ref_image_paths: Iterable[str | Path] | None = None,
    model: str | None = None,
) -> bool:
    """Generate one image through a Gemini-compatible runtime endpoint.

    The endpoint and key are intentionally required at call time.  Reference
    images, when supplied, are sent as inline image parts.
    """

    key = required_env("CUE_MEM_IMAGE_AIFAST_API_KEY")
    base_url = required_env("CUE_MEM_IMAGE_AIFAST_BASE_URL").rstrip("/")
    model_name = model or env_value(
        "CUE_MEM_IMAGE_AIFAST_MODEL", "gemini-3.1-flash-image-preview"
    )
    image_size = env_value("CUE_MEM_IMAGE_AIFAST_SIZE", "1K")

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for raw_path in ref_image_paths or []:
        part = _data_part(Path(raw_path))
        if part is not None:
            parts.append(part)

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": _aspect_ratio(),
                "imageSize": image_size,
            },
        },
    }
    url = f"{base_url}/v1beta/models/{model_name}:generateContent"
    destination = Path(save_path)

    for attempt in range(max(1, retries)):
        try:
            response = requests.post(
                url,
                params={"key": key},
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            images = _inline_images(response.json())
            if not images:
                raise RuntimeError("image provider returned no inline image")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(images[0]))
            return True
        except Exception as exc:  # pragma: no cover - provider/network behavior
            if attempt + 1 >= max(1, retries):
                print(f"[AIFAST] image generation failed: {exc}")
            else:
                time.sleep(min(2**attempt, 8))
    return False
