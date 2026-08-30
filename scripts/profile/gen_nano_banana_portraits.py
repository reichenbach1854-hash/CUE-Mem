"""gen_nano_banana_portraits.py

为 profiles 中的人物（Basic.Relationship）、宠物（Basic.Pets）和物品（根级 Items）
生成肖像图，支持四种图像后端（由 IMAGE_MODEL 选择）：

- ``jimeng_4_6``：火山引擎即梦 Seedream 4.6（异步提交 + 轮询 + URL 下载）
- ``openrouter_gemini_3_1_flash_image_preview``：OpenRouter
  ``google/gemini-3.1-flash-image-preview``（chat.completions + modalities 含 image）
- ``aifast_gemini_3_pro_image_preview``：AIFast Gemini 原生
  ``gemini-3-pro-image-preview``（generateContent + inlineData）
- ``local_free_t2i``：本地 OpenAI-compatible Free-T2I 服务（images.generate + b64_json）

切换后端：修改顶层 ``IMAGE_MODEL``，或命令行
``python profile/gen_nano_banana_portraits.py --image-model <jimeng_4_6|openrouter_gemini_3_1_flash_image_preview|aifast_gemini_3_pro_image_preview|local_free_t2i>``。
AIFast Gemini 模式示例：
``python gen_nano_banana_portraits.py --image-model aifast_gemini_3_pro_image_preview``。
本地 Free-T2I 模式示例：
``python gen_nano_banana_portraits.py --image-model local_free_t2i``。
小批量测试示例：
``python -m scripts.profile.gen_nano_banana_portraits --image-model aifast_gemini_3_pro_image_preview --max-profiles 2``。
各远程或本地服务的凭据和地址必须通过运行时环境变量提供。

Items 数据结构（与 gen_profile_w_items 输出一致）：
  - description: 实体锚点短语（来自 profile 各偏好条目的 entity_anchors）
  - source_subcategory: 可选，标明该锚点所属偏好子类
  - event / source_task_id: 可能为空，生图时不依赖

流程：提交任务 → 轮询 → 下载 → 回写 img_path → 输出 manifest
断点续跑：若输出 JSON / 图片 / manifest 已存在，会合并旧 img_path 并跳过已有图片。

保存文件名（在 generated_portraits 下）：
  profile_{profile_id}_{主人名stem}_{Relationship|Pets|Items}_{人物名|宠物名|物品描述stem}_{index}.png
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from scripts.common.llm import env_value, openai_client, required_env
from scripts.common.paths import project_path, resolve_path

TokenUsage = Dict[str, Optional[int]]

# ── 路径配置（相对于项目根目录）──────────────────────────────────────────────
PERSONA_FILE_PATH = str(project_path("profile", "profiles_with_anchors_with_items.json"))
SAVE_IMAGE_DIR = str(project_path("profile", "generated_portraits"))

# ── 图像后端：二选一 ─────────────────────────────────────────────────────────
MODEL_JIMENG_4_6 = "jimeng_4_6"
MODEL_OPENROUTER_GEMINI_IMAGE = "openrouter_gemini_3_1_flash_image_preview"
MODEL_AIFAST_GEMINI_IMAGE = "aifast_gemini_3_pro_image_preview"
MODEL_LOCAL_FREE_T2I = "local_free_t2i"
IMAGE_MODEL = MODEL_JIMENG_4_6

# OpenRouter/AIFast/本地服务配置只在运行时读取。
OPENROUTER_IMAGE_MODEL = env_value(
    "CUE_MEM_IMAGE_OPENROUTER_MODEL", "google/gemini-3.1-flash-image-preview"
)

# AIFast Gemini 原生生图接口（仅当 IMAGE_MODEL == MODEL_AIFAST_GEMINI_IMAGE 时使用）
AIFAST_IMAGE_MODEL = env_value("CUE_MEM_IMAGE_AIFAST_MODEL", "gemini-3-pro-image-preview")
AIFAST_IMAGE_SIZE = env_value("CUE_MEM_IMAGE_AIFAST_SIZE", "1K")

# 本地 OpenAI-compatible Free-T2I（仅当 IMAGE_MODEL == MODEL_LOCAL_FREE_T2I 时使用）
LOCAL_T2I_IMAGE_MODEL = env_value("CUE_MEM_IMAGE_LOCAL_MODEL", "free-t2i")
LOCAL_T2I_REQUEST_INTERVAL = float(os.environ.get("CUE_MEM_IMAGE_LOCAL_INTERVAL", "60"))
LOCAL_T2I_429_BACKOFF_BASE = float(os.environ.get("CUE_MEM_IMAGE_LOCAL_BACKOFF_BASE", "60"))
LOCAL_T2I_429_BACKOFF_MAX = float(os.environ.get("CUE_MEM_IMAGE_LOCAL_BACKOFF_MAX", "600"))
_LOCAL_T2I_RATE_LOCK = threading.Lock()
_LOCAL_T2I_LAST_REQUEST_AT = 0.0

# ── 火山引擎/TOS 凭证与地址只在请求时从环境变量读取 ─────────────────────────────

# ── 即梦4.6 API 固定参数 ──────────────────────────────────────────────────────
VOLCENGINE_REGION  = env_value("CUE_MEM_IMAGE_VOLC_REGION", "cn-north-1")
VOLCENGINE_SERVICE = env_value("CUE_MEM_IMAGE_VOLC_SERVICE", "cv")
VOLCENGINE_VERSION = env_value("CUE_MEM_IMAGE_VOLC_VERSION", "2022-08-31")
REQ_KEY = "jimeng_seedream46_cvtob"

# ── 生图配置 ──────────────────────────────────────────────────────────────────
IMAGE_WIDTH   = 1024
IMAGE_HEIGHT  = 1024
FORCE_SINGLE  = True
MAX_RETRIES   = 3
POLL_INTERVAL = 3      # 轮询间隔（秒）
POLL_TIMEOUT  = 180    # 单任务最长等待（秒）

MAX_PROFILES  = 0      # 0 = 处理全部 profiles
OVERWRITE     = False  # True = 已存在的图片也重新生成

# ── 指定重新生成的条目 ────────────────────────────────────────────────────────
# 非空时：只重新生成列出的条目（强制覆盖），其余全部跳过；原有批量逻辑不受影响。
# 空列表时：按 OVERWRITE 和 MAX_PROFILES 的原有逻辑运行全部任务。
# 格式：(profile_id, type, index)
#   type 取值：'Relationship' | 'Pets' | 'Items'
# 示例（取消注释并填写即可）：
#   REGEN_TARGETS = [(1, "Relationship", 2), (1, "Items", 3)]
REGEN_TARGETS: List[Tuple[int, str, int]] = []
# ─────────────────────────────────────────────────────────────────────────────
# 火山引擎 HMAC-SHA256 V4 签名
# ─────────────────────────────────────────────────────────────────────────────

def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _get_signing_key(secret_key: str, date_str: str, region: str, service: str) -> bytes:
    k = _hmac_sha256(secret_key.encode("utf-8"), date_str)
    k = _hmac_sha256(k, region)
    k = _hmac_sha256(k, service)
    k = _hmac_sha256(k, "request")
    return k


def _volcengine_request(action: str, body: dict) -> dict:
    access_key = required_env("CUE_MEM_IMAGE_VOLC_ACCESS_KEY")
    secret_key = required_env("CUE_MEM_IMAGE_VOLC_SECRET_KEY")
    base_url = required_env("CUE_MEM_IMAGE_VOLC_BASE_URL").rstrip("/")
    host = env_value("CUE_MEM_IMAGE_VOLC_HOST") or urlparse(base_url).netloc
    if not host:
        raise RuntimeError("CUE_MEM_IMAGE_VOLC_BASE_URL must include a host")
    method  = "POST"
    uri     = "/"
    query   = f"Action={action}&Version={VOLCENGINE_VERSION}"
    payload = json.dumps(body, ensure_ascii=False)

    payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    now    = datetime.now(timezone.utc)
    x_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_str = x_date[:8]

    headers_map = {
        "content-type":    "application/json",
        "host":            host,
        "x-content-sha256": payload_hash,
        "x-date":          x_date,
    }
    canonical_headers = "".join(f"{k}:{v}\n" for k, v in sorted(headers_map.items()))
    signed_headers    = ";".join(sorted(headers_map.keys()))

    canonical_request = "\n".join([
        method, uri, query,
        canonical_headers, signed_headers, payload_hash,
    ])
    credential_scope = f"{date_str}/{VOLCENGINE_REGION}/{VOLCENGINE_SERVICE}/request"
    string_to_sign   = "\n".join([
        "HMAC-SHA256", x_date, credential_scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])

    signing_key = _get_signing_key(
        secret_key, date_str, VOLCENGINE_REGION, VOLCENGINE_SERVICE
    )
    signature = hmac.new(
        signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    authorization = (
        f"HMAC-SHA256 Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )
    for _retry in range(5):
        resp = requests.post(
            f"{base_url}/?{query}",
            headers={
                "Authorization":    authorization,
                "Content-Type":     "application/json",
                "Host":             host,
                "X-Content-Sha256": payload_hash,
                "X-Date":           x_date,
            },
            data=payload.encode("utf-8"),
            timeout=30,
        )
        if resp.status_code == 429:
            wait = 10 * (2 ** _retry)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# 任务提交 & 轮询 & 下载
# ─────────────────────────────────────────────────────────────────────────────

def _submit_task(prompt: str) -> Optional[str]:
    body: Dict[str, Any] = {
        "req_key":      REQ_KEY,
        "prompt":       prompt[:800],
        "width":        IMAGE_WIDTH,
        "height":       IMAGE_HEIGHT,
        "force_single": FORCE_SINGLE,
    }
    data = _volcengine_request("CVSync2AsyncSubmitTask", body)
    if data.get("code") != 10000:
        print(f"    [SUBMIT ERR] code={data.get('code')} msg={data.get('message')}")
        return None
    return data["data"]["task_id"]


def _poll_task(task_id: str) -> Optional[str]:
    """轮询直到完成，返回第一张图片的 URL；失败返回 None。"""
    body = {
        "req_key":  REQ_KEY,
        "task_id":  task_id,
        "req_json": json.dumps({"return_url": True}),
    }
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        for _net_retry in range(3):
            try:
                data = _volcengine_request("CVSync2AsyncGetResult", body)
                break
            except requests.exceptions.ConnectionError:
                if _net_retry < 2:
                    time.sleep(3)
                else:
                    raise
        if data.get("code") != 10000:
            print(f"    [POLL ERR] code={data.get('code')} msg={data.get('message')}")
            return None
        task_data = data.get("data") or {}
        status    = task_data.get("status", "")
        if status == "done":
            urls = task_data.get("image_urls") or []
            return urls[0] if urls else None
        if status in ("not_found", "expired"):
            print(f"    [POLL] task {task_id}: status={status}")
            return None
        time.sleep(POLL_INTERVAL)
    print(f"    [TIMEOUT] task {task_id} exceeded {POLL_TIMEOUT}s")
    return None


def _download_image(url: str, save_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    [DOWNLOAD ERR] {exc}")
        return False


def _save_from_data_url(data_url: str, save_path: str) -> bool:
    """将 data:image/...;base64,... 写入磁盘。"""
    try:
        if "," not in data_url or not data_url.strip().lower().startswith("data:"):
            return False
        _, b64_part = data_url.split(",", 1)
        raw = base64.b64decode(b64_part)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True
    except Exception as exc:
        print(f"    [DATA_URL SAVE ERR] {exc}")
        return False


def _save_image_from_url_or_dataurl(src: str, save_path: str) -> bool:
    s = (src or "").strip()
    if not s:
        return False
    if s.lower().startswith("data:"):
        return _save_from_data_url(s, save_path)
    if urlparse(s).scheme in {"http", "https"}:
        return _download_image(s, save_path)
    print(f"    [SAVE ERR] unsupported image source prefix: {s[:80]!r}...")
    return False


def _empty_token_usage() -> TokenUsage:
    return {"input_tokens": None, "output_tokens": None, "total_tokens": None}


def _token_usage_from_openai_usage(usage: Any) -> TokenUsage:
    if usage is None:
        return _empty_token_usage()
    if hasattr(usage, "model_dump"):
        try:
            data = usage.model_dump()
        except Exception:
            data = {}
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {}

    input_tokens = (
        data.get("input_tokens")
        or data.get("prompt_tokens")
        or data.get("promptTokens")
    )
    output_tokens = (
        data.get("output_tokens")
        or data.get("completion_tokens")
        or data.get("completionTokens")
    )
    total_tokens = data.get("total_tokens") or data.get("totalTokens")
    return {
        "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
        "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
        "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
    }


def _token_usage_from_aifast_response(data: Any) -> TokenUsage:
    if not isinstance(data, dict):
        return _empty_token_usage()
    usage = data.get("usageMetadata") or data.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        return _empty_token_usage()
    input_tokens = usage.get("promptTokenCount") or usage.get("prompt_token_count")
    output_tokens = (
        usage.get("candidatesTokenCount")
        or usage.get("candidates_token_count")
        or usage.get("outputTokenCount")
        or usage.get("output_token_count")
    )
    total_tokens = usage.get("totalTokenCount") or usage.get("total_token_count")
    return {
        "input_tokens": input_tokens if isinstance(input_tokens, int) else None,
        "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
        "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
    }


def _print_token_usage(image_model: str, usage: TokenUsage) -> None:
    if all(usage.get(k) is None for k in ("input_tokens", "output_tokens", "total_tokens")):
        print(f"    [TOKENS] model={image_model} input=None output=None total=None")
        return
    print(
        f"    [TOKENS] model={image_model} "
        f"input={usage.get('input_tokens')} "
        f"output={usage.get('output_tokens')} "
        f"total={usage.get('total_tokens')}"
    )


def _extract_urls_from_openrouter_message(message: Any) -> List[str]:
    """从 OpenRouter / OpenAI chat 返回的 assistant message 中取出图片 URL 或 data URL。"""
    urls: List[str] = []
    if message is None:
        return urls
    if hasattr(message, "model_dump"):
        try:
            data = message.model_dump()
        except Exception:
            data = {}
    elif isinstance(message, dict):
        data = message
    else:
        data = {}
    images = data.get("images")
    if images is None:
        images = getattr(message, "images", None)
    if not images:
        return urls
    for img in images:
        if isinstance(img, dict):
            iu = img.get("image_url")
            if isinstance(iu, dict):
                u = iu.get("url")
            elif isinstance(iu, str):
                u = iu
            else:
                u = None
            if not u:
                u = img.get("url")
        else:
            iu = getattr(img, "image_url", None)
            u = getattr(iu, "url", None) if iu is not None else getattr(img, "url", None)
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    return urls


def _openrouter_generate_and_save(prompt: str, save_path: str) -> Tuple[bool, TokenUsage]:
    """调用 OpenRouter Gemini 图像模型，将第一张图写入 save_path。"""
    key = env_value("CUE_MEM_IMAGE_OPENROUTER_API_KEY")
    if not key:
        print("    [ERR] CUE_MEM_IMAGE_OPENROUTER_API_KEY 未设置")
        return False, _empty_token_usage()
    try:
        client = openai_client(
            api_key_env="CUE_MEM_IMAGE_OPENROUTER_API_KEY",
            base_url_env="CUE_MEM_IMAGE_OPENROUTER_BASE_URL",
        )
        completion = client.chat.completions.create(
            model=OPENROUTER_IMAGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            extra_body={"modalities": ["image", "text"]},
        )
    except Exception as exc:
        print(f"    [OPENROUTER ERR] request failed: {exc}")
        return False, _empty_token_usage()
    usage = _token_usage_from_openai_usage(getattr(completion, "usage", None))
    if not completion.choices:
        print("    [OPENROUTER ERR] empty choices")
        return False, usage
    message = completion.choices[0].message
    urls = _extract_urls_from_openrouter_message(message)
    if not urls:
        print("    [OPENROUTER ERR] no images in response (check modalities / model id)")
        return False, usage
    return _save_image_from_url_or_dataurl(urls[0], save_path), usage


def _aifast_aspect_ratio_from_dimensions(width: int, height: int) -> str:
    if width == height:
        return "1:1"
    ratio = width / max(height, 1)
    candidates = {
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
    }
    return min(candidates, key=lambda k: abs(candidates[k] - ratio))


def _extract_aifast_inline_images(data: Any) -> List[str]:
    """从 Gemini 原生 generateContent 响应中提取 inlineData.data base64 图片。"""
    images: List[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            inline = node.get("inlineData") or node.get("inline_data")
            if isinstance(inline, dict):
                mime = inline.get("mimeType") or inline.get("mime_type") or ""
                b64 = inline.get("data")
                if isinstance(b64, str) and b64.strip() and str(mime).startswith("image/"):
                    images.append(b64.strip())
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(data)
    return images


def _aifast_gemini_generate_and_save(prompt: str, save_path: str) -> Tuple[bool, TokenUsage]:
    """调用 AIFast Gemini 原生 generateContent 接口，将第一张 inlineData 图片写入 save_path。"""
    key = env_value("CUE_MEM_IMAGE_AIFAST_API_KEY")
    if not key:
        print("    [AIFAST ERR] CUE_MEM_IMAGE_AIFAST_API_KEY 未设置")
        return False, _empty_token_usage()

    base_url = (env_value("CUE_MEM_IMAGE_AIFAST_BASE_URL") or "").rstrip("/")
    if not base_url:
        print("    [AIFAST ERR] CUE_MEM_IMAGE_AIFAST_BASE_URL 未设置")
        return False, _empty_token_usage()
    model = (env_value("CUE_MEM_IMAGE_AIFAST_MODEL", AIFAST_IMAGE_MODEL) or "").strip()
    image_size = (env_value("CUE_MEM_IMAGE_AIFAST_SIZE", AIFAST_IMAGE_SIZE) or "").strip()
    aspect_ratio = _aifast_aspect_ratio_from_dimensions(IMAGE_WIDTH, IMAGE_HEIGHT)
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": image_size,
            },
        },
    }
    try:
        resp = requests.post(
            url,
            params={"key": key},
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        if resp.status_code != 200:
            print(f"    [AIFAST ERR] HTTP {resp.status_code}: {resp.text[:500]}")
            return False, _empty_token_usage()
        data = resp.json()
    except Exception as exc:
        print(f"    [AIFAST ERR] request failed: {exc}")
        return False, _empty_token_usage()

    usage = _token_usage_from_aifast_response(data)
    images = _extract_aifast_inline_images(data)
    if not images:
        text_parts: List[str] = []
        for candidate in data.get("candidates", []) if isinstance(data, dict) else []:
            for part in candidate.get("content", {}).get("parts", []):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())
        hint = f"; text response: {text_parts[0][:300]}" if text_parts else ""
        print(f"    [AIFAST ERR] no inline image in response{hint}")
        return False, usage

    try:
        raw = base64.b64decode(images[0])
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True, usage
    except Exception as exc:
        print(f"    [AIFAST ERR] save failed: {exc}")
        return False, usage


def _local_t2i_size_from_dimensions(width: int, height: int) -> str:
    """将脚本宽高转换成本地 Free-T2I 服务支持的 OpenAI size 字符串。"""
    exact_sizes = {
        (1024, 1024): "1024x1024",
        (1024, 1792): "1024x1792",
        (1792, 1024): "1792x1024",
        (512, 512): "512x512",
        (256, 256): "256x256",
    }
    exact = exact_sizes.get((width, height))
    if exact:
        return exact
    if width == height:
        fallback = "1024x1024"
    elif width > height:
        fallback = "1792x1024"
    else:
        fallback = "1024x1792"
    print(
        f"    [LOCAL_T2I WARN] unsupported IMAGE_WIDTH/IMAGE_HEIGHT="
        f"{width}x{height}; using nearest supported size {fallback}"
    )
    return fallback


class LocalT2IRateLimitError(RuntimeError):
    """本地 Free-T2I 上游返回 429 时用于触发指数退避。"""


def _wait_for_local_t2i_slot() -> None:
    """限制 local_free_t2i 请求发起频率，避免连续提交触发上游 429。"""
    global _LOCAL_T2I_LAST_REQUEST_AT
    if LOCAL_T2I_REQUEST_INTERVAL <= 0:
        return
    with _LOCAL_T2I_RATE_LOCK:
        now = time.time()
        elapsed = now - _LOCAL_T2I_LAST_REQUEST_AT
        wait = LOCAL_T2I_REQUEST_INTERVAL - elapsed
        if wait > 0:
            print(f"    [LOCAL_T2I WAIT] request interval cooldown: {wait:.1f}s")
            time.sleep(wait)
        _LOCAL_T2I_LAST_REQUEST_AT = time.time()


def _is_local_t2i_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return (
        status_code == 429
        or "upstream http 429" in text
        or "too frequent" in text
        or "rate limit" in text
    )


def _local_t2i_generate_and_save(prompt: str, save_path: str) -> Tuple[bool, TokenUsage]:
    """调用本地 OpenAI-compatible Free-T2I 服务，将 b64_json 解码写入 save_path。"""
    base_url = (env_value("CUE_MEM_IMAGE_LOCAL_BASE_URL") or "").strip()
    api_key = (env_value("CUE_MEM_IMAGE_LOCAL_API_KEY") or "").strip()
    if not base_url:
        print("    [LOCAL_T2I ERR] CUE_MEM_IMAGE_LOCAL_BASE_URL 未设置")
        return False, _empty_token_usage()
    if not api_key:
        print("    [LOCAL_T2I ERR] CUE_MEM_IMAGE_LOCAL_API_KEY 未设置")
        return False, _empty_token_usage()

    size = _local_t2i_size_from_dimensions(IMAGE_WIDTH, IMAGE_HEIGHT)
    try:
        _wait_for_local_t2i_slot()
        client = openai_client(
            api_key=api_key,
            base_url=base_url,
            api_key_env="CUE_MEM_IMAGE_LOCAL_API_KEY",
            base_url_env="CUE_MEM_IMAGE_LOCAL_BASE_URL",
        )
        result = client.images.generate(
            model=LOCAL_T2I_IMAGE_MODEL,
            prompt=prompt,
            n=1,
            size=size,
        )
    except Exception as exc:
        if _is_local_t2i_rate_limit_error(exc):
            print(f"    [LOCAL_T2I 429] request rate limited: {exc}")
            raise LocalT2IRateLimitError(str(exc)) from exc
        print(f"    [LOCAL_T2I ERR] request failed: {exc}")
        return False, _empty_token_usage()

    try:
        usage = _token_usage_from_openai_usage(getattr(result, "usage", None))
        if not result.data:
            print("    [LOCAL_T2I ERR] empty image data")
            return False, usage
        b64_json = getattr(result.data[0], "b64_json", None)
        if not b64_json:
            print("    [LOCAL_T2I ERR] missing b64_json in response")
            return False, usage
        raw = base64.b64decode(b64_json)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True, usage
    except Exception as exc:
        print(f"    [LOCAL_T2I ERR] save failed: {exc}")
        return False, _empty_token_usage()


def _jimeng_generate_and_save(prompt: str, save_path: str) -> Tuple[bool, TokenUsage]:
    """火山即梦：提交 → 轮询 → 下载。"""
    task_id = _submit_task(prompt)
    if not task_id:
        return False, _empty_token_usage()
    img_url = _poll_task(task_id)
    if not img_url:
        return False, _empty_token_usage()
    return _download_image(img_url, save_path), _empty_token_usage()


def _generate_image_to_path(image_model: str, prompt: str, save_path: str) -> Tuple[bool, TokenUsage]:
    if image_model == MODEL_JIMENG_4_6:
        return _jimeng_generate_and_save(prompt, save_path)
    if image_model == MODEL_OPENROUTER_GEMINI_IMAGE:
        return _openrouter_generate_and_save(prompt, save_path)
    if image_model == MODEL_AIFAST_GEMINI_IMAGE:
        return _aifast_gemini_generate_and_save(prompt, save_path)
    if image_model == MODEL_LOCAL_FREE_T2I:
        return _local_t2i_generate_and_save(prompt, save_path)
    print(f"    [ERR] 不支持的 IMAGE_MODEL: {image_model!r}")
    return False, _empty_token_usage()


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 构建
# ─────────────────────────────────────────────────────────────────────────────

CHINESE_FAMILY_PHOTO_ANCHORS = {
    "钉满女儿奖状的软木展示板",
    "贴有女儿从小学到中学照片的展示板",
    "原木色宽边相框黑白合影照",
    "原木色相框孙辈院中合影照片",
    "相框装裱的海边旅拍照片",
    "黑色实木相框军装合影照",
}

CHINESE_FIREFIGHTER_ANCHORS = {
    "红色磨损消防头盔",
    "金属消防员徽章",
}

def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_person_prompt(profile: Dict[str, Any], rel: Dict[str, Any]) -> str:
    rel_name   = clean_text(rel.get("name"))
    desc       = clean_text(rel.get("info") or rel.get("person"))
    appearance = clean_text(rel.get("appearance"))

    return (
        "Generate one realistic photographic portrait. "
        "Requirements: pure white background (#FFFFFF), single person, front-facing half-body, "
        "centered subject, clean composition, soft and even lighting, "
        "no obvious shadows, no environment elements, no props, no text, no watermark. "
        f"Portrait subject name: {rel_name}. "
        f"Person profile: {desc}. "
        f"Appearance focus: {appearance}. "
        "Faithfully render appearance details and temperament. Output one high-quality image."
    )


def build_pet_prompt(profile: Dict[str, Any], pet: Dict[str, Any]) -> str:
    owner_name = clean_text(profile.get("Basic", {}).get("name"))
    nickname   = clean_text(pet.get("name"))
    desc       = clean_text(pet.get("info") or pet.get("description"))
    appearance = clean_text(pet.get("appearance"))

    return (
        "Generate one realistic photographic pet portrait. "
        "Requirements: pure white background (#FFFFFF), one pet, front or slight side view, "
        "centered subject, clean composition, soft and even lighting, "
        "no obvious shadows, no environment elements, no props, no text, no watermark. "
        f"Owner: {owner_name}. "
        f"Pet name: {nickname}. "
        f"Pet description: {desc}. "
        f"Appearance focus: {appearance}. "
        "Faithfully render coat, posture, and facial features. Output one high-quality image."
    )


def build_item_prompt(profile: Dict[str, Any], item: Dict[str, Any]) -> str:
    """Items 仅保证有 description；source_subcategory 为可选上下文。"""
    description = clean_text(item.get("description"))
    subcat = clean_text(item.get("source_subcategory"))
    context = (
        f"Related lifestyle context (subcategory, not to render as scene): {subcat}. "
        if subcat
        else ""
    )
    extra_instruction = ""
    if description in CHINESE_FAMILY_PHOTO_ANCHORS:
        extra_instruction = (
            "If the object contains any printed, framed, or pinned photograph of people, "
            "the people shown inside that photo must look like an ordinary Chinese family: "
            "natural Chinese/East Asian faces, black or dark-brown hair, realistic everyday "
            "appearance, and no Caucasian or Western-looking facial features. "
        )
    elif description in CHINESE_FIREFIGHTER_ANCHORS:
        extra_instruction = (
            "The object must clearly match the visual identity of Chinese firefighters: "
            "use realistic Chinese fire-rescue styling, Chinese firefighter helmet/badge design cues, "
            "and avoid American, European, or generic Western fire department symbols or lettering. "
        )
    return (
        "Generate one realistic photographic object portrait. "
        "Requirements: pure white background (#FFFFFF), one object, front or slight side view, "
        "centered subject, clean composition, soft and even lighting, "
        "no obvious shadows, no environment elements, no props, no text, no watermark. "
        f"{context}"
        f"Object description: {description}. "
        f"{extra_instruction}"
        "Faithfully render object details and texture. Output one high-quality image."
    )


def safe_stem(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fa5]", "", text)
    return text[:40] if text else "unknown"


def portrait_file_basename(
    profile_id: int,
    name_stem: str,
    item_type: str,
    item_idx: int,
    item: Dict[str, Any],
) -> str:
    """
    生成保存用文件名（不含目录）：
    profile_{id}_{主人名stem}_{类型}_{人物名|宠物名|物品描述stem}_{序号}.png
    末尾仍保留序号，避免同名人物/重复描述 stem 冲突。
    """
    if item_type == "Relationship":
        label = clean_text(item.get("name"))
    elif item_type == "Pets":
        label = clean_text(item.get("name"))
    elif item_type == "Items":
        label = clean_text(item.get("description"))
    else:
        label = ""
    label_stem = safe_stem(label) if label else "unknown"
    return f"profile_{profile_id}_{name_stem}_{item_type}_{label_stem}_{item_idx}.png"


def _copy_img_paths_from_existing_profiles(
    profiles: List[Dict[str, Any]],
    existing_profiles: List[Dict[str, Any]],
) -> int:
    """把已有输出文件里的 img_path 合并回当前 profiles，避免断点续跑覆盖旧路径。"""
    copied = 0

    def item_match_key(item: Any, item_type: str) -> str:
        if not isinstance(item, dict):
            return ""
        if item_type in ("Relationship", "Pets"):
            return safe_stem(clean_text(item.get("name") or item.get("relation")))
        if item_type == "Items":
            return safe_stem(clean_text(item.get("description")))
        return ""

    def copy_list_paths(current_items: Any, existing_items: Any, item_type: str) -> None:
        nonlocal copied
        if not isinstance(current_items, list) or not isinstance(existing_items, list):
            return
        existing_by_key: Dict[str, Dict[str, Any]] = {}
        for existing_item in existing_items:
            key = item_match_key(existing_item, item_type)
            if key and key not in existing_by_key:
                existing_by_key[key] = existing_item
        for current_item in current_items:
            if not isinstance(current_item, dict):
                continue
            existing_item = existing_by_key.get(item_match_key(current_item, item_type))
            if not isinstance(existing_item, dict):
                continue
            old_path = clean_text(existing_item.get("img_path"))
            if old_path:
                current_item["img_path"] = old_path
                copied += 1

    for current_profile, existing_profile in zip(profiles, existing_profiles):
        if not isinstance(current_profile, dict) or not isinstance(existing_profile, dict):
            continue
        current_basic = current_profile.get("Basic", {})
        existing_basic = existing_profile.get("Basic", {})
        if isinstance(current_basic, dict) and isinstance(existing_basic, dict):
            copy_list_paths(current_basic.get("Relationship"), existing_basic.get("Relationship"), "Relationship")
            copy_list_paths(current_basic.get("Pets"), existing_basic.get("Pets"), "Pets")
        copy_list_paths(current_profile.get("Items"), existing_profile.get("Items"), "Items")
    return copied


def iter_tasks(profile: Dict[str, Any]) -> List[Tuple[str, int, Dict[str, Any], str]]:
    """枚举 profile 中所有需要生成图片的条目，返回 (type, idx, item, prompt) 列表。"""
    tasks: List[Tuple[str, int, Dict[str, Any], str]] = []
    basic = profile.get("Basic", {}) or {}

    for idx, rel in enumerate(basic.get("Relationship", []) or []):
        if not isinstance(rel, dict) or not clean_text(rel.get("appearance")):
            continue
        tasks.append(("Relationship", idx, rel, build_person_prompt(profile, rel)))

    for idx, pet in enumerate(basic.get("Pets", []) or []):
        if not isinstance(pet, dict) or not clean_text(pet.get("appearance")):
            continue
        tasks.append(("Pets", idx, pet, build_pet_prompt(profile, pet)))

    for idx, item in enumerate(profile.get("Items", []) or []):
        if not isinstance(item, dict):
            continue
        # 新 schema：以 description 为主；兼容旧数据中带长 event 的 Items
        if not clean_text(item.get("description")):
            continue
        tasks.append(("Items", idx, item, build_item_prompt(profile, item)))

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# 单任务生成
# ─────────────────────────────────────────────────────────────────────────────

def session(task: Tuple, profile_id: int, name_stem: str, image_model: str) -> Dict[str, Any]:
    item_type, item_idx, item, prompt = task
    file_name = portrait_file_basename(profile_id, name_stem, item_type, item_idx, item)
    save_path = os.path.join(SAVE_IMAGE_DIR, file_name)

    src_name = manifest_source_name(item_type, item)

    existing_img_path = clean_text(item.get("img_path"))
    if existing_img_path and os.path.exists(existing_img_path) and not OVERWRITE:
        return {
            "profile_id": profile_id,
            "type":       item_type,
            "index":      item_idx,
            "file":       existing_img_path,
            "status":     "skipped_exists",
            "source_name": src_name,
            "image_model": image_model,
            **_empty_token_usage(),
        }

    if os.path.exists(save_path) and not OVERWRITE:
        return {
            "profile_id": profile_id,
            "type":       item_type,
            "index":      item_idx,
            "file":       save_path,
            "status":     "skipped_exists",
            "source_name": src_name,
            "image_model": image_model,
            **_empty_token_usage(),
        }

    last_error = "unknown"
    last_usage = _empty_token_usage()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ok, usage = _generate_image_to_path(image_model, prompt, save_path)
            last_usage = usage
            _print_token_usage(image_model, usage)
            if ok:
                return {
                    "profile_id":  profile_id,
                    "type":        item_type,
                    "index":       item_idx,
                    "file":        save_path,
                    "status":      "ok",
                    "source_name": src_name,
                    "image_model": image_model,
                    **usage,
                }
            last_error = "generate_failed"
            if attempt < MAX_RETRIES:
                time.sleep(60 * attempt)
        except LocalT2IRateLimitError as exc:
            last_error = f"rate_limited: {exc}"
            if attempt < MAX_RETRIES:
                wait = min(
                    LOCAL_T2I_429_BACKOFF_MAX,
                    LOCAL_T2I_429_BACKOFF_BASE * (2 ** (attempt - 1)),
                )
                print(f"    [LOCAL_T2I 429] exponential backoff: {wait:.1f}s")
                time.sleep(wait)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_RETRIES:
                time.sleep(3 * attempt)

    return {
        "profile_id": profile_id,
        "type":       item_type,
        "index":      item_idx,
        "file":       "",
        "status":     f"failed: {last_error}",
        "source_name": src_name,
        "image_model": image_model,
        **last_usage,
    }


def manifest_source_name(item_type: str, item: Dict[str, Any]) -> str:
    if item_type == "Items":
        desc = clean_text(item.get("description"))
        sub = clean_text(item.get("source_subcategory"))
        return f"{desc} ({sub})" if sub else desc
    return clean_text(
        item.get("name") or item.get("relation") or item.get("description") or ""
    )


def build_current_manifest_index(profiles: List[Dict[str, Any]]) -> Dict[Tuple[int, str, int], str]:
    """Return current valid manifest keys and expected source_name values."""
    current: Dict[Tuple[int, str, int], str] = {}
    for profile_id, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            continue
        for task in iter_tasks(profile):
            item_type, item_idx, item_obj = task[0], task[1], task[2]
            current[(profile_id, item_type, item_idx)] = manifest_source_name(item_type, item_obj)
    return current


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main(
    image_model: Optional[str] = None,
    max_profiles: Optional[int] = None,
    profile_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    global PERSONA_FILE_PATH, SAVE_IMAGE_DIR
    if profile_path is not None:
        PERSONA_FILE_PATH = str(resolve_path(profile_path))
    if output_dir is not None:
        SAVE_IMAGE_DIR = str(resolve_path(output_dir))
    resolved_model = (image_model or IMAGE_MODEL).strip()
    resolved_max_profiles = MAX_PROFILES if max_profiles is None else max_profiles
    if resolved_max_profiles < 0:
        raise ValueError(f"--max-profiles 不能为负数: {resolved_max_profiles}")
    valid_models = (
        MODEL_JIMENG_4_6,
        MODEL_OPENROUTER_GEMINI_IMAGE,
        MODEL_AIFAST_GEMINI_IMAGE,
        MODEL_LOCAL_FREE_T2I,
    )
    if resolved_model not in valid_models:
        raise ValueError(
            f"不支持的 IMAGE_MODEL={resolved_model!r}，请使用 "
            f"{MODEL_JIMENG_4_6!r}、{MODEL_OPENROUTER_GEMINI_IMAGE!r}、"
            f"{MODEL_AIFAST_GEMINI_IMAGE!r} "
            f"或 {MODEL_LOCAL_FREE_T2I!r}"
        )
    if resolved_model == MODEL_OPENROUTER_GEMINI_IMAGE:
        if not env_value("CUE_MEM_IMAGE_OPENROUTER_API_KEY"):
            raise ValueError("OpenRouter 模式需要设置 CUE_MEM_IMAGE_OPENROUTER_API_KEY")
        if not env_value("CUE_MEM_IMAGE_OPENROUTER_BASE_URL"):
            raise ValueError("OpenRouter 模式需要设置 CUE_MEM_IMAGE_OPENROUTER_BASE_URL")
    if resolved_model == MODEL_AIFAST_GEMINI_IMAGE:
        if not env_value("CUE_MEM_IMAGE_AIFAST_API_KEY"):
            raise ValueError("AIFast 模式需要设置 CUE_MEM_IMAGE_AIFAST_API_KEY")
        if not env_value("CUE_MEM_IMAGE_AIFAST_BASE_URL"):
            raise ValueError("AIFast 模式需要设置 CUE_MEM_IMAGE_AIFAST_BASE_URL")
    if resolved_model == MODEL_LOCAL_FREE_T2I:
        if not env_value("CUE_MEM_IMAGE_LOCAL_API_KEY"):
            raise ValueError("本地 T2I 模式需要设置 CUE_MEM_IMAGE_LOCAL_API_KEY")
        if not env_value("CUE_MEM_IMAGE_LOCAL_BASE_URL"):
            raise ValueError("本地 T2I 模式需要设置 CUE_MEM_IMAGE_LOCAL_BASE_URL")
    if resolved_model == MODEL_JIMENG_4_6:
        for env_name in (
            "CUE_MEM_IMAGE_VOLC_ACCESS_KEY",
            "CUE_MEM_IMAGE_VOLC_SECRET_KEY",
            "CUE_MEM_IMAGE_VOLC_BASE_URL",
        ):
            if not env_value(env_name):
                raise ValueError(f"即梦模式需要设置 {env_name}")

    os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)
    print(f"[Config] IMAGE_MODEL = {resolved_model}")
    print(f"[Config] MAX_PROFILES = {resolved_max_profiles} (0 = all)")

    persona_path = Path(PERSONA_FILE_PATH)
    if not persona_path.is_file():
        raise FileNotFoundError(f"Profile 文件不存在: {persona_path}")

    # 推导输出路径（与输入同目录）
    in_path = Path(PERSONA_FILE_PATH).resolve()
    if in_path.name.endswith("_with_items.json"):
        output_path = str(in_path.with_name(in_path.name.replace("_with_items.json", "_with_images_entity.json")))
    else:
        output_path = str(in_path.with_name(f"{in_path.stem}_with_images_entity{in_path.suffix}"))

    if Path(output_path).resolve() == in_path:
        raise ValueError(f"输出路径与输入路径相同，拒绝覆盖: {PERSONA_FILE_PATH}")

    regen_set: set = {(p, t, i) for p, t, i in REGEN_TARGETS}
    with open(persona_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    # ── 断点续跑：从已有输出文件合并 img_path，避免重新写 output 时清空旧路径 ──
    if Path(output_path).exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_profiles = json.load(f)
            copied = _copy_img_paths_from_existing_profiles(profiles, existing_profiles)
            print(f"[Resume] 已从已有输出合并 img_path: {copied} 条 → {output_path}")
        except Exception as exc:
            print(f"[WARN] 读取已有输出以恢复 img_path 失败 ({exc})，将从输入文件继续")

    work_profiles = profiles[:resolved_max_profiles] if resolved_max_profiles > 0 else profiles

    manifest: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for profile_id, profile in enumerate(work_profiles):
            profile_name = clean_text(profile.get("Basic", {}).get("name", "unknown"))
            name_stem    = safe_stem(profile_name)
            for task in iter_tasks(profile):
                item_type, item_idx, item_obj = task[0], task[1], task[2]
                key = (profile_id, item_type, item_idx)

                if regen_set:
                    # REGEN_TARGETS 非空：只提交命中的条目，并临时强制覆盖
                    if key not in regen_set:
                        continue
                    # 临时覆盖 OVERWRITE 标志：直接把 save_path 的旧文件删掉，
                    # session() 内的 `if exists and not OVERWRITE` 分支就不会触发
                    file_name = portrait_file_basename(
                        profile_id, name_stem, item_type, item_idx, item_obj
                    )
                    save_path = os.path.join(SAVE_IMAGE_DIR, file_name)
                    if os.path.exists(save_path):
                        os.remove(save_path)
                        print(f"[REGEN] 已删除旧文件 → {save_path}")

                futures.append(
                    executor.submit(session, task, profile_id, name_stem, resolved_model)
                )

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Generating portraits",
        ):
            result = future.result()
            manifest.append(result)

            # 回写 img_path
            p_id      = result["profile_id"]
            item_type = result["type"]
            item_idx  = result["index"]
            img_file  = result["file"] if result["status"] in ("ok", "skipped_exists") else ""
            if not img_file:
                continue
            try:
                if item_type in ("Relationship", "Pets"):
                    profiles[p_id]["Basic"][item_type][item_idx]["img_path"] = img_file
                else:
                    profiles[p_id][item_type][item_idx]["img_path"] = img_file
            except (KeyError, IndexError, TypeError) as exc:
                print(
                    f"[WARN] img_path 回写失败: profile={p_id} "
                    f"type={item_type} index={item_idx} err={type(exc).__name__}: {exc}"
                )

    # 写 manifest：断点续跑时把新结果合并进已有 manifest，避免覆盖其他条目。
    # 同时清理已经不对应当前 profile 偏好/实体的旧 manifest 条目，避免 gallery 展示孤立旧图片。
    manifest_path = os.path.join(SAVE_IMAGE_DIR, "manifest.json")
    current_manifest_index = build_current_manifest_index(profiles)
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                old_manifest = json.load(f)
            filtered_old_manifest = []
            dropped_old = 0
            for row in old_manifest:
                try:
                    key = (row["profile_id"], row["type"], row["index"])
                except (KeyError, TypeError):
                    dropped_old += 1
                    continue
                expected_source = current_manifest_index.get(key)
                if expected_source is None:
                    dropped_old += 1
                    continue
                if clean_text(row.get("source_name")) != expected_source:
                    dropped_old += 1
                    continue
                filtered_old_manifest.append(row)
            old_manifest = filtered_old_manifest
            # 以 (profile_id, type, index) 为键构建旧条目索引
            old_index = {
                (r["profile_id"], r["type"], r["index"]): i
                for i, r in enumerate(old_manifest)
            }
            for new_r in manifest:
                key = (new_r["profile_id"], new_r["type"], new_r["index"])
                if key in old_index:
                    old_manifest[old_index[key]] = new_r   # 原地替换
                else:
                    old_manifest.append(new_r)             # 追加新条目
            manifest = old_manifest
            print(
                f"[Manifest] 已合并旧 manifest，当前共 {len(manifest)} 条"
                f"，清理无对应偏好的旧条目 {dropped_old} 条"
            )
        except Exception as exc:
            print(f"[WARN] 合并旧 manifest 失败 ({exc})，将直接写入新 manifest")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 写入 profiles（output_path 已在文件开头推导）
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    ok_count      = sum(1 for r in manifest if r["status"] == "ok")
    skipped_count = sum(1 for r in manifest if r["status"] == "skipped_exists")
    failed_count  = sum(1 for r in manifest if r["status"].startswith("failed"))

    print(f"Done: total={len(futures)}, ok={ok_count}, skipped={skipped_count}, failed={failed_count}")
    print(f"Manifest: {manifest_path}")
    print(f"Output:   {output_path}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(description="为 Relationship / Pets / Items 生成肖像图")
    _parser.add_argument(
        "--image-model",
        choices=[
            MODEL_JIMENG_4_6,
            MODEL_OPENROUTER_GEMINI_IMAGE,
            MODEL_AIFAST_GEMINI_IMAGE,
            MODEL_LOCAL_FREE_T2I,
        ],
        default=None,
        help="覆盖脚本中的 IMAGE_MODEL 常量",
    )
    _parser.add_argument(
        "--max-profiles",
        type=int,
        default=None,
        help="只处理前 N 个 profile；0 或不传表示处理全部",
    )
    _parser.add_argument(
        "--profile-path",
        default=None,
        help="profile JSON/JSONL input, relative to the project root",
    )
    _parser.add_argument(
        "--output-dir",
        default=None,
        help="directory for generated images, relative to the project root",
    )
    _args = _parser.parse_args()
    main(
        image_model=_args.image_model,
        max_profiles=_args.max_profiles,
        profile_path=_args.profile_path,
        output_dir=_args.output_dir,
    )
