"""gen_images_from_descriptions.py

批量根据事件中的 ``user_shared_image_description`` 生成图片，支持三种后端（由 ``IMAGE_MODEL`` 控制）：

- ``jimeng_4_6``：火山引擎即梦 Seedream 4.6（异步 + 轮询；可选 TOS 参考图 URL）
- ``openrouter_gemini_3_1_flash_image_preview``：OpenRouter
  ``google/gemini-3.1-flash-image-preview``（仅文本 prompt + modalities；**不使用** TOS 参考图 URL）
- ``aifast_gemini_3_pro_image_preview``：AIFast Gemini 原生
  ``gemini-3-pro-image-preview``（generateContent + inlineData；可使用本地参考图）

流程：
    1. 扫描 profiles，收集所有有 user_shared_image_description 的事件
    2. 以 (p_id, task_id) 为粒度去重，构建 prompt
    3. （仅即梦）本地参考图上传到 TOS，获得公网 URL，写入 ``image_urls``
    4. 调用所选后端生成图 → 保存到 ``OUTPUT_DIR``
    5. 回写 img_path 到 profiles，输出 manifest.json

**仅重新生成**：将 ``REGEN_TASK_IDS`` 设为非空 task_id 列表，或使用命令行
``--task-id <id>``（可多次）；将只处理这些 task，并删除已存在的输出文件后强制重画。

TOS 参考图功能需要在运行环境中提供 TOS 凭据、地址和 bucket 配置；
这些值不写入代码。

若不需要参考图功能，将 USE_REFERENCE_IMAGES 设为 False 即可。
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import concurrent.futures
import requests
from tqdm import tqdm
from urllib.parse import urlparse

from scripts.common.llm import env_value, openai_client, required_env
from scripts.common.paths import project_path, resolve_path

# ── 路径配置（相对于项目根目录）──────────────────────────────────────────────
_PROFILE_DIR = project_path("profile")
PROFILE_ENTITY_PATH = _PROFILE_DIR / "profiles_with_anchors_with_images_entity.json"
PROFILE_ALL_PATH = _PROFILE_DIR / "profiles_with_anchors_with_images_all.json"
PROFILE_PATH = str(PROFILE_ALL_PATH if PROFILE_ALL_PATH.exists() else PROFILE_ENTITY_PATH)
OUTPUT_DIR = str(project_path("event", "images"))

# ── 图像后端 ─────────────────────────────────────────────────────────────────
MODEL_JIMENG_4_6 = "jimeng_4_6"
MODEL_OPENROUTER_GEMINI_IMAGE = "openrouter_gemini_3_1_flash_image_preview"
MODEL_AIFAST_GEMINI_IMAGE = "aifast_gemini_3_pro_image_preview"
IMAGE_MODEL = MODEL_JIMENG_4_6

OPENROUTER_IMAGE_MODEL = env_value(
    "CUE_MEM_IMAGE_OPENROUTER_MODEL", "google/gemini-3.1-flash-image-preview"
)
AIFAST_IMAGE_MODEL = env_value("CUE_MEM_IMAGE_AIFAST_MODEL", "gemini-3.1-flash-image-preview")
AIFAST_IMAGE_SIZE = env_value("CUE_MEM_IMAGE_AIFAST_SIZE", "1K")

# 非空时：仅重新生成这些 task_id（字符串），并删除已存在的输出 PNG 后强制重画
REGEN_TASK_IDS: List[str] = []

# ── TOS 配置（用于上传本地参考图以获取公网 URL）────────────────────────────────
USE_REFERENCE_IMAGES = os.environ.get("CUE_MEM_IMAGE_USE_REFERENCE", "1").lower() not in {"0", "false", "no"}
TOS_BUCKET = env_value("CUE_MEM_IMAGE_TOS_BUCKET")
TOS_REGION = env_value("CUE_MEM_IMAGE_TOS_REGION", "cn-beijing")
TOS_ENDPOINT = env_value("CUE_MEM_IMAGE_TOS_ENDPOINT")
TOS_PUBLIC_BASE_URL = env_value("CUE_MEM_IMAGE_TOS_PUBLIC_BASE_URL")
TOS_PREFIX = env_value("CUE_MEM_IMAGE_TOS_PREFIX", "ref_images/")

# ── 即梦4.6 API 固定参数 ──────────────────────────────────────────────────────
VOLCENGINE_REGION  = env_value("CUE_MEM_IMAGE_VOLC_REGION", "cn-north-1")
VOLCENGINE_SERVICE = env_value("CUE_MEM_IMAGE_VOLC_SERVICE", "cv")
VOLCENGINE_VERSION = env_value("CUE_MEM_IMAGE_VOLC_VERSION", "2022-08-31")
REQ_KEY = "jimeng_seedream46_cvtob"

# ── 生图配置 ──────────────────────────────────────────────────────────────────
IMAGE_WIDTH   = 1024
IMAGE_HEIGHT  = 1024
FORCE_SINGLE  = True   # 强制单图输出，速度快、价格低
MAX_RETRIES   = 6
POLL_INTERVAL = 3      # 轮询间隔（秒）
POLL_TIMEOUT  = 360    # 单任务最长等待（秒）
MAX_WORKERS   = 2      # 并发数（避免触发 API 并发限额）

TOP_LEVEL_CATEGORIES = [
    'FoodAndDrink', 'HomeAndSpace', 'BodyAndHealth',
    'HobbiesAndEntertainment', 'WorkAndLearning', 'MobilityAndTravel',
]
BASIC_CATEGORIES = ['Relationship', 'Pets']


# ─────────────────────────────────────────────────────────────────────────────
# TOS 上传：本地参考图 → 公网 URL（进程内 LRU 缓存，同文件不重复上传）
# ─────────────────────────────────────────────────────────────────────────────

_tos_url_cache: Dict[str, str] = {}   # local_path → public_url


def _get_tos_client():
    """懒加载 TOS 客户端（需要 pip install tos）。"""
    try:
        import tos
    except ImportError:
        raise ImportError(
            "缺少 TOS SDK，请执行：pip install tos\n"
            "或将 USE_REFERENCE_IMAGES 设为 False 跳过参考图功能。"
        )
    return tos.TosClientV2(
        ak=required_env("CUE_MEM_IMAGE_TOS_ACCESS_KEY"),
        sk=required_env("CUE_MEM_IMAGE_TOS_SECRET_KEY"),
        endpoint=required_env("CUE_MEM_IMAGE_TOS_ENDPOINT"),
        region=TOS_REGION,
    )


def upload_to_tos(local_path: Path) -> Optional[str]:
    """
    将本地图片上传到 TOS（公有读 Bucket），返回公网 URL。
    同一文件在进程内只上传一次（缓存）。
    """
    if not USE_REFERENCE_IMAGES:
        return None

    key = str(local_path.resolve())
    if key in _tos_url_cache:
        return _tos_url_cache[key]

    if not local_path.exists():
        return None

    try:
        client   = _get_tos_client()
        # TOS Key：前缀 + 文件名，用文件 sha1 防止同名不同内容冲突
        with open(local_path, "rb") as f:
            file_bytes = f.read()
        sha1     = hashlib.sha1(file_bytes).hexdigest()[:12]
        suffix   = local_path.suffix or ".jpg"
        tos_key  = f"{TOS_PREFIX}{sha1}{suffix}"

        bucket = required_env("CUE_MEM_IMAGE_TOS_BUCKET")
        client.put_object(
            bucket=bucket,
            key=tos_key,
            content=file_bytes,
        )
        public_base = required_env("CUE_MEM_IMAGE_TOS_PUBLIC_BASE_URL").rstrip("/")
        url = f"{public_base}/{tos_key}"
        _tos_url_cache[key] = url
        return url
    except Exception as exc:
        print(f"    [TOS WARN] 上传失败 {local_path.name}: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 火山引擎 HMAC-SHA256 V4 签名
# 请求签名按供应商的 V4 规范实现。
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
    """向火山引擎 Visual API 发起一次经过 V4 签名的 POST 请求。"""
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

    # 按字母序排列参与签名的 Headers
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
        "HMAC-SHA256",
        x_date,
        credential_scope,
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
# 任务提交 & 轮询
# ─────────────────────────────────────────────────────────────────────────────

def _submit_task(prompt: str, image_urls: Optional[List[str]] = None) -> Optional[str]:
    """提交生图任务，返回 task_id；失败返回 None。"""
    body: Dict[str, Any] = {
        "req_key":      REQ_KEY,
        "prompt":       prompt[:800],   # API 限制 800 字符
        "width":        IMAGE_WIDTH,
        "height":       IMAGE_HEIGHT,
        "force_single": FORCE_SINGLE,
    }
    if image_urls:
        body["image_urls"] = image_urls

    data = _volcengine_request("CVSync2AsyncSubmitTask", body)
    if data.get("code") != 10000:
        print(f"    [SUBMIT ERR] code={data.get('code')} msg={data.get('message')} "
              f"request_id={data.get('request_id')}")
        return None
    return data["data"]["task_id"]


def _poll_task(task_id: str) -> Optional[List[str]]:
    """轮询任务，返回图片 URL 列表；超时或失败返回 None。"""
    body = {
        "req_key":  REQ_KEY,
        "task_id":  task_id,
        "req_json": json.dumps({"return_url": True}),
    }
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        # 网络抖动（SSL EOF / 连接重置）最多重试 3 次，不消耗外层重试机会
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
            return urls if urls else None
        if status in ("not_found", "expired"):
            print(f"    [POLL] task {task_id}: status={status}")
            return None
        # in_queue / generating → continue polling
        time.sleep(POLL_INTERVAL)

    print(f"    [TIMEOUT] task {task_id} exceeded {POLL_TIMEOUT}s")
    return None


def _download_image(url: str, save_path: Path) -> bool:
    """从 URL 下载图片并保存到本地。"""
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    [DOWNLOAD ERR] {exc}")
        return False


def _save_from_data_url(data_url: str, save_path: Path) -> bool:
    try:
        if "," not in data_url or not data_url.strip().lower().startswith("data:"):
            return False
        _, b64_part = data_url.split(",", 1)
        raw = base64.b64decode(b64_part)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True
    except Exception as exc:
        print(f"    [DATA_URL SAVE ERR] {exc}")
        return False


def _save_image_from_url_or_dataurl(src: str, save_path: Path) -> bool:
    s = (src or "").strip()
    if not s:
        return False
    if s.lower().startswith("data:"):
        return _save_from_data_url(s, save_path)
    if urlparse(s).scheme in {"http", "https"}:
        return _download_image(s, save_path)
    print(f"    [SAVE ERR] unsupported image source: {s[:80]!r}...")
    return False


def _extract_urls_from_openrouter_message(message: Any) -> List[str]:
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


def _openrouter_generate_to_path(
    prompt: str,
    save_path: Path,
    ref_paths: Optional[List[Path]] = None,
) -> bool:
    """
    调用 OpenRouter Gemini 图像生成。

    ref_paths: 本地参考图路径列表（可选）。非 None 时将图片 base64 编码后作为
               image_url 内容块拼入多模态消息，让模型参考人物/宠物/物品外貌。
    """
    key = env_value("CUE_MEM_IMAGE_OPENROUTER_API_KEY")
    if not key:
        print("    [ERR] CUE_MEM_IMAGE_OPENROUTER_API_KEY 未设置")
        return False

    # 构建消息内容：先放参考图（若有），最后放文本 prompt
    content: Any
    valid_refs = []
    if ref_paths:
        for rp in ref_paths:
            if rp is None:
                continue
            try:
                p = Path(rp)
                if not p.is_file():
                    continue
                suffix = p.suffix.lower().lstrip(".")
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png", "webp": "image/webp",
                        "gif": "image/gif"}.get(suffix, "image/jpeg")
                b64 = base64.b64encode(p.read_bytes()).decode("ascii")
                valid_refs.append(f"data:{mime};base64,{b64}")
            except Exception as exc:
                print(f"    [OPENROUTER WARN] 参考图编码失败 {rp}: {exc}")

    if valid_refs:
        content = [
            {"type": "image_url", "image_url": {"url": url}}
            for url in valid_refs
        ] + [{"type": "text", "text": prompt}]
        print(f"    [OpenRouter] 附带 {len(valid_refs)} 张参考图（base64）")
    else:
        content = prompt

    try:
        client = openai_client(
            api_key_env="CUE_MEM_IMAGE_OPENROUTER_API_KEY",
            base_url_env="CUE_MEM_IMAGE_OPENROUTER_BASE_URL",
        )
        completion = client.chat.completions.create(
            model=OPENROUTER_IMAGE_MODEL,
            messages=[{"role": "user", "content": content}],
            extra_body={"modalities": ["image", "text"]},
        )
    except Exception as exc:
        print(f"    [OPENROUTER ERR] {exc}")
        return False
    if not completion.choices:
        print("    [OPENROUTER ERR] empty choices")
        return False
    message = completion.choices[0].message
    urls = _extract_urls_from_openrouter_message(message)
    if not urls:
        print("    [OPENROUTER ERR] no images in response")
        return False
    return _save_image_from_url_or_dataurl(urls[0], save_path)


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


def _mime_for_image_path(path: Path) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/png")


def _aifast_inline_parts_from_paths(ref_paths: Optional[List[Path]]) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if not ref_paths:
        return parts
    for rp in ref_paths:
        if rp is None:
            continue
        try:
            path = Path(rp)
            if not path.is_file():
                continue
            parts.append(
                {
                    "inlineData": {
                        "mimeType": _mime_for_image_path(path),
                        "data": base64.b64encode(path.read_bytes()).decode("ascii"),
                    }
                }
            )
        except Exception as exc:
            print(f"    [AIFAST WARN] 参考图编码失败 {rp}: {exc}")
    return parts


def _aifast_gemini_generate_to_path(
    prompt: str,
    save_path: Path,
    ref_paths: Optional[List[Path]] = None,
) -> bool:
    key = env_value("CUE_MEM_IMAGE_AIFAST_API_KEY")
    if not key:
        print("    [AIFAST ERR] CUE_MEM_IMAGE_AIFAST_API_KEY 未设置")
        return False

    base_url = (env_value("CUE_MEM_IMAGE_AIFAST_BASE_URL") or "").rstrip("/")
    if not base_url:
        print("    [AIFAST ERR] CUE_MEM_IMAGE_AIFAST_BASE_URL 未设置")
        return False
    model = (env_value("CUE_MEM_IMAGE_AIFAST_MODEL", AIFAST_IMAGE_MODEL) or "").strip()
    image_size = (env_value("CUE_MEM_IMAGE_AIFAST_SIZE", AIFAST_IMAGE_SIZE) or "").strip()
    aspect_ratio = _aifast_aspect_ratio_from_dimensions(IMAGE_WIDTH, IMAGE_HEIGHT)
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    parts = _aifast_inline_parts_from_paths(ref_paths)
    if parts:
        print(f"    [AIFAST] 附带 {len(parts)} 张参考图（inlineData）")
    parts.append({"text": prompt})
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
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
            return False
        data = resp.json()
    except Exception as exc:
        print(f"    [AIFAST ERR] request failed: {exc}")
        return False

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
        return False

    try:
        raw = base64.b64decode(images[0])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True
    except Exception as exc:
        print(f"    [AIFAST ERR] save failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_records(profile_path: Path):
    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    img_to_path: Dict[int, Dict[str, Any]] = {}
    for i, profile in enumerate(profiles):
        img_to_path[i] = {}
        basic = profile.get("Basic", {}) or {}
        for item in (basic.get("Relationship", []) or []):
            img_path = item.get("img_path")
            for key in (item.get("name", ""), item.get("relation", "")):
                key = str(key or "").strip()
                if key and img_path:
                    img_to_path[i][key] = img_path
        for item in (basic.get("Pets", []) or []):
            key = str(item.get("name", "") or "").strip()
            if key and item.get("img_path"):
                img_to_path[i][key] = item.get("img_path")
        for item in (profile.get("Items", []) or []):
            desc = str(item.get("description", "") or "").strip()
            if desc and item.get("img_path"):
                img_to_path[i][desc[:15]] = item.get("img_path")
                img_to_path[i][desc] = item.get("img_path")

    return profiles, img_to_path


def resolve_entity_anchor_ref_path(
    p_id: int,
    anchor: str,
    img_to_path: Dict[int, Dict[str, Any]],
) -> Optional[Any]:
    """Resolve an entity anchor to a reference image path.

    Priority:
      1. Existing exact Items-style lookup by anchor[:15].
      2. Exact full-anchor lookup.
      3. Fallback containment lookup for names, e.g. "银灰虎斑猫Pixel" -> "Pixel".
    """
    anchor = str(anchor or "").strip()
    if not anchor:
        return None
    lookup = img_to_path.get(p_id) or {}

    raw = lookup.get(anchor[:15]) or lookup.get(anchor)
    if raw:
        return raw

    anchor_folded = anchor.casefold()
    for key in sorted((k for k in lookup if str(k).strip()), key=lambda x: len(str(x)), reverse=True):
        key_text = str(key).strip()
        if len(key_text) < 2:
            continue
        if key_text.casefold() in anchor_folded:
            return lookup.get(key)
    return None


def safe_stem(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fa5]", "", text)
    return text[:80] if text else "item"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt 构建
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(
    scene_desc: str,
    image_desc: str,
    anchor_desc: Optional[str],
    person_ref_name: Optional[str] = None,
    pet_ref_name:    Optional[str] = None,
) -> str:
    consistency_parts: List[str] = []

    if person_ref_name:
        consistency_parts.append(
            f"The person '{person_ref_name}' appears in the scene. "
            f"Keep their identity consistent: same face, hairstyle, body type, age, and skin tone."
        )
    if pet_ref_name:
        consistency_parts.append(
            f"The pet '{pet_ref_name}' appears in the scene. "
            f"Keep the pet's identity consistent: same coat color, pattern, body shape, and markings."
        )

    anchor_text = (
        f"Entity anchor: {anchor_desc}. Keep this item's appearance (material, color, shape) consistent. "
        if anchor_desc and "none" not in str(anchor_desc).lower()
        else ""
    )

    consistency_text = " ".join(consistency_parts)

    return (
        "Generate one highly realistic and clear photographic image, "
        "captured from a first-person perspective as if taken by a user with their smartphone. "
        "Subject in sharp focus. Natural bright lighting. No text, watermark, or logo. "
        f"Image description: {image_desc}. "
        f"{anchor_text}"
        f"{consistency_text}"
        "Important Notes:The photo must be taken from a first-person perspective; the photographer must not appear in the image."
        "Output one high-quality, sharp, clear image."
    )


def collect_reference_urls_and_status_lines(
    *,
    image_model: str,
    use_reference_images: bool,
    p_id: int,
    task_id: str,
    info: Dict[str, Any],
) -> Tuple[List[str], List[str]]:
    """
    解析人物/宠物/实体锚点的本地参考图路径；在即梦且 use_reference_images 时上传到 TOS。

    参考图路径均已在 task_groups 构建阶段从 profile 中解析好，存于 info 字典：
      info["person_ref"]  : Optional[Path]  — Relationship 肖像
      info["pet_ref"]     : Optional[Path]  — Pets 肖像
      info["anchor_refs"] : List[Path]      — 偏好 entity_anchors → Items 肖像

    返回 (传给即梦的 image_urls 列表, 供控制台打印的状态行)。
    """
    lines: List[str] = []
    ref_urls: List[str] = []

    def _local_line(label: str, path_obj: Optional[Any]) -> None:
        if path_obj is None:
            lines.append(f"{label}(本地): 无")
            return
        try:
            p = Path(path_obj)
        except (TypeError, ValueError):
            lines.append(f"{label}(本地): 路径无效 → {path_obj!r}")
            return
        if p.is_file():
            lines.append(f"{label}(本地): 已找到 → {p.name}")
        else:
            lines.append(f"{label}(本地): 路径存在但文件不存在 → {p}")

    _local_line("人物参考图", info.get("person_ref"))
    _local_line("宠物参考图", info.get("pet_ref"))
    anchor_refs = [
        Path(p) for p in (info.get("anchor_refs") or [])
        if p is not None and Path(p).is_file()
    ]
    anchor_desc_text = info.get("anchor_desc", "")
    if anchor_refs:
        for idx, anchor_ref in enumerate(anchor_refs, 1):
            label = f"实体锚点参考图 {idx}"
            if anchor_desc_text:
                label += f"（{anchor_desc_text}）" if len(anchor_refs) == 1 else ""
            _local_line(label, anchor_ref)
    else:
        if anchor_desc_text:
            lines.append(f"实体锚点参考图(本地): 无（{anchor_desc_text!r} 对应的 Items 无 img_path）")
        else:
            lines.append("实体锚点参考图(本地): 无（偏好 entity_anchors 为空）")

    if image_model == MODEL_OPENROUTER_GEMINI_IMAGE:
        if not use_reference_images:
            lines.append("API 参考图: USE_REFERENCE_IMAGES=False，OpenRouter 不附带参考图。")
            return [], lines
        lines.append("API 参考图: OpenRouter 模式使用本地参考图（base64），以上本地路径即为传入图片。")
        return [], lines   # ref_paths 由调用方直接传入 generate_one_image，此处无需返回 TOS URL

    if image_model == MODEL_AIFAST_GEMINI_IMAGE:
        if not use_reference_images:
            lines.append("API 参考图: USE_REFERENCE_IMAGES=False，AIFast 不附带参考图。")
            return [], lines
        lines.append("API 参考图: AIFast Gemini 模式使用本地参考图（inlineData），以上本地路径即为传入图片。")
        return [], lines

    if not use_reference_images:
        lines.append("API 参考图: USE_REFERENCE_IMAGES=False，即梦请求不附带 image_urls。")
        return [], lines

    upload_refs: List[Tuple[str, Path]] = []
    for label, key in [("人物参考图", "person_ref"), ("宠物参考图", "pet_ref")]:
        lp = info.get(key)
        if lp is not None:
            upload_refs.append((label, Path(lp)))
    for idx, anchor_ref in enumerate(anchor_refs, 1):
        upload_refs.append((f"实体锚点参考图 {idx}", anchor_ref))

    for label, p in upload_refs:
        if not p.is_file():
            lines.append(f"{label} TOS: 跳过（本地文件不可用）")
            continue
        url = upload_to_tos(p)
        if url:
            ref_urls.append(url)
            lines.append(f"{label} TOS: 上传成功，已加入 image_urls")
        else:
            lines.append(f"{label} TOS: 上传失败（本地文件可用）")

    if ref_urls:
        lines.append(f"API 参考图: 即梦将附带 {len(ref_urls)} 个参考图 URL（image_urls）。")
    else:
        lines.append("API 参考图: 即梦本次请求 image_urls 为空（无成功上传的参考图）。")

    return ref_urls, lines


# ─────────────────────────────────────────────────────────────────────────────
# 单任务生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_one_image(
    prompt: str,
    save_path: Path,
    p_id: int,
    task_id: str,
    occurrences: List[Tuple[str, int, int]],
    image_urls: Optional[List[str]],
    image_model: str,
    ref_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """
    根据 image_model 调用即梦4.6 或 OpenRouter Gemini 图像接口，将结果写入 save_path。
    image_urls : 即梦路径中使用的 TOS 公网 URL 列表。
    ref_paths  : OpenRouter 路径中使用的本地参考图路径列表（base64 编码后传入）。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    output_name = save_path.name

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if image_model == MODEL_JIMENG_4_6:
                if attempt == 1:
                    n_ref = len(image_urls) if image_urls else 0
                    print(
                        f"    [即梦] p_id={p_id} task_id={task_id} "
                        f"输出文件={output_name} "
                        f"参考图 URL 数量={n_ref} "
                        f"({'已加载并传入 API' if n_ref else '未传入参考图'})"
                    )
                jimeng_task_id = _submit_task(prompt, image_urls)
                if not jimeng_task_id:
                    time.sleep(2 * attempt)
                    continue
                urls = _poll_task(jimeng_task_id)
                if not urls:
                    time.sleep(2 * attempt)
                    continue
                if _download_image(urls[0], save_path):
                    print(f"    [OK] p_id={p_id} task_id={task_id} saved -> {output_name}")
                    return {
                        "status":       "ok",
                        "file":         str(save_path),
                        "p_id":         p_id,
                        "task_id":      task_id,
                        "occurrences":  occurrences,
                        "image_model":  image_model,
                    }
            elif image_model == MODEL_OPENROUTER_GEMINI_IMAGE:
                if attempt == 1:
                    n_ref = len([p for p in (ref_paths or []) if p and Path(p).is_file()])
                    print(
                        f"    [OpenRouter] p_id={p_id} task_id={task_id} "
                        f"输出文件={output_name} "
                        f"参考图 {n_ref} 张（base64）"
                        f"({'已传入' if n_ref else '无参考图'})"
                    )
                if _openrouter_generate_to_path(prompt, save_path, ref_paths=ref_paths):
                    print(f"    [OK] p_id={p_id} task_id={task_id} saved -> {output_name}")
                    return {
                        "status":       "ok",
                        "file":         str(save_path),
                        "p_id":         p_id,
                        "task_id":      task_id,
                        "occurrences":  occurrences,
                        "image_model":  image_model,
                    }
            elif image_model == MODEL_AIFAST_GEMINI_IMAGE:
                if attempt == 1:
                    n_ref = len([p for p in (ref_paths or []) if p and Path(p).is_file()])
                    print(
                        f"    [AIFast] p_id={p_id} task_id={task_id} "
                        f"输出文件={output_name} "
                        f"参考图 {n_ref} 张（inlineData）"
                        f"({'已传入' if n_ref else '无参考图'})"
                    )
                if _aifast_gemini_generate_to_path(prompt, save_path, ref_paths=ref_paths):
                    print(f"    [OK] p_id={p_id} task_id={task_id} saved -> {output_name}")
                    return {
                        "status":       "ok",
                        "file":         str(save_path),
                        "p_id":         p_id,
                        "task_id":      task_id,
                        "occurrences":  occurrences,
                        "image_model":  image_model,
                    }
            else:
                print(f"    [ERR] unknown IMAGE_MODEL: {image_model!r}")
                break
            time.sleep(2 * attempt)
        except Exception as exc:
            print(f"    [ERR] p_id={p_id} task={task_id} attempt={attempt}/{MAX_RETRIES}: {exc}")
            time.sleep(3 * attempt)

    return {
        "status":       "failed",
        "file":         "",
        "p_id":         p_id,
        "task_id":      task_id,
        "occurrences":  occurrences,
        "image_model":  image_model,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 辅助工具
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_record_container(profile: Dict[str, Any], category: str):
    if category in BASIC_CATEGORIES:
        basic = (profile.get("Basic", {}) or {}).get(category, []) or []
        top = profile.get(category, []) or []
        return basic + [r for r in top if r not in basic]
    return profile.get(category, []) or []


def _writeback_img_path(profiles, p_id, occurrences, file_path):
    target = profiles[p_id]
    for cat, rec_i, evt_i in occurrences:
        container = (
            target.setdefault("Basic", {}).setdefault(cat, [])
            if cat in BASIC_CATEGORIES
            else target.setdefault(cat, [])
        )
        try:
            container[rec_i]["events"][evt_i]["img_path"] = file_path
        except (KeyError, IndexError, TypeError) as exc:
            print(f"[WARN] img_path 回写失败 p_id={p_id} cat={cat} rec={rec_i} evt={evt_i}: {exc}")


def _safe_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    """Write JSON robustly on Windows without relying on truncate()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=indent)
    payload_bytes = payload.encode("utf-8")
    try:
        if path.exists():
            old_bytes = path.read_bytes()
            if old_bytes == payload_bytes:
                return
            if len(payload_bytes) < len(old_bytes):
                # JSON parsers accept trailing whitespace; this avoids truncate()
                # on Windows files that refuse file truncation with Errno 13/22.
                payload_bytes += b" " * (len(old_bytes) - len(payload_bytes))
            with open(path, "r+b") as f:
                f.seek(0)
                f.write(payload_bytes)
        else:
            path.write_bytes(payload_bytes)
    except OSError as exc:
        raise OSError(
            f"JSON 写入失败: {path.resolve()} (raw={str(path)!r}); {exc}"
        ) from exc


def parse_profile_id_filter(raw_values: Optional[List[str]]) -> Optional[set[int]]:
    if not raw_values:
        return None
    result: set[int] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                result.add(int(part))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def run(
    image_model: Optional[str] = None,
    regen_task_ids: Optional[List[str]] = None,
    max_profiles: Optional[int] = None,
    only_profile_ids: Optional[List[str]] = None,
    profile_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    resolved_model = (image_model or IMAGE_MODEL).strip()
    resolved_max_profiles = 0 if max_profiles is None else max_profiles
    only_profile_id_set = parse_profile_id_filter(only_profile_ids)
    resolved_profile_path = resolve_path(profile_path or PROFILE_PATH)
    if resolved_max_profiles < 0:
        raise ValueError(f"--max-profiles 不能为负数: {resolved_max_profiles}")
    if resolved_model not in (MODEL_JIMENG_4_6, MODEL_OPENROUTER_GEMINI_IMAGE, MODEL_AIFAST_GEMINI_IMAGE):
        raise ValueError(
            f"不支持的 IMAGE_MODEL={resolved_model!r}，请使用 "
            f"{MODEL_JIMENG_4_6!r}、{MODEL_OPENROUTER_GEMINI_IMAGE!r} "
            f"或 {MODEL_AIFAST_GEMINI_IMAGE!r}"
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
    if resolved_model == MODEL_JIMENG_4_6:
        for env_name in (
            "CUE_MEM_IMAGE_VOLC_ACCESS_KEY",
            "CUE_MEM_IMAGE_VOLC_SECRET_KEY",
            "CUE_MEM_IMAGE_VOLC_BASE_URL",
        ):
            if not env_value(env_name):
                raise ValueError(f"即梦模式需要设置 {env_name}")

    regen_source = REGEN_TASK_IDS if regen_task_ids is None else regen_task_ids
    regen_set = {str(x).strip() for x in (regen_source or []) if str(x).strip()}
    regen_mode = bool(regen_set)

    if not resolved_profile_path.is_file():
        raise FileNotFoundError(f"Profile file not found: {resolved_profile_path}")

    print(f"[Config] IMAGE_MODEL = {resolved_model}")
    print(f"[Config] MAX_PROFILES = {resolved_max_profiles} (0 = all)")
    print(f"[Config] PROFILE_PATH = {resolved_profile_path}")
    if only_profile_id_set is not None:
        print(f"[Config] ONLY_PROFILE_IDS = {sorted(only_profile_id_set)}")
    if regen_mode:
        print(f"[Config] REGEN task_ids ({len(regen_set)}): {sorted(regen_set)}")
    if resolved_model == MODEL_OPENROUTER_GEMINI_IMAGE and USE_REFERENCE_IMAGES:
        print("[Config] OpenRouter 模式不使用 TOS 参考图 URL，仅用文本 prompt。")
    if resolved_model == MODEL_AIFAST_GEMINI_IMAGE and USE_REFERENCE_IMAGES:
        print("[Config] AIFast Gemini 模式不使用 TOS 参考图 URL，使用本地参考图 inlineData。")

    profiles, img_to_path = load_records(resolved_profile_path)
    output_dir = resolve_path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 以 (p_id, task_id) 为粒度聚合所有出现位置 ─────────────────────────────
    task_groups: Dict[Tuple[int, str], Dict[str, Any]] = {}

    for p_id, profile in enumerate(profiles):
        if resolved_max_profiles > 0 and p_id >= resolved_max_profiles:
            continue
        if only_profile_id_set is not None and p_id not in only_profile_id_set:
            continue
        for category in TOP_LEVEL_CATEGORIES + BASIC_CATEGORIES:
            records = _resolve_record_container(profile, category)
            for rec_idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                events = rec.get("events", [])
                if not isinstance(events, list):
                    continue

                rec_person_name: Optional[str] = None
                rec_pet_name:    Optional[str] = None
                rec_person_ref:  Optional[Path] = None
                rec_pet_ref:     Optional[Path] = None
                if category == "Relationship":
                    rec_person_name = rec.get("name") or rec.get("relation")
                    raw = rec.get("img_path")
                    if raw:
                        try:
                            p = Path(raw)
                            rec_person_ref = p if p.exists() else None
                        except (TypeError, ValueError):
                            pass
                elif category == "Pets":
                    rec_pet_name = rec.get("name")
                    raw = rec.get("img_path")
                    if raw:
                        try:
                            p = Path(raw)
                            rec_pet_ref = p if p.exists() else None
                        except (TypeError, ValueError):
                            pass

                for evt_idx, evt in enumerate(events):
                    if not isinstance(evt, dict):
                        continue
                    image_desc = str(evt.get("user_shared_image_description", "")).strip()
                    if not image_desc or image_desc.lower() == "none":
                        continue
                    task_id = evt.get("task_id")
                    if not task_id:
                        continue

                    key = (p_id, str(task_id))
                    if key not in task_groups:
                        task_groups[key] = {
                            "event":        evt,
                            "occurrences":  [],
                            "person_name":  None,
                            "pet_name":     None,
                            "person_ref":   None,   # Path
                            "pet_ref":      None,   # Path
                            "anchor_refs":  [],     # List[Path] — 来自事件 entity_anchors 的 Items 参考图
                            "anchor_desc":  "",     # str — 锚点文字描述（用于 prompt）
                        }
                    task_groups[key]["occurrences"].append((category, rec_idx, evt_idx))
                    if rec_person_name and not task_groups[key]["person_name"]:
                        task_groups[key]["person_name"] = rec_person_name
                    if rec_pet_name and not task_groups[key]["pet_name"]:
                        task_groups[key]["pet_name"] = rec_pet_name
                    if rec_person_ref and not task_groups[key]["person_ref"]:
                        task_groups[key]["person_ref"] = rec_person_ref
                    if rec_pet_ref and not task_groups[key]["pet_ref"]:
                        task_groups[key]["pet_ref"] = rec_pet_ref
                    # 从事件的 entity_anchors 字段中查找所有可用 Items 参考图。
                    anchors_for_prompt: List[str] = []
                    existing_anchor_ref_paths = {
                        str(Path(p).resolve()) for p in (task_groups[key].get("anchor_refs") or [])
                    }
                    for anchor_text in (evt.get("entity_anchors") or []):
                        if not isinstance(anchor_text, str) or not anchor_text.strip():
                            continue
                        anchor = anchor_text.strip()
                        anchors_for_prompt.append(anchor)
                        raw = resolve_entity_anchor_ref_path(p_id, anchor, img_to_path)
                        if not raw:
                            continue
                        try:
                            ap = Path(raw)
                            if ap.is_file():
                                resolved = str(ap.resolve())
                                if resolved not in existing_anchor_ref_paths:
                                    task_groups[key]["anchor_refs"].append(ap)
                                    existing_anchor_ref_paths.add(resolved)
                        except (TypeError, ValueError):
                            pass
                    if anchors_for_prompt:
                        old_desc = str(task_groups[key].get("anchor_desc") or "").strip()
                        all_descs = []
                        if old_desc:
                            all_descs.extend(x.strip() for x in old_desc.split("；") if x.strip())
                        all_descs.extend(anchors_for_prompt)
                        task_groups[key]["anchor_desc"] = "；".join(dict.fromkeys(all_descs))

    if regen_mode:
        filtered = {k: v for k, v in task_groups.items() if str(k[1]) in regen_set}
        found_ids = {str(k[1]) for k in filtered}
        missing = regen_set - found_ids
        if missing:
            print(f"[WARN] REGEN 中以下 task_id 在数据中未找到: {sorted(missing)}")
        task_groups = filtered

    print(f"unique (p_id, task_id) groups to process: {len(task_groups)}")
    if not task_groups:
        print("No image tasks to process; manifest/profile outputs are left unchanged.")
        print(
            f"done: total=0, success=0, skipped=0, failed=0, "
            f"manifest={output_dir / 'manifest.json'}"
        )
        return

    manifest: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for (p_id, task_id), info in task_groups.items():
            evt         = info["event"]
            image_desc  = str(evt.get("user_shared_image_description", "")).strip()
            scene_desc  = str(evt.get("scene_description", "")).strip()
            anchor_desc = info.get("anchor_desc", "")

            safe_tid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(task_id))
            out_path = output_dir / f"pid_{p_id:04d}_task_{safe_tid}.png"
            print(f"[输出映射] p_id={p_id} task_id={task_id} -> {out_path.name}")

            if out_path.exists() and regen_mode:
                print(f"[REGEN] existing file will be overwritten only after successful generation -> {out_path.name}")

            if out_path.exists() and not regen_mode:
                manifest.append({
                    "p_id": p_id,
                    "task_id": task_id,
                    "occurrences": info["occurrences"],
                    "file": str(out_path),
                    "status": "skipped_exists",
                    "image_model": resolved_model,
                })
                _writeback_img_path(profiles, p_id, info["occurrences"], str(out_path))
                print(
                    f"[参考图状态] p_id={p_id} task_id={task_id} "
                    f"跳过生成（文件已存在 {out_path.name}），未执行参考图解析与 TOS 上传"
                )
                continue

            ref_urls, ref_status_lines = collect_reference_urls_and_status_lines(
                image_model=resolved_model,
                use_reference_images=USE_REFERENCE_IMAGES,
                p_id=p_id,
                task_id=str(task_id),
                info=info,
            )

            # OpenRouter/AIFast: 直接用本地 Path 对象（base64 编码后传入）
            if resolved_model in (MODEL_OPENROUTER_GEMINI_IMAGE, MODEL_AIFAST_GEMINI_IMAGE) and USE_REFERENCE_IMAGES:
                anchor_ref_paths = [
                    p for p in (info.get("anchor_refs") or [])
                    if p is not None and Path(p).is_file()
                ]
                local_ref_paths: List[Path] = [
                    p for p in [info.get("person_ref"), info.get("pet_ref"), *anchor_ref_paths]
                    if p is not None and Path(p).is_file()
                ]
            else:
                local_ref_paths = []

            print(f"[参考图状态] p_id={p_id} task_id={task_id}")
            for _ln in ref_status_lines:
                print(f"  {_ln}")

            # prompt 里的文字一致性约束：有参考图 URL（即梦）或有本地参考图（OpenRouter/AIFast）时都省略文字描述
            has_any_ref = bool(ref_urls) or bool(local_ref_paths)
            prompt = build_prompt(
                scene_desc=scene_desc,
                image_desc=image_desc,
                anchor_desc=anchor_desc,
                person_ref_name=None if has_any_ref else info["person_name"],
                pet_ref_name=None    if has_any_ref else info["pet_name"],
            )

            futures.append(
                executor.submit(
                    generate_one_image,
                    prompt,
                    out_path,
                    p_id,
                    task_id,
                    info["occurrences"],
                    ref_urls or None,
                    resolved_model,
                    local_ref_paths or None,
                )
            )

        print(f"All tasks submitted ({len(futures)} new), waiting for completion...")
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            manifest.append({
                "p_id":        result["p_id"],
                "task_id":     result["task_id"],
                "occurrences": result["occurrences"],
                "file":        result["file"],
                "status":      result["status"],
                "image_model": result.get("image_model", resolved_model),
            })
            if result["status"] == "ok":
                _writeback_img_path(
                    profiles, result["p_id"],
                    result.get("occurrences", []), result["file"],
                )

    manifest_path = output_dir / "manifest.json"
    _safe_write_json(manifest_path, manifest, indent=2)

    out_profile = PROFILE_ALL_PATH
    _safe_write_json(out_profile, profiles, indent=2)

    ok_count   = sum(1 for x in manifest if x["status"] == "ok")
    skip_count = sum(1 for x in manifest if x["status"] == "skipped_exists")
    fail_count = sum(1 for x in manifest if x["status"] == "failed")
    print(
        f"done: total={len(manifest)}, success={ok_count}, "
        f"skipped={skip_count}, failed={fail_count}, manifest={manifest_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从事件描述批量生图（即梦 / OpenRouter / AIFast）")
    parser.add_argument(
        "--image-model",
        choices=[MODEL_JIMENG_4_6, MODEL_OPENROUTER_GEMINI_IMAGE, MODEL_AIFAST_GEMINI_IMAGE],
        default=None,
        help="覆盖脚本中的 IMAGE_MODEL",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        dest="regen_task_ids",
        default=None,
        help="仅重新生成指定 task_id（可多次传入）；覆盖脚本中的 REGEN_TASK_IDS",
    )
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=None,
        help="只处理前 N 个 profile；0 或不传表示处理全部",
    )
    parser.add_argument(
        "--only_profile_ids",
        "--only-profile-ids",
        nargs="*",
        default=None,
        help="只处理指定 p_id，支持空格或逗号形式，例如 --only_profile_ids 1 2 3 或 --only_profile_ids 1,2,3",
    )
    parser.add_argument(
        "--profile-path",
        default=None,
        help="Override profile JSON used for scanning events; default prefers profiles_with_anchors_with_images_all.json",
    )
    parser.add_argument("--output-dir", default=None, help="directory for generated images")
    args = parser.parse_args()
    run(
        image_model=args.image_model,
        regen_task_ids=args.regen_task_ids,
        max_profiles=args.max_profiles,
        only_profile_ids=args.only_profile_ids,
        profile_path=args.profile_path,
        output_dir=args.output_dir,
    )
