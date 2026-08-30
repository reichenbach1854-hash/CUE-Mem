"""Embedding 编码器封装：统一文本/图像/音频向量接口。"""

from __future__ import annotations

import base64
import logging
import mimetypes
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import (
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_API_BASE,
    GEMINI_EMBEDDING_API_KEY,
    GEMINI_EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    IMAGEBIND_EMBEDDING_DIM,
    IMAGEBIND_MODEL,
    redact_sensitive_text,
    required_runtime_value,
)
from .data_loader import build_text_for_turn, resolve_media_path

logger = logging.getLogger(__name__)


class ImageBindEncoder:
    """ImageBind 多模态编码器。

    所有模态映射到同一 1024 维空间，天然支持跨模态检索。
    """

    def __init__(self, device: str = "cuda:0", data_dir: str | Path | None = None):
        from imagebind import data as imagebind_data
        from imagebind.models import imagebind_model
        from imagebind.models.imagebind_model import ModalityType

        self.device = device
        self.provider = "imagebind"
        self.model_name = IMAGEBIND_MODEL
        self.embedding_dim = IMAGEBIND_EMBEDDING_DIM
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
        self._data = imagebind_data
        self._modality_type = ModalityType
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)
        # 模态使用统计，用于检查 MM-Index 是否真的编码了图片和音频
        self.stats: dict[str, int] = defaultdict(int)

    def reset_stats(self) -> None:
        """重置统计计数器，可在每个 profile/mode 开始前调用。"""
        self.stats = defaultdict(int)

    def get_stats(self) -> dict[str, int]:
        """返回当前统计快照。"""
        return dict(self.stats)

    def print_stats(self, prefix: str = "") -> None:
        """打印模态使用统计，方便检查 MM-Index 质量。"""
        tag = f"[{prefix}] " if prefix else ""
        total = self.stats.get("total_turns", 0)
        provider = getattr(self, "provider", "embedding")
        print(f"\n{tag}=== {provider} encode_turn_multimodal 统计 (total={total}) ===")
        categories = [
            ("text_only",            "纯文本（无图像无音频）"),
            ("text_image",           "文本 + 图像"),
            ("text_audio",           "文本 + 音频"),
            ("text_image_audio",     "文本 + 图像 + 音频"),
            ("image_encode_failed",  "图像编码失败"),
            ("audio_encode_failed",  "音频编码失败"),
            ("image_path_not_found", "图像路径不存在"),
            ("audio_path_not_found", "音频路径不存在"),
        ]
        for key, label in categories:
            cnt = self.stats.get(key, 0)
            if cnt > 0 or key in ("text_only", "text_image", "text_audio", "text_image_audio"):
                pct = f" ({cnt/total*100:.1f}%)" if total > 0 else ""
                print(f"  {label:<28}: {cnt:>5}{pct}")
        print()

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """文本 → [1024] 归一化向量。"""
        modality = self._modality_type.TEXT
        inputs = {modality: self._data.load_and_transform_text([text], self.device)}
        emb = self.model(inputs)[modality][0].detach().cpu()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_image(self, image_path: str | Path) -> torch.Tensor:
        """图像 → [1024] 归一化向量。"""
        modality = self._modality_type.VISION
        inputs = {modality: self._data.load_and_transform_vision_data(
            [str(image_path)], self.device
        )}
        emb = self.model(inputs)[modality][0].detach().cpu()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_audio(self, audio_path: str | Path) -> torch.Tensor:
        """音频 → [1024] 归一化向量。"""
        modality = self._modality_type.AUDIO
        inputs = {modality: self._data.load_and_transform_audio_data(
            [str(audio_path)], self.device
        )}
        emb = self.model(inputs)[modality][0].detach().cpu()
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_texts_batch(self, texts: list[str]) -> torch.Tensor:
        """批量文本编码 → [N, 1024]。"""
        if not texts:
            return torch.empty(0, self.embedding_dim)
        modality = self._modality_type.TEXT
        inputs = {modality: self._data.load_and_transform_text(texts, self.device)}
        embs = self.model(inputs)[modality].detach().cpu()
        return F.normalize(embs, dim=-1)

    @torch.no_grad()
    def encode_images_batch(self, image_paths: list[str | Path]) -> torch.Tensor:
        """批量图像编码 → [N, 1024]。"""
        if not image_paths:
            return torch.empty(0, self.embedding_dim)
        modality = self._modality_type.VISION
        inputs = {modality: self._data.load_and_transform_vision_data(
            [str(p) for p in image_paths], self.device
        )}
        embs = self.model(inputs)[modality].detach().cpu()
        return F.normalize(embs, dim=-1)

    @torch.no_grad()
    def encode_audios_batch(self, audio_paths: list[str | Path]) -> torch.Tensor:
        """批量音频编码 → [N, 1024]。"""
        if not audio_paths:
            return torch.empty(0, self.embedding_dim)
        modality = self._modality_type.AUDIO
        inputs = {modality: self._data.load_and_transform_audio_data(
            [str(p) for p in audio_paths], self.device
        )}
        embs = self.model(inputs)[modality].detach().cpu()
        return F.normalize(embs, dim=-1)

    def encode_text_image(self, text: str, image_path: str | Path) -> torch.Tensor:
        """ImageBind fallback: text/image 分别编码后做归一化均值融合。"""
        text_emb = self.encode_text(text)
        image_emb = self.encode_image(image_path)
        return F.normalize((text_emb + image_emb) / 2, dim=-1)

    @property
    def text_image_composition_method(self) -> str:
        return "mean_fusion"

    def encode_turn_text_only(self, turn: dict) -> torch.Tensor:
        """Text-Index: 将 turn 的全部信息拼为文本后编码。

        包含 voice_caption / user_text + assistant + image_caption。
        """
        text = build_text_for_turn(turn)
        return self.encode_text(text)

    def encode_turn_multimodal(self, turn: dict) -> torch.Tensor:
        """MM-Index legacy: 对 turn 存在的每种模态分别编码后归一化均值融合。

        text_emb  = encode_text(user_text + assistant)  # 始终有
        img_emb   = encode_image(image_path)            # 可选
        aud_emb   = encode_audio(voice_path)            # 可选
        fused     = normalize(mean([text_emb, img_emb?, aud_emb?]))
        """
        turn_id = turn.get("turn_id", "unknown")
        session_id = turn.get("session_id", "unknown")

        self.stats["total_turns"] += 1
        embeddings: list[torch.Tensor] = []

        # 文本: MM-Index 只使用原始文本，不使用 voice_caption 转写。
        text_parts = []
        if turn["user_text"]:
            text_parts.append(f"用户: {turn['user_text']}")
        if turn["assistant"]:
            text_parts.append(f"助手: {turn['assistant']}")
        text = "\n".join(text_parts) if text_parts else "空"
        text_emb = self.encode_text(text)
        embeddings.append(text_emb)

        got_image = False
        got_audio = False

        # 图像
        if turn["image_paths"]:
            raw_path = turn["image_paths"][0]
            img_path = resolve_media_path(raw_path, self.data_dir)
            if img_path and img_path.exists():
                try:
                    img_emb = self.encode_image(img_path)
                    embeddings.append(img_emb)
                    got_image = True
                except Exception as e:  # noqa: BLE001
                    self.stats["image_encode_failed"] += 1
                    logger.warning(
                        "图像编码失败 | turn_id=%s session_id=%s | "
                        "raw_path=%s resolved=%s | %s",
                        turn_id, session_id, raw_path, img_path, e
                    )
            else:
                self.stats["image_path_not_found"] += 1
                logger.warning(
                    "图像路径不存在 | turn_id=%s session_id=%s | "
                    "raw_path=%s resolved=%s",
                    turn_id, session_id, raw_path, img_path
                )

        # 音频
        if turn["voice_paths"]:
            raw_path = turn["voice_paths"][0]
            aud_path = resolve_media_path(raw_path, self.data_dir)
            if aud_path and aud_path.exists():
                try:
                    aud_emb = self.encode_audio(aud_path)
                    embeddings.append(aud_emb)
                    got_audio = True
                except Exception as e:  # noqa: BLE001
                    self.stats["audio_encode_failed"] += 1
                    logger.warning(
                        "音频编码失败 | turn_id=%s session_id=%s | "
                        "raw_path=%s resolved=%s | %s",
                        turn_id, session_id, raw_path, aud_path, e
                    )
            else:
                self.stats["audio_path_not_found"] += 1
                logger.warning(
                    "音频路径不存在 | turn_id=%s session_id=%s | "
                    "raw_path=%s resolved=%s",
                    turn_id, session_id, raw_path, aud_path
                )

        # 模态组合统计
        if got_image and got_audio:
            self.stats["text_image_audio"] += 1
        elif got_image:
            self.stats["text_image"] += 1
        elif got_audio:
            self.stats["text_audio"] += 1
        else:
            self.stats["text_only"] += 1

        fused = torch.stack(embeddings).mean(dim=0)
        return F.normalize(fused, dim=-1)

    def encode_turn_multimodal_items(self, turn: dict) -> dict[str, list[dict]]:
        """MM-Index: 为一个 turn 生成分模态 embedding items。

        每个 turn 始终生成 1 个 text item；每张图片和每段音频各自生成
        独立 item。调用方负责补充 item_id 和持久化 metadata。
        """
        turn_id = turn.get("turn_id", "unknown")
        session_id = turn.get("session_id", "unknown")

        self.stats["total_turns"] += 1
        items: dict[str, list[dict]] = {
            "text": [],
            "image": [],
            "audio": [],
        }

        text_parts = []
        if turn["user_text"]:
            text_parts.append(f"用户: {turn['user_text']}")
        if turn["assistant"]:
            text_parts.append(f"助手: {turn['assistant']}")
        text = "\n".join(text_parts) if text_parts else "空"
        items["text"].append({
            "embedding": self.encode_text(text),
            "path": None,
        })

        got_image = False
        got_audio = False

        for raw_path in turn["image_paths"]:
            img_path = resolve_media_path(raw_path, self.data_dir)
            if img_path and img_path.exists():
                try:
                    items["image"].append({
                        "embedding": self.encode_image(img_path),
                        "path": str(img_path),
                    })
                    got_image = True
                except Exception as e:  # noqa: BLE001
                    self.stats["image_encode_failed"] += 1
                    logger.warning(
                        "图像编码失败 | turn_id=%s session_id=%s | "
                        "raw_path=%s resolved=%s | %s",
                        turn_id, session_id, raw_path, img_path, e
                    )
            else:
                self.stats["image_path_not_found"] += 1
                logger.warning(
                    "图像路径不存在 | turn_id=%s session_id=%s | "
                    "raw_path=%s resolved=%s",
                    turn_id, session_id, raw_path, img_path
                )

        for raw_path in turn["voice_paths"]:
            aud_path = resolve_media_path(raw_path, self.data_dir)
            if aud_path and aud_path.exists():
                try:
                    items["audio"].append({
                        "embedding": self.encode_audio(aud_path),
                        "path": str(aud_path),
                    })
                    got_audio = True
                except Exception as e:  # noqa: BLE001
                    self.stats["audio_encode_failed"] += 1
                    logger.warning(
                        "音频编码失败 | turn_id=%s session_id=%s | "
                        "raw_path=%s resolved=%s | %s",
                        turn_id, session_id, raw_path, aud_path, e
                    )
            else:
                self.stats["audio_path_not_found"] += 1
                logger.warning(
                    "音频路径不存在 | turn_id=%s session_id=%s | "
                    "raw_path=%s resolved=%s",
                    turn_id, session_id, raw_path, aud_path
                )

        if got_image and got_audio:
            self.stats["text_image_audio"] += 1
        elif got_image:
            self.stats["text_image"] += 1
        elif got_audio:
            self.stats["text_audio"] += 1
        else:
            self.stats["text_only"] += 1

        return items

    def encode_query_text(self, question: str,
                          option_captions: dict[str, str] | None = None) -> torch.Tensor:
        """编码 QA 查询 (Text-Index 模式)。"""
        text = question
        if option_captions:
            for letter in sorted(option_captions):
                text += f"\n{letter}. {option_captions[letter]}"
        return self.encode_text(text)

    def encode_query_multimodal(self, question: str,
                                option_captions: dict[str, str] | None = None) -> torch.Tensor:
        """编码 QA 查询 (MM-Index 模式)。

        查询始终只用文本编码，避免选项图片泄漏答案信息到检索阶段。
        """
        return self.encode_query_text(question, option_captions)

    def encode_query_multimodal_items(self, qa: dict) -> dict[str, list[dict]]:
        """编码 MM-Index query 的分模态 items。

        文本 query 始终只使用 question；图片类 QA 额外编码 A/B/C/D
        四张 option_images。图片编码失败时才 fallback 到对应 caption。
        """
        query_items: dict[str, list[dict]] = {
            "text": [{
                "embedding": self.encode_text(qa.get("question", "")),
                "metadata": {"source": "question"},
            }],
            "image": [],
            "audio": [],
        }

        point = qa.get("point", "")
        if not point.endswith("_img"):
            return query_items

        option_images = qa.get("option_images") or {}
        captions = qa.get("option_captions") or qa.get("question_image_descriptions") or {}

        for letter in sorted(option_images):
            raw_path = option_images[letter]
            img_path = resolve_media_path(raw_path, self.data_dir)
            if img_path and img_path.exists():
                try:
                    query_items["image"].append({
                        "embedding": self.encode_image(img_path),
                        "metadata": {
                            "source": "option_image",
                            "letter": letter,
                            "path": str(img_path),
                            "mtime": img_path.stat().st_mtime,
                        },
                    })
                    continue
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "选项图片 query 编码失败 | qa_id=%s option=%s "
                        "raw_path=%s resolved=%s | %s",
                        qa.get("qa_id", ""), letter, raw_path, img_path, e
                    )
            else:
                logger.warning(
                    "选项图片 query 路径不存在 | qa_id=%s option=%s "
                    "raw_path=%s resolved=%s",
                    qa.get("qa_id", ""), letter, raw_path, img_path
                )

            caption = captions.get(letter, "")
            if caption:
                query_items["text"].append({
                    "embedding": self.encode_text(f"选项 {letter}: {caption}"),
                    "metadata": {
                        "source": "option_caption_fallback",
                        "letter": letter,
                    },
                })

        return query_items

    @staticmethod
    def _extract_text_options(qa: dict) -> dict[str, str]:
        """兼容常见 QA 选项字段，统一返回 A/B/C/D 文本映射。"""
        for field in (
            "options",
            "choices",
            "option_texts",
            "option_captions",
            "question_options",
        ):
            raw = qa.get(field)
            if isinstance(raw, dict):
                options = {}
                for letter in "ABCD":
                    value = raw.get(letter)
                    if value is None:
                        value = raw.get(letter.lower())
                    if isinstance(value, dict):
                        value = (
                            value.get("text")
                            or value.get("content")
                            or value.get("caption")
                            or value.get("label")
                        )
                    if value is not None and str(value).strip():
                        options[letter] = str(value).strip()
                if options:
                    return options
            elif isinstance(raw, list):
                options = {
                    letter: str(value).strip()
                    for letter, value in zip("ABCD", raw)
                    if value is not None and str(value).strip()
                }
                if options:
                    return options
        question = str(qa.get("question", ""))
        if question:
            # Current benchmark text QAs often inline options in the question:
            # "A. ...\nB. ...\nC. ...\nD. ...\n请在 A/B/C/D 中选择..."
            pattern = re.compile(
                r"(?:^|\n)\s*([ABCD])\s*[\.．、:：]\s*"
                r"(.+?)(?=(?:\n\s*[ABCD]\s*[\.．、:：])|\n\s*请在\s*A/B/C/D|$)",
                re.DOTALL,
            )
            options = {}
            for letter, text in pattern.findall(question):
                cleaned = re.sub(r"\s+", " ", text).strip()
                if cleaned:
                    options[letter.upper()] = cleaned
            if options:
                return options
        return {}

    def encode_query_composed_items(self, qa: dict) -> dict[str, list[dict]]:
        """生成 option-conditioned composed query embeddings。"""
        question = str(qa.get("question", ""))
        point = str(qa.get("point", ""))
        query_items: dict[str, list[dict]] = {"composed": []}

        if point.endswith("_img"):
            option_images = qa.get("option_images") or {}
            captions = (
                qa.get("option_captions")
                or qa.get("question_image_descriptions")
                or {}
            )
            letters = [
                letter
                for letter in "ABCD"
                if letter in option_images
                or (isinstance(captions, dict) and letter in captions)
            ]
            for letter in letters:
                raw_path = option_images.get(letter)
                img_path = (
                    resolve_media_path(raw_path, self.data_dir) if raw_path else None
                )
                if img_path and img_path.exists():
                    try:
                        query_items["composed"].append({
                            "embedding": self.encode_text_image(question, img_path),
                            "metadata": {
                                "source": "question_plus_option_image",
                                "letter": letter,
                                "point": point,
                                "path": str(img_path),
                                "mtime": img_path.stat().st_mtime,
                                "size": img_path.stat().st_size,
                                "composition_method": self.text_image_composition_method,
                            },
                        })
                        continue
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "组合 query 编码失败 | qa_id=%s option=%s "
                            "raw_path=%s resolved=%s | %s",
                            qa.get("qa_id", ""), letter, raw_path, img_path, exc,
                        )
                elif raw_path:
                    logger.warning(
                        "组合 query 图片不存在 | qa_id=%s option=%s "
                        "raw_path=%s resolved=%s",
                        qa.get("qa_id", ""), letter, raw_path, img_path,
                    )

                caption = captions.get(letter, "") if isinstance(captions, dict) else ""
                if caption:
                    query_items["composed"].append({
                        "embedding": self.encode_text(
                            f"{question}\n选项 {letter}: {caption}"
                        ),
                        "metadata": {
                            "source": "question_plus_option_image",
                            "letter": letter,
                            "point": point,
                            "fallback": "option_caption",
                            "composition_method": "text_fallback",
                        },
                    })

        else:
            options = self._extract_text_options(qa)
            for letter in "ABCD":
                if letter not in options:
                    continue
                query_items["composed"].append({
                    "embedding": self.encode_text(
                        f"{question}\n选项 {letter}: {options[letter]}"
                    ),
                    "metadata": {
                        "source": "question_plus_option_text",
                        "letter": letter,
                        "point": point,
                        "composition_method": "text_concatenation",
                    },
                })

        if not query_items["composed"]:
            query_items["composed"].append({
                "embedding": self.encode_text(question),
                "metadata": {
                    "source": "question_text_only_fallback",
                    "point": point,
                    "fallback": "missing_options",
                    "composition_method": "text_only",
                },
            })
        return query_items


class GeminiEmbeddingEncoder(ImageBindEncoder):
    """Gemini Embedding 2 编码器。

    只替换底层 text/image/audio embedding 调用；turn 文本构造、分模态
    item 构建和统计逻辑复用 ImageBindEncoder 的实现。
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str = GEMINI_EMBEDDING_MODEL,
        embedding_dim: int = GEMINI_EMBEDDING_DIM,
        data_dir: str | Path | None = None,
        **_: Any,
    ):
        import httpx

        self.provider = "gemini"
        self.model_name = model
        self.embedding_dim = embedding_dim
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
        self.api_base = required_runtime_value(
            api_base or GEMINI_EMBEDDING_API_BASE,
            argument_name="Gemini embedding API base",
            env_names=(
                "RQ3_GEMINI_EMBEDDING_API_BASE",
                "GEMINI_EMBEDDING_API_BASE",
                "OPENROUTER_API_BASE",
            ),
        ).rstrip("/")
        self.api_key = required_runtime_value(
            api_key or GEMINI_EMBEDDING_API_KEY,
            argument_name="Gemini embedding API key",
            env_names=(
                "RQ3_GEMINI_EMBEDDING_API_KEY",
                "GEMINI_EMBEDDING_API_KEY",
                "GEMINI_API_KEY",
                "OPENROUTER_API_KEY",
            ),
        )
        try:
            self.client = httpx.Client(
                timeout=120.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        except Exception as exc:  # noqa: BLE001
            safe_error = redact_sensitive_text(exc, self.api_key, self.api_base)
            raise RuntimeError(
                f"failed to initialize embedding client: {safe_error}"
            ) from None
        self.stats: dict[str, int] = defaultdict(int)

    def _embedding_kwargs(self, input_data: Any) -> dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "input": input_data,
        }
        if self.embedding_dim:
            kwargs["dimensions"] = self.embedding_dim
        return kwargs

    def _post_embeddings(self, input_data: Any) -> dict[str, Any]:
        response = self.client.post(
            f"{self.api_base}/embeddings",
            json=self._embedding_kwargs(input_data),
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"raw_text": response.text}

        if response.status_code >= 400:
            raise RuntimeError(
                "Gemini embedding API request failed: "
                f"status={response.status_code}, response="
                f"{redact_sensitive_text(response.text[:1000], self.api_key, self.api_base)}"
            )
        return payload

    @staticmethod
    def _extract_embeddings(payload: dict[str, Any]) -> list[list[float]]:
        data = payload.get("data")
        if isinstance(data, list) and data:
            embeddings = []
            for item in data:
                embedding = item.get("embedding") if isinstance(item, dict) else None
                if embedding is None:
                    raise RuntimeError("Gemini embedding API response item has no embedding")
                embeddings.append(embedding)
            return embeddings

        # Google-style fallback, useful if a compatible proxy returns native shape.
        if isinstance(payload.get("embedding"), dict):
            values = payload["embedding"].get("values")
            if values is not None:
                return [values]
        if isinstance(payload.get("embeddings"), list) and payload["embeddings"]:
            embeddings = []
            for item in payload["embeddings"]:
                if isinstance(item, dict) and "values" in item:
                    embeddings.append(item["values"])
                elif isinstance(item, dict) and "embedding" in item:
                    embeddings.append(item["embedding"])
            if embeddings:
                return embeddings

        raise RuntimeError("Gemini embedding API returned no embeddings")

    def _embed(self, input_data: Any, wrap_batch: bool = True) -> torch.Tensor:
        payload = self._post_embeddings([input_data] if wrap_batch else input_data)
        emb = torch.tensor(self._extract_embeddings(payload)[0], dtype=torch.float32)
        return F.normalize(emb.detach().cpu(), dim=-1)

    @staticmethod
    def _file_to_data_url(path: str | Path, default_mime: str) -> str:
        p = Path(path)
        mime = mimetypes.guess_type(p.name)[0] or default_mime
        data_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{data_b64}"

    @staticmethod
    def _file_to_b64(path: str | Path) -> str:
        return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

    @staticmethod
    def _audio_clip_format(path: str | Path) -> str:
        ext = Path(path).suffix.lower().lstrip(".")
        if ext in {"wav", "mp3", "flac", "ogg", "m4a", "aac", "opus"}:
            return ext
        return "wav"

    def encode_text(self, text: str) -> torch.Tensor:
        """文本 → [D] 归一化向量。"""
        return self._embed(text)

    def encode_image(self, image_path: str | Path) -> torch.Tensor:
        """图像原始文件 → [D] 归一化向量。"""
        data_url = self._file_to_data_url(image_path, "image/png")
        return self._embed(
            [{
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    }
                ]
            }],
            wrap_batch=False,
        )

    def encode_text_image(self, text: str, image_path: str | Path) -> torch.Tensor:
        """Gemini joint text+image input → [D] 归一化向量。"""
        data_url = self._file_to_data_url(image_path, "image/png")
        return self._embed(
            [{
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ]
            }],
            wrap_batch=False,
        )

    @property
    def text_image_composition_method(self) -> str:
        return "gemini_joint_text_image"

    def encode_audio(self, audio_path: str | Path) -> torch.Tensor:
        """音频原始文件 → [D] 归一化向量。"""
        fmt = self._audio_clip_format(audio_path)
        mime = mimetypes.guess_type(str(audio_path))[0] or f"audio/{fmt}"
        data_url = f"data:{mime};base64,{self._file_to_b64(audio_path)}"
        return self._embed(
            [{
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_url,
                            "format": fmt,
                        },
                    }
                ]
            }],
            wrap_batch=False,
        )

    def encode_texts_batch(self, texts: list[str]) -> torch.Tensor:
        """批量文本编码 → [N, D]。"""
        if not texts:
            return torch.empty(0, self.embedding_dim)
        payload = self._post_embeddings(texts)
        embs = torch.tensor(self._extract_embeddings(payload), dtype=torch.float32)
        return F.normalize(embs.detach().cpu(), dim=-1)

    def encode_images_batch(self, image_paths: list[str | Path]) -> torch.Tensor:
        """批量图像编码 → [N, D]。"""
        if not image_paths:
            return torch.empty(0, self.embedding_dim)
        return torch.stack([self.encode_image(path) for path in image_paths])

    def encode_audios_batch(self, audio_paths: list[str | Path]) -> torch.Tensor:
        """批量音频编码 → [N, D]。"""
        if not audio_paths:
            return torch.empty(0, self.embedding_dim)
        return torch.stack([self.encode_audio(path) for path in audio_paths])


def create_encoder(
    provider: str = EMBEDDING_PROVIDER,
    device: str = "cuda:0",
    gemini_api_key: str | None = None,
    gemini_api_base: str | None = None,
    gemini_model: str = GEMINI_EMBEDDING_MODEL,
    gemini_embedding_dim: int = GEMINI_EMBEDDING_DIM,
    data_dir: str | Path | None = None,
):
    """根据 provider 创建 embedding encoder。"""
    normalized = provider.lower()
    if normalized == "imagebind":
        return ImageBindEncoder(device=device, data_dir=data_dir)
    if normalized == "gemini":
        return GeminiEmbeddingEncoder(
            api_key=gemini_api_key,
            api_base=gemini_api_base,
            model=gemini_model,
            embedding_dim=gemini_embedding_dim,
            data_dir=data_dir,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")
