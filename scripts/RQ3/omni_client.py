"""Qwen3-Omni API 调用封装。支持文本、图像、音频混合输入。"""

from __future__ import annotations

import re
import sys
import time
from typing import Any

from .config import (
    ALIYUN_OMNI_API_BASE,
    ALIYUN_OMNI_API_KEY,
    ALIYUN_OMNI_MODEL,
    EVAL_MODEL_PROVIDER,
    OMNI_API_BASE,
    OMNI_API_KEY,
    OMNI_MODEL,
    PROMPT_DIR,
    redact_sensitive_text,
    required_runtime_value,
)


def _load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


SYSTEM_PROMPT = _load_prompt("sys_prompt")

FORMAT_PROMPTS = {
    "pref_text": _load_prompt("preference_prompt"),
    "pref_img": _load_prompt("preference_prompt"),
    "rec_text": _load_prompt("recommendation_prompt"),
    "rec_img": _load_prompt("recommendation_prompt"),
    "entity_text": _load_prompt("entity_prompt"),
    "entity_img": _load_prompt("entity_prompt"),
    "refusal_text": _load_prompt("refusal_prompt"),
}

# 追加到 format_prompt 末尾，要求模型同时输出选项字母和推理依据
REASONING_FORMAT = (
    "\n\n## 输出格式\n"
    "第一行输出你选择的选项字母（A、B、C 或 D）。\n"
    "第二行输出「依据session：」后跟你做出判断所依赖的所有记忆的 session 日期"
    "（即记忆中 timestamp 字段的值），用逗号分隔。\n"
    "第三行起输出你的判断依据：结合上述 session 日期，说明你是如何从对话文本、"
    "图片描述还是语音消息中推断出用户的偏好或习惯，从而选出该选项的。\n"
    "示例：\n"
    "A\n"
    "依据session：2024-03-15, 2024-04-02\n"
    "依据：在 2024-03-15 的对话中，用户通过语音消息提到喜欢清淡口味；"
    "在 2024-04-02 的对话中，用户分享了一张日式料理的图片并表达了喜爱，因此选择A。"
)


def _split_answer_reasoning(raw: str) -> tuple[str, str, list[str]]:
    """将模型原始输出拆分为选项字母、推理依据和依据 session 日期列表。"""
    if not raw or not raw.strip():
        return raw, "", []

    lines = raw.strip().splitlines()
    choice_line = lines[0].strip()

    reasoning_sessions: list[str] = []
    reasoning_lines: list[str] = []

    for line in lines[1:]:
        stripped = line.strip()
        matched = False

        for prefix in ("依据session：", "依据session:", "依据 session：", "依据 session:"):
            if stripped.lower().startswith(prefix.lower()):
                session_str = stripped[len(prefix):].strip()
                reasoning_sessions = [
                    s.strip()
                    for s in re.split(r"[,，;；\s]+", session_str)
                    if s.strip()
                ]
                matched = True
                break

        if not matched:
            reasoning_lines.append(line)

    reasoning = "\n".join(reasoning_lines).strip()

    for prefix in ("依据：", "依据:", "判断依据：", "判断依据:"):
        if reasoning.startswith(prefix):
            reasoning = reasoning[len(prefix):].strip()
            break

    return choice_line, reasoning, reasoning_sessions


class OmniClient:
    """Qwen3-Omni 问答客户端。"""

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        provider: str = EVAL_MODEL_PROVIDER,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ):
        self.provider = provider.lower()
        if self.provider == "aliyun":
            model = model or ALIYUN_OMNI_MODEL
            api_base = api_base or ALIYUN_OMNI_API_BASE
            api_key = api_key or ALIYUN_OMNI_API_KEY
            base_env_names = ("RQ3_ALIYUN_OMNI_API_BASE",)
            key_env_names = ("RQ3_ALIYUN_OMNI_API_KEY", "DASHSCOPE_API_KEY")
        elif self.provider == "vllm":
            model = model or OMNI_MODEL
            api_base = api_base or OMNI_API_BASE
            api_key = api_key or OMNI_API_KEY
            base_env_names = ("RQ3_OMNI_API_BASE", "CUE_MEM_LLM_BASE_URL")
            key_env_names = ("RQ3_OMNI_API_KEY", "CUE_MEM_LLM_API_KEY")
        else:
            raise ValueError(f"Unsupported eval provider: {provider}")

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
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def answer_qa(
        self,
        qa: dict,
        memory_content: str | list[dict],
        question_content: str | list[dict],
        use_mode: str,
        with_reasoning: bool = False,
    ) -> dict[str, Any]:
        """调用 Qwen3-Omni 回答一个 QA。

        Args:
            qa: 原始 QA dict
            memory_content: Text-Use 为 str；MM-Use 为 content blocks list
            question_content: Text-Use 为 str；MM-Use 为 content blocks list
            use_mode: 'text' | 'multimodal'
            with_reasoning: 是否要求模型输出推理依据

        Returns:
            {
                'model_answer': str,          提取的选项字母 (A/B/C/D)
                'raw_response': str,          模型原始回复
                'reasoning': str,             推理依据（with_reasoning=True 时有值）
                'reasoning_sessions': list,   依据的 session 日期列表
            }
        """
        point = qa.get("point", "pref_text")
        format_prompt = FORMAT_PROMPTS.get(point, "")

        if with_reasoning and format_prompt:
            format_prompt = format_prompt.rstrip() + REASONING_FORMAT

        # with_reasoning 时输出更长，调高 max_tokens
        max_tokens = 512 if with_reasoning else self.max_tokens

        if use_mode == "text":
            user_content = f"{format_prompt}\n\n{memory_content}\n\n{question_content}"
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        else:
            # MM-Use: content blocks
            blocks: list[dict] = []

            if format_prompt:
                blocks.append({"type": "text", "text": format_prompt})

            if isinstance(memory_content, list):
                blocks.extend(memory_content)
            else:
                blocks.append({"type": "text", "text": str(memory_content)})

            blocks.append({"type": "text", "text": "\n"})

            if isinstance(question_content, list):
                blocks.extend(question_content)
            else:
                blocks.append({"type": "text", "text": str(question_content)})

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": blocks},
            ]

        raw = self._call_model(messages, max_tokens)

        reasoning = ""
        reasoning_sessions: list[str] = []

        if with_reasoning and raw and not raw.startswith("[ERROR]"):
            choice_part, reasoning, reasoning_sessions = _split_answer_reasoning(raw)
            model_answer = _extract_answer(choice_part)
        else:
            model_answer = _extract_answer(raw)

        return {
            "model_answer": model_answer,
            "raw_response": raw,
            "reasoning": reasoning,
            "reasoning_sessions": reasoning_sessions,
        }

    def _call_model(self, messages: list[dict], max_tokens: int) -> str:
        try:
            if self.provider == "aliyun":
                return self._call_with_retries(messages, max_tokens)
            return self._call_vllm(messages, max_tokens)
        except Exception as e:  # noqa: BLE001
            print(
                f"[OMNI ERROR] provider={self.provider} model={self.model} "
                f"blocks={getattr(self, '_last_aliyun_block_counts', {})}: "
                f"{redact_sensitive_text(e, self.api_key, self.api_base)}",
                file=sys.stderr,
                flush=True,
            )
            return f"[ERROR] {redact_sensitive_text(e, self.api_key, self.api_base)}"

    def _call_with_retries(self, messages: list[dict], max_tokens: int) -> str:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                return self._call_aliyun(messages, max_tokens)
            except Exception as e:
                last_error = e
                msg = redact_sensitive_text(e, self.api_key, self.api_base)
                if "insufficient_quota" in msg or "InvalidParameter" in msg:
                    raise
                if "Request rate increased too quickly" not in msg and "429" not in msg:
                    raise
                wait = min(30, 2 ** attempt)
                print(
                    f"[OMNI RETRY] provider=aliyun model={self.model} "
                    f"attempt={attempt + 1}/4 wait={wait}s error={msg}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        raise last_error or RuntimeError("Aliyun request failed after retries")

    def _call_vllm(self, messages: list[dict], max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
            # Disable Qwen3-Omni's thinking mode for deterministic QA evaluation.
            # Without this, the model generates unbounded <think> tokens before
            # responding, causing requests to hang for minutes.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (response.choices[0].message.content or "").strip()

    def _call_aliyun(self, messages: list[dict], max_tokens: int) -> str:
        normalized = self._normalize_messages_for_aliyun(messages)
        self._last_aliyun_block_counts = self._message_block_counts(normalized)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=normalized,
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

    @staticmethod
    def _message_block_counts(messages: list[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                typ = block.get("type", "<missing>")
                counts[typ] = counts.get(typ, 0) + 1
        return counts

    @staticmethod
    def _normalize_messages_for_aliyun(messages: list[dict]) -> list[dict]:
        normalized = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                content = [OmniClient._normalize_block_for_aliyun(b) for b in content]
            normalized.append({**msg, "content": content})
        return normalized

    @staticmethod
    def _normalize_block_for_aliyun(block: dict) -> dict:
        if block.get("type") != "audio_url":
            return block

        url = (block.get("audio_url") or {}).get("url", "")
        if not url.startswith("data:") or ";base64," not in url:
            return block

        header, _data = url.split(",", 1)
        mime = header.removeprefix("data:").split(";")[0]
        fmt = mime.split("/")[-1] if "/" in mime else "wav"
        return {
            "type": "input_audio",
            "input_audio": {
                "data": url,
                "format": fmt,
            },
        }


def _extract_answer(text: str) -> str:
    """从模型回复中提取 A/B/C/D。"""
    text = text.strip()

    if text in ("A", "B", "C", "D"):
        return text

    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)

    for ch in text:
        if ch in "ABCD":
            return ch

    return text[:1] if text else ""
