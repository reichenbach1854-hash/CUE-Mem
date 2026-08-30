"""
    RQ1/RQ2 benchmark 主运行脚本。

这个脚本负责把整理好的历史多轮对话和 QA 数据送入不同 Memory 方法进行评测。
整体流程如下：

1. 从 benchmark/data/dialog/{data_name}.json 读取数据集。
   数据集需要包含：
   - character_profile: 用户画像，用于构造 user 名称；
   - multi_session_dialogues: 多 session 历史对话；
   - human-annotated QAs: 待评测问题列表。

2. 将历史对话转换为 memory 模块统一接收的格式。
   - 文本轮次会保留 user/assistant 文本；
   - 图像轮次会保留 image path、image caption、image_id；
   - 音频轮次会保留 voice path、voice caption、voice_id；
   - 对于纯文本 memory，图像/音频 caption 会被拼接进 text 字段。

3. 根据 --memory_name 初始化对应 memory 模块，并把历史对话逐条写入 memory。

4. 遍历 QA，先用问题召回相关记忆，再调用 LLM/VLM 生成答案。
   对于 entity/preference/recommendation/refusal 等 category，会追加对应 prompt
   约束模型只输出 A/B/C/D。

5. 保存回答结果与可选的运行效率统计。

注意：脚本依赖 benchmark/memengine 这个本地包，因此建议从
    RQ1_RQ2 benchmark package can be run from the repository root:
    python run_bench.py --data_name history_with_qa --sample 5 ...
"""

import hashlib
import logging
import json
import random
import re
import sys
import pickle
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from benchmark.memengine.function.prompt import format_prompt, render_prompt
from benchmark.paths import (
    BENCHMARK_ROOT,
    DATA_ROOT,
    DIALOG_ROOT,
    IMAGE_ROOT,
    MEMORY_CACHE_ROOT,
    PROMPT_ROOT,
    RESULT_ROOT,
    VOICE_ROOT,
    memory_cache_path,
)
from benchmark.security import redact_runtime_text, safe_runtime_error
from scripts.common.llm import openai_client

from memengine import MemoryConfig
from default_config.DefaultMemoryConfig import DEFAULT_FUMEMORY, DEFAULT_LTMEMORY, DEFAULT_STMEMORY, DEFAULT_GAMEMORY, DEFAULT_NGMEMORY, DEFAULT_AUGUSTUSMEMORY, DEFAULT_UNIVERSALRAGMEMORY, DEFAULT_MMFUMEMORY
from default_config.DefaultMMMemoryConfig import DEFAULT_MMMEMORY  
from default_config.DefaultMemoryConfig import DEFAULT_MGMEMORY, DEFAULT_RFMEMORY, DEFAULT_AMEMORY, DEFAULT_MEMORYOSMEMORY
import time
import threading
import concurrent.futures
from tqdm import tqdm
import os
import argparse
import base64
import warnings
import numpy as np
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 路径与目录配置
# ---------------------------------------------------------------------------
# SCRIPT_DIR 指向当前 run_bench.py 所在目录：benchmark/run
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT is the externally configurable benchmark root, not the code package.
PROJECT_ROOT = str(BENCHMARK_ROOT)

# benchmark 数据目录约定：
# - data/dialog  : 存放待评测数据集 JSON；
# - data/image   : 存放 Mem-Gallery 原始图像资源；
# - data/voice   : 存放语音资源；
# - prompt       : 存放不同 QA category 的输出格式约束。
DATA_DIR = str(DATA_ROOT)
DIALOG_DIR = str(DIALOG_ROOT)
IMAGE_DIR = str(IMAGE_ROOT)
VOICE_DIR = str(VOICE_ROOT)
#RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
# 当前默认写入 result_debug，避免覆盖正式评测结果。
RESULT_DIR = str(RESULT_ROOT)
PROMPT_DIR = str(PROMPT_ROOT)

OPENAI_APIKEY = None
OPENAI_APIBASE = None
OPENAI_MODEL = None


def get_available_datasets(dialog_dir=DIALOG_DIR):
    """
    扫描 data/dialog 目录，返回所有可用数据集名。

    run_bench.py 的 --data_name 参数不带 .json 后缀；例如：
      data/dialog/history_with_qa.json -> --data_name history_with_qa

    Returns:
        list: 去掉 .json 后缀后的数据集名列表。
    """
    datasets = []
    if os.path.exists(dialog_dir):
        for filename in os.listdir(dialog_dir):
            if filename.endswith('.json'):
                # Skip result files and other special files
                if '_results_' in filename or '_evaluate_result_' in filename:
                    continue
                # Extract dataset name: extract "DatasetName" from "DatasetName.json"
                dataset_name = filename.replace('.json', '')
                datasets.append(dataset_name)
    return sorted(datasets)


def _json_hash(obj):
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _amem_cache_path(data_name, processed_dialogs, memory_config, model_name, config_label):
    cache_key = _json_hash({
        "version": 1,
        "data_name": data_name,
        "config_label": config_label,
        "model_name": model_name,
        "dialog_hash": _json_hash(processed_dialogs),
        "embedding_model": memory_config.get("embedding_model"),
        "llm_backend": memory_config.get("llm_backend"),
        "llm_model": memory_config.get("llm_model"),
        "api_base": memory_config.get("api_base"),
        "evo_threshold": memory_config.get("evo_threshold"),
        "evo_interval": memory_config.get("evo_interval"),
    })
    cache_dir = str(memory_cache_path("AMemMemory"))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{data_name}_{cache_key}.pkl")


def _try_load_amem_cache(memory_agent, cache_path, processed_dialog_count):
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if int(payload.get("processed_dialog_count", -1)) != int(processed_dialog_count):
            print(f"[AMem cache] ignored stale cache: {cache_path}")
            return False
        memory_agent.memory.import_cache(payload["memory"])
        note_count = len(memory_agent.memory.memory_system.memories)
        print(f"[AMem cache] loaded {note_count} notes from {cache_path}")
        return True
    except Exception as e:
        print(f"[AMem cache] failed to load {cache_path}: {redact_runtime_text(e)}; rebuilding memory")
        return False


def _try_save_amem_cache(memory_agent, cache_path, processed_dialog_count, memory_duration):
    try:
        payload = {
            "version": 1,
            "processed_dialog_count": int(processed_dialog_count),
            "memory_time_seconds": float(memory_duration),
            "memory": memory_agent.memory.export_cache(),
        }
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        note_count = len(memory_agent.memory.memory_system.memories)
        print(f"[AMem cache] saved {note_count} notes -> {cache_path}")
    except Exception as e:
        print(f"[AMem cache] failed to save {cache_path}: {redact_runtime_text(e)}")


def _memoryos_cache_path(data_name, processed_dialogs, memory_config, model_name, config_label):
    cache_key = _json_hash({
        "version": 1,
        "data_name": data_name,
        "config_label": config_label,
        "model_name": model_name,
        "dialog_hash": _json_hash(processed_dialogs),
        "embedding_model": memory_config.get("embedding_model"),
        "embedding_model_kwargs": memory_config.get("embedding_model_kwargs"),
        "llm_model": memory_config.get("llm_model"),
        "api_base": memory_config.get("api_base"),
        "short_term_capacity": memory_config.get("short_term_capacity"),
        "mid_term_capacity": memory_config.get("mid_term_capacity"),
        "long_term_knowledge_capacity": memory_config.get("long_term_knowledge_capacity"),
        "retrieval_queue_capacity": memory_config.get("retrieval_queue_capacity"),
        "mid_term_heat_threshold": memory_config.get("mid_term_heat_threshold"),
        "mid_term_similarity_threshold": memory_config.get("mid_term_similarity_threshold"),
        "fast_mode": memory_config.get("fast_mode"),
        "fast_batch_size": memory_config.get("fast_batch_size"),
        "fast_use_llm_summary": memory_config.get("fast_use_llm_summary"),
    })
    cache_dir = str(memory_cache_path("MemoryOSMemory"))
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{data_name}_{cache_key}.pkl")


def _try_load_memoryos_cache(memory_agent, cache_path, processed_dialog_count):
    if not os.path.exists(cache_path):
        return False
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        if int(payload.get("processed_dialog_count", -1)) != int(processed_dialog_count):
            print(f"[MemoryOS cache] ignored stale cache: {cache_path}")
            return False
        memory_agent.memory.import_cache(payload["memory"])
        short_count = len(memory_agent.memory.memoryos.short_term_memory.get_all())
        mid_count = len(memory_agent.memory.memoryos.mid_term_memory.sessions)
        print(f"[MemoryOS cache] loaded short_term={short_count}, mid_term_sessions={mid_count} from {cache_path}")
        return True
    except Exception as e:
        print(f"[MemoryOS cache] failed to load {cache_path}: {redact_runtime_text(e)}; rebuilding memory")
        return False


def _try_save_memoryos_cache(memory_agent, cache_path, processed_dialog_count, memory_duration):
    try:
        payload = {
            "version": 1,
            "processed_dialog_count": int(processed_dialog_count),
            "memory_time_seconds": float(memory_duration),
            "memory": memory_agent.memory.export_cache(),
        }
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        short_count = len(memory_agent.memory.memoryos.short_term_memory.get_all())
        mid_count = len(memory_agent.memory.memoryos.mid_term_memory.sessions)
        print(f"[MemoryOS cache] saved short_term={short_count}, mid_term_sessions={mid_count} -> {cache_path}")
    except Exception as e:
        print(f"[MemoryOS cache] failed to save {cache_path}: {redact_runtime_text(e)}")


# Prompt 文件缓存，避免每条 QA 都重复读磁盘。
_PROMPT_CACHE = {}

def load_prompt_file(filename):
    """
    从 benchmark/prompt 目录读取 prompt 文件，并做内存缓存。

    Args:
        filename: prompt 文件名，例如 sys_prompt.txt。

    Returns:
        str: prompt 文本；文件不存在或读取失败时返回空字符串。
    """
    if filename in _PROMPT_CACHE:
        return _PROMPT_CACHE[filename]
    
    prompt_path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            _PROMPT_CACHE[filename] = content
            return content
    except FileNotFoundError:
        print(f"Warning: Prompt file not found: {prompt_path}")
        _PROMPT_CACHE[filename] = ""
        return ""
    except Exception as e:
        print(f"Error loading prompt file {filename}: {redact_runtime_text(e)}")
        _PROMPT_CACHE[filename] = ""
        return ""

def _load_all_prompts():
    """脚本启动时预加载常用 prompt，便于后续按 category 快速取用。"""
    prompt_files = [
        "sys_prompt.txt", "ar_prompt.txt", "cd_prompt.txt", "vs_prompt.txt",
        "entity_prompt.txt", "preference_prompt.txt",
        "recommendation_prompt.txt", "refusal_prompt.txt",
    ]
    for filename in prompt_files:
        load_prompt_file(filename)

# Execute preloading
_load_all_prompts()


# Load SystemPrompt from file
SystemPrompt = load_prompt_file("sys_prompt.txt")
if not SystemPrompt:
    # If file does not exist, use default value as fallback
    SystemPrompt = """Your task is to answer questions in a concise manner with the help of memory content.
When the question is: \"What did the charity race raise awareness for?\", you should not answer in the form of: \"The charity race raised awareness for mental health.\" Instead, it should be: \"Mental health.\", as this is more concise.
"""

TextMsgPrompt = """
The retrieved memory contents are as follows:

{memory_context}
""" 

MsgStartPromptWOMemory = """
The retrieved memory contents are as follows:

"""

MMMemoryDialoguePrompt = """
{textual_context}
image:
image_id: {image_id}
image_content:
"""


DialogueAgentPrompt = """
Your task is to answer the question about the conversation between {speaker_a} and {speaker_b} in a concise manner with the help of memory content.
Please only provide the content of the answer, without including introductory phrases like 'answer:'.
For questions that require answering a date or time, strictly follow the format and provide a specific date or time whenever possible.
Generate answers primarily concise, yet complete enough to accurately answer the questions.

The current question is as follows:
{observation} {format_constraint}
"""

DialogueAgentPromptImage = """
Here is the attached image of the question:
"""

def seed_everything(seed=42):
    """固定 Python / numpy / torch 的随机种子，提升小批量测试的可复现性。"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")


def _ensure_list(value):
    """把 None / 标量 / list 统一转成 list，方便媒体字段按数组处理。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _safe_get_list_item(values, index=0, default=""):
    """安全读取列表第 index 个元素；缺失时返回 default。"""
    values = _ensure_list(values)
    if 0 <= index < len(values):
        return values[index]
    return default


def resolve_media_path(media_path, media_root, media_prefix, fallback_root=None):
    """
    将数据集中的媒体路径解析为本地可访问路径。

    支持三类路径：
    1. 绝对路径：直接 normpath 后返回；
    2. Benchmark 标准相对路径：例如 ../image/D00/D00-001.png；
    3. 其他相对路径：从 fallback_root 或 media_root 下解析。

    这样同一套逻辑可以兼容历史数据和本项目新整理的数据。
    """
    if not media_path:
        return ""

    media_path = str(media_path)

    if os.path.isabs(media_path):
        return os.path.normpath(media_path)

    if media_path.startswith(media_prefix):
        rel_path = media_path.replace(media_prefix, "", 1)
        return os.path.normpath(os.path.join(media_root, rel_path))

    if fallback_root is None:
        fallback_root = media_root

    return os.path.normpath(os.path.join(fallback_root, media_path))


def resolve_image_path(image_path, fallback_root=None):
    """解析图像路径；默认根目录为 benchmark/data/image。"""
    return resolve_media_path(
        image_path,
        media_root=IMAGE_DIR,
        media_prefix="../image/",
        fallback_root=fallback_root or IMAGE_DIR,
    )


def resolve_voice_path(voice_path, fallback_root=None):
    """解析音频路径；默认根目录为 benchmark/data/voice。"""
    return resolve_media_path(
        voice_path,
        media_root=VOICE_DIR,
        media_prefix="../voice/",
        fallback_root=fallback_root or VOICE_DIR,
    )


def encode_image(image_path):
    """
    将图像文件读取为 base64 字符串，供 OpenAI-compatible VLM API 使用。

    注意：这里仅负责图像编码，不会压缩或转换图像格式。
    如果图像不存在或读取失败，会抛出异常，调用方决定是否跳过。
    """
    final_path = resolve_image_path(image_path, fallback_root=IMAGE_DIR)

    try:
        if not os.path.exists(final_path):
            raise FileNotFoundError(f"Image file not found: {final_path} (original path: {image_path})")
        with open(final_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
            if not encoded:
                raise ValueError(f"Encoded image is empty for {final_path}")
            return encoded
    except FileNotFoundError as e:
        print(f"Error: {redact_runtime_text(e)}")
        raise
    except PermissionError as e:
        print(f"Error: Permission denied when reading image {redact_runtime_text(final_path)}: {redact_runtime_text(e)}")
        raise
    except Exception as e:
        print(f"Error encoding image {redact_runtime_text(image_path)} (tried {redact_runtime_text(final_path)}): {redact_runtime_text(e)}")
        raise


class VLMAgent():
    """
    对 OpenAI-compatible Chat Completions API 的轻量封装。

    该类只负责"最终回答模型"的调用：
    - self.run(): 统一处理 API 请求、重试和响应解析；
    - fast_run_with_mm_memory(): 将召回到的多模态 memory 拼成 VLM 消息；
    - fast_run_with_textual_memory(): 将召回到的文本 memory 拼成纯文本/可选图像消息。

    这里的 multimodal 主要指"发送图像给 VLM"。当前音频不会直接作为二进制
    发给回答模型，而是通过 voice_caption 写入 memory 文本。
    """
    def __init__(self, model_name=None, seed=None, api_key=None, base_url=None):
        # Credentials and endpoint are supplied only at runtime.
        self.client = openai_client(api_key=api_key, base_url=base_url)
        self.model_name = model_name or OPENAI_MODEL
        self.seed = seed
        # Check if it's a Gemini series model
        self.is_gemini = 'gemini' in self.model_name.lower()

    def parse_response(self, response):
        """从 Chat Completions 响应中抽取 message.content，并做结构校验。"""
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError("API response has no choices")
        if not hasattr(response.choices[0], 'message'):
            raise ValueError("API response choice has no message")
        if not hasattr(response.choices[0].message, 'content'):
            raise ValueError("API response message has no content")
        return {'result': response.choices[0].message.content}

    def run(self, message_list):
        """
        调用模型 API，并在临时失败时指数退避重试。

        Args:
            message_list: OpenAI Chat Completions 格式 messages。

        Returns:
            {"result": "..."}，保持和原脚本调用方的返回格式一致。
        """
        # 构造 API 参数。temperature=0 用于稳定评测输出。
        api_params = {
            'model': self.model_name,
            'messages': message_list,
            'temperature': 0.0,  # Ensure deterministic generation
            'max_tokens': 1024,  # 1024 足够输出推理+选项；2048 在 detailed caption 下易超上限
        }

        # 只有部分本地/兼容 Qwen API 支持 seed；其他 API 传 seed 可能报错。
        if self.seed is not None and "qwen" in self.model_name.lower():
            api_params['seed'] = self.seed
        # Qwen3 系列（含 Omni 和非 Omni）：通过 chat_template_kwargs 关闭思考模式
        if "qwen3" in self.model_name.lower():
            api_params['extra_body'] = {"chat_template_kwargs": {"enable_thinking": False}}

        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**api_params)
                parsed_response = self.parse_response(response)
                return parsed_response
            except Exception as e:
                err_str = str(e)
                # 输入超长（400 input_tokens）：减半 max_tokens 后立即重试，不计入重试次数
                if '400' in err_str and 'input_tokens' in err_str:
                    cur = api_params.get('max_tokens', 1024)
                    if cur > 128:
                        api_params['max_tokens'] = max(128, cur // 2)
                        print(f"[context too long] max_tokens {cur} -> {api_params['max_tokens']}, retrying...")
                        continue
                if attempt < max_retries - 1:
                    print(
                        f"API call failed (attempt {attempt + 1}/{max_retries}): "
                        f"{redact_runtime_text(e)}. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise safe_runtime_error(
                        f"API call failed after {max_retries} attempts", e
                    ) from None
    
    def fast_run_with_mm_memory(self, memory_dict_list, query_prompt, query_img=None):
        """
        使用多模态 memory 生成回答。

        memory_dict_list 是 memory.recall() 返回的候选记忆列表，每条通常包含：
        - text: 该轮对话文本；
        - image: 可选，含 path/caption/img_id；
        - voice: 可选，当前只作为元数据保存在 memory 中，不直接编码给 VLM。

        对于 image memory，会把文本上下文和 image_id 一起加入消息，再把图像
        base64 后作为 image_url 发送给支持视觉输入的模型。
        """
        conversation_info_flow = []
        print("memory_dict_list: ", memory_dict_list)

        # 无召回结果时，模型只能根据问题本身作答；如果问题附带图像，则一并发送。
        if not memory_dict_list or len(memory_dict_list) == 0:
            if query_img:
                query_img_path = query_img.get('path') if isinstance(query_img, dict) else None
                user_content = [
                    {"type": "text", "text": query_prompt},
                ]
                if query_img_path:
                    try:
                        user_content.append({"type": "text", "text": DialogueAgentPromptImage})
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(query_img_path)}"}})
                    except Exception as e:
                        print(f"Warning: Failed to encode query image at {redact_runtime_text(query_img_path)}: {redact_runtime_text(e)}. Continuing without image.")
                else:
                    # If entered if query_img branch but no path, this is a data anomaly
                    raise ValueError(f"query_img is provided but path is missing. query_img: {query_img}")
                
                response = self.run([
                    {"role": "system", "content": SystemPrompt},
                    {"role": "user", "content": user_content}
                ])
            else:
                response = self.run([
                    {"role": "system", "content": SystemPrompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": query_prompt}
                    ]}
                ])
            return response.get('result', '')

        for memory_dict in memory_dict_list:
            if memory_dict.get('image'):
                # 多模态 memory：先加入文字说明，再追加图像本体。
                textual_content = format_prompt(
                    MMMemoryDialoguePrompt,
                    {
                        'textual_context': memory_dict.get('text', ''),
                        'image_id': memory_dict.get('image', {}).get('img_id', ''),
                    },
                    input_variables=['textual_context', 'image_id'],
                )
                conversation_info_flow.append({"type": "text", "text": textual_content})
                # Safely get image path
                image_path = memory_dict.get('image', {}).get('path')
                if image_path:
                    try:
                        conversation_info_flow.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}})
                    except Exception as e:
                        raise ValueError(f"Warning: Failed to encode image at {redact_runtime_text(image_path)}: {redact_runtime_text(e)}. Skipping image.")
                else:
                    # If entered if memory_dict.get('image') branch but no path, this is a data anomaly
                    raise ValueError(f"Image data found in memory_dict but path is missing. memory_dict: {memory_dict}")
            else:
                # 纯文本 memory：直接把召回文本加入上下文。
                conversation_info_flow.append({"type": "text", "text": memory_dict.get('text')})
        

        if query_img:
            query_img_path = query_img.get('path') if isinstance(query_img, dict) else None
            user_content = [
                {"type": "text", "text": MsgStartPromptWOMemory},
                *conversation_info_flow,
                {"type": "text", "text": query_prompt},
            ]
            if query_img_path:
                try:
                    user_content.append({"type": "text", "text": DialogueAgentPromptImage})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(query_img_path)}"}})
                except Exception as e:
                    raise ValueError(f"Warning: Failed to encode query image at {redact_runtime_text(query_img_path)}: {redact_runtime_text(e)}. Continuing without image.")
            else:
                # If entered if query_img branch but no path, this is a data anomaly
                raise ValueError(f"query_img is provided but path is missing. query_img: {query_img}")
            
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": user_content}
            ])
        else:
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": [
                    {"type": "text", "text": MsgStartPromptWOMemory},
                    *conversation_info_flow,
                    {"type": "text", "text": query_prompt}
                ]}
            ])
        return response.get('result', '')

    def fast_run_with_textual_memory(self, text_memory_prompt, query_prompt, query_img=None):
        """
        使用文本化 memory 生成回答。

        FUMemory/STMemory/LTMemory 等非多模态 memory 最终都会把召回结果转成
        text_memory_prompt。若问题本身附带图像，仍可以将该问题图像作为额外输入。
        """
        if query_img:
            query_img_path = query_img.get('path') if isinstance(query_img, dict) else None
            user_content = [
                {"type": "text", "text": text_memory_prompt},
                {"type": "text", "text": query_prompt},
            ]
            if query_img_path:
                try:
                    user_content.append({"type": "text", "text": DialogueAgentPromptImage})
                    user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(query_img_path)}"}})
                except Exception as e:
                    print(f"Warning: Failed to encode query image at {redact_runtime_text(query_img_path)}: {redact_runtime_text(e)}. Continuing without image.")
            else:
                # If entered if query_img branch but no path, this is a data anomaly
                raise ValueError(f"query_img is provided but path is missing. query_img: {query_img}")
            
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": user_content}
            ])
        else:
            response = self.run([
                {"role": "system", "content": SystemPrompt},
                {"role": "user", "content": [
                    {"type": "text", "text": text_memory_prompt},
                    {"type": "text", "text": query_prompt}
                ]}
            ])
        return response.get('result', '')


# ----- Dialogue Agent -----
class DialogueAgent():
    """
    将"Memory 模块"和"回答模型 VLMAgent"串起来的评测代理。

    该类提供三个核心动作：
    1. memory_store(): 把历史对话写入 memory；
    2. memory_recall(): 根据当前问题召回相关历史记忆；
    3. response(): 将召回结果和问题拼接成 prompt，调用 VLMAgent 生成答案。
    """
    def __init__(self, memory_name, DialogueAgentMemoryConfig, model_name=None, seed=None):
        self.vlm = VLMAgent(
            model_name=model_name,
            seed=seed,
            api_key=OPENAI_APIKEY,
            base_url=OPENAI_APIBASE,
        )
        self.memory_name = memory_name
        # memory_name 是命令行传入的字符串，例如 FUMemory / MMMemory。
        # Resolve only the allow-listed class exported by memengine.
        from memengine import __all__ as memory_exports

        if memory_name not in memory_exports:
            raise ValueError(f"Unsupported memory name: {memory_name}")
        import memengine

        memory_class = getattr(memengine, memory_name)
        self.memory = memory_class(MemoryConfig(DialogueAgentMemoryConfig))
        
        # Get is_multimodal attribute from config, provide default value for backward compatibility
        self.is_multimodal = DialogueAgentMemoryConfig.get('is_multimodal', False)
    
    def reset(self):
        """清空 memory 状态，确保每个数据集从空记忆开始写入。"""
        self.memory.reset()
    
    def memory_store(self, message_dict):
        """
        将一轮历史对话写入 memory。

        对于多模态 memory：
          直接保存结构化 dict，保留 image/voice 附件字段。

        对于文本 memory：
          把图像 caption、音频 caption 拼接到 text 中，使纯文本检索器也能
          使用视觉/音频信息。
        """
        if self.is_multimodal:
            # Multimodal memory: store original multimodal fields.
            # The current VLM path supports images. Voice metadata is preserved in storage
            # but audio files are not directly encoded by VLMAgent.
            self.memory.store(message_dict)
        else:
            # Textual memory: use generated captions instead of raw media.
            # For voice-message turns, process_conversation already puts
            # user_voice_message_caption into text, so avoid duplicating the same voice caption.
            text_str = message_dict.get('text', '') if isinstance(message_dict, dict) else message_dict

            if isinstance(message_dict, dict):
                image_info = message_dict.get('image')
                if image_info:
                    img_id = image_info.get('img_id', '') if isinstance(image_info, dict) else ''
                    img_caption = image_info.get('caption', '') if isinstance(image_info, dict) else ''
                    if img_id or img_caption:
                        text_str += (
                            '\nimage:'
                            + '\nimage_id: ' + str(img_id)
                            + '\nimage_caption: ' + str(img_caption)
                        )

                voice_info = message_dict.get('voice')
                if voice_info:
                    voice_id = voice_info.get('voice_id', '') if isinstance(voice_info, dict) else ''
                    voice_caption = voice_info.get('caption', '') if isinstance(voice_info, dict) else ''
                    # Avoid duplicating user_voice_message_caption when it is identical to voice_caption.
                    if (voice_id or voice_caption) and (not voice_caption or voice_caption not in text_str):
                        text_str += (
                            '\nvoice:'
                            + '\nvoice_id: ' + str(voice_id)
                            + '\nvoice_caption: ' + str(voice_caption)
                        )

                message_dict['text'] = text_str

            self.memory.store(message_dict)

    def memory_recall(self, observation, observation_image=None, observation_voice=None):
        """
        根据当前 QA 问题召回相关 memory。

        observation 是题目文本；observation_image / observation_voice 是问题本身
        可能附带的媒体。当前四类 MCQ 通常只有文本问题，但保留该接口兼容原
        Mem-Gallery 的图像/语音问题格式。
        """
        if self.is_multimodal:
            observation_dict = {
                'text': observation,
                'image': observation_image
            }
            # Preserve voice metadata for future multimodal retrievers, but current encoders
            # may ignore it. The image path remains the only media encoded by VLMAgent.
            if observation_voice:
                observation_dict['voice'] = observation_voice
            return self.memory.recall(observation_dict)
        else:
            # For non-multimodal memory, add media captions to textual query.
            if observation_image:
                caption = observation_image.get('caption')
                if caption:
                    observation += '\nquestion\'s image:' + '\nimage_caption: ' + caption
            if observation_voice:
                caption = observation_voice.get('caption')
                if caption:
                    observation += '\nquestion\'s voice:' + '\nvoice_caption: ' + caption
            return self.memory.recall(observation)

    def response(self, memory_result, observation, speaker_a, speaker_b, observation_image=None, format_constraint=None):
        """
        根据召回 memory 和问题生成最终答案。

        format_constraint 来自不同 category 的 prompt 文件，例如：
        - preference_prompt.txt
        - entity_prompt.txt
        - recommendation_prompt.txt
        - refusal_prompt.txt

        这些 prompt 会追加到问题后，约束模型只输出 A/B/C/D。
        """
        # Handle different types of memory return
        memory_context = memory_result

        # Build format constraint string (if any)
        format_constraint_str = ""
        if format_constraint:
            format_constraint_str = "\n\n" + format_constraint
        
        query_prompt = format_prompt(
            DialogueAgentPrompt,
            {
                'observation': observation,
                'speaker_a': speaker_a,
                'speaker_b': speaker_b,
                'format_constraint': format_constraint_str,
            },
            input_variables=['observation', 'speaker_a', 'speaker_b', 'format_constraint'],
        )

        if self.is_multimodal:
            res = self.vlm.fast_run_with_mm_memory(memory_context, query_prompt, observation_image)
        else:
            # Other caption-based memories use textual memory
            text_memory_prompt = format_prompt(
                TextMsgPrompt,
                {'memory_context': memory_context},
                input_variables=['memory_context'],
            )
            res = self.vlm.fast_run_with_textual_memory(text_memory_prompt, query_prompt, observation_image)
        return res


def _infer_input_modality(has_user_voice, has_user_text, has_image, has_voice):
    """根据当前轮次包含的字段推断输入模态，用于调试和结果分析。"""
    if has_user_voice and has_image:
        return "voice_image"
    if has_user_voice:
        return "voice"
    if has_user_text and has_image:
        return "text_image"
    if has_user_text:
        return "text"
    if has_image:
        return "image"
    if has_voice:
        return "voice"
    return "unknown"


def _build_image_object(dialog, data_dir):
    """
    从一轮对话中抽取图像信息。

    输入字段约定：
      input_image   : 图像路径列表；
      image_caption : 图像 caption 列表；
      image_id      : 图像 ID 列表，例如 D00-001。

    返回：
      first_image       : 第一张图像，兼容旧 memory 接口；
      image_attachments : 全部图像附件，保留多图信息用于扩展。
    """
    img_list = _ensure_list(dialog.get("input_image", []))
    img_caption_list = _ensure_list(dialog.get("image_caption", []))
    img_id_list = _ensure_list(dialog.get("image_id", []))

    image_objects = []
    for idx, img_path in enumerate(img_list):
        if not img_path:
            continue
        try:
            image_obj = {
                "path": resolve_image_path(img_path, fallback_root=data_dir),
                "caption": _safe_get_list_item(img_caption_list, idx, ""),
                "img_id": _safe_get_list_item(img_id_list, idx, ""),
            }
            image_objects.append(image_obj)
        except Exception as e:
            print(f"Warning: Failed to process image {redact_runtime_text(img_path)} in dialog {dialog.get('round', 'unknown')}: {redact_runtime_text(e)}. Skipping image.")

    first_image = image_objects[0] if image_objects else None
    image_attachments = [
        {
            "type": "image",
            "id": img.get("img_id", ""),
            "path": img.get("path", ""),
            "caption": img.get("caption", ""),
        }
        for img in image_objects
    ]
    return first_image, image_attachments


def _build_voice_object(dialog, data_dir):
    """
    从一轮对话中抽取音频信息。

    输入字段约定：
      input_voice_message : 音频路径列表；
      voice_caption       : 音频 caption / 转写文本列表；
      voice_id            : 音频 ID 列表，例如 D00-001。

    当前回答模型不会直接接收音频文件，voice_caption 会在文本 memory 中
    拼入 text，或作为 voice 元数据保留在多模态 memory 中。
    """
    voice_list = _ensure_list(dialog.get("input_voice_message", []))
    voice_caption_list = _ensure_list(dialog.get("voice_caption", []))
    voice_id_list = _ensure_list(dialog.get("voice_id", []))

    voice_objects = []
    for idx, voice_path in enumerate(voice_list):
        if not voice_path:
            continue
        try:
            voice_obj = {
                "path": resolve_voice_path(voice_path, fallback_root=data_dir),
                "caption": _safe_get_list_item(voice_caption_list, idx, ""),
                "voice_id": _safe_get_list_item(voice_id_list, idx, ""),
            }
            voice_objects.append(voice_obj)
        except Exception as e:
            print(f"Warning: Failed to process voice {redact_runtime_text(voice_path)} in dialog {dialog.get('round', 'unknown')}: {redact_runtime_text(e)}. Skipping voice.")

    first_voice = voice_objects[0] if voice_objects else None
    voice_attachments = [
        {
            "type": "voice",
            "id": voice.get("voice_id", ""),
            "path": voice.get("path", ""),
            "caption": voice.get("caption", ""),
        }
        for voice in voice_objects
    ]
    return first_voice, voice_attachments


def process_conversation(conversation_data, data_dir=None, character_profile=None):
    """
    Process conversation data into memory-system format.

    Supported input schemas:
    1. Existing text/image schema:
       {
         "dialogues": [
           {"round": "D00:00", "user": "...", "assistant": "...",
            "input_image": [...], "image_caption": [...], "image_id": [...]}
         ]
       }

    2. New voice-message schema:
       {
         "dialogues": [
           {"round": "D12:00",
            "user_voice_message_caption": "...",
            "assistant": "...",
            "input_voice_message": [...], "voice_caption": [...], "voice_id": [...]}
         ]
       }

    Output item format:
       {
         "text": "user_voice_message_caption: ...\nassistant: ..." or "user (...): ...\nassistant: ...",
         "image": {"path": ..., "caption": ..., "img_id": ...} or None,
         "voice": {"path": ..., "caption": ..., "voice_id": ...} or None,
         "attachments": [{"type": "image"/"voice", "id": ..., "path": ..., "caption": ...}, ...],
         "timestamp": ...,
         "dialogue_id": ...,
         "session_id": ...,
         "input_modality": ...
       }
    """
    if data_dir is None:
        data_dir = DATA_DIR

    processed = []

    # Dynamically set speaker_a based on character_profile.
    if character_profile and character_profile.get("name"):
        speaker_a = f"user ({character_profile.get('name')})"
    else:
        speaker_a = "user"
    speaker_b = "assistant"

    # conversation_data 是 session 列表。每个 session 含 session_id/date/dialogues。
    for session_idx, session_data in enumerate(conversation_data):
        if not isinstance(session_data, dict):
            continue

        session_id = session_data.get("session_id", "")
        session_date = session_data.get("date", "")
        # 新整理的数据使用 "dialogues"，部分旧数据可能使用 "dialog_list"。
        dialogues = session_data.get("dialogues", session_data.get("dialog_list", []))

        for dialog in dialogues:
            if not isinstance(dialog, dict):
                continue

            user_text = dialog.get("user", "")
            user_voice_caption = dialog.get("user_voice_message_caption", "")
            assistant_text = dialog.get("assistant", "")

            image_obj, image_attachments = _build_image_object(dialog, data_dir=data_dir)
            voice_obj, voice_attachments = _build_voice_object(dialog, data_dir=data_dir)
            attachments = image_attachments + voice_attachments

            has_user_voice = bool(user_voice_caption)
            has_user_text = bool(user_text)
            has_image = image_obj is not None
            has_voice = voice_obj is not None

            # 如果当前轮既没有文本也没有媒体，则不写入 memory。
            if not (has_user_voice or has_user_text or assistant_text or has_image or has_voice):
                continue

            text_parts = []
            # 核心规则：如果存在用户语音 caption，则优先使用语音 caption。
            # 这样 voice-message turn 会以"听到的内容"为用户输入，而不是空文本。
            if user_voice_caption:
                text_parts.append(f"user_voice_message_caption: {user_voice_caption}")
            elif user_text:
                text_parts.append(f"{speaker_a}: {user_text}")

            if assistant_text:
                text_parts.append(f"{speaker_b}: {assistant_text}")

            combined_text = "\n".join(text_parts)

            processed.append({
                # text 是 memory 存储和检索的主要文本字段。
                "text": combined_text,
                # image/voice 是首个媒体对象，兼容原 memory 接口。
                "image": image_obj,
                "voice": voice_obj,
                # attachments 保留全部媒体对象，方便后续扩展多图/多音频检索。
                "attachments": attachments,
                "timestamp": session_date,
                "dialogue_id": dialog.get("round", ""),
                "session_id": session_id,
                "input_modality": _infer_input_modality(
                    has_user_voice=has_user_voice,
                    has_user_text=has_user_text,
                    has_image=has_image,
                    has_voice=has_voice,
                ),
            })

    return processed

def get_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def sample_qa_pairs_by_category(qa_pairs, sample_per_category):
    """
    对 QA 做小样本抽样：每个 category 最多取 N 条，并保持原始顺序。

    这里的 category 来自 qa["point"]，例如：
      entity / preference / recommendation / refusal

    设计成"每类 N 条"而不是"总共 N 条"，是为了混合数据集快速冒烟测试时
    能同时覆盖四种 QA 类型。
    """
    if sample_per_category is None or sample_per_category <= 0:
        return qa_pairs

    sampled = []
    counts = {}
    for qa in qa_pairs:
        category = qa.get("point", "") or "unknown"
        current = counts.get(category, 0)
        if current >= sample_per_category:
            continue
        sampled.append(qa)
        counts[category] = current + 1

    print(f"QA sample mode: at most {sample_per_category} item(s) per category")
    print(f"  original QA count: {len(qa_pairs)}")
    print(f"  sampled QA count : {len(sampled)}")
    for category in sorted(counts.keys()):
        print(f"  {category}: {counts[category]}")
    return sampled


def _extract_choice(answer: str) -> str:
    """从模型输出中提取 A/B/C/D 选项字母。"""
    if not answer:
        return ""
    s = answer.strip()
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = re.search(r'\b([A-D])\b', s.upper())
    return m.group(1) if m else s.upper()


def compute_accuracy_summary(results: list) -> dict:
    """
    按 category（point 字段）和 explicit/implicit 统计正确率。

    entity 的 qa_type 映射规则（见 build_bench_input.py 注释）：
      Relationship / Pets → explicit
      Items               → 读取 entity_explicitness 字段
    preference / refusal / recommendation 的 qa_type 本身就是 explicit / implicit。
    """
    from collections import defaultdict

    ENTITY_EXPLICIT_TYPES = {"Relationship", "Pets"}

    def get_explicitness(item: dict) -> str:
        cat = (item.get("category") or "").lower()
        qa_type = item.get("qa_type", "")
        if cat in ("entity", "entity_img", "entity_text"):
            if qa_type in ENTITY_EXPLICIT_TYPES:
                return "explicit"
            if qa_type == "Items":
                return item.get("entity_explicitness") or "unknown"
            return "unknown"
        if qa_type in ("explicit", "implicit"):
            return qa_type
        return "unknown"

    # stats[category][split] = [correct, total]
    stats: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))

    for item in results:
        cat      = (item.get("category") or "").lower()
        pred     = _extract_choice(item.get("system_answer", ""))
        gold     = _extract_choice(item.get("original_answer", ""))
        correct  = 1 if pred == gold else 0

        explicitness = get_explicitness(item)

        stats[cat]["overall"][0] += correct
        stats[cat]["overall"][1] += 1
        stats[cat][explicitness][0] += correct
        stats[cat][explicitness][1] += 1

    all_correct = sum(v["overall"][0] for v in stats.values())
    all_total   = sum(v["overall"][1] for v in stats.values())

    def make_entry(correct, total):
        return {"correct": correct, "total": total,
                "accuracy": round(correct / total * 100, 2) if total > 0 else 0.0}

    summary = {}
    for cat in sorted(stats.keys()):
        cat_entry = {}
        for split in ["overall", "explicit", "implicit", "mixed", "unknown"]:
            c, t = stats[cat][split]
            if t > 0:
                cat_entry[split] = make_entry(c, t)
        summary[cat] = cat_entry

    summary["__overall__"] = {"overall": make_entry(all_correct, all_total)}
    return summary


def print_accuracy_summary(summary: dict) -> None:
    """将正确率摘要以对齐表格形式打印到控制台。"""
    W = 72
    print(f"\n{'='*W}")
    print("Accuracy Summary")
    print(f"{'='*W}")
    print(f"{'Category':<20} {'Split':<12} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*W}")
    for cat in sorted(k for k in summary if k != "__overall__"):
        first = True
        for split in ["overall", "explicit", "implicit", "mixed", "unknown"]:
            if split not in summary[cat]:
                continue
            d     = summary[cat][split]
            label = cat if first else ""
            first = False
            print(f"{label:<20} {split:<12} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>9.2f}%")
        print(f"{'-'*W}")
    if "__overall__" in summary:
        d = summary["__overall__"]["overall"]
        print(f"{'TOTAL':<20} {'overall':<12} {d['correct']:>8} {d['total']:>8} {d['accuracy']:>9.2f}%")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# 增量保存 / 断点续跑 工具函数
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 10


def _atomic_save_json(data, path):
    """原子写入 JSON：先写临时文件再 rename，避免写到一半断电导致文件损坏。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _qa_resume_key_from_result(record: dict) -> tuple[str, str]:
    """Return the stable resume key for a saved result record."""
    return (
        str(record.get("qa_id", "")),
        str(record.get("category") or record.get("point") or ""),
    )


def _qa_resume_key_from_input(qa: dict, fallback_idx: int) -> tuple[str, str]:
    """Return the stable resume key for an input QA item."""
    return (
        str(qa.get("qa_id", f"qa_{fallback_idx}")),
        str(qa.get("point") or qa.get("category") or ""),
    )


def _load_completed_results(path):
    """加载已完成的结果用于断点续跑。返回 {(qa_id, category): result_dict}。

    只有 system_answer 非空的记录才视为"已完成"；
    API 调用失败导致 system_answer 为空的条目会被重新评测。
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        completed = {}
        skipped_empty = 0
        for r in records:
            if "qa_id" not in r:
                continue
            if not r.get("system_answer", "").strip():
                skipped_empty += 1
                continue
            key = _qa_resume_key_from_result(r)
            if not key[0] or not key[1]:
                print(
                    "[resume] Warning: result record missing qa_id/category; "
                    f"will not use it for resume matching: {r.get('qa_id')!r}"
                )
                continue
            completed[key] = r
        if skipped_empty:
            print(f"[resume] {skipped_empty} items have empty system_answer, will re-evaluate them.")
        return completed
    except Exception as e:
        print(f"WARN: failed to load checkpoint {path}: {redact_runtime_text(e)}")
        return {}


def run_mm_bench(llm_name, memory_name, DialogueAgentMemoryConfig, data_name, save_results, save_efficiency, eval_retrieval_metrics=False, eval_topk=5, model_name=None, seed=None, sample=None, save_memory_context=False, max_workers=8, qa_category=None, image_option_mode="caption", with_reasoning=True):
    """
    单个数据集的完整评测流程。

    Args:
        llm_name: 命令行选择的模型别名，用于结果目录命名。
        memory_name: 使用的 memory 方法名，例如 FUMemory / MMMemory。
        DialogueAgentMemoryConfig: 对应 memory 的配置 dict。
        data_name: data/dialog/{data_name}.json 的文件名（不含后缀）。
        save_results: 是否保存每条 QA 的模型输出。
        save_efficiency: 是否保存运行耗时统计。
        eval_retrieval_metrics: 是否额外保存检索到的 memory id 和 gold clue。
        eval_topk: 检索指标 top-k 参数（当前函数只透传/记录）。
        model_name: 实际传给 OpenAI-compatible API 的模型名。
        seed: 随机种子。
        sample: 每个 category 抽样数量；None 表示全量评测。
        save_memory_context: 是否把每条 QA 检索到的记忆内容写入结果文件。
    """
    # data_name_underscore 用于输出文件命名；保留原变量名以兼容旧代码。
    data_name_underscore = data_name
    
    # 判断 memory 是否支持多模态。文本 memory 会使用 caption；多模态 memory 可保留图像对象。
    is_multimodal = DialogueAgentMemoryConfig.get('is_multimodal', False)

    sample_suffix = f"_sample{sample}" if sample is not None and sample > 0 else ""
    category_suffix = f"_{qa_category}" if qa_category else ""
    output_file = os.path.join(
        RESULT_DIR,
        llm_name,
        memory_name,
        f"{data_name_underscore}{sample_suffix}{category_suffix}_results.json"
    )

    # Extract directory path
    output_dir = os.path.dirname(output_file)

    # Create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载 benchmark/data/dialog/{data_name}.json。
    json_filename = f"{data_name}.json"
    json_path = os.path.join(DIALOG_DIR, json_filename)
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        print(f"Loaded {json_filename} dataset from {json_path}")
    except FileNotFoundError:
        print(f"Can't find {json_path}")
        return
    except Exception as e:
        print(f"Error loading dataset: {redact_runtime_text(e)}")
        return
    
    # 初始化 DialogueAgent：内部包含 memory 模块和最终回答模型。
    memory_agent = DialogueAgent(memory_name, DialogueAgentMemoryConfig, model_name=model_name, seed=seed)
    memory_agent.reset()

    # Get sample_id from character_profile
    character_profile = dataset.get("character_profile", {})
    sample_id = character_profile.get("name", data_name)

    # run_bench.py 约定的数据集字段：
    # - multi_session_dialogues: 历史对话；
    # - human-annotated QAs: 待评测 QA。
    conversation_data = dataset.get("multi_session_dialogues", [])
    qa_pairs = dataset.get("human-annotated QAs", [])

    if qa_category:
        original_count = len(qa_pairs)
        qa_cat_lower = qa_category.lower()
        qa_pairs = [qa for qa in qa_pairs if (qa.get("point", "") or "").lower().startswith(qa_cat_lower)]
        print(f"QA category filter: '{qa_category}' -> {len(qa_pairs)}/{original_count} items kept")
        if not qa_pairs:
            print(f"No QA items match category '{qa_category}', skip")
            return

    qa_pairs = sample_qa_pairs_by_category(qa_pairs, sample)

    # Process conversation data
    processed_dialogs = process_conversation(conversation_data, data_dir=DATA_DIR, character_profile=character_profile)

    if not processed_dialogs:
        print(f"No valid conversation data in {sample_id}, skip")
        return

    # Dynamically set speaker_a based on character_profile
    if character_profile and character_profile.get("name"):
        speaker_a = f"user ({character_profile.get('name')})"
    else:
        speaker_a = "user"
    speaker_b = "assistant"

    # ── 断点续跑：加载已完成的结果 ──
    completed_map = _load_completed_results(output_file) if save_results else {}
    if completed_map:
        print(f"[resume] Found {len(completed_map)} completed QA items, will skip them.")

    # 如果所有 QA 都已完成，直接跳过整个数据集
    all_qa_keys = {_qa_resume_key_from_input(qa, i) for i, qa in enumerate(qa_pairs)}
    if save_results and all_qa_keys and all_qa_keys.issubset(completed_map.keys()):
        print(f"[skip] All {len(all_qa_keys)} QA items already completed for {data_name}. Skipping entirely.")
        return

    # 第一步：把历史对话逐条写入 memory。慢速外部 memory 按数据集+配置缓存。
    memory_start = time.time()
    memory_cache_path = None
    memory_cache_loaded = False
    if memory_name == "AMemMemory":
        dialog_config_label = os.path.relpath(DIALOG_DIR, DATA_DIR).replace(os.sep, "__")
        memory_cache_path = _amem_cache_path(
            data_name,
            processed_dialogs,
            DialogueAgentMemoryConfig,
            model_name or llm_name,
            dialog_config_label,
        )
        memory_cache_loaded = _try_load_amem_cache(memory_agent, memory_cache_path, len(processed_dialogs))
    elif memory_name == "MemoryOSMemory":
        dialog_config_label = os.path.relpath(DIALOG_DIR, DATA_DIR).replace(os.sep, "__")
        memory_cache_path = _memoryos_cache_path(
            data_name,
            processed_dialogs,
            DialogueAgentMemoryConfig,
            model_name or llm_name,
            dialog_config_label,
        )
        memory_cache_loaded = _try_load_memoryos_cache(memory_agent, memory_cache_path, len(processed_dialogs))

    memory_store_failures = 0
    if not memory_cache_loaded:
        for dialog in tqdm(processed_dialogs, desc="Processing dialogs", total=len(processed_dialogs)):
            try:
                memory_agent.memory_store(dialog)
            except Exception as e:
                memory_store_failures += 1
                print(f"Warning: Failed to store dialog {dialog.get('dialogue_id', 'unknown')}: {redact_runtime_text(e)}. Skipping...")
                continue
        if hasattr(memory_agent.memory, "finalize_store"):
            memory_agent.memory.finalize_store()
    memory_duration = time.time() - memory_start if processed_dialogs else 0.0

    if memory_name == "AMemMemory" and memory_cache_path and not memory_cache_loaded:
        if memory_store_failures == 0:
            _try_save_amem_cache(memory_agent, memory_cache_path, len(processed_dialogs), memory_duration)
        else:
            print(f"[AMem cache] not saved because {memory_store_failures} dialog(s) failed to store")
    elif memory_name == "MemoryOSMemory" and memory_cache_path and not memory_cache_loaded:
        if memory_store_failures == 0:
            _try_save_memoryos_cache(memory_agent, memory_cache_path, len(processed_dialogs), memory_duration)
        else:
            print(f"[MemoryOS cache] not saved because {memory_store_failures} dialog(s) failed to store")

    # ── 第二步 Phase-1：串行召回 memory（部分 memory 有状态副作用，不可并行） ──
    qa_count = len(qa_pairs)
    prepared_tasks = []
    skipped = 0

    for qa_idx, qa in tqdm(enumerate(qa_pairs), desc="Recalling memory", total=qa_count):
        qa_id = qa.get("qa_id", f"qa_{qa_idx}")
        resume_key = _qa_resume_key_from_input(qa, qa_idx)

        if resume_key in completed_map:
            skipped += 1
            continue

        question = qa.get("question", "")
        if not question:
            continue

        # ── Image-option QA: 将选项图片的描述注入问题文本 ──
        # 优先使用 option_captions（VLM 生成的中文 caption），
        # 退而使用 question_image_descriptions（Stage 1 英文 image prompt）
        option_captions = qa.get("option_captions") or qa.get("question_image_descriptions")
        if option_captions and isinstance(option_captions, dict) and any(option_captions.get(l) for l in "ABCD"):
            if image_option_mode == "caption":
                desc_lines = []
                for letter in "ABCD":
                    desc = option_captions.get(letter, "")
                    if desc:
                        desc_lines.append(f"{letter}. {desc}")
                if desc_lines:
                    insert_text = "\n".join(desc_lines)
                    suffix = "请在 A/B/C/D 中选择最符合的选项。"
                    q_stripped = question.rstrip()
                    if q_stripped.endswith(suffix):
                        question = q_stripped[:-len(suffix)].rstrip() + "\n\n" + insert_text + "\n\n" + suffix
                    else:
                        question = q_stripped + "\n\n" + insert_text
            elif image_option_mode == "vlm":
                # TODO: VLM 模式——将 4 张选项图片直接编码发送给 VLM
                print(f"[{qa_id}] VLM image-option mode not yet implemented, falling back to caption")
                desc_lines = []
                for letter in "ABCD":
                    desc = option_captions.get(letter, "")
                    if desc:
                        desc_lines.append(f"{letter}. {desc}")
                if desc_lines:
                    insert_text = "\n".join(desc_lines)
                    suffix = "请在 A/B/C/D 中选择最符合的选项。"
                    q_stripped = question.rstrip()
                    if q_stripped.endswith(suffix):
                        question = q_stripped[:-len(suffix)].rstrip() + "\n\n" + insert_text + "\n\n" + suffix
                    else:
                        question = q_stripped + "\n\n" + insert_text

        question_image = None
        question_image_caption = None
        question_voice = None
        question_voice_caption = None

        if qa.get("question_image"):
            question_image_path = qa.get("question_image", "")
            question_image = resolve_image_path(question_image_path, fallback_root=IMAGE_DIR)
            question_image_caption = qa.get("image_caption", None)

        if qa.get("question_voice_message"):
            question_voice_path = qa.get("question_voice_message", "")
            question_voice = resolve_voice_path(question_voice_path, fallback_root=VOICE_DIR)
            question_voice_caption = qa.get("voice_caption", qa.get("question_voice_message_caption", None))

        original_answer = qa.get("answer", "")
        category = qa.get("point", "")
        qa_type = qa.get("qa_type", "")

        format_constraint = None
        if category:
            category_upper = category.upper()
            _fc_map = {
                "AR": "ar_prompt.txt", "CD": "cd_prompt.txt", "VS": "vs_prompt.txt",
                "ENTITY": "entity_prompt.txt", "PREFERENCE": "preference_prompt.txt",
                "PREFERENCE_SAME_CATEGORY": "preference_prompt.txt",
                "PREFERENCE_CROSS_CATEGORY": "preference_prompt.txt",
                "RECOMMENDATION": "recommendation_prompt.txt",
                "RECOMMENDATION_SAME_CATEGORY": "recommendation_prompt.txt",
                "RECOMMENDATION_CROSS_CATEGORY": "recommendation_prompt.txt",
                "REFUSAL": "refusal_prompt.txt",
                "OVERTHINKING": "overthinking_prompt.txt",
                # 新 point 名（build_bench_input.py 的 category 模式产出）
                "PREF_IMG": "preference_prompt.txt",
                "PREF_TEXT": "preference_prompt.txt",
                "REC_IMG": "recommendation_prompt.txt",
                "REC_TEXT": "recommendation_prompt.txt",
                "ENTITY_IMG": "entity_prompt.txt",
                "ENTITY_TEXT": "entity_prompt.txt",
                "REFUSAL_TEXT": "refusal_prompt.txt",
                "AUDIO_CONTEXT": "preference_prompt.txt",
            }
            fc_file = _fc_map.get(category_upper)
            if fc_file:
                format_constraint = load_prompt_file(fc_file)

        if with_reasoning and format_constraint:
            format_constraint = (
                format_constraint.rstrip()
                + "\n\n"
                + "## 输出格式\n"
                + "第一行输出你选择的选项字母（A、B、C 或 D）。\n"
                + "第二行输出「依据session：」后跟你做出判断所依赖的所有记忆的 session 日期（即记忆中 timestamp 字段的值），用逗号分隔。\n"
                + "第三行起输出你的判断依据：结合上述 session 日期，说明你是如何从对话文本、图片描述还是语音消息中推断出用户的偏好或习惯，从而选出该选项的。\n"
                + "示例：\n"
                + "A\n"
                + "依据session：2024-03-15, 2024-04-02\n"
                + "依据：在 2024-03-15 的对话中，用户通过语音消息提到喜欢清淡口味；在 2024-04-02 的对话中，用户分享了一张日式料理的图片并表达了喜爱，因此选择A最符合用户的饮食偏好。"
            )

        qa_session_id = qa.get("session_id", "")
        if isinstance(qa_session_id, list) and len(qa_session_id) > 0:
            qa_session_id = qa_session_id[0]

        # 召回 memory（串行，保证线程安全）
        try:
            observation_image = None
            if question_image:
                observation_image = {'path': question_image}
                if question_image_caption:
                    observation_image['caption'] = question_image_caption

            observation_voice = None
            if question_voice:
                observation_voice = {'path': question_voice}
                if question_voice_caption:
                    observation_voice['caption'] = question_voice_caption

            memory_context = memory_agent.memory_recall(question, observation_image, observation_voice)
            if memory_context is None:
                memory_context = []
                print(f"Warning: Memory context is None for question: {question[:50]}...")
        except Exception as e:
            print(f"Error retrieving memory for question: {question[:50]}... Error: {redact_runtime_text(e)}")
            memory_context = []

        retrieved_ids = []
        if eval_retrieval_metrics:
            try:
                retrieved_ids = getattr(memory_agent.memory.recall_op, 'last_retrieved_ids', []) or []
            except AttributeError:
                retrieved_ids = []
            except Exception as e:
                print(f"Warning: Failed to get retrieved_ids: {redact_runtime_text(e)}")
                retrieved_ids = []
            try:
                retrieved_ids = [str(x) for x in retrieved_ids]
                seen = set()
                retrieved_ids = [x for x in retrieved_ids if not (x in seen or seen.add(x))]
            except Exception:
                retrieved_ids = []

        clue_ids = qa.get("clue", [])
        if not isinstance(clue_ids, list):
            clue_ids = []

        response_observation_image = None
        if question_image:
            response_observation_image = {'path': question_image}
            if question_image_caption:
                response_observation_image['caption'] = question_image_caption

        prepared_tasks.append({
            "qa_id": qa_id,
            "memory_context": memory_context,
            "question": question,
            "speaker_a": speaker_a,
            "speaker_b": speaker_b,
            "obs_image": response_observation_image,
            "format_constraint": format_constraint,
            "original_answer": original_answer,
            "category": category,
            "qa_type": qa_type,
            "entity_explicitness": qa.get("entity_explicitness", ""),
            "entity_source_refs": qa.get("entity_source_refs", []),
            "entity_name": qa.get("entity_name", ""),
            "entity_anchor_lookup_name": qa.get("entity_anchor_lookup_name", ""),
            "session_id": qa_session_id,
            "retrieved_ids": retrieved_ids,
            "clue_ids": clue_ids,
            "sample_id": sample_id,
        })

    if skipped:
        print(f"[resume] Skipped {skipped} already-completed items, {len(prepared_tasks)} remaining.")

    # ── 第二步 Phase-2：并发调用 LLM 生成回答 ──
    def _split_answer_reasoning(raw: str) -> tuple[str, str, list]:
        """将 LLM 原始输出拆分为选项字母、推理依据和依据 session 日期列表。"""
        if not raw or not raw.strip():
            return raw, "", []
        lines = raw.strip().splitlines()
        choice_line = lines[0].strip()

        reasoning_sessions: list[str] = []
        reasoning_lines: list[str] = []
        for line in lines[1:]:
            stripped = line.strip()
            for prefix in ("依据session：", "依据session:", "依据 session：", "依据 session:"):
                if stripped.lower().startswith(prefix.lower()):
                    session_str = stripped[len(prefix):].strip()
                    reasoning_sessions = [s.strip() for s in re.split(r'[,，;；\s]+', session_str) if s.strip()]
                    break
            else:
                reasoning_lines.append(line)

        reasoning = "\n".join(reasoning_lines).strip()
        for prefix in ("依据：", "依据:", "判断依据：", "判断依据:"):
            if reasoning.startswith(prefix):
                reasoning = reasoning[len(prefix):].strip()
                break
        return choice_line, reasoning, reasoning_sessions

    def _respond_one(task):
        """单条 QA 的 LLM 回答（线程安全：只读 memory_context + 无状态 API 调用）。"""
        try:
            system_answer = memory_agent.response(
                task["memory_context"],
                task["question"],
                task["speaker_a"],
                task["speaker_b"],
                task["obs_image"],
                format_constraint=task["format_constraint"],
            )
            if system_answer is None:
                system_answer = ""
                print(f"Warning: System answer is None for question: {task['question'][:50]}...")
        except Exception as e:
            print(f"Error generating response for question: {task['question'][:50]}... Error: {redact_runtime_text(e)}")
            system_answer = ""

        reasoning = ""
        reasoning_sessions = []
        if with_reasoning and system_answer:
            choice_part, reasoning, reasoning_sessions = _split_answer_reasoning(system_answer)
            system_answer = choice_part

        result_item = {
            "qa_id": task["qa_id"],
            "sample_id": task["sample_id"],
            "session_id": task["session_id"],
            "speaker_a": task["speaker_a"],
            "speaker_b": task["speaker_b"],
            "question": task["question"],
            "system_answer": system_answer,
            "original_answer": task["original_answer"],
            "category": task["category"],
            "qa_type": task["qa_type"],
            "entity_explicitness": task.get("entity_explicitness", ""),
            "entity_source_refs": task.get("entity_source_refs", []),
            "entity_name": task.get("entity_name", ""),
            "entity_anchor_lookup_name": task.get("entity_anchor_lookup_name", ""),
            "timestamp": get_timestamp(),
        }
        if with_reasoning:
            result_item["reasoning"] = reasoning
            result_item["reasoning_sessions"] = reasoning_sessions
        if eval_retrieval_metrics:
            result_item["retrieved_ids"] = task["retrieved_ids"]
            result_item["clue"] = task["clue_ids"]
        if save_memory_context:
            result_item["memory_context"] = task["memory_context"]
        return result_item

    new_results = []
    lock = threading.Lock()
    qa_start = time.time()

    if prepared_tasks:
        actual_workers = min(max_workers, len(prepared_tasks))
        print(f"LLM response phase: {len(prepared_tasks)} items × {actual_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
            futures = {executor.submit(_respond_one, t): t for t in prepared_tasks}
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(prepared_tasks),
                desc=f"LLM responses ({actual_workers} workers)",
            ):
                result = future.result()
                with lock:
                    new_results.append(result)
                    if save_results and len(new_results) % CHECKPOINT_EVERY == 0:
                        all_so_far = list(completed_map.values()) + new_results
                        _atomic_save_json(all_so_far, output_file)
                        print(f"  [checkpoint] saved {len(all_so_far)} results")

    qa_duration = time.time() - qa_start if prepared_tasks else 0.0

    # ── 合并已完成 + 新完成的结果 ──
    results = list(completed_map.values()) + new_results

    # Final saving
    if save_results and results:
        try:
            _atomic_save_json(results, output_file)
            print(f"Results saved to {output_file} ({len(results)} items)")
        except Exception as e:
            print(f"Error saving result: {redact_runtime_text(e)}")

    if save_efficiency:
        efficiency_metrics = {
            "sample_id": sample_id,
            "llm_name": llm_name,
            "memory_name": memory_name,
            "is_multimodal": is_multimodal,
            "conversation_turns": len(processed_dialogs),
            "qa_count": qa_count,
            "qa_completed": len(results),
            "max_workers": max_workers,
            "memory_time_seconds": memory_duration,
            "qa_time_seconds": qa_duration,
            "total_time_seconds": memory_duration + qa_duration,
            "recorded_at": get_timestamp(),
        }
        efficiency_file = os.path.join(
            os.path.dirname(output_file),
            f"{data_name_underscore}{sample_suffix}{category_suffix}_efficiency.json",
        )
        try:
            with open(efficiency_file, "w", encoding="utf-8") as f:
                json.dump(efficiency_metrics, f, ensure_ascii=False, indent=2)
            print(f"Efficiency metrics saved to {efficiency_file}")
        except Exception as e:
            print(f"Error saving efficiency metrics: {redact_runtime_text(e)}")

    # -----------------------------------------------------------------------
    # 正确率统计：按 category 及 explicit / implicit 分别统计
    # -----------------------------------------------------------------------
    if results:
        accuracy_summary = compute_accuracy_summary(results)
        print_accuracy_summary(accuracy_summary)

        accuracy_file = os.path.join(
            os.path.dirname(output_file),
            f"{data_name_underscore}{sample_suffix}{category_suffix}_accuracy.json",
        )
        try:
            with open(accuracy_file, "w", encoding="utf-8") as f:
                json.dump(accuracy_summary, f, ensure_ascii=False, indent=2)
            print(f"Accuracy summary saved to {accuracy_file}")
        except Exception as e:
            print(f"Error saving accuracy summary: {redact_runtime_text(e)}")


if __name__ == '__main__':
    ############################ 命令行参数 ####################################
    parser = argparse.ArgumentParser()
    # llm_name 是高层别名，下面会映射到具体 OPENAI_MODEL / API base。
    parser.add_argument('--llm_name', default=['qwen2-5-vl-7b'], nargs='+',
                        choices=['qwen2-5-7b', 'qwen2-5-vl-3b', 'qwen2-5-vl-7b', 'qwen2-5-vl-32b', 'gpt-5.4-mini', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'qwen3.6-35b-a3b'])
    # memory_name 控制使用哪一种记忆模块。注意部分 memory 需要额外依赖（torch/transformers 等）。
    parser.add_argument('--memory_name', default=['MMMemory'], nargs='+',
                        choices=["FUMemory", "STMemory", "LTMemory", "GAMemory", "MGMemory", "RFMemory", "MMMemory", "MMFUMemory", "NGMemory", "AUGUSTUSMemory", "UniversalRAGMemory", "AMemMemory", "MemoryOSMemory"])
    # data_name 对应 benchmark/data/dialog/{data_name}.json。
    parser.add_argument('--data_name', default=None, nargs='+',
                        help='Dataset name(s) without .json suffix. Accepts one or more names, e.g. --data_name history_with_qa_p0 history_with_qa_p1. If not specified and --all_datasets is used, will process all datasets.')
    parser.add_argument('--all_datasets', action="store_true", help='Process all available datasets in the dialog directory')
    parser.add_argument('--save_results', action="store_true", help='Save QA performance results JSON')
    parser.add_argument('--save_efficiency', action="store_true", help='Save efficiency metrics JSON')
    parser.add_argument('--eval_retrieval_metrics', action="store_true", help='Evaluate retrieval metrics (mAP@K, Recall@K, HitRate@K, Precision@K)')
    parser.add_argument('--eval_topk', type=int, default=10, help='Top-K cutoff for retrieval metrics')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--sample', type=int, default=None, help='Sample at most N QA items per category for quick testing')
    parser.add_argument('--qa_category', type=str, nargs='+', default=None,
                        help='Only evaluate QA items of specific categories (e.g., preference entity recommendation refusal). Case-insensitive. If not set, all categories are evaluated.')
    parser.add_argument('--save_memory_context', action="store_true", help='Save retrieved memory context for each QA item into results JSON')
    parser.add_argument('--max_workers', type=int, default=8, help='Max concurrent LLM response threads (default: 4)')
    parser.add_argument('--image_option_mode', type=str, default='caption', choices=['caption', 'vlm'],
                        help='How to handle image-option QA: caption (inject text descriptions) or vlm (send images to VLM, not yet implemented)')
    parser.add_argument('--with_reasoning', action="store_true",
                        help='让 LLM 同时输出判断依据（推断出的用户偏好/习惯），保存到结果文件的 reasoning 字段')
    parser.add_argument('--caption_category', type=str, default=None, nargs='+',
                        choices=['base', 'brief', 'medium', 'detailed'],
                        help='Caption 粒度（可多个）：从 dialog/{category}/ 读取数据，结果写入 result_debug/{category}/。'
                             'base 为无图片/音频 caption 变化的基础模式')
    parser.add_argument('--audio_caption', type=str, default=None, nargs='+',
                        help='音频模型名（可多个）：从 dialog/audio_caption/{model}/ 读取数据，结果写入 result_debug/audio_caption/{model}/。'
                             '例如：--audio_caption moss_audio_8b qwen3_asr_1.7b')
    parser.add_argument('--api_key', default=None,
                        help='Runtime API key override; prefer CUE_MEM_LLM_API_KEY')
    parser.add_argument('--base_url', default=None,
                        help='Runtime API endpoint override; prefer CUE_MEM_LLM_BASE_URL')
    args = parser.parse_args()

    seed_everything(args.seed)

    # -----------------------------------------------------------------------
    # LLM 配置映射
    # -----------------------------------------------------------------------
    def _get_llm_config(llm_name):
        """Return model credentials from CLI overrides or runtime environment."""
        supported = {
            'qwen2-5-7b', 'qwen2-5-vl-3b', 'qwen2-5-vl-7b', 'qwen2-5-vl-32b',
            'gpt-5.4-mini', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
            'qwen3.6-35b-a3b',
        }
        if llm_name not in supported:
            raise ValueError(f"Unsupported LLM name: {llm_name}")

        api_key = args.api_key or os.environ.get('CUE_MEM_VLLM_API_KEY')
        api_key = api_key or os.environ.get('CUE_MEM_LLM_API_KEY')
        base_url = args.base_url or os.environ.get('CUE_MEM_VLLM_BASE_URL')
        base_url = base_url or os.environ.get('CUE_MEM_LLM_BASE_URL')
        model_name = os.environ.get('CUE_MEM_LLM_MODEL', '').strip() or llm_name
        if llm_name == 'qwen3.6-35b-a3b' and not os.environ.get('CUE_MEM_LLM_MODEL'):
            model_name = 'qwen36-35b-a3b-fp8'
        return api_key, base_url, model_name

    # -----------------------------------------------------------------------
    # 选择数据集
    # -----------------------------------------------------------------------
    if args.all_datasets:
        datasets = []  # 在每个 config 循环内按 DIALOG_DIR 扫描
    else:
        if args.data_name is None:
            print("Error: Either --data_name or --all_datasets must be specified")
            sys.exit(1)
        datasets = args.data_name

    # -----------------------------------------------------------------------
    # Memory 配置映射
    # -----------------------------------------------------------------------

    # 外部 API 模型集合：这些模型的 memory store/recall LLM 需要使用外部 API
    _EXTERNAL_API_LLMS = {
        'qwen3.6-35b-a3b',
        'gpt-5.4-mini',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
    }

    def _patch_memory_llm(config, api_key: str, base_url: str, model_name: str):
        """递归替换 memory config 中所有 APILLM 的连接信息。
        用于让 memory store/recall 阶段与评测阶段使用同一套外部 API。"""
        import copy
        config = copy.deepcopy(config)

        def _patch(obj):
            if isinstance(obj, dict):
                if obj.get('method') == 'APILLM':
                    obj['api_key'] = api_key
                    obj['base_url'] = base_url
                    obj['name'] = model_name
                for v in obj.values():
                    _patch(v)
            elif isinstance(obj, list):
                for item in obj:
                    _patch(item)

        _patch(config)
        return config

    def _get_memory_config(memory_name: str, llm_name: str = None):
        """根据 memory_name 返回对应的配置 dict。
        若 llm_name 为外部 API 模型，则同步替换 memory 内部的 APILLM 连接信息。"""
        if memory_name == 'FUMemory':
            return DEFAULT_FUMEMORY
        elif memory_name == 'STMemory':
            return DEFAULT_STMEMORY
        elif memory_name == 'LTMemory':
            return DEFAULT_LTMEMORY
        elif memory_name == 'GAMemory':
            return DEFAULT_GAMEMORY
        elif memory_name == 'MGMemory':
            return DEFAULT_MGMEMORY
        elif memory_name == 'RFMemory':
            return DEFAULT_RFMEMORY
        elif memory_name == 'MMMemory':
            return DEFAULT_MMMEMORY
        elif memory_name == 'MMFUMemory':
            import copy
            cfg = copy.deepcopy(DEFAULT_MMFUMEMORY)
            if llm_name in ['qwen2-5-7b', 'qwen2-5-vl-3b', 'qwen2-5-vl-7b', 'qwen2-5-vl-32b']:
                cfg['recall']['truncation']['tokens_per_image'] = 256
            elif llm_name in ['gpt-5.4-mini', 'gemini-2.5-flash', 'gemini-2.5-flash-lite']:
                cfg['recall']['truncation']['tokens_per_image'] = 576
            print(f"MMFUMemory configured with tokens_per_image={cfg['recall']['truncation']['tokens_per_image']} for model {llm_name}")
            return cfg
        elif memory_name == 'NGMemory':
            return DEFAULT_NGMEMORY
        elif memory_name == 'AUGUSTUSMemory':
            return DEFAULT_AUGUSTUSMEMORY
        elif memory_name == 'UniversalRAGMemory':
            print("Using UniversalRAGMemory with dynamic routing (no, document, image)")
            return DEFAULT_UNIVERSALRAGMEMORY
        elif memory_name == 'AMemMemory':
            return DEFAULT_AMEMORY
        elif memory_name == 'MemoryOSMemory':
            return DEFAULT_MEMORYOSMEMORY
        else:
            raise ValueError(f"Unsupported memory name: {memory_name}")

    def _get_memory_config_patched(memory_name: str, llm_name: str = None):
        """获取 memory config，并在需要时将内部 APILLM 替换为与评测 LLM 相同的外部 API。"""
        cfg = _get_memory_config(memory_name, llm_name)
        if llm_name in _EXTERNAL_API_LLMS:
            api_key, base_url, model_name = _get_llm_config(llm_name)
            cfg = _patch_memory_llm(cfg, api_key, base_url, model_name)
            if memory_name == 'AMemMemory':
                cfg['llm_backend'] = 'openai'
                cfg['llm_model'] = model_name
                cfg['api_key'] = api_key
                cfg['api_base'] = base_url
            elif memory_name == 'MemoryOSMemory':
                cfg['llm_model'] = model_name
                cfg['api_key'] = api_key
                cfg['api_base'] = base_url
            print(f"[memory LLM] runtime provider configured: model={model_name}")
        return cfg

    # -----------------------------------------------------------------------
    # 逐个 llm × memory × qa_category × 数据集运行评测
    # -----------------------------------------------------------------------
    llm_names = args.llm_name
    memory_names = args.memory_name
    qa_categories = args.qa_category if args.qa_category else [None]

    # -----------------------------------------------------------------------
    # 构建统一的 (dialog_dir, result_dir, config_label) 列表
    # -----------------------------------------------------------------------
    all_configs = []
    for cap_cat in (args.caption_category or []):
        all_configs.append((
            os.path.join(DATA_DIR, "dialog", cap_cat),
            os.path.join(RESULT_DIR, cap_cat),
            f"caption_category={cap_cat}",
        ))
    for audio_cap in (args.audio_caption or []):
        all_configs.append((
            os.path.join(DATA_DIR, "dialog", "audio_caption", audio_cap),
            os.path.join(RESULT_DIR, "audio_caption", audio_cap),
            f"audio_caption={audio_cap}",
        ))
    if not all_configs:
        all_configs.append((
            os.path.join(DATA_DIR, "dialog"),
            RESULT_DIR,
            "default",
        ))

    print(f"\nLLM models to evaluate: {', '.join(llm_names)}")
    print(f"Memory methods to evaluate: {', '.join(memory_names)}")
    print(f"QA categories to evaluate: {', '.join(c for c in qa_categories if c) if qa_categories != [None] else 'all'}")
    print(f"Data configs to evaluate: {', '.join(label for _, _, label in all_configs)}")

    combo_idx = 0

    for DIALOG_DIR, RESULT_DIR, config_label in all_configs:
        print(f"\n[{config_label}] DIALOG_DIR -> {DIALOG_DIR}")
        print(f"[{config_label}] RESULT_DIR -> {RESULT_DIR}")

        # --all_datasets 时需要按当前 DIALOG_DIR 重新扫描
        if args.all_datasets:
            datasets = get_available_datasets(DIALOG_DIR)
            if not datasets:
                print(f"No datasets found in {DIALOG_DIR}, skipping.")
                continue
            print(f"Found {len(datasets)} datasets in {DIALOG_DIR}: {', '.join(datasets)}")

        for llm_name in llm_names:
            OPENAI_APIKEY, OPENAI_APIBASE, OPENAI_MODEL = _get_llm_config(llm_name)

            for memory_name in memory_names:
                DialogueAgentMemoryConfig = _get_memory_config_patched(memory_name, llm_name)

                for qa_category in qa_categories:
                    for data_name in datasets:
                        combo_idx += 1
                        cat_label = qa_category if qa_category else 'all'
                        print(f"\n{'='*80}")
                        print(f"[{combo_idx}] config={config_label}  llm={llm_name}  memory={memory_name}  category={cat_label}  dataset={data_name}")
                        print(f"{'='*80}")
                        try:
                            run_mm_bench(
                                llm_name,
                                memory_name,
                                DialogueAgentMemoryConfig,
                                data_name,
                                args.save_results,
                                args.save_efficiency,
                                args.eval_retrieval_metrics,
                                args.eval_topk,
                                model_name=OPENAI_MODEL,
                                seed=args.seed,
                                sample=args.sample,
                                save_memory_context=args.save_memory_context,
                                max_workers=args.max_workers,
                                qa_category=qa_category,
                                image_option_mode=args.image_option_mode,
                                with_reasoning=args.with_reasoning,
                            )
                        except Exception as e:
                            print(f"Error processing dataset {data_name} with {memory_name}: {redact_runtime_text(e)}")
                            continue

    print(f"\n{'='*80}")
    print(f"Completed: {combo_idx} evaluation runs")
    print(f"{'='*80}")
