import os
import json
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from memengine.memory.BaseMemory import ExplicitMemory
from benchmark.paths import BENCHMARK_ROOT, REPOSITORY_ROOT


def _get_config_value(config_obj: Any, name: str, default: Any = None) -> Any:
    return getattr(getattr(config_obj, "args", config_obj), name, default)


def _observation_to_text(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    if not isinstance(observation, dict):
        return str(observation)
    return str(observation.get("text", "")).strip()


def _split_user_assistant(text: str) -> Tuple[str, str]:
    user_lines: List[str] = []
    assistant_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower().startswith("assistant:"):
            assistant_lines.append(stripped.split(":", 1)[1].strip())
        else:
            user_lines.append(stripped)
    return "\n".join(user_lines).strip(), "\n".join(assistant_lines).strip()


class MemoryOSMemory(ExplicitMemory):
    """
    Text-only RQ1/RQ2 adapter for MemoryOS.

    It stores caption-enriched dialogue turns into MemoryOS and calls the
    Retriever directly during QA. It intentionally does not call
    Memoryos.get_response(), because run_bench.py owns final answer generation
    and the MemoryOS get_response() method mutates memory with the test query.
    """

    DIALOGUE_PREFIX_RE = re.compile(r"^\[dialogue_id=(?P<dialogue_id>[^\]]+)\]\s*")

    def __init__(self, config) -> None:
        super().__init__(config)
        self.top_k = int(_get_config_value(config, "top_k", 10))
        self.top_k_sessions = int(_get_config_value(config, "top_k_sessions", self.top_k))
        self.top_k_knowledge = int(_get_config_value(config, "top_k_knowledge", self.top_k))
        self.user_id = _get_config_value(config, "user_id", "memgallery_user")
        self.assistant_id = _get_config_value(config, "assistant_id", "memgallery_assistant")
        self.llm_model = _get_config_value(config, "llm_model", os.getenv("CUE_MEM_LLM_MODEL", ""))
        self.api_key = (
            os.getenv("MEMORYOS_API_KEY")
            or _get_config_value(config, "api_key", None)
            or os.getenv("CUE_MEM_LLM_API_KEY")
        )
        self.api_base = (
            os.getenv("MEMORYOS_API_BASE")
            or _get_config_value(config, "api_base", None)
            or os.getenv("CUE_MEM_LLM_BASE_URL")
        )
        self.embedding_model = _get_config_value(config, "embedding_model", "BAAI/bge-m3")
        self.embedding_model_kwargs = _get_config_value(config, "embedding_model_kwargs", None)
        self.short_term_capacity = int(_get_config_value(config, "short_term_capacity", 10))
        self.mid_term_capacity = int(_get_config_value(config, "mid_term_capacity", 2000))
        self.long_term_knowledge_capacity = int(_get_config_value(config, "long_term_knowledge_capacity", 100))
        self.retrieval_queue_capacity = int(_get_config_value(config, "retrieval_queue_capacity", 7))
        self.mid_term_heat_threshold = float(_get_config_value(config, "mid_term_heat_threshold", 5.0))
        self.mid_term_similarity_threshold = float(_get_config_value(config, "mid_term_similarity_threshold", 0.6))
        self.fast_mode = bool(_get_config_value(config, "fast_mode", False))
        self.fast_batch_size = max(1, int(_get_config_value(config, "fast_batch_size", 15)))
        self.fast_use_llm_summary = bool(_get_config_value(config, "fast_use_llm_summary", True))

        self.recall_op = SimpleNamespace(last_retrieved_ids=[])
        self._fast_buffer: List[Dict[str, Any]] = []
        self._last_fast_page: Dict[str, Any] | None = None
        self._instance_id = uuid.uuid4().hex[:12]
        self.storage_root = self._resolve_storage_root(_get_config_value(config, "storage_root", ".memory_cache/MemoryOS"))
        self._build_memoryos()

    def _resolve_storage_root(self, storage_root: str) -> Path:
        path = Path(str(storage_root))
        if not path.is_absolute():
            path = BENCHMARK_ROOT / path
        return path

    def _build_memoryos(self) -> None:
        repo_root = REPOSITORY_ROOT
        memoryos_src = repo_root / "MemoryOS" / "memoryos-mcp"
        if not memoryos_src.exists():
            raise FileNotFoundError(f"MemoryOS source directory not found: {memoryos_src}")
        memoryos_path = str(memoryos_src)
        if memoryos_path not in sys.path:
            sys.path.insert(0, memoryos_path)

        from memoryos.memoryos import Memoryos

        data_storage_path = self.storage_root / self._instance_id
        data_storage_path.mkdir(parents=True, exist_ok=True)

        self.memoryos = Memoryos(
            user_id=self.user_id,
            assistant_id=self.assistant_id,
            openai_api_key=self.api_key,
            openai_base_url=self.api_base,
            data_storage_path=str(data_storage_path),
            short_term_capacity=self.short_term_capacity,
            mid_term_capacity=self.mid_term_capacity,
            long_term_knowledge_capacity=self.long_term_knowledge_capacity,
            retrieval_queue_capacity=self.retrieval_queue_capacity,
            mid_term_heat_threshold=self.mid_term_heat_threshold,
            mid_term_similarity_threshold=self.mid_term_similarity_threshold,
            llm_model=self.llm_model,
            embedding_model_name=self.embedding_model,
            embedding_model_kwargs=self.embedding_model_kwargs,
        )

    def reset(self) -> None:
        self.recall_op.last_retrieved_ids = []
        self._fast_buffer = []
        self._last_fast_page = None
        self._instance_id = uuid.uuid4().hex[:12]
        self._build_memoryos()

    def store(self, observation) -> None:
        text = _observation_to_text(observation)
        if not text:
            return

        dialogue_id = observation.get("dialogue_id") if isinstance(observation, dict) else None
        timestamp = observation.get("timestamp") if isinstance(observation, dict) else None
        user_input, agent_response = _split_user_assistant(text)
        if dialogue_id:
            user_input = f"[dialogue_id={dialogue_id}] {user_input}"

        qa_pair = {
            "user_input": user_input,
            "agent_response": agent_response,
            "timestamp": timestamp,
        }

        if self.fast_mode:
            self._fast_buffer.append(qa_pair)
            if len(self._fast_buffer) >= self.fast_batch_size:
                self._flush_fast_buffer()
            return

        self.memoryos.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp,
            meta_data={"dialogue_id": dialogue_id} if dialogue_id else None,
        )

    def _memoryos_utils(self):
        try:
            from memoryos.utils import generate_id, get_timestamp, gpt_generate_multi_summary
        except ImportError:
            from utils import generate_id, get_timestamp, gpt_generate_multi_summary
        return generate_id, get_timestamp, gpt_generate_multi_summary

    def _session_id_from_user_input(self, user_input: str) -> str:
        dialogue_id, _ = self._strip_dialogue_prefix(user_input or "")
        if not dialogue_id:
            return ""
        return dialogue_id.split(":", 1)[0]

    def _flush_fast_buffer(self) -> None:
        if not self._fast_buffer:
            return

        generate_id, get_timestamp, gpt_generate_multi_summary = self._memoryos_utils()
        pages: List[Dict[str, Any]] = []
        prev_page = self._last_fast_page

        for qa_pair in self._fast_buffer:
            page = {
                "page_id": generate_id("page"),
                "user_input": qa_pair.get("user_input", ""),
                "agent_response": qa_pair.get("agent_response", ""),
                "timestamp": qa_pair.get("timestamp") or get_timestamp(),
                "preloaded": False,
                # Fast benchmark mode skips hot-session profile updates.
                "analyzed": True,
                "pre_page": None,
                "next_page": None,
                "meta_info": {
                    "source": "memoryos_fast_mode",
                    "dialogue_id": self._strip_dialogue_prefix(qa_pair.get("user_input", ""))[0],
                },
                "page_keywords": [],
            }

            if prev_page and self._session_id_from_user_input(prev_page.get("user_input", "")) == self._session_id_from_user_input(page["user_input"]):
                page["pre_page"] = prev_page.get("page_id")
                prev_page["next_page"] = page["page_id"]

            pages.append(page)
            prev_page = page

        self._last_fast_page = pages[-1]
        self._fast_buffer = []

        input_text = "\n".join(
            f"User: {p.get('user_input', '')}\nAssistant: {p.get('agent_response', '')}"
            for p in pages
        )

        summaries = []
        if self.fast_use_llm_summary:
            multi_summary_result = gpt_generate_multi_summary(input_text, self.memoryos.client, model=self.llm_model)
            summaries = (multi_summary_result or {}).get("summaries") or []

        if not summaries:
            first_id = self._strip_dialogue_prefix(pages[0].get("user_input", ""))[0] or "unknown"
            last_id = self._strip_dialogue_prefix(pages[-1].get("user_input", ""))[0] or first_id
            summaries = [{
                "theme": "conversation_batch",
                "content": f"Conversation batch from {first_id} to {last_id}.",
                "keywords": [],
            }]

        for summary_item in summaries:
            theme_summary = summary_item.get("content") or "Conversation batch."
            theme_keywords = summary_item.get("keywords") or []
            self.memoryos.mid_term_memory.insert_pages_into_session(
                summary_for_new_pages=theme_summary,
                keywords_for_new_pages=theme_keywords,
                pages_to_insert=pages,
                similarity_threshold=self.mid_term_similarity_threshold,
            )

    def finalize_store(self) -> None:
        if self.fast_mode:
            self._flush_fast_buffer()

    def _read_json_file(self, path: str, default: Any) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    def _write_json_file(self, path: str, data: Any) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def export_cache(self) -> Dict[str, Any]:
        self.finalize_store()
        self.memoryos.short_term_memory.save()
        self.memoryos.mid_term_memory.save()
        self.memoryos.user_long_term_memory.save()
        self.memoryos.assistant_long_term_memory.save()
        return {
            "version": 1,
            "fast_mode": self.fast_mode,
            "fast_batch_size": self.fast_batch_size,
            "files": {
                "short_term": self._read_json_file(self.memoryos.short_term_memory.file_path, []),
                "mid_term": self._read_json_file(self.memoryos.mid_term_memory.file_path, {"sessions": {}, "access_frequency": {}}),
                "user_long_term": self._read_json_file(self.memoryos.user_long_term_memory.file_path, {}),
                "assistant_long_term": self._read_json_file(self.memoryos.assistant_long_term_memory.file_path, {}),
            },
            "last_fast_page": self._last_fast_page,
            "last_evicted_page_for_continuity": self.memoryos.updater.last_evicted_page_for_continuity,
        }

    def import_cache(self, payload: Dict[str, Any]) -> None:
        files = payload.get("files", {}) if isinstance(payload, dict) else {}
        self._write_json_file(self.memoryos.short_term_memory.file_path, files.get("short_term", []))
        self._write_json_file(self.memoryos.mid_term_memory.file_path, files.get("mid_term", {"sessions": {}, "access_frequency": {}}))
        self._write_json_file(self.memoryos.user_long_term_memory.file_path, files.get("user_long_term", {}))
        self._write_json_file(self.memoryos.assistant_long_term_memory.file_path, files.get("assistant_long_term", {}))

        self.memoryos.short_term_memory.load()
        self.memoryos.mid_term_memory.load()
        self.memoryos.user_long_term_memory.load()
        self.memoryos.assistant_long_term_memory.load()
        self.memoryos.updater.last_evicted_page_for_continuity = payload.get("last_evicted_page_for_continuity")
        self._last_fast_page = payload.get("last_fast_page")
        self._fast_buffer = []

    def _strip_dialogue_prefix(self, text: str) -> Tuple[str, str]:
        match = self.DIALOGUE_PREFIX_RE.match(text or "")
        if not match:
            return "", text or ""
        return match.group("dialogue_id"), self.DIALOGUE_PREFIX_RE.sub("", text or "", count=1)

    def _format_page(self, page: Dict[str, Any], score: Any = None) -> Tuple[str, str]:
        raw_user = str(page.get("user_input", ""))
        dialogue_id, user_input = self._strip_dialogue_prefix(raw_user)
        agent_response = str(page.get("agent_response", ""))
        timestamp = page.get("timestamp", "")
        score_line = f"\nscore: {score}" if score is not None else ""
        text = (
            f"dialogue_id: {dialogue_id or 'unknown'}\n"
            f"timestamp: {timestamp}\n"
            f"User: {user_input}\n"
            f"Assistant: {agent_response}{score_line}"
        )
        return dialogue_id, text

    def recall(self, query) -> str:
        query_text = _observation_to_text(query)
        if not query_text:
            self.recall_op.last_retrieved_ids = []
            return "None"

        results = self.memoryos.retriever.retrieve_context(
            user_query=query_text,
            user_id=self.user_id,
            top_k_sessions=self.top_k_sessions,
            top_k_knowledge=self.top_k_knowledge,
        )

        context_blocks: List[str] = []
        retrieved_ids: List[str] = []
        seen_ids = set()

        for qa_pair in self.memoryos.short_term_memory.get_all():
            dialogue_id, block = self._format_page(qa_pair)
            if dialogue_id and dialogue_id not in seen_ids:
                retrieved_ids.append(dialogue_id)
                seen_ids.add(dialogue_id)
            context_blocks.append("[Short-term Memory]\n" + block)

        for session in results.get("retrieved_pages", []) or []:
            session_summary = session.get("session_summary", "")
            for matched in session.get("matched_pages", []) or []:
                page = matched.get("page_data", matched)
                score = matched.get("score")
                dialogue_id, block = self._format_page(page, score=score)
                if dialogue_id and dialogue_id not in seen_ids:
                    retrieved_ids.append(dialogue_id)
                    seen_ids.add(dialogue_id)
                if session_summary:
                    block = f"session_summary: {session_summary}\n{block}"
                context_blocks.append("[Mid-term Memory]\n" + block)

        for knowledge in results.get("retrieved_user_knowledge", []) or []:
            context_blocks.append(
                "[User Knowledge]\n"
                f"timestamp: {knowledge.get('timestamp', '')}\n"
                f"{knowledge.get('knowledge', '')}"
            )

        self.recall_op.last_retrieved_ids = retrieved_ids[: self.top_k]
        return "\n\n".join(context_blocks[: self.top_k]) if context_blocks else "None"

    def display(self) -> None:
        short_count = len(self.memoryos.short_term_memory.get_all())
        mid_count = len(getattr(self.memoryos.mid_term_memory, "sessions", {}))
        print(f"MemoryOSMemory short_term={short_count}, mid_term_sessions={mid_count}")

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass
