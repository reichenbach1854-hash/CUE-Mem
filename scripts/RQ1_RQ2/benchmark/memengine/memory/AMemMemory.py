import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

from memengine.memory.BaseMemory import ExplicitMemory
from benchmark.paths import REPOSITORY_ROOT


def _get_config_value(config_obj: Any, name: str, default: Any = None) -> Any:
    return getattr(getattr(config_obj, "args", config_obj), name, default)


def _observation_to_text(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    if not isinstance(observation, dict):
        return str(observation)
    return str(observation.get("text", "")).strip()


class AMemMemory(ExplicitMemory):
    """
    Text-only RQ1/RQ2 adapter for A-Mem.

    The benchmark passes caption-enriched text to non-multimodal memory backends.
    This adapter stores each processed dialogue turn as one A-Mem note and
    returns A-Mem's retrieved notes as textual memory context.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self.top_k = int(_get_config_value(config, "top_k", 10))
        self.model_name = _get_config_value(config, "embedding_model", "BAAI/bge-m3")
        self.llm_backend = _get_config_value(config, "llm_backend", "openai")
        self.llm_model = _get_config_value(config, "llm_model", os.getenv("CUE_MEM_LLM_MODEL", ""))
        self.api_key = (
            os.getenv("AMEM_API_KEY")
            or _get_config_value(config, "api_key", None)
            or os.getenv("CUE_MEM_LLM_API_KEY")
        )
        self.api_base = (
            os.getenv("AMEM_API_BASE")
            or _get_config_value(config, "api_base", None)
            or os.getenv("CUE_MEM_LLM_BASE_URL")
        )
        self.evo_threshold = int(_get_config_value(config, "evo_threshold", 100))
        self.evo_interval = int(_get_config_value(config, "evo_interval", 15))
        self.check_connection = bool(_get_config_value(config, "check_connection", False))

        # run_bench.py reads memory.recall_op.last_retrieved_ids for retrieval metrics.
        self.recall_op = SimpleNamespace(last_retrieved_ids=[])
        self._note_id_to_dialogue_id: Dict[str, str] = {}
        self._build_memory_system()

    def _build_memory_system(self) -> None:
        repo_root = REPOSITORY_ROOT
        amem_dir = repo_root / "A-mem"
        if not amem_dir.exists():
            raise FileNotFoundError(f"A-Mem source directory not found: {amem_dir}")
        amem_path = str(amem_dir)
        if amem_path not in sys.path:
            sys.path.insert(0, amem_path)

        from memory_layer_robust import RobustAgenticMemorySystem

        self.memory_system = RobustAgenticMemorySystem(
            model_name=self.model_name,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            api_key=self.api_key,
            api_base=self.api_base,
            evo_threshold=self.evo_threshold,
            evo_interval=self.evo_interval,
            check_connection=self.check_connection,
        )

    def reset(self) -> None:
        self._note_id_to_dialogue_id = {}
        self.recall_op.last_retrieved_ids = []
        self._build_memory_system()

    def store(self, observation) -> None:
        text = _observation_to_text(observation)
        if not text:
            return

        timestamp = observation.get("timestamp") if isinstance(observation, dict) else None
        dialogue_id = observation.get("dialogue_id") if isinstance(observation, dict) else None
        note_id = self.memory_system.add_note(content=text, time=timestamp)
        if dialogue_id:
            self._note_id_to_dialogue_id[note_id] = str(dialogue_id)

    def recall(self, query) -> str:
        query_text = _observation_to_text(query)
        if not query_text:
            self.recall_op.last_retrieved_ids = []
            return "None"

        memory_str, indices = self.memory_system.find_related_memories(query_text, k=self.top_k)
        note_ids: List[str] = list(self.memory_system.memories.keys())
        retrieved_ids: List[str] = []
        for idx in indices:
            try:
                note_id = note_ids[int(idx)]
            except Exception:
                continue
            dialogue_id = self._note_id_to_dialogue_id.get(note_id)
            if dialogue_id:
                retrieved_ids.append(dialogue_id)
        self.recall_op.last_retrieved_ids = retrieved_ids
        return memory_str or "None"

    def export_cache(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "memories": self.memory_system.memories,
            "note_id_to_dialogue_id": self._note_id_to_dialogue_id,
            "evo_cnt": getattr(self.memory_system, "evo_cnt", 0),
            "note_counter": getattr(self.memory_system, "_note_counter", len(self.memory_system.memories)),
            "model_name": self.model_name,
            "llm_backend": self.llm_backend,
            "llm_model": self.llm_model,
            "evo_threshold": self.evo_threshold,
            "evo_interval": self.evo_interval,
        }

    def import_cache(self, payload: Mapping[str, Any]) -> None:
        if int(payload.get("version", 0)) != 1:
            raise ValueError(f"Unsupported AMem cache version: {payload.get('version')}")
        self.memory_system.memories = dict(payload.get("memories") or {})
        self._note_id_to_dialogue_id = dict(payload.get("note_id_to_dialogue_id") or {})
        self.memory_system.evo_cnt = int(payload.get("evo_cnt", 0))
        self.memory_system._note_counter = int(payload.get("note_counter", len(self.memory_system.memories)))
        self._rebuild_retriever()

    def _rebuild_retriever(self) -> None:
        retriever_cls = self.memory_system.retriever.__class__
        self.memory_system.retriever = retriever_cls(self.model_name)
        documents = []
        for note in self.memory_system.memories.values():
            documents.append(
                "content:" + note.content +
                " context:" + note.context +
                " keywords: " + ", ".join(note.keywords) +
                " tags: " + ", ".join(note.tags)
            )
        if documents:
            self.memory_system.retriever.add_documents(documents)

    def display(self) -> None:
        print(f"AMemMemory notes={len(self.memory_system.memories)}")

    def manage(self, operation, **kwargs) -> None:
        pass

    def optimize(self, **kwargs) -> None:
        pass
