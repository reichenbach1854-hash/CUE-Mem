"""检索模块：从 MemoryIndex 中检索 top-k 最相关的 turn。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .config import RQ3_ROOT
from .data_loader import clue_to_turn_ids, resolve_media_path
from .indexer import MODALITIES, MemoryIndex


class MemoryRetriever:
    """给定 query 从 MemoryIndex 检索 top-k 相关 turn。

    Text-Index 使用原 turn-level cosine 检索。
    MM-Index 分别检索 text/image/audio item，再通过 RRF 融合为 turn-level 排名。
    """

    def __init__(
        self,
        index: MemoryIndex,
        encoder: Any,
        top_k: int = 5,
        cache_dir: Path | str | None = None,
        query_cache_dir: Path | str | None = None,
        data_dir: Path | str | None = None,
        rrf_c: int = 60,
        query_embedding_mode: str = "composed",
    ):
        if query_embedding_mode not in {"composed", "separate"}:
            raise ValueError(
                "query_embedding_mode must be 'composed' or 'separate'"
            )
        if index.index_mode == "multimodal" and query_embedding_mode != "separate":
            raise ValueError(
                "Separate multimodal index requires "
                "query_embedding_mode='separate'."
            )
        self.index = index
        self.encoder = encoder
        self.top_k = top_k
        self.rrf_c = rrf_c
        self.query_embedding_mode = query_embedding_mode
        # text cache: [(idx, score), ...]
        # multimodal cache: [(turn_id, score), ...]
        self._mem_cache: dict[str, list] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
        self.query_cache_dir = (
            Path(query_cache_dir).expanduser().resolve()
            if query_cache_dir
            else self._default_query_cache_dir()
        )
        self.query_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, qa: dict, top_k: int) -> str:
        provider = self.index.embedding_provider or getattr(self.encoder, "provider", "unknown")
        model = self.index.embedding_model or getattr(self.encoder, "model_name", "unknown")
        dim = self.index.embedding_dim or getattr(self.encoder, "embedding_dim", "unknown")
        qa_id = self._safe_cache_component(str(qa.get("qa_id", "unknown")))
        config = {
            "provider": provider,
            "model": model,
            "dim": dim,
            "index_mode": self.index.index_mode,
            "query_embedding_mode": self.query_embedding_mode,
            "query_parser_version": 2,
            "question_fingerprint": self._question_fingerprint(qa),
            "top_k": top_k,
        }
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        # Provider/index/query mode 已包含在父目录和 hash 中。短文件名避免
        # Windows 传统 MAX_PATH=260 导致 write_text 抛 FileNotFoundError。
        return f"{qa_id[:48]}__{config_hash}__top{top_k}"

    @staticmethod
    def _safe_cache_component(value: str) -> str:
        safe = value
        for char in '<>:"/\\|?*':
            safe = safe.replace(char, "_")
        return safe.strip(". ") or "unknown"

    def _provider_model_dim_key(self) -> str:
        provider = self.index.embedding_provider or getattr(self.encoder, "provider", "unknown")
        model = self.index.embedding_model or getattr(self.encoder, "model_name", "unknown")
        dim = self.index.embedding_dim or getattr(self.encoder, "embedding_dim", "unknown")
        key = f"{provider}__{model}__dim{dim}"
        return key.replace("/", "_").replace("\\", "_").replace(":", "_")

    def _default_query_cache_dir(self) -> Path:
        provider = self.index.embedding_provider or getattr(self.encoder, "provider", "unknown")
        if self.cache_dir is not None:
            return self.cache_dir / "query_embeddings" / provider / self._provider_model_dim_key()
        return RQ3_ROOT / "cache" / "query_embeddings" / provider / self._provider_model_dim_key()

    def _load_cache(self, key: str) -> list | None:
        """先查内存，再查磁盘。"""
        if key in self._mem_cache:
            return self._mem_cache[key]
        if self.cache_dir:
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                data = json.loads(path.read_text())
                self._mem_cache[key] = data
                return data
        return None

    def _save_cache(self, key: str, data: list) -> None:
        self._mem_cache[key] = data
        if self.cache_dir:
            path = self.cache_dir / f"{key}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(data), encoding="utf-8")
            tmp_path.replace(path)

    def _encode_query(self, qa: dict) -> torch.Tensor:
        """Text-Index 查询编码。MM-Index 使用 query items。"""
        question = qa.get("question", "")
        option_captions = qa.get("option_captions")
        return self.encoder.encode_query_text(question, option_captions)

    def retrieve(self, qa: dict, top_k: int | None = None) -> list[dict]:
        """检索与 QA 最相关的 top-k 个 turn。

        返回: [{
            'turn_id': str,
            'session_id': str,
            'score': float,
            'turn_data': dict,
            'rank': int,
        }, ...]
        """
        k = top_k or self.top_k
        qa_id = qa.get("qa_id", "")
        cache_key = self._cache_key(qa, k) if qa_id else None

        cached = self._load_cache(cache_key) if cache_key else None
        if cached is not None:
            if self.index.index_mode == "text":
                return self._build_text_results(cached)
            if self.index.index_mode == "unified_multimodal":
                return self._build_unified_results(cached)
            return self._build_multimodal_results(cached)

        if self.index.index_mode == "text":
            if self.query_embedding_mode == "composed":
                query_items = self._load_or_encode_query_items(qa)
                raw = self._retrieve_text_rrf_raw(query_items, k)
            else:
                query_emb = self._encode_query(qa)
                self._check_query_dim(query_emb)
                raw = self._retrieve_text_raw(query_emb, k)
            if cache_key:
                self._save_cache(cache_key, raw)
            return self._build_text_results(raw)

        query_items = self._load_or_encode_query_items(qa)
        if self.index.index_mode == "unified_multimodal":
            raw = self._retrieve_unified_multimodal_raw(query_items, k)
        else:
            raw = self._retrieve_multimodal_rrf_raw(query_items, k)
        if cache_key:
            self._save_cache(cache_key, raw)
        if self.index.index_mode == "unified_multimodal":
            return self._build_unified_results(raw)
        return self._build_multimodal_results(raw)

    def _retrieve_text_raw(self, query_emb: torch.Tensor, k: int) -> list[tuple[int, float]]:
        if self.index.embeddings is None:
            raise ValueError("Text index embeddings are not loaded.")
        self._check_embedding_dim(self.index.embeddings, query_emb)
        scores = torch.matmul(self.index.embeddings, query_emb)
        top_scores, top_indices = torch.topk(scores, min(k, len(self.index)))
        return list(zip(top_indices.tolist(), top_scores.tolist()))

    def _retrieve_text_rrf_raw(
        self,
        query_items: dict[str, list[dict]],
        k: int,
    ) -> list[tuple[int, float, list[dict]]]:
        """用 composed multi-query 检索 Text-Index，并通过 RRF 融合。"""
        embeddings = self.index.embeddings
        if embeddings is None:
            raise ValueError("Text index embeddings are not loaded.")

        candidate_k = max(20, k * 4)
        scores_by_idx: dict[int, float] = {}
        items_by_idx: dict[int, list[dict]] = {}

        for group_items in query_items.values():
            for query_item in group_items:
                query_emb = query_item["embedding"]
                self._check_query_dim(query_emb)
                self._check_embedding_dim(embeddings, query_emb)

                scores = torch.matmul(embeddings, query_emb)
                top_n = min(candidate_k, embeddings.shape[0])
                top_scores, top_indices = torch.topk(scores, top_n)

                for rank, (idx, item_score) in enumerate(
                    zip(top_indices.tolist(), top_scores.tolist()),
                    start=1,
                ):
                    scores_by_idx[idx] = (
                        scores_by_idx.get(idx, 0.0)
                        + 1.0 / (self.rrf_c + rank)
                    )
                    turn_id = self.index.turn_ids[idx]
                    items_by_idx.setdefault(idx, []).append({
                        "item_id": f"{turn_id}::text::{idx}",
                        "modality": "text",
                        "path": None,
                        "fallback": None,
                        "score": item_score,
                        "rank": rank,
                        "query_metadata": query_item.get("metadata", {}),
                    })

        ranked = sorted(
            scores_by_idx.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            (idx, score, items_by_idx.get(idx, []))
            for idx, score in ranked[:k]
        ]

    def _retrieve_multimodal_rrf_raw(
        self,
        query_items: dict[str, list[dict]],
        k: int,
    ) -> list[tuple[str, float]]:
        candidate_k = max(20, k * 4)
        scores_by_turn: dict[str, float] = {}

        for modality in MODALITIES:
            embeddings = self.index.modality_embeddings.get(modality)
            metadata = self.index.modality_metadata.get(modality, [])
            if embeddings is None or embeddings.shape[0] == 0:
                continue
            for query_item in query_items.get(modality, []):
                query_emb = query_item["embedding"]
                self._check_query_dim(query_emb)
                self._check_embedding_dim(embeddings, query_emb)
                ranked_turns = self._rank_turns_for_query_embedding(
                    embeddings,
                    metadata,
                    query_emb,
                    candidate_k,
                )
                for rank, turn_id in enumerate(ranked_turns, start=1):
                    scores_by_turn[turn_id] = scores_by_turn.get(turn_id, 0.0) + (
                        1.0 / (self.rrf_c + rank)
                    )

        ranked = sorted(scores_by_turn.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def _retrieve_unified_multimodal_raw(
        self,
        query_items: dict[str, list[dict]],
        k: int,
    ) -> list[tuple[str, float, list[dict]]]:
        embeddings = self.index.item_embeddings
        metadata = self.index.item_metadata
        if embeddings is None or embeddings.shape[0] == 0:
            return []

        candidate_k = max(20, k * 4)
        scores_by_turn: dict[str, float] = {}
        items_by_turn: dict[str, list[dict]] = {}

        for group_items in query_items.values():
            for query_item in group_items:
                query_emb = query_item["embedding"]
                self._check_query_dim(query_emb)
                self._check_embedding_dim(embeddings, query_emb)
                scores = torch.matmul(embeddings, query_emb)
                top_n = min(candidate_k, embeddings.shape[0])
                top_scores, top_indices = torch.topk(scores, top_n)

                seen_turns: set[str] = set()
                turn_rank = 0
                for idx, item_score in zip(
                    top_indices.tolist(),
                    top_scores.tolist(),
                ):
                    meta = metadata[idx]
                    turn_id = meta["turn_id"]
                    if turn_id in seen_turns:
                        continue
                    seen_turns.add(turn_id)
                    turn_rank += 1
                    scores_by_turn[turn_id] = (
                        scores_by_turn.get(turn_id, 0.0)
                        + 1.0 / (self.rrf_c + turn_rank)
                    )
                    items_by_turn.setdefault(turn_id, []).append({
                        "item_id": meta["item_id"],
                        "modality": meta["modality"],
                        "path": meta.get("path"),
                        "fallback": meta.get("fallback"),
                        "score": item_score,
                        "rank": turn_rank,
                        "query_metadata": query_item.get("metadata", {}),
                    })

        ranked = sorted(
            scores_by_turn.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return [
            (turn_id, score, items_by_turn.get(turn_id, []))
            for turn_id, score in ranked[:k]
        ]

    @staticmethod
    def _rank_turns_for_query_embedding(
        embeddings: torch.Tensor,
        metadata: list[dict],
        query_emb: torch.Tensor,
        candidate_k: int,
    ) -> list[str]:
        scores = torch.matmul(embeddings, query_emb)
        top_n = min(candidate_k, embeddings.shape[0])
        _, top_indices = torch.topk(scores, top_n)

        ranked_turns: list[str] = []
        seen_turns: set[str] = set()
        for idx in top_indices.tolist():
            turn_id = metadata[idx]["turn_id"]
            if turn_id in seen_turns:
                continue
            seen_turns.add(turn_id)
            ranked_turns.append(turn_id)
        return ranked_turns

    def _load_or_encode_query_items(self, qa: dict) -> dict[str, list[dict]]:
        qa_id = qa.get("qa_id", "")
        cache_path = self._query_cache_path(qa)
        if qa_id and cache_path.exists():
            try:
                return self._load_query_items(cache_path, qa)
            except Exception:  # noqa: BLE001
                # Corrupt or stale query cache should not block retrieval.
                cache_path.unlink(missing_ok=True)

        if self.query_embedding_mode == "composed":
            query_items = self.encoder.encode_query_composed_items(qa)
        else:
            query_items = self.encoder.encode_query_multimodal_items(qa)
        if qa_id:
            self._save_query_items(cache_path, qa, query_items)
        return query_items

    def _query_cache_path(self, qa: dict) -> Path:
        qa_id = qa.get("qa_id", "unknown")
        safe_qa_id = str(qa_id).replace("/", "_").replace("\\", "_").replace(":", "_")
        fingerprint = self._question_fingerprint(qa)[:12]
        return (
            self.query_cache_dir
            / self.index.index_mode
            / self.query_embedding_mode
            / f"{safe_qa_id}__{fingerprint}.pt"
        )

    def _save_query_items(
        self,
        cache_path: Path,
        qa: dict,
        query_items: dict[str, list[dict]],
    ) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "qa_id": qa.get("qa_id", ""),
            "embedding_provider": self.index.embedding_provider,
            "embedding_model": self.index.embedding_model,
            "embedding_dim": self.index.embedding_dim,
            "index_mode": self.index.index_mode,
            "query_embedding_mode": self.query_embedding_mode,
            "option_image_fingerprints": self._option_image_fingerprints(qa),
            "question_fingerprint": self._question_fingerprint(qa),
            "question_text": qa.get("question", ""),
            "text_options_fingerprint": self.encoder._extract_text_options(qa),
            "items": {
                group: [
                    {
                        "embedding": item["embedding"],
                        "metadata": item.get("metadata", {}),
                    }
                    for item in items
                ]
                for group, items in query_items.items()
            },
        }
        torch.save(payload, cache_path)

    def _load_query_items(self, cache_path: Path, qa: dict) -> dict[str, list[dict]]:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("embedding_provider") != self.index.embedding_provider:
            raise ValueError("Query cache provider mismatch")
        if payload.get("embedding_model") != self.index.embedding_model:
            raise ValueError("Query cache model mismatch")
        if payload.get("embedding_dim") != self.index.embedding_dim:
            raise ValueError("Query cache dim mismatch")
        if payload.get("index_mode") != self.index.index_mode:
            raise ValueError("Query cache index mode mismatch")
        if payload.get("query_embedding_mode") != self.query_embedding_mode:
            raise ValueError("Query cache embedding mode mismatch")
        if payload.get("option_image_fingerprints") != self._option_image_fingerprints(qa):
            raise ValueError("Query cache option image fingerprint mismatch")
        if payload.get("question_fingerprint") != self._question_fingerprint(qa):
            raise ValueError("Query cache question/options fingerprint mismatch")
        return payload["items"]

    def _question_fingerprint(self, qa: dict) -> str:
        payload = {
            "question": qa.get("question", ""),
            "point": qa.get("point", ""),
            "text_options": self.encoder._extract_text_options(qa),
            "option_images": self._option_image_fingerprints(qa),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _option_image_fingerprints(self, qa: dict) -> dict[str, dict]:
        fingerprints: dict[str, dict] = {}
        option_images = qa.get("option_images") or {}
        for letter, raw_path in option_images.items():
            path = resolve_media_path(raw_path, self.data_dir)
            if path and path.exists():
                stat = path.stat()
                fingerprints[letter] = {
                    "path": str(path),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                }
            else:
                fingerprints[letter] = {"path": str(raw_path), "missing": True}
        return fingerprints

    def _check_query_dim(self, query_emb: torch.Tensor) -> None:
        expected = self.index.embedding_dim
        if expected is not None and query_emb.shape[-1] != expected:
            raise ValueError(
                "Query embedding dimension does not match loaded index: "
                f"query_dim={query_emb.shape[-1]}, index_dim={expected}. "
                "Please regenerate embedding cache with the same provider/model/dim."
            )

    @staticmethod
    def _check_embedding_dim(index_embeddings: torch.Tensor, query_emb: torch.Tensor) -> None:
        if index_embeddings.shape[-1] != query_emb.shape[-1]:
            raise ValueError(
                "Index embedding dimension does not match query embedding: "
                f"index_dim={index_embeddings.shape[-1]}, query_dim={query_emb.shape[-1]}. "
                "Please regenerate embedding cache with the same provider/model/dim."
            )

    def _build_text_results(self, raw: list) -> list[dict]:
        results = []
        for rank, raw_item in enumerate(raw):
            if len(raw_item) == 3:
                idx, score, retrieved_items = raw_item
            else:
                idx, score = raw_item
                turn_id = self.index.turn_ids[idx]
                retrieved_items = [{
                    "item_id": f"{turn_id}::text::{idx}",
                    "modality": "text",
                    "path": None,
                    "fallback": None,
                    "score": score,
                    "rank": rank + 1,
                    "query_metadata": {
                        "source": "single_text_query",
                        "composition_method": "text_only",
                    },
                }]
            turn_data = self.index.turns[idx]
            results.append({
                "turn_id": self.index.turn_ids[idx],
                "session_id": turn_data["session_id"],
                "score": score,
                "turn_data": turn_data,
                "rank": rank,
                "retrieved_items": retrieved_items,
            })
        return results

    def _build_multimodal_results(self, raw: list) -> list[dict]:
        results = []
        for rank, (turn_id, score) in enumerate(raw):
            turn_data = self.index.get_turn(turn_id)
            if turn_data is None:
                continue
            results.append({
                "turn_id": turn_id,
                "session_id": turn_data["session_id"],
                "score": score,
                "turn_data": turn_data,
                "rank": rank,
            })
        return results

    def _build_unified_results(self, raw: list) -> list[dict]:
        results = []
        for rank, (turn_id, score, retrieved_items) in enumerate(raw):
            turn_data = self.index.get_turn(turn_id)
            if turn_data is None:
                continue
            results.append({
                "turn_id": turn_id,
                "session_id": turn_data["session_id"],
                "score": score,
                "turn_data": turn_data,
                "rank": rank,
                "retrieved_items": retrieved_items,
            })
        return results

    def retrieve_with_evidence_eval(
        self,
        qa: dict,
        sessions: list[dict],
        k_list: list[int] | None = None,
    ) -> dict:
        """检索并计算 evidence 召回指标。"""
        if k_list is None:
            k_list = [1, 3, 5, 10]

        max_k = max(k_list)
        retrieved = self.retrieve(qa, top_k=max_k)

        clue = qa.get("clue", [])
        evidence_turns = clue_to_turn_ids(clue, sessions)

        recall_at_k = {}
        precision_at_k = {}
        for k in k_list:
            top_k_ids = {r["turn_id"] for r in retrieved[:k]}
            if evidence_turns:
                hits = top_k_ids & evidence_turns
                recall_at_k[k] = len(hits) / len(evidence_turns)
                precision_at_k[k] = len(hits) / k
            else:
                recall_at_k[k] = 0.0
                precision_at_k[k] = 0.0

        return {
            "retrieved": retrieved,
            "recall@k": recall_at_k,
            "precision@k": precision_at_k,
            "evidence_turn_ids": evidence_turns,
        }
