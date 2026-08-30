"""构建并管理 RQ3 memory embedding 索引。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .data_loader import flatten_turns

MODALITIES = ("text", "image", "audio")


class MemoryIndex:
    """一个 profile 的 memory embedding 索引。

    index_mode == "text" 时使用原 turn-level 单矩阵索引。
    index_mode == "multimodal" 时使用 text/image/audio 分模态 item 索引。
    index_mode == "unified_multimodal" 时将全部模态 item 放入共享矩阵。
    """

    def __init__(
        self,
        embeddings: torch.Tensor | None,
        turn_ids: list[str],
        turns: list[dict],
        index_mode: str,
        modality_embeddings: dict[str, torch.Tensor] | None = None,
        modality_metadata: dict[str, list[dict]] | None = None,
        item_embeddings: torch.Tensor | None = None,
        item_metadata: list[dict] | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_dim: int | None = None,
    ):
        """
        Args:
            embeddings: text index 的 [N, D] turn-level embedding。
            turn_ids: turn_id 列表。text 模式与 embeddings 行对齐；multimodal
                模式为所有 turn 的列表。
            turns: 完整 turn 数据列表。
            index_mode: 'text' | 'multimodal'。
            modality_embeddings: multimodal 模式的分模态 embedding。
            modality_metadata: multimodal 模式的分模态 metadata。
            item_embeddings: unified_multimodal 模式的共享 item embedding。
            item_metadata: unified_multimodal 模式的共享 item metadata。
        """
        self.embeddings = embeddings
        self.turn_ids = turn_ids
        self.turns = turns
        self.index_mode = index_mode
        self.modality_embeddings = modality_embeddings or {}
        self.modality_metadata = modality_metadata or {}
        self.item_embeddings = item_embeddings
        self.item_metadata = item_metadata or []
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim or self._infer_embedding_dim()
        self._turn_map = {t["turn_id"]: t for t in turns}

    def __len__(self) -> int:
        return len(self.turn_ids)

    def get_turn(self, turn_id: str) -> dict | None:
        return self._turn_map.get(turn_id)

    @classmethod
    def build(
        cls,
        sessions: list[dict],
        encoder: Any,
        index_mode: str,
        cache_dir: Path | None = None,
        skip_existing: bool = True,
        separate_cache_dir: Path | None = None,
    ) -> MemoryIndex:
        """遍历所有 turn 编码后构建索引。"""
        if cache_dir and skip_existing and cls._cache_exists(cache_dir, index_mode):
            cached = cls.load(cache_dir, index_mode, sessions)
            if cls._is_encoder_compatible(cached, encoder):
                return cached
            print(
                "  [cache invalid] provider/model/dim mismatch; "
                "rebuilding index."
            )

        if index_mode == "text":
            index = cls._build_text_index(sessions, encoder, cache_dir)
        elif index_mode == "multimodal":
            index = cls._build_multimodal_index(sessions, encoder, cache_dir)
        elif index_mode == "unified_multimodal":
            if separate_cache_dir and cls._cache_exists(separate_cache_dir, "multimodal"):
                try:
                    index = cls._build_unified_from_separate_cache(
                        separate_cache_dir,
                        sessions,
                        encoder,
                    )
                except ValueError as exc:
                    print(
                        "  Separate cache cannot be reused; "
                        f"encoding unified index normally. Reason: {exc}"
                    )
                    index = cls._build_unified_multimodal_index(
                        sessions,
                        encoder,
                        cache_dir,
                    )
            else:
                index = cls._build_unified_multimodal_index(
                    sessions,
                    encoder,
                    cache_dir,
                )
        else:
            raise ValueError(f"Unsupported index_mode: {index_mode}")

        if cache_dir:
            index.save(cache_dir)
            cls._clear_partial_cache(cache_dir, index_mode)

        return index

    @staticmethod
    def _is_encoder_compatible(index: MemoryIndex, encoder: Any) -> bool:
        return (
            index.embedding_provider == getattr(encoder, "provider", None)
            and index.embedding_model == getattr(encoder, "model_name", None)
            and int(index.embedding_dim or 0)
            == int(getattr(encoder, "embedding_dim", 0))
        )

    @classmethod
    def _build_unified_multimodal_index(
        cls,
        sessions: list[dict],
        encoder: Any,
        cache_dir: Path | None = None,
    ) -> MemoryIndex:
        turns = flatten_turns(sessions)
        turn_ids = [turn["turn_id"] for turn in turns]
        emb_list, metadata, completed = cls._load_unified_partial(cache_dir)

        for turn in tqdm(turns, desc="Encoding (unified multimodal items)"):
            if turn["turn_id"] in completed:
                continue
            items = encoder.encode_turn_multimodal_items(turn)
            for modality in MODALITIES:
                for item in items[modality]:
                    meta = {
                        "item_id": f"{turn['turn_id']}::{modality}::{len(metadata)}",
                        "turn_id": turn["turn_id"],
                        "session_id": turn["session_id"],
                        "modality": modality,
                        "turn_data": turn,
                    }
                    if item.get("path"):
                        meta["path"] = item["path"]
                    if item.get("fallback"):
                        meta["fallback"] = item["fallback"]
                    emb_list.append(item["embedding"])
                    metadata.append(meta)
            completed.add(turn["turn_id"])
            cls._save_unified_partial(
                cache_dir,
                emb_list,
                metadata,
                completed,
                encoder,
            )

        embedding_dim = (
            int(emb_list[0].shape[-1])
            if emb_list
            else int(encoder.embedding_dim)
        )
        item_embeddings = (
            torch.stack(emb_list)
            if emb_list
            else torch.empty(0, embedding_dim)
        )
        return cls(
            embeddings=None,
            turn_ids=turn_ids,
            turns=turns,
            index_mode="unified_multimodal",
            item_embeddings=item_embeddings,
            item_metadata=metadata,
            embedding_provider=getattr(encoder, "provider", None),
            embedding_model=getattr(encoder, "model_name", None),
            embedding_dim=embedding_dim,
        )

    @classmethod
    def _build_unified_from_separate_cache(
        cls,
        separate_cache_dir: Path,
        sessions: list[dict],
        encoder: Any,
    ) -> MemoryIndex:
        turns = flatten_turns(sessions)
        separate = cls._load_multimodal(separate_cache_dir, turns)
        expected = (
            getattr(encoder, "provider", None),
            getattr(encoder, "model_name", None),
            int(getattr(encoder, "embedding_dim", 0)),
        )
        actual = (
            separate.embedding_provider,
            separate.embedding_model,
            int(separate.embedding_dim or 0),
        )
        if actual != expected:
            raise ValueError(
                "Separate cache is incompatible with current encoder: "
                f"cache={actual}, encoder={expected}"
            )

        tensors = []
        metadata = []
        for modality in MODALITIES:
            embeddings = separate.modality_embeddings[modality]
            tensors.append(embeddings)
            for old_meta in separate.modality_metadata[modality]:
                meta = dict(old_meta)
                meta["item_id"] = (
                    meta.get("item_id")
                    or f"{meta['turn_id']}::{modality}::{len(metadata)}"
                )
                meta["modality"] = modality
                metadata.append(meta)
        item_embeddings = torch.cat(tensors, dim=0)
        print(
            "  Reused separate multimodal cache: "
            f"{item_embeddings.shape[0]} items"
        )
        return cls(
            embeddings=None,
            turn_ids=[turn["turn_id"] for turn in turns],
            turns=turns,
            index_mode="unified_multimodal",
            item_embeddings=item_embeddings,
            item_metadata=metadata,
            embedding_provider=separate.embedding_provider,
            embedding_model=separate.embedding_model,
            embedding_dim=separate.embedding_dim,
        )

    @classmethod
    def _build_text_index(
        cls,
        sessions: list[dict],
        encoder: Any,
        cache_dir: Path | None = None,
    ) -> MemoryIndex:
        turns = flatten_turns(sessions)
        turn_ids, emb_list = cls._load_text_partial(cache_dir)
        completed = set(turn_ids)

        for turn in tqdm(turns, desc="Encoding (text)"):
            if turn["turn_id"] in completed:
                continue
            emb_list.append(encoder.encode_turn_text_only(turn))
            turn_ids.append(turn["turn_id"])
            completed.add(turn["turn_id"])
            cls._save_text_partial(cache_dir, turn_ids, emb_list, encoder)

        embeddings = torch.stack(emb_list)
        return cls(
            embeddings,
            turn_ids,
            turns,
            "text",
            embedding_provider=getattr(encoder, "provider", None),
            embedding_model=getattr(encoder, "model_name", None),
            embedding_dim=embeddings.shape[1],
        )

    @classmethod
    def _build_multimodal_index(
        cls,
        sessions: list[dict],
        encoder: Any,
        cache_dir: Path | None = None,
    ) -> MemoryIndex:
        turns = flatten_turns(sessions)
        turn_ids = [turn["turn_id"] for turn in turns]
        emb_lists, metadata, completed = cls._load_multimodal_partial(cache_dir)

        for turn in tqdm(turns, desc="Encoding (multimodal items)"):
            if turn["turn_id"] in completed:
                continue
            items = encoder.encode_turn_multimodal_items(turn)
            for modality in MODALITIES:
                for item in items[modality]:
                    item_id = f"{turn['turn_id']}::{modality}::{len(metadata[modality])}"
                    emb_lists[modality].append(item["embedding"])
                    meta = {
                        "item_id": item_id,
                        "turn_id": turn["turn_id"],
                        "session_id": turn["session_id"],
                        "modality": modality,
                        "turn_data": turn,
                    }
                    if item.get("path"):
                        meta["path"] = item["path"]
                    if item.get("fallback"):
                        meta["fallback"] = item["fallback"]
                    metadata[modality].append(meta)
            completed.add(turn["turn_id"])
            cls._save_multimodal_partial(cache_dir, emb_lists, metadata, completed, encoder)

        embedding_dim = cls._resolve_encoder_dim(encoder, emb_lists)
        embeddings = {}
        for modality in MODALITIES:
            if emb_lists[modality]:
                embeddings[modality] = torch.stack(emb_lists[modality])
            else:
                embeddings[modality] = torch.empty(0, embedding_dim)
        return cls(
            embeddings=None,
            turn_ids=turn_ids,
            turns=turns,
            index_mode="multimodal",
            modality_embeddings=embeddings,
            modality_metadata=metadata,
            embedding_provider=getattr(encoder, "provider", None),
            embedding_model=getattr(encoder, "model_name", None),
            embedding_dim=embedding_dim,
        )

    @staticmethod
    def _resolve_encoder_dim(encoder: Any, emb_lists: dict[str, list[torch.Tensor]]) -> int:
        for modality in MODALITIES:
            if emb_lists[modality]:
                return int(emb_lists[modality][0].shape[-1])
        dim = getattr(encoder, "embedding_dim", None)
        if dim is None:
            raise ValueError("Cannot infer embedding dimension from encoder or embeddings.")
        return int(dim)

    def _infer_embedding_dim(self) -> int | None:
        if self.embeddings is not None and self.embeddings.ndim == 2:
            return int(self.embeddings.shape[1])
        for embeddings in self.modality_embeddings.values():
            if embeddings.ndim == 2:
                return int(embeddings.shape[1])
        if self.item_embeddings is not None and self.item_embeddings.ndim == 2:
            return int(self.item_embeddings.shape[1])
        return None

    @staticmethod
    def _text_partial_paths(cache_dir: Path) -> tuple[Path, Path]:
        return (
            cache_dir / "text_embeddings.partial.pt",
            cache_dir / "text_metadata.partial.json",
        )

    @staticmethod
    def _multimodal_partial_paths(cache_dir: Path, modality: str) -> tuple[Path, Path]:
        return (
            cache_dir / f"{modality}_embeddings.partial.pt",
            cache_dir / f"{modality}_metadata.partial.json",
        )

    @staticmethod
    def _multimodal_progress_path(cache_dir: Path) -> Path:
        return cache_dir / "multimodal_progress.partial.json"

    @staticmethod
    def _unified_partial_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
        return (
            cache_dir / "all_embeddings.partial.pt",
            cache_dir / "all_metadata.partial.json",
            cache_dir / "unified_progress.partial.json",
        )

    @classmethod
    def _load_unified_partial(
        cls,
        cache_dir: Path | None,
    ) -> tuple[list[torch.Tensor], list[dict], set[str]]:
        if cache_dir is None:
            return [], [], set()
        emb_path, meta_path, progress_path = cls._unified_partial_paths(cache_dir)
        if not emb_path.exists() or not meta_path.exists() or not progress_path.exists():
            return [], [], set()
        embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_obj = json.load(f)
        with open(progress_path, "r", encoding="utf-8") as f:
            progress = json.load(f)
        print(
            "  [resume] unified multimodal partial: "
            f"{len(progress.get('completed_turn_ids', []))} turns"
        )
        return (
            [embeddings[i] for i in range(embeddings.shape[0])],
            meta_obj.get("items", []),
            set(progress.get("completed_turn_ids", [])),
        )

    @classmethod
    def _save_unified_partial(
        cls,
        cache_dir: Path | None,
        emb_list: list[torch.Tensor],
        metadata: list[dict],
        completed: set[str],
        encoder: Any,
    ) -> None:
        if cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, meta_path, progress_path = cls._unified_partial_paths(cache_dir)
        dim = int(
            emb_list[0].shape[-1]
            if emb_list
            else encoder.embedding_dim
        )
        embeddings = (
            torch.stack(emb_list) if emb_list else torch.empty(0, dim)
        )
        torch.save(embeddings, emb_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "complete": False,
                "index_mode": "unified_multimodal",
                "num_items": len(metadata),
                "embedding_provider": getattr(encoder, "provider", None),
                "embedding_model": getattr(encoder, "model_name", None),
                "embedding_dim": dim,
                "items": metadata,
            }, f, ensure_ascii=False, indent=2)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({
                "complete": False,
                "completed_turn_ids": sorted(completed),
                "num_completed_turns": len(completed),
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def _load_text_partial(
        cls,
        cache_dir: Path | None,
    ) -> tuple[list[str], list[torch.Tensor]]:
        if cache_dir is None:
            return [], []
        emb_path, meta_path = cls._text_partial_paths(cache_dir)
        if not emb_path.exists() or not meta_path.exists():
            return [], []

        embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        turn_ids = meta.get("turn_ids", [])
        emb_list = [embeddings[i] for i in range(embeddings.shape[0])]
        print(f"  [resume] text partial: {len(turn_ids)} turns")
        return turn_ids, emb_list

    @classmethod
    def _save_text_partial(
        cls,
        cache_dir: Path | None,
        turn_ids: list[str],
        emb_list: list[torch.Tensor],
        encoder: Any,
    ) -> None:
        if cache_dir is None or not emb_list:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path, meta_path = cls._text_partial_paths(cache_dir)
        torch.save(torch.stack(emb_list), emb_path)
        meta = {
            "complete": False,
            "index_mode": "text",
            "num_turns": len(turn_ids),
            "turn_ids": turn_ids,
            "embedding_provider": getattr(encoder, "provider", None),
            "embedding_model": getattr(encoder, "model_name", None),
            "embedding_dim": int(emb_list[0].shape[-1]),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def _load_multimodal_partial(
        cls,
        cache_dir: Path | None,
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, list[dict]], set[str]]:
        emb_lists: dict[str, list[torch.Tensor]] = {m: [] for m in MODALITIES}
        metadata: dict[str, list[dict]] = {m: [] for m in MODALITIES}
        completed: set[str] = set()
        if cache_dir is None:
            return emb_lists, metadata, completed

        progress_path = cls._multimodal_progress_path(cache_dir)
        if progress_path.exists():
            with open(progress_path, "r", encoding="utf-8") as f:
                progress = json.load(f)
            completed = set(progress.get("completed_turn_ids", []))

        loaded_any = False
        for modality in MODALITIES:
            emb_path, meta_path = cls._multimodal_partial_paths(cache_dir, modality)
            if emb_path.exists() and meta_path.exists():
                embeddings = torch.load(emb_path, map_location="cpu", weights_only=True)
                emb_lists[modality] = [
                    embeddings[i] for i in range(embeddings.shape[0])
                ]
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_obj = json.load(f)
                metadata[modality] = meta_obj.get("items", meta_obj)
                loaded_any = True

        if loaded_any:
            print(f"  [resume] multimodal partial: {len(completed)} turns")
        return emb_lists, metadata, completed

    @classmethod
    def _save_multimodal_partial(
        cls,
        cache_dir: Path | None,
        emb_lists: dict[str, list[torch.Tensor]],
        metadata: dict[str, list[dict]],
        completed: set[str],
        encoder: Any,
    ) -> None:
        if cache_dir is None:
            return
        cache_dir.mkdir(parents=True, exist_ok=True)
        embedding_dim = cls._resolve_encoder_dim(encoder, emb_lists)

        for modality in MODALITIES:
            emb_path, meta_path = cls._multimodal_partial_paths(cache_dir, modality)
            embeddings = (
                torch.stack(emb_lists[modality])
                if emb_lists[modality]
                else torch.empty(0, embedding_dim)
            )
            torch.save(embeddings, emb_path)
            meta = {
                "complete": False,
                "index_mode": "multimodal",
                "modality": modality,
                "num_items": len(metadata[modality]),
                "embedding_provider": getattr(encoder, "provider", None),
                "embedding_model": getattr(encoder, "model_name", None),
                "embedding_dim": embedding_dim,
                "items": metadata[modality],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        progress = {
            "complete": False,
            "completed_turn_ids": sorted(completed),
            "num_completed_turns": len(completed),
        }
        with open(cls._multimodal_progress_path(cache_dir), "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    @classmethod
    def _clear_partial_cache(cls, cache_dir: Path, index_mode: str) -> None:
        paths: list[Path] = []
        if index_mode == "text":
            paths.extend(cls._text_partial_paths(cache_dir))
        elif index_mode == "multimodal":
            for modality in MODALITIES:
                paths.extend(cls._multimodal_partial_paths(cache_dir, modality))
            paths.append(cls._multimodal_progress_path(cache_dir))
        elif index_mode == "unified_multimodal":
            paths.extend(cls._unified_partial_paths(cache_dir))

        for path in paths:
            if path.exists():
                path.unlink()

    @staticmethod
    def _cache_exists(cache_dir: Path, index_mode: str) -> bool:
        if index_mode == "text":
            return (
                (cache_dir / "text_embeddings.pt").exists()
                and (cache_dir / "text_metadata.json").exists()
            )
        if index_mode == "multimodal":
            return all(
                (cache_dir / f"{modality}_embeddings.pt").exists()
                and (cache_dir / f"{modality}_metadata.json").exists()
                for modality in MODALITIES
            )
        if index_mode == "unified_multimodal":
            return (
                (cache_dir / "all_embeddings.pt").exists()
                and (cache_dir / "all_metadata.json").exists()
            )
        return False

    def save(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if self.index_mode == "text":
            self._save_text(cache_dir)
        elif self.index_mode == "multimodal":
            self._save_multimodal(cache_dir)
        elif self.index_mode == "unified_multimodal":
            self._save_unified_multimodal(cache_dir)
        else:
            raise ValueError(f"Unsupported index_mode: {self.index_mode}")

    def _save_text(self, cache_dir: Path) -> None:
        torch.save(self.embeddings, cache_dir / "text_embeddings.pt")
        meta = {
            "complete": True,
            "index_mode": self.index_mode,
            "num_turns": len(self.turn_ids),
            "turn_ids": self.turn_ids,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }
        with open(cache_dir / "text_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _save_multimodal(self, cache_dir: Path) -> None:
        for modality in MODALITIES:
            torch.save(
                self.modality_embeddings[modality],
                cache_dir / f"{modality}_embeddings.pt",
            )
            with open(cache_dir / f"{modality}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "complete": True,
                        "index_mode": self.index_mode,
                        "modality": modality,
                        "num_items": len(self.modality_metadata[modality]),
                        "embedding_provider": self.embedding_provider,
                        "embedding_model": self.embedding_model,
                        "embedding_dim": self.embedding_dim,
                        "items": self.modality_metadata[modality],
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

    def _save_unified_multimodal(self, cache_dir: Path) -> None:
        torch.save(self.item_embeddings, cache_dir / "all_embeddings.pt")
        with open(cache_dir / "all_metadata.json", "w", encoding="utf-8") as f:
            json.dump({
                "complete": True,
                "index_mode": self.index_mode,
                "num_items": len(self.item_metadata),
                "embedding_provider": self.embedding_provider,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "items": self.item_metadata,
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        index_mode: str,
        sessions: list[dict],
    ) -> MemoryIndex:
        turns = flatten_turns(sessions)
        if index_mode == "text":
            return cls._load_text(cache_dir, turns)
        if index_mode == "multimodal":
            return cls._load_multimodal(cache_dir, turns)
        if index_mode == "unified_multimodal":
            return cls._load_unified_multimodal(cache_dir, turns)
        raise ValueError(f"Unsupported index_mode: {index_mode}")

    @classmethod
    def _load_text(cls, cache_dir: Path, turns: list[dict]) -> MemoryIndex:
        embeddings = torch.load(
            cache_dir / "text_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        with open(cache_dir / "text_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return cls(
            embeddings,
            meta["turn_ids"],
            turns,
            "text",
            embedding_provider=meta.get("embedding_provider"),
            embedding_model=meta.get("embedding_model"),
            embedding_dim=meta.get("embedding_dim", embeddings.shape[1]),
        )

    @classmethod
    def _load_multimodal(cls, cache_dir: Path, turns: list[dict]) -> MemoryIndex:
        embeddings: dict[str, torch.Tensor] = {}
        metadata: dict[str, list[dict]] = {}
        turn_map = {turn["turn_id"]: turn for turn in turns}

        for modality in MODALITIES:
            embeddings[modality] = torch.load(
                cache_dir / f"{modality}_embeddings.pt",
                map_location="cpu",
                weights_only=True,
            )
            with open(cache_dir / f"{modality}_metadata.json", "r", encoding="utf-8") as f:
                meta_obj = json.load(f)
            if isinstance(meta_obj, dict) and "items" in meta_obj:
                metadata[modality] = meta_obj["items"]
                provider = meta_obj.get("embedding_provider")
                model = meta_obj.get("embedding_model")
                dim = meta_obj.get("embedding_dim", embeddings[modality].shape[1])
            else:
                metadata[modality] = meta_obj
                provider = None
                model = None
                dim = embeddings[modality].shape[1]
            for meta in metadata[modality]:
                turn_data = turn_map.get(meta["turn_id"], meta.get("turn_data", {}))
                meta["turn_data"] = turn_data

        return cls(
            embeddings=None,
            turn_ids=[turn["turn_id"] for turn in turns],
            turns=turns,
            index_mode="multimodal",
            modality_embeddings=embeddings,
            modality_metadata=metadata,
            embedding_provider=provider,
            embedding_model=model,
            embedding_dim=dim,
        )

    @classmethod
    def _load_unified_multimodal(
        cls,
        cache_dir: Path,
        turns: list[dict],
    ) -> MemoryIndex:
        embeddings = torch.load(
            cache_dir / "all_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        )
        with open(cache_dir / "all_metadata.json", "r", encoding="utf-8") as f:
            meta_obj = json.load(f)
        metadata = meta_obj.get("items", [])
        turn_map = {turn["turn_id"]: turn for turn in turns}
        for meta in metadata:
            meta["turn_data"] = turn_map.get(
                meta["turn_id"],
                meta.get("turn_data", {}),
            )
        return cls(
            embeddings=None,
            turn_ids=[turn["turn_id"] for turn in turns],
            turns=turns,
            index_mode="unified_multimodal",
            item_embeddings=embeddings,
            item_metadata=metadata,
            embedding_provider=meta_obj.get("embedding_provider"),
            embedding_model=meta_obj.get("embedding_model"),
            embedding_dim=meta_obj.get("embedding_dim", embeddings.shape[1]),
        )
