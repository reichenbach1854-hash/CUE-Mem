"""Step 1: 预计算所有 turn 的 embedding。

对每个 profile 分别计算两种模式的 embedding:
  - text:        encode_turn_text_only()  → embeddings/{provider}/text_only/p{N}/
  - multimodal:  分模态 text/image/audio item index → embeddings/{provider}/multimodal/p{N}/
  - unified_multimodal: 全模态共享 item index → embeddings/{provider}/unified_multimodal/p{N}/

用法:
    python -m scripts.RQ3.step1_encode_embeddings
    python scripts/RQ3/step1_encode_embeddings.py --device cuda:1 --skip-existing
    python -m scripts.RQ3.step1_encode_embeddings --profiles 0 1
    python -m scripts.RQ3.step1_encode_embeddings --modes text
    python -m scripts.RQ3.step1_encode_embeddings --sample 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Direct execution from a checkout: make the sibling package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import (
    DATA_DIR,
    EMBEDDING_DIR,
    EMBEDDING_PROVIDER,
    GEMINI_EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    IMAGEBIND_DEVICE,
    profile_paths,
    resolve_path,
)
from .data_loader import load_profile
from .encoder import create_encoder
from .indexer import MemoryIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预计算 RQ3 embedding")
    parser.add_argument("--embedding-provider", type=str, default=EMBEDDING_PROVIDER,
                        choices=["imagebind", "gemini"],
                        help="embedding 后端")
    parser.add_argument("--device", type=str, default=IMAGEBIND_DEVICE)
    parser.add_argument("--gemini-api-base", type=str, default=None)
    parser.add_argument("--gemini-api-key", type=str, default=None)
    parser.add_argument("--gemini-model", type=str, default=GEMINI_EMBEDDING_MODEL)
    parser.add_argument("--gemini-embedding-dim", type=int, default=GEMINI_EMBEDDING_DIM)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="RQ3 data directory; defaults to the project-relative configuration.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="Directory containing history_with_qa_pN.json files.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Directory for embedding indices and the encoding summary.",
    )
    parser.add_argument(
        "--profile-files",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit profile JSON files; overrides --history-dir.",
    )
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的缓存")
    parser.add_argument("--profiles", type=int, nargs="+", default=None,
                        help="只处理指定 profile (0/1/2)")
    parser.add_argument("--sample", type=int, default=None,
                        help="每个 profile 只编码前 N 个 turn（调试用）")
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=["unified_multimodal"],
        choices=["text", "multimodal", "unified_multimodal"],
                        help="编码模式")
    parser.add_argument(
        "--reuse-separate-cache-for-unified",
        action="store_true",
        help="用已有 multimodal 分模态缓存拼接 unified cache，避免重新编码",
    )
    return parser.parse_args()


def _embedding_cache_dir(
    embedding_dir: Path,
    provider: str,
    mode: str,
    p_id: int,
    sample: int | None = None,
) -> Path:
    mode_dir = {
        "text": "text_only",
        "multimodal": "multimodal",
        "unified_multimodal": "unified_multimodal",
    }[mode]
    profile_dir = f"p{p_id}" if sample is None else f"p{p_id}_sample{sample}"
    return embedding_dir / provider / mode_dir / profile_dir


def _limit_sessions_by_turns(sessions: list[dict], max_turns: int | None) -> list[dict]:
    """返回只包含前 max_turns 个 turn 的 sessions 浅拷贝。"""
    if max_turns is None:
        return sessions
    if max_turns <= 0:
        raise ValueError("--sample must be a positive integer")

    limited_sessions = []
    remaining = max_turns
    for sess in sessions:
        if remaining <= 0:
            break
        turns = sess["turns"][:remaining]
        if turns:
            copied = dict(sess)
            copied["turns"] = turns
            limited_sessions.append(copied)
            remaining -= len(turns)
    return limited_sessions


def _runtime_inputs(args: argparse.Namespace) -> tuple[Path, list[Path], Path]:
    data_dir = resolve_path(args.data_dir, DATA_DIR)
    history_dir = resolve_path(args.history_dir, data_dir / "history_dialogue")
    embedding_dir = resolve_path(args.embedding_dir, EMBEDDING_DIR)
    if args.profile_files:
        profiles = [resolve_path(path) for path in args.profile_files]
    else:
        profiles = profile_paths(history_dir)
    if not profiles:
        raise ValueError("no profile files configured")
    return data_dir, profiles, embedding_dir


def main():
    args = parse_args()
    data_dir, profile_files, embedding_dir = _runtime_inputs(args)

    # 启用 WARNING 级别日志，使 encoder.py 中的模态失败警告可见
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    print(f"Embedding provider: {args.embedding_provider}")
    print(f"Device: {args.device}")
    print(f"Modes: {args.modes}")
    if args.sample is not None:
        print(f"Sample turns per profile: {args.sample}")
    print("Loading embedding encoder ...")
    encoder = create_encoder(
        provider=args.embedding_provider,
        device=args.device,
        gemini_api_key=args.gemini_api_key,
        gemini_api_base=args.gemini_api_base,
        gemini_model=args.gemini_model,
        gemini_embedding_dim=args.gemini_embedding_dim,
        data_dir=data_dir,
    )
    print(
        "Embedding encoder loaded: "
        f"provider={encoder.provider}, model={encoder.model_name}, dim={encoder.embedding_dim}\n"
    )

    profile_indices = (
        args.profiles if args.profiles is not None else list(range(len(profile_files)))
    )

    summary = {}

    for p_id in profile_indices:
        if p_id < 0 or p_id >= len(profile_files):
            print(f"[WARN] invalid profile index {p_id}, skipping")
            continue
        json_path = profile_files[p_id]
        if not json_path.exists():
            print(f"[WARN] {json_path} not found, skipping p{p_id}")
            continue

        print(f"{'='*60}")
        print(f"Profile p{p_id}: {json_path.name}")
        profile_data = load_profile(json_path)
        all_sessions = profile_data["sessions"]
        total_turns = sum(len(s["turns"]) for s in all_sessions)
        sessions = _limit_sessions_by_turns(all_sessions, args.sample)
        encoded_turns = sum(len(s["turns"]) for s in sessions)
        print(f"  Sessions: {len(all_sessions)}, Turns: {total_turns}")
        if args.sample is not None:
            print(f"  Sampled turns: {encoded_turns}")

        for mode in args.modes:
            cache_dir = _embedding_cache_dir(
                embedding_dir,
                args.embedding_provider,
                mode,
                p_id,
                args.sample,
            )

            print(f"\n  Mode: {mode} → {cache_dir}")

            # multimodal 模式编码前重置统计，确保每次统计只属于当前 profile/mode
            if mode in {"multimodal", "unified_multimodal"}:
                encoder.reset_stats()

            separate_cache_dir = None
            if (
                mode == "unified_multimodal"
                and args.reuse_separate_cache_for_unified
            ):
                separate_cache_dir = _embedding_cache_dir(
                    embedding_dir,
                    args.embedding_provider,
                    "multimodal",
                    p_id,
                    args.sample,
                )
                print(f"  Separate cache candidate: {separate_cache_dir}")

            index = MemoryIndex.build(
                sessions=sessions,
                encoder=encoder,
                index_mode=mode,
                cache_dir=cache_dir,
                skip_existing=args.skip_existing,
                separate_cache_dir=separate_cache_dir,
            )

            if mode == "text":
                print(f"  Index built: {len(index)} turns, shape={index.embeddings.shape}")
            elif mode == "multimodal":
                shapes = {
                    modality: list(emb.shape)
                    for modality, emb in index.modality_embeddings.items()
                }
                print(f"  Index built: {len(index)} turns, modality_shapes={shapes}")
            else:
                print(
                    f"  Index built: {len(index)} turns, "
                    f"all_shape={list(index.item_embeddings.shape)}"
                )

            # multimodal 模式编码完成后打印模态统计，验证图片和音频是否真实参与了编码
            if mode in {"multimodal", "unified_multimodal"}:
                encoder.print_stats(prefix=f"p{p_id}/{mode}")
                modal_stats = encoder.get_stats()
                mode_summary = {
                    "turns": len(index),
                    "cache_dir": str(cache_dir),
                    "embedding_provider": index.embedding_provider,
                    "embedding_model": index.embedding_model,
                    "embedding_dim": index.embedding_dim,
                    "modal_stats": modal_stats,
                }
                if mode == "multimodal":
                    mode_summary["modality_shapes"] = {
                        modality: list(emb.shape)
                        for modality, emb in index.modality_embeddings.items()
                    }
                else:
                    mode_summary["all_shape"] = list(index.item_embeddings.shape)
                    mode_summary["num_items"] = len(index.item_metadata)
                summary[f"{args.embedding_provider}_p{p_id}_{mode}"] = mode_summary
            else:
                summary[f"{args.embedding_provider}_p{p_id}_{mode}"] = {
                    "turns": len(index),
                    "shape": list(index.embeddings.shape),
                    "cache_dir": str(cache_dir),
                    "embedding_provider": index.embedding_provider,
                    "embedding_model": index.embedding_model,
                    "embedding_dim": index.embedding_dim,
                }

    # 保存全局 summary
    summary_path = embedding_dir / "encoding_summary.json"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
