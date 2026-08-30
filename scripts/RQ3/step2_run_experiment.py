"""Step 2: 运行 RQ3 主实验 — 4 种 pipeline 变体 × 3 profiles。

用法:
    python -m scripts.RQ3.step2_run_experiment --variants TT TM MT MM
    python -m scripts.RQ3.step2_run_experiment --variants TT --top-k 5
    python -m scripts.RQ3.step2_run_experiment --variants MM --profiles 0 --sample 10
    python -m scripts.RQ3.step2_run_experiment --variants TT --resume
    python -m scripts.RQ3.step2_run_experiment --variants TT --with-reasoning
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import (
    ALIYUN_OMNI_MODEL,
    DATA_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_DIR,
    EMBEDDING_PROVIDER,
    EVAL_MODEL_PROVIDER,
    GEMINI_EMBEDDING_DIM,
    GEMINI_EMBEDDING_MODEL,
    IMAGEBIND_DEVICE,
    OMNI_MODEL,
    RESULT_DIR,
    RETRIEVAL_TOP_K_LIST,
    VARIANTS,
    profile_paths,
    redact_sensitive_text,
    resolve_path,
)
from .data_loader import load_profile
from .encoder import create_encoder
from .indexer import MemoryIndex
from .memory_formatter import (
    format_memory_multimodal,
    format_memory_text,
    format_question_multimodal,
    format_question_text,
)
from .omni_client import OmniClient
from .retriever import MemoryRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RQ3 主实验")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=VARIANTS,
        choices=VARIANTS,
        help="要运行的 pipeline 变体",
    )
    parser.add_argument(
        "--profiles",
        type=int,
        nargs="+",
        default=None,
        help="只处理指定 profile",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="检索 top-k",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="每个 profile 只测试前 N 条 QA（调试用）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="LLM 并发调用数",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑: 跳过已有结果的 QA",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="每完成 N 条 QA 保存一次结果；默认 1 表示逐条增量保存",
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default=EMBEDDING_PROVIDER,
        choices=["imagebind", "gemini"],
        help="embedding 后端",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=IMAGEBIND_DEVICE,
    )
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
        "--profile-files",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit profile JSON files; overrides --history-dir.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Directory containing embedding indices and retrieval caches.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Directory for experiment result files.",
    )
    parser.add_argument(
        "--retrieval-cache-dir",
        type=Path,
        default=None,
        help="Root for retrieval result caches; defaults under --embedding-dir.",
    )
    parser.add_argument(
        "--query-cache-dir",
        type=Path,
        default=None,
        help="Root for encoded QA query caches; defaults under --embedding-dir.",
    )
    parser.add_argument(
        "--memory-index-mode",
        choices=["unified_multimodal", "multimodal", "text"],
        default="unified_multimodal",
        help=(
            "M-Index 使用的 memory index；默认 unified_multimodal。"
            "旧基线使用 multimodal。"
        ),
    )
    parser.add_argument(
        "--query-embedding-mode",
        choices=["composed", "separate"],
        default="composed",
        help="M-Index query 编码模式；默认 option-conditioned composed。",
    )
    parser.add_argument(
        "--eval-provider",
        type=str,
        default=EVAL_MODEL_PROVIDER,
        choices=["vllm", "aliyun"],
        help="评测模型后端",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help="评测模型名称；vllm 默认 OMNI_MODEL，aliyun 默认 ALIYUN_OMNI_MODEL",
    )
    parser.add_argument(
        "--result-subdir",
        type=str,
        default="",
        help=(
            "可选结果子目录，写入 RQ3/results/<provider_model>/<result-subdir>/"
            "<variant>/pN_results.json；用于隔离 unified_embedding 等实验。"
        ),
    )
    parser.add_argument(
        "--omni-model",
        type=str,
        default=None,
        help="兼容旧参数，等价于 --eval-model",
    )
    parser.add_argument(
        "--omni-api-base",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--omni-api-key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--with-reasoning",
        action="store_true",
        help="让模型同时输出推理依据，保存到结果的 reasoning 字段",
    )
    parser.add_argument(
        "--qa-retries",
        type=int,
        default=3,
        help="单条 QA 遇到 429/限流类错误时的重试次数",
    )
    parser.add_argument(
        "--qa-retry-base-wait",
        type=float,
        default=2.0,
        help="QA 级重试的基础等待秒数，按指数退避",
    )
    parser.add_argument(
        "--verbose-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在 console 打印 QA/API/多模态输入错误上下文，默认开启",
    )
    parser.add_argument(
        "--log-mm-inputs",
        action="store_true",
        help="逐条打印 TM/MM 的多模态输入 block 统计（调试用）",
    )
    return parser.parse_args()


def _variant_index_mode(variant: str, memory_index_mode: str) -> str:
    """TT/TM → text；MT/MM → CLI 指定的 M-Index。"""
    return "text" if variant.startswith("T") else memory_index_mode


def _variant_use_mode(variant: str) -> str:
    """TT/MT → text, TM/MM → multimodal"""
    return "text" if variant.endswith("T") else "multimodal"


def _embedding_cache_dir(
    embedding_dir: Path,
    provider: str,
    index_mode: str,
    p_id: int,
) -> Path:
    mode_dir = {
        "text": "text_only",
        "multimodal": "multimodal",
        "unified_multimodal": "unified_multimodal",
    }[index_mode]
    return embedding_dir / provider / mode_dir / f"p{p_id}"


def _safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _eval_result_root(
    result_dir: Path,
    provider: str,
    model: str,
    result_subdir: str = "",
) -> Path:
    root = result_dir / f"{provider}_{_safe_name(model)}"
    result_subdir = result_subdir.strip().strip("/\\")
    if result_subdir:
        root = root / _safe_name(result_subdir)
    return root


def _resolve_eval_model(args: argparse.Namespace) -> str:
    if args.eval_model:
        return args.eval_model
    if args.omni_model:
        return args.omni_model
    if args.eval_provider == "aliyun":
        return ALIYUN_OMNI_MODEL
    return OMNI_MODEL


def _configure_runtime_paths(args: argparse.Namespace) -> None:
    """Resolve every filesystem input/output relative to the project root."""

    args.data_dir = resolve_path(args.data_dir, DATA_DIR)
    args.history_dir = resolve_path(
        args.history_dir,
        args.data_dir / "history_dialogue",
    )
    args.embedding_dir = resolve_path(args.embedding_dir, EMBEDDING_DIR)
    args.result_dir = resolve_path(args.result_dir, RESULT_DIR)
    args.retrieval_cache_dir = resolve_path(
        args.retrieval_cache_dir,
        args.embedding_dir / "retrieval_cache",
    )
    args.query_cache_dir = resolve_path(
        args.query_cache_dir,
        args.embedding_dir / "query_cache",
    )
    args.profile_files = (
        [resolve_path(path) for path in args.profile_files]
        if args.profile_files
        else profile_paths(args.history_dir)
    )


def _qa_uid(item: dict, p_id: int) -> str:
    """Stable QA identity for resume/checkpoint.

    RQ3 data may reuse the same qa_id across different point values, e.g.
    pref_img/rec_img/pref_text.  Resume must therefore key by profile+point+qa_id,
    not qa_id alone.
    """
    existing = str(item.get("qa_uid", "")).strip()
    if existing:
        return existing
    qa_id = str(item.get("qa_id", "")).strip()
    point = str(item.get("point", "")).strip()
    if not qa_id:
        return ""
    return f"p{p_id}::{point}::{qa_id}"


def _load_existing_results(result_path: Path, p_id: int) -> dict[str, dict]:
    if not result_path.exists():
        return {}
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    if isinstance(results, dict) and isinstance(results.get("results"), list):
        results = results["results"]
    if not isinstance(results, list):
        print(f"  [WARN] Unsupported existing result shape: {result_path}")
        return {}

    completed: dict[str, dict] = {}
    skipped = 0
    duplicates = 0
    for r in results:
        if not isinstance(r, dict):
            continue
        uid = _qa_uid(r, p_id)
        if not uid:
            continue
        if not str(r.get("model_answer", "")).strip():  # 空答案视为未完成
            skipped += 1
            continue
        if uid in completed:
            duplicates += 1
        r["qa_uid"] = uid
        completed[uid] = r
    if skipped:
        print(f"  [resume] {skipped} items with empty model_answer will be re-evaluated.")
    if duplicates:
        print(f"  [resume] {duplicates} duplicate qa_uid rows found; keeping the last non-empty answer.")
    return completed


def _ordered_results(
    qas: list[dict],
    result_by_uid: dict[str, dict],
    p_id: int,
) -> list[dict]:
    """Export results in source QA order, preserving extra existing rows.

    Extra rows can happen when --sample is used with an existing full result file;
    preserving them avoids truncating completed runs during a sampled resume.
    """
    ordered: list[dict] = []
    seen: set[str] = set()
    for qa in qas:
        uid = _qa_uid(qa, p_id)
        if uid in result_by_uid:
            row = result_by_uid[uid]
            row["qa_uid"] = uid
            ordered.append(row)
            seen.add(uid)
    for uid, row in result_by_uid.items():
        if uid not in seen:
            row["qa_uid"] = uid
            ordered.append(row)
    return ordered


def _print_resume_summary(
    *,
    qas: list[dict],
    existing: dict[str, dict],
    pending_qas: list[dict],
    p_id: int,
) -> None:
    expected_uids = {_qa_uid(q, p_id) for q in qas if _qa_uid(q, p_id)}
    completed_uids = set(existing)
    extra = completed_uids - expected_uids
    missing_by_point = Counter(str(q.get("point", "")) for q in pending_qas)
    print(
        "  Resuming: "
        f"completed={len(completed_uids & expected_uids)}, "
        f"pending={len(pending_qas)}, extra_existing={len(extra)}"
    )
    if missing_by_point:
        detail = ", ".join(
            f"{point}={count}" for point, count in sorted(missing_by_point.items())
        )
        print(f"  Pending by point: {detail}")


def _save_results(results: list[dict], result_path: Path) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(result_path)


def _maybe_save_checkpoint(
    results: list[dict],
    result_path: Path,
    completed_since_save: int,
    checkpoint_every: int,
) -> int:
    if checkpoint_every <= 0:
        return completed_since_save
    if completed_since_save >= checkpoint_every:
        _save_results(results, result_path)
        return 0
    return completed_since_save


def _block_type_counts(blocks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for block in blocks:
        typ = block.get("type", "<missing>")
        counts[typ] = counts.get(typ, 0) + 1
    return counts


def _is_retryable_error_text(text: str) -> bool:
    lowered = text.lower()
    retry_markers = (
        "429",
        "rate limit",
        "rate_limit",
        "too quickly",
        "request rate increased",
        "throttl",
        "temporarily unavailable",
        "timeout",
        "timed out",
    )
    return any(marker in lowered for marker in retry_markers)


def _retry_wait(base_wait: float, attempt: int) -> float:
    return min(60.0, base_wait * (2 ** attempt))


def _print_error_context(
    *,
    stage: str,
    qa: dict,
    variant: str,
    p_id: int,
    use_mode: str,
    retrieved: list[dict] | None = None,
    memory_content: Any = None,
    question_content: Any = None,
    error: Any = None,
    redaction_values: tuple[str | None, ...] = (),
) -> None:
    qa_id = qa.get("qa_id", "")
    print(
        f"\n[STEP2 ERROR] stage={stage} variant={variant} p{p_id} "
        f"qa_id={qa_id} use_mode={use_mode}",
        file=sys.stderr,
        flush=True,
    )
    if error is not None:
        print(
            "  error="
            f"{redact_sensitive_text(error, *redaction_values)}",
            file=sys.stderr,
            flush=True,
        )
    if retrieved is not None:
        print(
            "  retrieved_turn_ids="
            f"{[item.get('turn_id') for item in retrieved]}",
            file=sys.stderr,
            flush=True,
        )
        for item in retrieved:
            turn = item.get("turn_data", {})
            print(
                "  turn="
                f"{turn.get('turn_id')} session={turn.get('session_id')} "
                f"images={len(turn.get('image_paths') or [])} "
                f"audio={len(turn.get('voice_paths') or [])} "
                f"has_voice_caption={bool(turn.get('voice_caption'))}",
                file=sys.stderr,
                flush=True,
            )
    if isinstance(memory_content, list):
        print(
            f"  memory_blocks={_block_type_counts(memory_content)}",
            file=sys.stderr,
            flush=True,
        )
    if isinstance(question_content, list):
        print(
            f"  question_blocks={_block_type_counts(question_content)}",
            file=sys.stderr,
            flush=True,
        )


def _print_mm_input_context(
    *,
    qa: dict,
    variant: str,
    p_id: int,
    retrieved: list[dict],
    memory_content: list[dict],
    question_content: list[dict],
) -> None:
    print(
        f"[MM INPUT] variant={variant} p{p_id} qa_id={qa.get('qa_id', '')} "
        f"retrieved={[item.get('turn_id') for item in retrieved]} "
        f"memory_blocks={_block_type_counts(memory_content)} "
        f"question_blocks={_block_type_counts(question_content)}",
        flush=True,
    )


def process_one_qa(
    qa: dict,
    retriever: MemoryRetriever,
    omni: OmniClient,
    sessions: list[dict],
    use_mode: str,
    top_k: int,
    with_reasoning: bool = False,
    variant: str = "",
    p_id: int = -1,
    verbose_errors: bool = True,
    log_mm_inputs: bool = False,
    qa_retries: int = 3,
    qa_retry_base_wait: float = 2.0,
) -> dict:
    """处理单个 QA: 检索 → 格式化 → 回答。"""

    retrieved: list[dict] = []
    memory_content: Any = None
    question_content: Any = None
    try:
        # 1. 检索 + evidence 评估
        retrieval_result = retriever.retrieve_with_evidence_eval(
            qa,
            sessions,
            k_list=RETRIEVAL_TOP_K_LIST,
        )
        retrieved = retrieval_result["retrieved"][:top_k]

        # 2. 根据 use_mode 格式化 memory 和 question
        if use_mode == "text":
            memory_content = format_memory_text(retrieved)
            question_content = format_question_text(qa)
        else:
            mm_memory = format_memory_multimodal(
                retrieved,
                data_dir=retriever.data_dir,
            )
            memory_content = mm_memory["content_blocks"]
            question_content = format_question_multimodal(
                qa,
                retriever.data_dir,
            )
            if log_mm_inputs:
                _print_mm_input_context(
                    qa=qa,
                    variant=variant,
                    p_id=p_id,
                    retrieved=retrieved,
                    memory_content=memory_content,
                    question_content=question_content,
                )

        # 3. 调用 LLM。OmniClient 内部有 provider 级重试；这里再做 QA 级
        # retry，处理最终仍返回 [ERROR] 429/限流的情况。
        llm_result = {}
        last_error_text = ""
        for attempt in range(qa_retries + 1):
            llm_result = omni.answer_qa(
                qa,
                memory_content,
                question_content,
                use_mode,
                with_reasoning=with_reasoning,
            )
            raw_response = str(llm_result.get("raw_response", ""))
            if not raw_response.startswith("[ERROR]"):
                break
            if not _is_retryable_error_text(raw_response) or attempt >= qa_retries:
                break
            last_error_text = raw_response
            wait = _retry_wait(qa_retry_base_wait, attempt)
            print(
                f"[QA RETRY] variant={variant} p{p_id} qa_id={qa.get('qa_id', '')} "
                f"attempt={attempt + 1}/{qa_retries} wait={wait:.1f}s "
                f"error={last_error_text[:300]}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    except Exception as e:
        if _is_retryable_error_text(str(e)):
            for attempt in range(qa_retries):
                wait = _retry_wait(qa_retry_base_wait, attempt)
                print(
                    f"[QA RETRY] variant={variant} p{p_id} qa_id={qa.get('qa_id', '')} "
                    f"exception attempt={attempt + 1}/{qa_retries} wait={wait:.1f}s "
                    f"error={redact_sensitive_text(e, omni.api_key, omni.api_base)[:300]}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
                try:
                    return process_one_qa(
                        qa,
                        retriever,
                        omni,
                        sessions,
                        use_mode,
                        top_k,
                        with_reasoning=with_reasoning,
                        variant=variant,
                        p_id=p_id,
                        verbose_errors=verbose_errors,
                        log_mm_inputs=log_mm_inputs,
                        qa_retries=0,
                        qa_retry_base_wait=qa_retry_base_wait,
                    )
                except Exception as retry_e:  # noqa: BLE001
                    e = retry_e
                    if not _is_retryable_error_text(str(e)):
                        break
        if verbose_errors:
            _print_error_context(
                stage="process_one_qa_exception",
                qa=qa,
                variant=variant,
                p_id=p_id,
                use_mode=use_mode,
                retrieved=retrieved,
                memory_content=memory_content,
                question_content=question_content,
                error=e,
                redaction_values=(omni.api_key, omni.api_base),
            )
        raise

    if verbose_errors and str(llm_result.get("raw_response", "")).startswith("[ERROR]"):
        _print_error_context(
            stage="llm_error_response",
            qa=qa,
            variant=variant,
            p_id=p_id,
            use_mode=use_mode,
            retrieved=retrieved,
            memory_content=memory_content,
            question_content=question_content,
            error=llm_result.get("raw_response", ""),
            redaction_values=(omni.api_key, omni.api_base),
        )

    # 4. 组装结果
    result = {
        "qa_uid": _qa_uid(qa, p_id),
        "qa_id": qa.get("qa_id", ""),
        "question": qa.get("question", ""),
        "answer": qa.get("answer", ""),
        "model_answer": llm_result["model_answer"],
        "raw_response": llm_result["raw_response"],
        "correct": llm_result["model_answer"] == qa.get("answer", ""),
        "point": qa.get("point", ""),
        "qa_type": qa.get("qa_type", ""),
        "category": qa.get("category", ""),
        "retrieved_turn_ids": [r["turn_id"] for r in retrieved],
        "retrieved_scores": [r["score"] for r in retrieved],
        "retrieved_items": [
            {
                "turn_id": r["turn_id"],
                "items": r.get("retrieved_items", []),
            }
            for r in retrieved
            if r.get("retrieved_items")
        ],
        "clue": qa.get("clue", []),
        "matched_session_ids": qa.get("matched_session_ids", []),
        "recall@k": retrieval_result["recall@k"],
        "precision@k": retrieval_result["precision@k"],
    }

    if with_reasoning:
        result["reasoning"] = llm_result.get("reasoning", "")
        result["reasoning_sessions"] = llm_result.get("reasoning_sessions", [])

    return result


def run_variant_profile(
    variant: str,
    p_id: int,
    encoder: Any,
    omni: OmniClient,
    args: argparse.Namespace,
) -> None:
    """运行一个 variant × profile 的完整实验。"""
    index_mode = _variant_index_mode(variant, args.memory_index_mode)
    use_mode = _variant_use_mode(variant)

    json_path = args.profile_files[p_id]
    if not json_path.exists():
        print(f"  [WARN] {json_path} not found, skipping")
        return

    print(f"\n  Loading profile p{p_id} ...")
    profile_data = load_profile(json_path)
    sessions = profile_data["sessions"]
    qas = profile_data["qas"]

    if args.sample:
        qas = qas[: args.sample]

    eval_model = _resolve_eval_model(args)
    result_path = (
        _eval_result_root(
            args.result_dir,
            args.eval_provider,
            eval_model,
            args.result_subdir,
        )
        / variant
        / f"p{p_id}_results.json"
    )

    # 断点续跑
    existing = _load_existing_results(result_path, p_id) if args.resume else {}
    pending_qas = [q for q in qas if _qa_uid(q, p_id) not in existing]

    if args.resume:
        _print_resume_summary(
            qas=qas,
            existing=existing,
            pending_qas=pending_qas,
            p_id=p_id,
        )

    # 加载 embedding 索引
    cache_dir = _embedding_cache_dir(
        args.embedding_dir,
        args.embedding_provider,
        index_mode,
        p_id,
    )

    print(f"  Loading index ({index_mode}) from {cache_dir} ...")
    index = MemoryIndex.load(cache_dir, index_mode, sessions)

    if index_mode == "multimodal":
        # 旧分模态 MM-Index 仅支持 same-modality separate query。
        query_mode = "separate"
    else:
        # Text-Index 与 unified shared-space index 都支持 composed query。
        query_mode = args.query_embedding_mode
    retrieval_cache_dir = (
        args.retrieval_cache_dir
        / args.embedding_provider
        / index_mode
        / f"query_{query_mode}"
        / f"p{p_id}"
    )
    query_cache_dir = (
        args.query_cache_dir
        / args.embedding_provider
        / index_mode
        / f"query_{query_mode}"
        / f"p{p_id}"
    )
    retriever = MemoryRetriever(
        index,
        encoder,
        top_k=args.top_k,
        cache_dir=retrieval_cache_dir,
        query_cache_dir=query_cache_dir,
        data_dir=args.data_dir,
        query_embedding_mode=query_mode,
    )

    print(f"  Index loaded: {len(index)} turns")
    print(f"  Retrieval cache: {retrieval_cache_dir}")
    print(f"  Query embedding mode: {query_mode}")
    print(f"  Result checkpoint: every {args.checkpoint_every} QA(s)")

    # 处理 QA
    result_by_uid = dict(existing)
    completed_since_save = 0

    if args.max_workers <= 1:
        for qa in tqdm(pending_qas, desc=f"  {variant}/p{p_id}"):
            result = process_one_qa(
                qa,
                retriever,
                omni,
                sessions,
                use_mode,
                args.top_k,
                with_reasoning=args.with_reasoning,
                variant=variant,
                p_id=p_id,
                verbose_errors=args.verbose_errors,
                log_mm_inputs=args.log_mm_inputs,
                qa_retries=args.qa_retries,
                qa_retry_base_wait=args.qa_retry_base_wait,
            )
            result_by_uid[_qa_uid(qa, p_id)] = result
            completed_since_save += 1

            completed_since_save = _maybe_save_checkpoint(
                _ordered_results(qas, result_by_uid, p_id),
                result_path,
                completed_since_save,
                args.checkpoint_every,
            )

    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_one_qa,
                    qa,
                    retriever,
                    omni,
                    sessions,
                    use_mode,
                    args.top_k,
                    args.with_reasoning,
                    variant,
                    p_id,
                    args.verbose_errors,
                    args.log_mm_inputs,
                    args.qa_retries,
                    args.qa_retry_base_wait,
                ): qa
                for qa in pending_qas
            }

            pbar = tqdm(total=len(pending_qas), desc=f"  {variant}/p{p_id}")

            for future in as_completed(futures):
                try:
                    result = future.result()
                    qa = futures[future]
                    result_by_uid[_qa_uid(qa, p_id)] = result
                    completed_since_save += 1

                except Exception as e:  # noqa: BLE001
                    qa = futures[future]
                    safe_error = redact_sensitive_text(e, omni.api_key, omni.api_base)
                    print(f"  [ERROR] {qa.get('qa_id')}: {safe_error}")
                    result_by_uid[_qa_uid(qa, p_id)] = {
                        "qa_uid": _qa_uid(qa, p_id),
                        "qa_id": qa.get("qa_id", ""),
                        "question": qa.get("question", ""),
                        "answer": qa.get("answer", ""),
                        "model_answer": "",
                        "correct": False,
                        "point": qa.get("point", ""),
                        "qa_type": qa.get("qa_type", ""),
                        "category": qa.get("category", ""),
                        "error": safe_error,
                    }
                    completed_since_save += 1

                pbar.update(1)

                completed_since_save = _maybe_save_checkpoint(
                    _ordered_results(qas, result_by_uid, p_id),
                    result_path,
                    completed_since_save,
                    args.checkpoint_every,
                )

            pbar.close()

    results = _ordered_results(qas, result_by_uid, p_id)
    _save_results(results, result_path)

    correct = sum(1 for r in results if r.get("correct"))
    total = len(results)
    acc = correct / total * 100 if total > 0 else 0.0

    print(f"  Results saved: {result_path}")
    print(f"  Accuracy: {correct}/{total} = {acc:.1f}%")


def main() -> None:
    args = parse_args()
    _configure_runtime_paths(args)

    print("RQ3 Experiment")
    print(f"  Variants: {args.variants}")
    print(f"  Top-k: {args.top_k}")
    print(f"  Embedding provider: {args.embedding_provider}")
    print(f"  Memory index mode: {args.memory_index_mode}")
    print(f"  Query embedding mode: {args.query_embedding_mode}")
    print(f"  Eval provider: {args.eval_provider}")
    print(f"  Eval model: {_resolve_eval_model(args)}")
    print(
        "  Result root: "
        f"{_eval_result_root(args.result_dir, args.eval_provider, _resolve_eval_model(args), args.result_subdir)}"
    )
    print(f"  Device: {args.device}")
    print(f"  With reasoning: {args.with_reasoning}")
    if (
        args.memory_index_mode == "multimodal"
        and args.query_embedding_mode != "separate"
        and any(variant.startswith("M") for variant in args.variants)
    ):
        print(
            "  [WARN] Legacy multimodal index only supports separate query; "
            "MT/MM will use query mode 'separate'."
        )
    if args.eval_provider == "aliyun" and args.max_workers > 1:
        print(
            "  [WARN] Aliyun omni is sensitive to burst traffic; "
            "use --max-workers 1 if you see rate-limit errors."
        )

    print("\nLoading embedding encoder ...")
    encoder = create_encoder(
        provider=args.embedding_provider,
        device=args.device,
        gemini_api_key=args.gemini_api_key,
        gemini_api_base=args.gemini_api_base,
        gemini_model=args.gemini_model,
        gemini_embedding_dim=args.gemini_embedding_dim,
        data_dir=args.data_dir,
    )

    omni_kwargs = {"provider": args.eval_provider, "model": _resolve_eval_model(args)}
    if args.omni_api_base:
        omni_kwargs["api_base"] = args.omni_api_base
    if args.omni_api_key:
        omni_kwargs["api_key"] = args.omni_api_key

    omni = OmniClient(**omni_kwargs)
    print("Clients ready.\n")

    if args.profiles is not None:
        profile_indices = args.profiles
    else:
        profile_indices = list(range(len(args.profile_files)))

    for variant in args.variants:
        print(f"\n{'=' * 60}")
        print(
            f"Variant: {variant} "
            f"(index={_variant_index_mode(variant, args.memory_index_mode)}, "
            f"use={_variant_use_mode(variant)})"
        )
        print(f"{'=' * 60}")

        for p_id in profile_indices:
            t0 = time.time()
            run_variant_profile(variant, p_id, encoder, omni, args)
            elapsed = time.time() - t0
            print(f"  Time: {elapsed:.1f}s")

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
