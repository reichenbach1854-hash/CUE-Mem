"""Generate user-message speech with a reference voice per profile."""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any

from scripts.common.io import load_record_list
from scripts.common.paths import project_path, resolve_path


DEFAULT_DIALOGUE_PATH = project_path("event", "dialogue_with_anchors_rechecked.jsonl")
DEFAULT_VOICE_STYLE_DIR = project_path("profile", "voice_style")
DEFAULT_OUTPUT_DIR = project_path("event", "voice_message_005_019")
DEFAULT_MODEL_PATH = Path("Qwen3-TTS-12Hz-1.7B-Base")


def find_reference_audio(p_id: int, voice_style_dir: Path) -> str | None:
    matches = glob.glob(str(voice_style_dir / f"{p_id}_*_voice_design.wav"))
    return matches[0] if matches else None


def worker_fn(
    worker_id: int,
    gpu_id: int,
    task_queue: Queue,
    result_queue: Queue,
    output_dir: str,
    model_path: str,
) -> None:
    """Load one model per process to avoid CUDA fork issues."""

    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        result_queue.put(("fatal", "", f"missing TTS dependency: {exc}"))
        return

    print(f"[Worker {worker_id}] loading model on cuda:{gpu_id} ...")
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=f"cuda:{gpu_id}",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    print(f"[Worker {worker_id}] ready.")

    while True:
        task = task_queue.get()
        if task is None:
            break
        _, task_id, turn_idx, text, ref_audio_path = task
        out_path = Path(output_dir) / f"{task_id}_turn{turn_idx}.wav"
        if out_path.exists():
            result_queue.put(("skip", str(out_path), None))
            continue
        try:
            wavs, sample_rate = model.generate_voice_clone(
                text=text,
                language="Chinese",
                ref_audio=ref_audio_path,
                x_vector_only_mode=True,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_path, wavs[0], sample_rate)
            result_queue.put(("ok", str(out_path), None))
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            result_queue.put(("error", str(out_path), str(exc)))


def collect_tasks(dialogues: list[dict[str, Any]], voice_style_dir: Path) -> list[tuple[Any, ...]]:
    tasks: list[tuple[Any, ...]] = []
    reference_cache: dict[int, str | None] = {}
    for event_record in dialogues:
        p_id = event_record.get("p_id")
        task_id = event_record.get("task_id")
        event = event_record.get("event") or {}
        if not isinstance(p_id, int) or not task_id or not isinstance(event, dict):
            continue
        dialog = event.get("dialog") or []
        tts_turns = [
            (idx, turn)
            for idx, turn in enumerate(dialog)
            if isinstance(turn, dict)
            and turn.get("role") == "user"
            and turn.get("background_audio")
        ]
        if not tts_turns:
            continue
        if p_id not in reference_cache:
            reference_cache[p_id] = find_reference_audio(p_id, voice_style_dir)
        reference = reference_cache[p_id]
        if reference is None:
            print(f"[Skip] no reference audio for p_id={p_id}")
            continue
        for turn_idx, turn in tts_turns:
            tasks.append((p_id, task_id, turn_idx, turn.get("content", ""), reference))
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogue", default=str(DEFAULT_DIALOGUE_PATH))
    parser.add_argument("--voice-style-dir", default=str(DEFAULT_VOICE_STYLE_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers_per_gpu < 1:
        raise ValueError("--workers-per-gpu must be positive")
    dialogue_path = resolve_path(args.dialogue)
    voice_style_dir = resolve_path(args.voice_style_dir)
    output_dir = resolve_path(args.output_dir)
    model_path = resolve_path(args.model_path)
    dialogues = load_record_list(dialogue_path)
    tasks = collect_tasks(dialogues, voice_style_dir)
    print(f"[Info] total tasks: {len(tasks)}")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install torch to generate voice messages") from exc
    num_gpus = max(1, torch.cuda.device_count())
    num_workers = num_gpus * args.workers_per_gpu
    output_dir.mkdir(parents=True, exist_ok=True)

    task_queue: Queue = mp.Queue()
    result_queue: Queue = mp.Queue()
    workers: list[Process] = []
    for worker_id in range(num_workers):
        process = Process(
            target=worker_fn,
            args=(
                worker_id,
                worker_id % num_gpus,
                task_queue,
                result_queue,
                str(output_dir),
                str(model_path),
            ),
            daemon=True,
        )
        process.start()
        workers.append(process)

    for task in tasks:
        task_queue.put(task)
    for _ in workers:
        task_queue.put(None)

    done = skipped = failed = 0
    while done + skipped + failed < len(tasks):
        status, path, error = result_queue.get()
        if status == "ok":
            done += 1
            print(f"[Save] {path} ({done + skipped + failed}/{len(tasks)})")
        elif status == "skip":
            skipped += 1
            print(f"[Skip] {path} ({done + skipped + failed}/{len(tasks)})")
        else:
            failed += 1
            print(f"[Error] {path}: {error}")

    for process in workers:
        process.join()
    print(f"完成: total={len(tasks)} generated={done} skipped={skipped} failed={failed}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
