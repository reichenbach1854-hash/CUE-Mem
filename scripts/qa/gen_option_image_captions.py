"""为图片类 QA 的 A/B/C/D 选项图片生成 Caption。

读取 qa_pref_image_mcq.json、qa_rec_image_mcq.json
和 qa_entity_image_mcq.json，用 VLM 为每个选项图片生成中文描述，
分别保存到:
  - qa/pref_img_captions.json
  - qa/rec_img_captions.json
  - qa/entity_img_captions.json

用法:
    python qa/gen_option_image_captions.py
    python qa/gen_option_image_captions.py --pref-only
    python qa/gen_option_image_captions.py --rec-only
    python qa/gen_option_image_captions.py --entity-only
    python qa/gen_option_image_captions.py --sample 5
    python qa/gen_option_image_captions.py --workers 4

输出格式 (list of dict):
    [
        {
            "qa_id": "0-FoodAndDrink-0",
            "option_captions": {
                "A": "一间门面简洁的小餐馆……",
                "B": "一家设计感十足的新开小馆……",
                "C": "一家灯火通明的大型餐厅……",
                "D": "一处温馨的街角小店……"
            }
        },
        ...
    ]
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import mimetypes
import os
import re
import threading
import time
from pathlib import Path

from tqdm import tqdm

from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.common.paths import resolve_path
from scripts.qa.config import qa_path

PREF_IMAGE_QA = qa_path("qa_pref_image_mcq.json")
REC_IMAGE_QA = qa_path("qa_rec_image_mcq.json")
ENTITY_IMAGE_QA = qa_path("qa_entity_image_mcq.json")

PREF_OUTPUT = qa_path("pref_img_captions.json")
REC_OUTPUT = qa_path("rec_img_captions.json")
ENTITY_OUTPUT = qa_path("entity_img_captions.json")

# The key and optional endpoint are supplied at runtime.
API_BASE_URL = env_value("CUE_MEM_QA_IMAGE_BASE_URL") or env_value("CUE_MEM_LLM_BASE_URL")
API_KEY = env_value("CUE_MEM_QA_IMAGE_API_KEY") or env_value("CUE_MEM_LLM_API_KEY")
DEFAULT_MODEL = env_value("CUE_MEM_QA_IMAGE_MODEL", "gpt-5.4")


CAPTION_PROMPT = (
    "为这张图片生成一条简洁的图片说明。"
    "仅描述可见的细节，且仅输出纯文本，不要输出序号、标点前缀或额外说明。"
    "请用中文回答。"
)

CHECKPOINT_EVERY = 20
MAX_RETRIES = 3


def parse_profile_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    ids: set[int] = set()
    for part in raw.replace("，", ",").split(","):
        part = part.strip()
        if part:
            ids.add(int(part))
    return ids


def get_profile_id(qa: dict) -> int | None:
    for key in ("p_id", "profile_id"):
        value = qa.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)

    qa_id = str(qa.get("qa_id", ""))
    match = re.match(r"^p?(\d+)(?:-|::)", qa_id)
    if match:
        return int(match.group(1))
    return None


def filter_by_profile_ids(records: list[dict], only_profile_ids: set[int] | None) -> list[dict]:
    if only_profile_ids is None:
        return records
    return [r for r in records if get_profile_id(r) in only_profile_ids]


def resolve_image_path(raw: str) -> str:
    if not raw or not raw.strip():
        return ""
    return str(resolve_path(raw.strip().replace("\\", "/")).resolve())


def _usage_from_response(api_res) -> tuple[int, int]:
    u = getattr(api_res, "usage", None)
    pt = usage_value(u, "prompt_tokens")
    ct = usage_value(u, "completion_tokens")
    tt = getattr(u, "total_tokens", None)
    p, c = pt, ct
    if tt and p == 0 and c == 0:
        return int(tt), 0
    return p, c


def caption_one_image(
    task_id: str,
    img_path: str,
    *,
    model: str,
    api_base_url: str,
    api_key: str,
) -> tuple[str, str, int, int]:
    local_img = resolve_image_path(img_path)
    if not local_img or not os.path.exists(local_img):
        print(f"[MISS] {task_id}: {img_path} -> {local_img}")
        return task_id, "", 0, 0

    mime, _ = mimetypes.guess_type(local_img)
    if not mime:
        mime = "image/png"

    with open(local_img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:{mime};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": CAPTION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    client = openai_client(api_key=api_key, base_url=api_base_url)
    for attempt in range(MAX_RETRIES):
        try:
            api_res = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            text = message_content_to_text(api_res.choices[0].message.content).strip()
            pt, ct = _usage_from_response(api_res)
            return task_id, text, pt, ct
        except Exception as e:
            wait = 2 ** attempt
            print(f"[RETRY] {task_id} attempt {attempt + 1}: {e}; wait {wait}s")
            time.sleep(wait)

    print(f"[FAIL] {task_id}")
    return task_id, "", 0, 0


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        records = load_records(path)
        return {r["qa_id"]: r for r in records}
    except Exception as e:
        print(f"WARN: failed to load {path}: {e}")
        return {}


def atomic_save(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


def run_caption_phase(
    qa_records: list[dict],
    output_path: Path,
    *,
    model: str,
    workers: int,
    sample: int | None = None,
    label: str = "",
) -> dict:
    existing = load_existing(output_path)
    if existing:
        print(f"[{label}][resume] {len(existing)} records already done")

    tasks: list[tuple[str, str, str]] = []
    for qa in qa_records:
        qa_id = qa.get("qa_id", "")
        if qa_id in existing:
            captions = existing[qa_id].get("option_captions", {})
            if all(captions.get(l) for l in "ABCD"):
                continue
        option_images = qa.get("option_images", {})
        for letter in "ABCD":
            img_path = option_images.get(letter, "")
            if not img_path:
                continue
            tid = f"{qa_id}_{letter}"
            if qa_id in existing and existing[qa_id].get("option_captions", {}).get(letter):
                continue
            tasks.append((tid, qa_id, letter, img_path))

    if sample is not None:
        qa_ids_sampled = []
        seen = set()
        for _, qid, _, _ in tasks:
            if qid not in seen:
                seen.add(qid)
                qa_ids_sampled.append(qid)
                if len(qa_ids_sampled) >= sample:
                    break
        tasks = [(tid, qid, l, p) for tid, qid, l, p in tasks if qid in set(qa_ids_sampled)]

    if not tasks:
        print(f"[{label}] All captions already generated, nothing to do")
        return {"requests": 0, "successful": 0, "prompt_tokens": 0, "completion_tokens": 0}

    print(f"[{label}] {len(tasks)} images to caption ({len(set(t[1] for t in tasks))} QA items)")

    result_map: dict[str, dict] = dict(existing)
    lock = threading.Lock()
    total_pt = total_ct = 0
    completed = 0
    successful = 0

    future_to_meta: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for tid, qa_id, letter, img_path in tasks:
            fut = ex.submit(
                caption_one_image,
                tid,
                img_path,
                model=model,
                api_base_url=API_BASE_URL,
                api_key=API_KEY,
            )
            future_to_meta[fut] = (tid, qa_id, letter)

        for fut in tqdm(
            concurrent.futures.as_completed(future_to_meta),
            total=len(future_to_meta),
            desc=f"{label} Caption",
        ):
            _, caption, pt, ct = fut.result()
            tid, qa_id, letter = future_to_meta[fut]

            with lock:
                total_pt += pt
                total_ct += ct
                if caption:
                    successful += 1
                    if qa_id not in result_map:
                        result_map[qa_id] = {"qa_id": qa_id, "option_captions": {}}
                    result_map[qa_id]["option_captions"][letter] = caption
                completed += 1
                if completed % CHECKPOINT_EVERY == 0:
                    sorted_records = sorted(result_map.values(), key=lambda r: r["qa_id"])
                    atomic_save(sorted_records, output_path)
                    tqdm.write(f"[{label}] checkpoint ({completed}/{len(tasks)})")

    sorted_records = sorted(result_map.values(), key=lambda r: r["qa_id"])
    atomic_save(sorted_records, output_path)

    summary = {
        "model": model,
        "base_url_configured_at_runtime": bool(API_BASE_URL),
        "requests": len(tasks),
        "successful": successful,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "total_tokens": total_pt + total_ct,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="为图片类 QA 选项图片生成 Caption",
    )
    parser.add_argument("--model", type=str, default=None,
                        help=f"VLM 模型名（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--workers", type=int, default=8, help="并发数")
    parser.add_argument("--sample", type=int, default=None,
                        help="每类仅处理前 N 个 QA（按 qa_id 排序）")
    parser.add_argument(
        "--only_profile_ids",
        type=str,
        default=None,
        help="只处理指定 p_id，逗号分隔，例如 5,6,7；已有输出会按 qa_id 保留并合并",
    )
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--pref-only", action="store_true", help="仅处理 preference 图片 QA")
    g.add_argument("--rec-only", action="store_true", help="仅处理 recommendation 图片 QA")
    g.add_argument("--entity-only", action="store_true", help="仅处理 entity 图片 QA")
    args = parser.parse_args()

    model = (args.model or "").strip() or DEFAULT_MODEL
    only_profile_ids = parse_profile_ids(args.only_profile_ids)
    print(f"[config] model={model}  base_url_configured={bool(API_BASE_URL)}")
    print(f"[config] workers={args.workers}  sample={args.sample}")
    if only_profile_ids is not None:
        print(f"[config] only_profile_ids={sorted(only_profile_ids)}")

    run_pref = not args.rec_only and not args.entity_only
    run_rec = not args.pref_only and not args.entity_only
    run_entity = not args.pref_only and not args.rec_only

    if run_pref:
        if PREF_IMAGE_QA.exists():
            with open(PREF_IMAGE_QA, "r", encoding="utf-8") as f:
                pref_records = json.load(f)
            total = len(pref_records)
            pref_records = filter_by_profile_ids(pref_records, only_profile_ids)
            print(f"\nLoaded {total} preference image QA records; selected={len(pref_records)}")
            summary = run_caption_phase(
                pref_records, PREF_OUTPUT,
                model=model, workers=args.workers, sample=args.sample,
                label="pref",
            )
            print(f"[pref] Done: {summary['successful']}/{summary['requests']} ok, "
                  f"tokens={summary['total_tokens']} -> {PREF_OUTPUT}")
        else:
            print(f"WARN: {PREF_IMAGE_QA} not found, skip preference")

    if run_rec:
        if REC_IMAGE_QA.exists():
            with open(REC_IMAGE_QA, "r", encoding="utf-8") as f:
                rec_records = json.load(f)
            total = len(rec_records)
            rec_records = filter_by_profile_ids(rec_records, only_profile_ids)
            print(f"\nLoaded {total} recommendation image QA records; selected={len(rec_records)}")
            summary = run_caption_phase(
                rec_records, REC_OUTPUT,
                model=model, workers=args.workers, sample=args.sample,
                label="rec",
            )
            print(f"[rec] Done: {summary['successful']}/{summary['requests']} ok, "
                  f"tokens={summary['total_tokens']} -> {REC_OUTPUT}")
        else:
            print(f"WARN: {REC_IMAGE_QA} not found, skip recommendation")

    if run_entity:
        if ENTITY_IMAGE_QA.exists():
            with open(ENTITY_IMAGE_QA, "r", encoding="utf-8") as f:
                entity_records = json.load(f)
            total = len(entity_records)
            entity_records = filter_by_profile_ids(entity_records, only_profile_ids)
            print(f"\nLoaded {total} entity image QA records; selected={len(entity_records)}")
            summary = run_caption_phase(
                entity_records, ENTITY_OUTPUT,
                model=model, workers=args.workers, sample=args.sample,
                label="entity",
            )
            print(f"[entity] Done: {summary['successful']}/{summary['requests']} ok, "
                  f"tokens={summary['total_tokens']} -> {ENTITY_OUTPUT}")
        else:
            print(f"WARN: {ENTITY_IMAGE_QA} not found, skip entity")

    print("\nAll done.")


if __name__ == "__main__":
    main()
