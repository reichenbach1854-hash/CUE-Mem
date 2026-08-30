"""gen_audio_by_kling.py

读取 dialogue_000_002.jsonl 中每条对话的 background_audio 字段，
调用 Kling 文生音效 API 批量生成背景音频，并输出与
background_audio_manifest.json 格式兼容的索引文件。

流程：
    1. 扫描所有 dialog 消息，收集唯一的 background_audio prompt
    2. 对每个 prompt，先检查本地缓存（event/background_audio/）
       - 命中 → 直接复用
       - 未命中 → 调用 Kling API 生成并下载
    3. 写出 manifest 文件，供 gen_audio_mix.py 合成使用
"""

import concurrent.futures
import json
import os
import re
import time

import requests
from tqdm import tqdm

from scripts.common.llm import required_env
from scripts.common.paths import project_path, resolve_path

# ──────────────────────────────────────────────────────────────────────────────
# 路径配置
# ──────────────────────────────────────────────────────────────────────────────
DIALOGUE_PATH = project_path("event", "dialogue_000_019_with_anchors.jsonl")
AUDIO_DIR = project_path("event", "background_audio")
MANIFEST_PATH = project_path("event", "background_audio_manifest_kling.json")

# Kling 凭据和服务地址只从环境变量读取：
# CUE_MEM_KLING_ACCESS_KEY / CUE_MEM_KLING_SECRET_KEY / CUE_MEM_KLING_BASE_URL

AUDIO_DURATION = 5.0    # 秒，范围 [3.0, 10.0]，精度 0.1
POLL_INTERVAL  = 3      # 轮询间隔（秒）
POLL_TIMEOUT   = 180    # 单个任务最长等待时间（秒）
DEFAULT_WORKERS = 3     # Kling 并发任务数；过高可能触发限流或网络错误


def parse_profile_id_filter(raw_values: list[str] | None) -> set[int] | None:
    """Parse --only_profile_ids values such as '5,6,7' or '5 6 7'."""
    if not raw_values:
        return None
    result: set[int] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                result.add(int(part))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Token / 鉴权
# ──────────────────────────────────────────────────────────────────────────────
def get_kling_token() -> str:
    """生成 Kling JWT Token（有效期 30 分钟）。"""
    import jwt

    payload = {
        "iss": required_env("CUE_MEM_KLING_ACCESS_KEY"),
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5,
    }
    return jwt.encode(
        payload,
        required_env("CUE_MEM_KLING_SECRET_KEY"),
        algorithm="HS256",
        headers={"alg": "HS256", "typ": "JWT"},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 文件名工具
# ──────────────────────────────────────────────────────────────────────────────
def sanitize_filename(prompt: str, max_len: int = 80) -> str:
    """将 prompt 转为合法文件名（不含扩展名）。保留中文字符。"""
    name = re.sub(r'[\\/*?:"<>|]', "", prompt)   # 去掉路径非法字符
    name = re.sub(r'\s+', "_", name.strip())       # 空白 → 下划线
    return name[:max_len]


def audio_path_for_prompt(prompt: str) -> str:
    """返回给定 prompt 对应的本地音频文件路径（.mp3）。"""
    return os.path.join(AUDIO_DIR, sanitize_filename(prompt) + ".mp3")


# ──────────────────────────────────────────────────────────────────────────────
# Kling API 调用
# ──────────────────────────────────────────────────────────────────────────────
def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def create_audio_task(prompt: str, token: str) -> str | None:
    """POST 创建文生音效任务，返回 task_id；失败返回 None。"""
    url = f"{required_env('CUE_MEM_KLING_BASE_URL').rstrip('/')}/v1/audio/text-to-audio"
    body = {"prompt": prompt, "duration": AUDIO_DURATION}
    try:
        resp = requests.post(url, headers=_headers(token), json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != 0:
            print(f"    [API ERR] {data.get('message')}")
            return None
        return data["data"]["task_id"]
    except Exception as exc:
        print(f"    [ERR] create_audio_task: {exc}")
        return None


def poll_audio_task(task_id: str, token: str) -> str | None:
    """轮询任务直到完成；返回 mp3 下载 URL，超时/失败返回 None。"""
    url = f"{required_env('CUE_MEM_KLING_BASE_URL').rstrip('/')}/v1/audio/text-to-audio/{task_id}"
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        try:
            resp = requests.get(url, headers=_headers(token), timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                print(f"    [API ERR] {data.get('message')}")
                return None
            status = data["data"]["task_status"]
            if status == "succeed":
                audios = (data["data"].get("task_result") or {}).get("audios", [])
                return audios[0].get("url_mp3") if audios else None
            if status == "failed":
                print(f"    [FAIL] task_status_msg: {data['data'].get('task_status_msg')}")
                return None
            time.sleep(POLL_INTERVAL)
        except Exception as exc:
            print(f"    [ERR] poll_audio_task: {exc}")
            return None
    print(f"    [TIMEOUT] task {task_id} exceeded {POLL_TIMEOUT}s")
    return None


def download_audio(url: str, save_path: str) -> bool:
    """从 URL 下载音频并保存到 save_path。"""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    [ERR] download_audio: {exc}")
        return False


def generate_audio_for_prompt(prompt: str, save_path: str) -> bool:
    """完整流程：创建任务 → 轮询 → 下载。返回是否成功。"""
    token = get_kling_token()
    task_id = create_audio_task(prompt, token)
    if not task_id:
        return False
    print(f"    task_id={task_id}, 轮询中...")
    token = get_kling_token()          # 刷新 token，防止长轮询时过期
    mp3_url = poll_audio_task(task_id, token)
    if not mp3_url:
        return False
    return download_audio(mp3_url, save_path)


def process_prompt_audio(prompt: str, save_path: str) -> tuple[str, str, str]:
    """
    处理单个 background_audio prompt。

    Returns:
        (prompt, status, save_path)
        status: cached | ok | failed
    """
    if os.path.exists(save_path):
        return prompt, "cached", save_path

    ok = generate_audio_for_prompt(prompt, save_path)
    return prompt, "ok" if ok else "failed", save_path


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────
def main(
    sample: int = 0,
    workers: int = DEFAULT_WORKERS,
    only_profile_ids: list[str] | None = None,
    dialogue_path: str | None = None,
    audio_dir: str | None = None,
    manifest_path: str | None = None,
):
    """
    Args:
        sample: 仅生成前 N 个唯一 prompt 的音频（0 = 全量）。
                可通过命令行参数 --sample N 指定，例如：
                    python gen_audio_by_kling.py --sample 5
        workers: 并发生成 prompt 数量。
        only_profile_ids: 仅处理指定 p_id；未选中的旧 manifest 条目会保留。
    """
    global DIALOGUE_PATH, AUDIO_DIR, MANIFEST_PATH
    if dialogue_path is not None:
        DIALOGUE_PATH = resolve_path(dialogue_path)
    if audio_dir is not None:
        AUDIO_DIR = resolve_path(audio_dir)
    if manifest_path is not None:
        MANIFEST_PATH = resolve_path(manifest_path)
    if workers < 1:
        raise ValueError(f"--workers 必须 >= 1，当前为 {workers}")
    only_profile_id_set = parse_profile_id_filter(only_profile_ids)
    os.makedirs(AUDIO_DIR, exist_ok=True)

    print(f"[1/3] 加载对话文件: {DIALOGUE_PATH}")
    with open(DIALOGUE_PATH, "r", encoding="utf-8") as f:
        events_data = json.load(f)
    if only_profile_id_set is not None:
        print(f"    ONLY_PROFILE_IDS = {sorted(only_profile_id_set)}")

    # ── 收集所有 background_audio 信息 ────────────────────────────────────────
    # prompt_to_path: 全局去重，prompt → 本地音频路径
    prompt_to_path: dict[str, str] = {}
    # manifest_map: task_id → manifest 条目
    manifest_map: dict[str, dict] = {}

    for entry in events_data:
        task_id = entry.get("task_id", "")
        p_id    = entry.get("p_id", -1)
        if only_profile_id_set is not None and p_id not in only_profile_id_set:
            continue
        dialog  = entry.get("event", {}).get("dialog", [])
        if not dialog:
            continue

        # 同一 task 内按 prompt 聚合 turn 索引
        prompt_turns: dict[str, list[int]] = {}
        for turn_idx, msg in enumerate(dialog):
            bg = msg.get("background_audio")
            if not bg:
                continue
            prompt_turns.setdefault(bg, []).append(turn_idx)
            prompt_to_path[bg] = audio_path_for_prompt(bg)

        if not prompt_turns:
            continue

        bg_path_list = [
            {
                "query": prompt,
                "path": audio_path_for_prompt(prompt),
                "tts_turn_indices": turns,
            }
            for prompt, turns in prompt_turns.items()
        ]

        manifest_map[task_id] = {
            "p_id": p_id,
            "task_id": task_id,
            "background_audio_path": bg_path_list,
            "status": "pending",
        }

    print(f"    共 {len(prompt_to_path)} 个唯一 prompt，涉及 {len(manifest_map)} 个 task")

    # ── 按 sample 限制处理数量 ─────────────────────────────────────────────────
    prompts_to_process = dict(
        list(prompt_to_path.items())[:sample] if sample > 0 else prompt_to_path
    )
    if sample > 0:
        print(f"    [SAMPLE 模式] 仅处理前 {sample} 个 prompt（共 {len(prompt_to_path)} 个）")

    # ── 并发逐 prompt 生成或复用音频 ─────────────────────────────────────────
    print(f"\n[2/3] 生成背景音频（保存至 {AUDIO_DIR}），workers={workers}")
    success_prompts: set[str] = set()
    failed_prompts:  set[str] = set()
    cached_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(process_prompt_audio, prompt, save_path): (prompt, save_path)
            for prompt, save_path in prompts_to_process.items()
        }
        for future in tqdm(
            concurrent.futures.as_completed(future_map),
            total=len(future_map),
            desc="Kling audio",
        ):
            prompt, save_path = future_map[future]
            try:
                prompt, status, save_path = future.result()
            except Exception as exc:
                tqdm.write(f"  [FAIL] {prompt!r} | unexpected error: {exc}")
                failed_prompts.add(prompt)
                continue

            if status == "cached":
                tqdm.write(f"  [CACHE] {prompt!r}")
                success_prompts.add(prompt)
                cached_count += 1
            elif status == "ok":
                tqdm.write(f"    [OK]  {prompt!r} → {save_path}")
                success_prompts.add(prompt)
            else:
                tqdm.write(f"    [FAIL] 无法生成音频: {prompt!r}")
                failed_prompts.add(prompt)

    # ── 更新 manifest status 并写出 ───────────────────────────────────────────
    print(f"\n[3/3] 写出 manifest: {MANIFEST_PATH}")
    for entry in manifest_map.values():
        prompts_in_entry = {bg["query"] for bg in entry["background_audio_path"]}
        ok_cnt = sum(1 for p in prompts_in_entry if p in success_prompts)
        if ok_cnt == len(prompts_in_entry):
            entry["status"] = "ok"
        elif ok_cnt > 0:
            entry["status"] = "partial"
        else:
            entry["status"] = "failed"

    manifest = list(manifest_map.values())
    if only_profile_id_set is not None and os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            old_manifest = json.load(f)
        kept_old = []
        for entry in old_manifest:
            old_pid = entry.get("p_id")
            if old_pid in only_profile_id_set:
                continue
            kept_old.append(entry)
        manifest = kept_old + manifest
        print(
            f"    增量 manifest: 保留旧条目 {len(kept_old)}，"
            f"写入/替换本次条目 {len(manifest_map)}"
        )
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n=== 完成 ===")
    print(f"  唯一 prompt 总数  : {len(prompt_to_path)}")
    print(f"  本地缓存复用      : {cached_count}")
    print(f"  新生成成功        : {len(success_prompts) - cached_count}")
    print(f"  生成失败          : {len(failed_prompts)}")
    print(f"  Manifest 条目数   : {len(manifest)}")
    print(f"  Manifest 路径     : {MANIFEST_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="用 Kling API 生成背景音效")
    parser.add_argument(
        "--sample", type=int, default=0,
        metavar="N",
        help="仅生成前 N 个唯一 prompt 的音频，用于效果检验（默认 0 = 全量）",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        metavar="N",
        help=f"并发生成的 Kling 音频任务数（默认 {DEFAULT_WORKERS}）",
    )
    parser.add_argument(
        "--only_profile_ids",
        "--only-profile-ids",
        nargs="*",
        default=None,
        help="只处理指定 p_id，支持空格或逗号形式，例如 --only_profile_ids 5 6 7 或 --only_profile_ids 5,6,7",
    )
    parser.add_argument("--dialogue", default=None, help="dialogue JSON/JSONL input")
    parser.add_argument("--audio-dir", default=None, help="generated audio directory")
    parser.add_argument("--manifest", default=None, help="manifest output path")
    args = parser.parse_args()
    main(
        sample=args.sample,
        workers=args.workers,
        only_profile_ids=args.only_profile_ids,
        dialogue_path=args.dialogue,
        audio_dir=args.audio_dir,
        manifest_path=args.manifest,
    )
