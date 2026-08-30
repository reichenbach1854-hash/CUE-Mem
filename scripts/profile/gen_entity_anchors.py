"""
gen_entity_anchors.py
─────────────────────
为 profile 文件中所有含 "visual" 证据的偏好逐条生成 entity_anchor 列表，
并将结果写回到新的输出文件中。

用法：
    python gen_entity_anchors.py --input profiles_implicit_first.jsonl
    python gen_entity_anchors.py --input profiles_implicit_first.jsonl --output profiles_with_anchors.jsonl
    python gen_entity_anchors.py --input profiles_implicit_first.jsonl --sample 5
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path

from tqdm import tqdm
try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency
    def repair_json(text: str) -> str:
        return text

from scripts.common.llm import openai_client
from scripts.common.paths import project_path, resolve_path

# ─────────────────────────────────────────────────────────────────────────────
# 配置（密钥和服务地址只从运行时环境读取）
# ─────────────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("CUE_MEM_LLM_MODEL", "claude-sonnet-4-6")

CATEGORY_KEYS = [
    "FoodAndDrink", "HomeAndSpace", "BodyAndHealth",
    "HobbiesAndEntertainment", "WorkAndLearning", "MobilityAndTravel", 
]

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_ENTITY_ANCHORS = r'''你的任务是为以下人物偏好生成 entity_anchor（实体锚点）列表。

## 【entity_anchor 定义】
entity_anchor 是能在**视觉画面中直接观察到**的具体实物，用于后续生成该场景的图片描述和视觉问答。

## 【生成规则】
1. **数量**：
   - expression_type 为 "explicit" 的偏好：生成 **2个** entity_anchor。
   - expression_type 为 "implicit" 的偏好：生成 **2个** entity_anchor。
2. **必须是具体实物**，禁止以下类型：
   - ❌ 人物、宠物（猫、狗等动物）
   - ❌ 建筑整体、风景（山、海、天空等）
   - ❌ 抽象概念（整洁感、氛围、风格等）
3. **有辨识度**：entity_anchor后续是要生成定妆照，作为图片问答题的线索的，必须足够有辨识度和代表性；
4. **给出至少一种属性**：描述时能自然带出以下至少一种属性：
   - 材质（木质、陶瓷、皮革、不锈钢、棉麻……）
   - 颜色（白色、原木色、深蓝……）
   - 形状/款式（圆形、方形、细颈、折叠……）
5. **与该偏好强相关**：entity_anchor 应能直接体现偏好的内容，而非泛泛的场景道具。
6. **体现用户主动选择与长期习惯**：entity_anchor 必须是用户长期使用、反复购买、固定保留或主动布置的个人物品，能够反映该人物的偏好、审美、职业习惯或生活方式。不要选择偶然出现在场景中的临时物品、公共环境设施、一次性消耗品或无法体现用户意愿的背景道具，例如散落的 A4 打印纸、车厢布帘、公共交通茶杯、普通票据、路边招牌等。
7. **格式**：每个 anchor 为 **不超过 20 个汉字的短语**，例如：
   - ✅ "原木色长书桌"、"白色陶瓷手冲壶"、"深蓝色瑜伽垫"、"黑色皮面记事本"
   - ❌ "书桌"（太泛）、"整洁的厨房"（不是实物）、"猫咪玩具"（宠物相关）
8. **完整性与辨识度**：entity_anchor必须是一个完整的实体，不能出现“xxx的一角”、“xxx的一部分”等模糊的描述；
## 【输出格式】
仅输出如下 JSON 对象，不要添加任何额外内容或解释：
```json
{{"entity_anchors": ["...", "..."]}}
```

## 【待处理的偏好】
{pref_str}
'''


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def collect_visual_prefs(profile: dict) -> list[dict]:
    """从 profile 中收集所有含 'visual' 证据的偏好，附带定位信息。"""
    visual_prefs = []
    for cat in CATEGORY_KEYS:
        items = profile.get(cat, [])
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            sources = item.get("evidence_sources", [])
            if "visual" in sources:
                visual_prefs.append({
                    "cat": cat,
                    "idx": idx,
                    "subcategory": item.get("subcategory", ""),
                    "preference": item.get("preference", ""),
                    "expression_type": item.get("expression_type", ""),
                    "analysis": item.get("analysis", []),
                })
    return visual_prefs


def build_pref_str(visual_pref: dict) -> str:
    """将单条偏好格式化为注入 prompt 的文本。"""
    visual_analysis = [a for a in visual_pref["analysis"] if "视觉" in a or "（视觉）" in a]
    analysis_str = "\n    ".join(visual_analysis) if visual_analysis else "（无单独视觉分析）"
    return (
        f"expression_type: {visual_pref['expression_type']}\n"
        f"subcategory: {visual_pref['subcategory']}\n"
        f"preference: {visual_pref['preference']}\n"
        f"visual_analysis: {analysis_str}"
    )


def call_llm_for_anchor(persona_id: int, pref_order: int, visual_pref: dict) -> tuple[list[str] | None, int, int]:
    """调用 LLM，为单条 preference 返回 entity_anchors 或 None（失败）。"""
    pref_str = build_pref_str(visual_pref)
    prompt = PROMPT_ENTITY_ANCHORS.format(pref_str=pref_str)

    for attempt in range(3):
        client = openai_client()
        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                reasoning_effort="medium",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )
            result = ""
            usage_info = None
            for chunk in api_res:
                if chunk.choices and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            raw = result.strip().replace("```json", "").replace("```", "")
            parsed = json.loads(repair_json(raw))

            if not isinstance(parsed, dict):
                print(
                    f"[Persona {persona_id} Pref {pref_order}] LLM returned non-dict "
                    f"(attempt {attempt}), retrying."
                )
                continue

            anchors = parsed.get("entity_anchors", [])
            if not isinstance(anchors, list) or not 2 <= len(anchors) <= 3:
                print(
                    f"[Persona {persona_id} Pref {pref_order}] Expected 2-3 anchors, "
                    f"got {len(anchors) if isinstance(anchors, list) else 'invalid'} "
                    f"(attempt {attempt}), retrying."
                )
                continue

            pt = usage_info.prompt_tokens if usage_info else 0
            ct = usage_info.completion_tokens if usage_info else 0
            return anchors, pt, ct

        except Exception as e:
            print(f"[Persona {persona_id} Pref {pref_order}] Error on attempt {attempt}: {e}")

    print(f"[Persona {persona_id} Pref {pref_order}] Failed to generate entity anchors after 3 attempts.")
    return None, 0, 0


def inject_anchor(profile: dict, visual_pref: dict, anchors: list[str]) -> dict:
    """将单条 entity_anchors 写回 profile 对应的偏好条目中。"""
    if not anchors:
        return profile
    cat, idx = visual_pref["cat"], visual_pref["idx"]
    target = profile.get(cat, [])[idx]
    if isinstance(target, dict):
        target["entity_anchors"] = anchors

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def process_persona(profile: dict) -> tuple[dict, int, int]:
    persona_id = profile.get("id", "?")
    visual_prefs = collect_visual_prefs(profile)
    if not visual_prefs:
        return profile, 0, 0

    total_pt = 0
    total_ct = 0
    for pref_order, visual_pref in enumerate(visual_prefs):
        anchors, pt, ct = call_llm_for_anchor(persona_id, pref_order, visual_pref)
        total_pt += pt
        total_ct += ct
        if anchors is None:
            continue
        profile = inject_anchor(profile, visual_pref, anchors)

    return profile, total_pt, total_ct


def parse_args():
    parser = argparse.ArgumentParser(description="Add entity_anchors to visual preferences in profile files.")
    parser.add_argument(
        "--input", "-i",
        default="profile/profiles_implicit_first.jsonl",
        help="Input profile file (.json or .jsonl), relative to the project root.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path, relative to the project root. Default: <input_stem>_with_anchors<ext>.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers. Default: 8.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process the first N profiles for small-batch debugging.",
    )
    parser.add_argument("--model", default=MODEL, help="LLM model name")
    return parser.parse_args()


def load_profiles(path: Path) -> list[dict]:
    def normalize_loaded_profiles(data):
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Profile file must contain a JSON object/list, got {type(data).__name__}.")

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        return normalize_loaded_profiles(json.loads(text))
    except json.JSONDecodeError as json_error:
        # Some generated profile files are pretty-printed JSON arrays with an
        # accidental extra closing bracket at EOF. Accept only that narrow case.
        decoder = json.JSONDecoder()
        try:
            data, end = decoder.raw_decode(text)
            trailing = text[end:].strip()
            if trailing and set(trailing) <= {"]"}:
                print(
                    f"Warning: ignored extra trailing closing bracket(s) in {path}: {trailing!r}"
                )
                return normalize_loaded_profiles(data)
        except json.JSONDecodeError:
            pass

        if text.lstrip().startswith("["):
            raise ValueError(
                f"Failed to parse JSON array file {path}: {json_error}"
            ) from json_error

        # 尝试 jsonl（每行一个 JSON）
        profiles = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                profiles.append(json.loads(line))
            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f"Failed to parse {path} as JSON ({json_error}) or JSONL "
                    f"(line {line_no}: {line_error})."
                ) from line_error
        return profiles


def save_profiles(profiles: list[dict], path: Path):
    path.write_text(json.dumps(profiles, indent=4, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()

    global MODEL
    MODEL = args.model
    input_path = resolve_path(args.input, project_path("profile", "profiles_implicit_first.jsonl"))
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if args.output:
        output_path = resolve_path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_with_anchors" + input_path.suffix)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    profiles = load_profiles(input_path)
    print(f"Loaded {len(profiles)} profiles.")

    if args.sample is not None:
        if args.sample <= 0:
            raise ValueError(f"--sample must be a positive integer, got {args.sample}.")
        profiles = profiles[:args.sample]
        print(f"Sample mode: processing first {len(profiles)} profiles.")

    results = []
    total_pt = 0
    total_ct = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_persona, p): p.get("id", i) for i, p in enumerate(profiles)}

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            updated_profile, pt, ct = future.result()
            results.append(updated_profile)
            total_pt += pt
            total_ct += ct

    results.sort(key=lambda x: x.get("id", 10**9))
    save_profiles(results, output_path)

    print(f"\nDone. Saved to {output_path}")
    print(f"Total Prompt Tokens   : {total_pt}")
    print(f"Total Completion Tokens: {total_ct}")
    print(f"Estimated Cost         : ${total_pt * 1e-6 + total_ct * 3e-6:.4f}")


if __name__ == "__main__":
    main()
