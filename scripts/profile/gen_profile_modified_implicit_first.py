import argparse
import json
import math
import os
import re
from tqdm import tqdm
import concurrent.futures
import random
try:
    from json_repair import repair_json
except ImportError:  # pragma: no cover - optional dependency
    def repair_json(text: str) -> str:
        return text
from pathlib import Path

from scripts.common.llm import openai_client
from scripts.common.paths import project_path, resolve_path


PERSONA_FILE_PATH = project_path("profile", "personas.jsonl")
SAVE_PROFILE_PATH = project_path("profile", "profiles_implicit_first.jsonl")
MODEL = os.environ.get("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")


TARGET_PERSONA_COUNT = 20

PERSONA_AUDIO_KEYWORDS = {
    0: ["猫叫声", "吸尘器声", "切菜声", "炒菜的声音", "咖啡机声","倒水声", "清晨鸟鸣叫的声音", "轻音乐的声音", "铅笔书写声", "自行车铃声", "闹钟声"],
    1: ["微波炉的声音", "键盘打字声", "鼠标点击声","铅笔书写声", "弹电吉他的声音", "掌声", "交谈声和说笑声", "地铁列车声", "咖啡机声", "倒水声"],
    2: ["铅笔书写声", "翻动书页的声音", "打印机声", "切菜声", "餐具碰撞声", "炒菜的声音", "倒水声", "吸尘器声", "狗叫声", "清晨鸟鸣叫的声音", "孩子们玩闹的声音"],
    3: ["清晨鸟鸣叫的声音", "铅笔书写声", "翻动书页的声音","键盘打字声", "鼠标点击声", "电话铃声", "狗叫声", "汽车启动的声音", "闹钟声", "古典音乐"],
    4: ["闹钟声", "跑步声", "打篮球的声音", "游泳水花声", "欢呼声", "打乒乓球的声音", "倒水声", "微波炉声", "摇滚歌曲","汽车发动声"],
    5: ["键盘打字声", "鼠标点击声", "汽车发动声", "酒杯相碰的声音", "汽车喇叭声", "电子舞曲", "倒啤酒的声音", "自行车铃声"],
    6: ["粉笔在黑板上书写的声音", "铅笔书写声","翻动书页的声音", "相机快门声", "爵士歌曲", "跑步声", "汽车发动声", "徒步声", "山林自然环境音"],
    7: ["咖啡机声", "沸腾声", "切菜声", "流行歌曲", "餐具碰撞声", "猫叫声", "拉小提琴的声音", "游泳水花声", "地铁列车声"],
    8: ["摇滚歌曲", "狗叫声", "微波炉声", "摩托车声", "打篮球的声音", "汽车喇叭声", "音乐现场欢呼声", "笑声", "吸尘器声", "吃零食的声音"],
    9: ["海浪声", "相机快门声", "飞机声", "行李箱轮子滚动的声音", "咀嚼声", "沸腾声", "游泳水花声", "古典音乐","鼠标点击声","铅笔书写声"],
    10: ["弹钢琴的声音", "咖啡机声", "倒水声", "打网球的声音", "键盘打字声", "鼠标点击声", "打印机声", "汽车发动声","古典音乐"],
    11: ["海浪声", "跑步声", "自行车铃声", "缝纫机的声音", "微波炉声", "狗叫声", "闹钟声", "地铁列车声", "爵士歌曲"],
    12: ["架子鼓声", "摇滚歌曲", "欢呼声", "人群欢呼声", "行李箱轮子滚动的声音", "飞机声", "跑步声", "拆外卖包装袋的声音", "黑胶唱片的音乐声"],
    13: ["电子舞曲", "游泳水花声", "跑步脚步声", "清晨鸟鸣叫的声音", "切菜声", "炒菜的声音","狗叫声", "地铁列车声", "闹钟声"],
    14: ["篮球拍地声", "键盘打字声", "鼠标点击声", "铅笔书写声","打印机声", "自行车铃声", "地铁列车声", "摇滚歌曲", "拉二胡的声音"],
    15: ["狗叫声", "相机快门声", "打网球的声音", "汽车发动声", "轻音乐的声音", "弹吉他的声音", "海浪声", "手机震动声"],
    16: ["飞机声", "电话铃声", "键盘打字声", "铅笔书写声", "沸腾声", "炒菜的声音", "微波炉声", "打乒乓球的声音","古典音乐"],
    17: ["铅笔书写声", "电话铃声", "游泳的声音", "汽车喇叭声", "弹钢琴的声音", "狗叫声", "爵士歌曲", "闹钟声"],
    18: ["跑步脚步声", "掌声", "音乐现场欢呼声", "酒杯相碰的声音", "弹吉他的声音", "相机快门声", "电子舞曲", "吸尘器声"],
    19: ["弹吉他的声音", "流行歌曲", "踢足球的声音", "翻动书页的声音", "咖啡机声", "切菜声", "倒水声", "闹钟声"],
}

PROMPT_PROFILE = '''基于给定的人设描述和MBTI类型，扩展并生成一个详细、结构化的人物档案。仅输出以下JSON格式。所有字段值必须是列表或字典，每个列表项必须是一个独立、具体、简洁的事实/特征。严格遵循字段、字段名称、顺序和嵌套结构。不要添加任何额外内容或解释。

```json
{{
  "Basic": {{
    "name": "xxx",
    "age": "xxx",
    "gender": "xxx",
    "occupation": "xxx",
    "education": "xxx",
    "voice_timbre": "xxx",
    "Relationship": [
      {{"relation": "...", "name": "...", "info": "gender;occupation;traits", "appearance": "Detailed physical description"}},
      ...
    ],
    "Pets": [
      {{"name": "...", "info": "species;breed;age;traits", "appearance": "Detailed physical description"}},
      ...
    ]
  }}
}}
```

要求：
1. 仅输出上述JSON结构。字段顺序和嵌套格式不得更改。内容必须基于人设描述，具有清晰的个体特征。
2. 所有字段值必须具体且明确。避免模糊或抽象表达。完整填写所有字段，不得留空。Basic中的"xxx"必须替换。如果人设中缺少姓名或教育等信息，请进行合理推断并补全。name字段应尽量多样且丰富。
3. "voice_timbre"字段必须用具体方式描述用户的声音风格，保持简洁但具体。
4. 若某些方面在人设描述中未明确提及，应通过合理推断进行补充，使档案完整且连贯。
5. 符合东亚文化圈的背景，且使用中文描述。

[Persona Description]
{persona}

[MBTI Type]
{mbti}
'''

# ─────────────────────────────────────────────────────────────────────────────
# 共享常量：类别说明 & JSON schema
# ─────────────────────────────────────────────────────────────────────────────
_CATEGORIES_DOC = """
## 【分类说明】各类别含义与子类别选项

### FoodAndDrink（饮食与饮品）
- 咖啡与茶 / 烹饪习惯 / 饮食风格 / 零食与甜品 / 外食偏好

### HomeAndSpace（居家与空间）
- 家居风格 / 植物与绿植 / 书架与收藏 / 整洁与收纳 / 香氛与氛围

### BodyAndHealth（身体与健康）
- 晨间习惯 / 运动方式 / 睡眠习惯 / 护肤与美妆

### HobbiesAndEntertainment（兴趣与娱乐）
- 乐器与音乐 / 影视与追剧 / 阅读 / 游戏 / 手工与创作 / 户外与自然

### WorkAndLearning（工作与学习）
- 工作环境 / 工作专注度 / 学习方式 / 职业特征

### MobilityAndTravel（出行与旅行）
- 日常通勤 / 旅行偏好 / 旅行风格 / 交通工具偏好
"""

_PREF_JSON_SCHEMA = """\
{{
  "FoodAndDrink":            [{{"subcategory":"...","preference":"...","expression_type":"...","evidence_sources":[...],"analysis":[...]}},...],
  "HomeAndSpace":            [...],
  "BodyAndHealth":           [...],
  "HobbiesAndEntertainment": [...],
  "WorkAndLearning":         [...],
  "MobilityAndTravel":       [...]
}}"""

# ─────────────────────────────────────────────────────────────────────────────
# 第一步：仅生成隐式偏好（6–8 条），无禁区参照
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_IMPLICIT_PREFS = r'''你的任务是为给定人物档案生成 **隐式偏好（implicit preferences）**。

## 【输出格式】
仅输出如下 JSON，不要添加任何额外内容或解释：
```json
{json_schema}
```

## 【数量要求】
- 全部类别合计生成 **6–8 条** 隐式偏好，分散到尽量多的不同类别中。
- 在全部隐式偏好中，`evidence_sources` 包含 `"visual"` 的条目**至少 4 条**。

## 【隐式偏好定义与规则】
1. **expression_type 必须全部为 "implicit"**。
2. **evidence_sources 绝不包含 "dialogue"**；仅使用 "audio" 和/或 "visual"。
3. 隐式偏好是用户**不会在对话中直接说出**、需要通过音频、图像边缘细节推断的习惯/倾向。
4. analysis 列表中每条说明如何通过 audio/visual 线索推断出该偏好：
   - audio 证据必须以 `(关键词)` 开头，例如 `(切菜声)清脆有节奏的切菜声暗示习惯自己下厨`
   - visual 证据示例：`（视觉）画面中桌面始终整洁，说明…`
注意：implicit preference的visual证据一定要足够隐蔽，不能太明显，只能出现在画面边缘、角落、阴影处等不显眼的位置，或者只出现物品的一部分，不要出现整个物品。
5. 对于辨识度很高的声音，如"切菜声"、"炒菜的声音"、"弹吉他的声音"、"弹吉他的声音"、"笔在纸上写字的声音"、"猫叫声"、"狗叫声"、"自行车铃声"、"翻动书页的声音"、"音乐声"等，单独出现即可辨别出来，则"evidence_sources"只出现"audio"即可，不要再加入"visual"。 只有对辨识度不高的声音，才需要"audio"和"visual"同时出现。

## 【人物与宠物一致性硬约束】
1. [User Profile] 中的 Basic、Relationship、Pets 是唯一可信的人物/宠物设定来源。
2. 严禁虚构或新增任何 Basic 信息以外的亲人、伴侣、朋友、同事、导师、孩子、宠物或动物。
3. 若要提到关系人物，必须只使用 Relationship 中已经列出的 relation/name/info；不得改变其性别、关系、职业或身份。
4. 若要提到宠物，必须只使用 Pets 中已经列出的 name/info；不得把未列出的猫、狗、鸟、兔等动物写成用户宠物。
5. 如果 Relationship 或 Pets 为空，不得生成任何“家中有某亲人/宠物陪伴”等相关偏好；音频中的动物声只能作为环境声，不得推断为用户饲养的宠物。

## 【核心约束：定向音频关键词引导】
你必须优先使用以下为当前 persona 预先分配的音频关键词来设计含 audio 的隐式偏好。

[Assigned Audio Keywords]
{assigned_audio_keywords}

规则：
- 至少覆盖已分配关键词中的一半；不要只反复使用其中 1–2 个。
- 每个关键词最多使用 1 次，禁止重复堆砌。
- 含 audio 的条目自然分散到不同主分类，不要全部堆在同一类别。
- 不要使用分配列表之外的音频关键词（除非人物设定强相关且无法用已分配关键词表达）。
- 可以 2–3 个关键词共同体现同一偏好，例如：
  - 切菜声 + 炒菜的声音 → 习惯在家自己做饭
  - 键盘打字声 + 鼠标点击声 → 长时间使用电脑
  - 飞机声 + 行李箱轮子滚动的声音 → 频繁出行
- 关键词与类别的自然对应参考：
  - 饮食：咖啡机声 / 切菜声 / 炒菜的声音 / 沸腾声 / 微波炉声 / 餐具碰撞声 / 倒水声 / 咀嚼声
  - 工作学习：键盘打字声 / 鼠标点击声 / 打印机声 / 铅笔书写声 / 粉笔声 / 电话铃声
  - 出行：地铁列车声 / 汽车发动声 / 汽车喇叭声 / 自行车铃声 / 飞机声 / 行李箱轮子滚动的声音 / 跑步声
  - 兴趣娱乐：弹吉他的声音 / 弹钢琴的声音 / 架子鼓声 / 打篮球的声音 / 打网球的声音 / 欢呼声 / 相机快门声
  - 居家自然：猫叫声 / 狗叫声 / 清晨鸟鸣叫的声音 / 吸尘器声 / 翻动书页的声音

{categories_doc}

[User Profile]
{profile_str}

[MBTI Type]
{mbti}
'''

# ─────────────────────────────────────────────────────────────────────────────
# 第二步：仅生成显式偏好（6–8 条），喂入已有隐式偏好作为"禁区"参照
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_EXPLICIT_PREFS = r'''你的任务是为给定人物档案生成 **显式偏好（explicit preferences）**。

## 【输出格式】
仅输出如下 JSON，不要添加任何额外内容或解释：
```json
{json_schema}
```

## 【数量要求】
- 全部类别合计生成 **6–8 条** 显式偏好，分散到尽量多的不同类别中。
- 在全部显式偏好中，`evidence_sources` 同时包含 `"dialogue"` 与 `"visual"` 的条目**至少 4 条**。

## 【显式偏好定义与规则】
1. **expression_type 必须全部为 "explicit"**。
2. **evidence_sources 必须包含 "dialogue"**（可同时包含 "visual"，但绝不包含 "audio"）。
3. 显式偏好是用户**会主动在对话中直接说出**的事实/喜好，例如：
   - "我喜欢手冲咖啡，每天早上必须来一杯"
   - "我觉得家里必须保持整洁，很难忍受杂乱"
   - 关系人物的存在事实（必须显式，不得仅靠音视频隐式引入）
4. analysis 列表中每条说明该偏好如何通过 dialogue 或 visual 体现：
   - dialogue 证据示例：`"她在对话中提到'…'，明确表达了…"`
   - visual 证据示例：`"（视觉）画面中出现了…，体现了…"`
5. 内容必须贴合人物档案和 MBTI，具体且不模糊。

## 【人物与宠物一致性硬约束】
1. [User Profile] 中的 Basic、Relationship、Pets 是唯一可信的人物/宠物设定来源。
2. 严禁虚构或新增任何 Basic 信息以外的亲人、伴侣、朋友、同事、导师、孩子、宠物或动物。
3. 若显式偏好涉及关系人物，必须只使用 Relationship 中已经列出的 relation/name/info；不得改变其性别、关系、职业或身份。
4. 若显式偏好涉及宠物，必须只使用 Pets 中已经列出的 name/info；不得把未列出的猫、狗、鸟、兔等动物写成用户宠物。
5. 如果 Relationship 或 Pets 为空，不得生成任何“我有某亲人/宠物”“家里有某宠物”等事实型偏好。

## 【最重要约束：与隐式偏好严格区分】

以下是**已经生成好的隐式偏好**，显式偏好必须与这些内容**完全无关，不得相似或高度关联**：

[已生成的隐式偏好]
{implicit_prefs_str}

**禁止出现的情况（举例）：**
- 隐式偏好已有"习惯在家自己下厨（切菜声/炒菜声推断）" → 显式偏好不得出现"喜欢烹饪"或"喜欢在家做饭"等相关内容
- 隐式偏好已有"每天固定煮咖啡（咖啡机声推断）" → 显式偏好不得出现任何咖啡相关内容
- 隐式偏好已有"频繁乘坐地铁通勤（地铁列车声推断）" → 显式偏好不得出现"喜欢公共交通"或通勤相关内容
- 更广泛地说：只要隐式偏好覆盖了某个**行为领域或生活习惯**，显式偏好就必须回避该领域

**正确做法：** 显式偏好应聚焦于隐式偏好未触及的生活侧面，选取用户会主动开口谈及的兴趣、价值观或喜恶。两者应覆盖互补的维度，而非对同一领域做"隐性行为 vs 口头表达"的重复。

## 【强制例外：关系人物基本信息】
- 若人物有家人/亲密关系人物，其**存在事实**即使与隐式偏好无关联，也**必须**安排为显式偏好（dialogue 来源），不受上述禁区约束。
- 该例外只适用于 Relationship 中已经列出的关系人物；不得借此新增未列出的亲人、伴侣、朋友、孩子或宠物。

{categories_doc}

[User Profile]
{profile_str}

[MBTI Type]
{mbti}
'''

AUDIO_KEYWORD_PATTERN = re.compile(r'^[\(（]([^)）]+)[\)）]')
VISUAL_ANALYSIS_LABELS = {"视觉"}
PREFERENCE_CATEGORY_KEYS = [
    "FoodAndDrink", "HomeAndSpace", "BodyAndHealth",
    "HobbiesAndEntertainment", "WorkAndLearning", "MobilityAndTravel",
]


def get_assigned_audio_keywords(persona_id: int):
    if persona_id not in PERSONA_AUDIO_KEYWORDS:
        raise KeyError(f"Missing assigned audio keywords for persona id {persona_id}")
    return PERSONA_AUDIO_KEYWORDS[persona_id]


def session_profile(persona_id, persona, mbti):
    for _ in range(3):
        usage_info = None
        prompt_tokens = 0
        completion_tokens = 0
        client = openai_client()
        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                reasoning_effort="high",
                messages=[{"role": "user", "content": PROMPT_PROFILE.format(persona=persona, mbti=mbti)}],
                stream=True,
                stream_options={"include_usage": True},
            )
            result = ""
            for chunk in api_res:
                if chunk.choices and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            response = result.strip().replace("```json", "").replace("```", "")
            response = json.loads(repair_json(response))
            response.update({"mbti": mbti, "id": persona_id, "persona": persona})

            if isinstance(response["Basic"]["name"], list):
                print(f"Warning: Persona {persona_id} has multiple names: {response['Basic']['name']}. ReMake.")
                continue

            if usage_info is not None:
                prompt_tokens += usage_info.prompt_tokens
                completion_tokens += usage_info.completion_tokens

            return response, prompt_tokens, completion_tokens

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for persona {persona_id}: {e}")
            print(response)

    return {}, 0, 0


def get_profile_str(data):
    scalar_keys = ["name", "age", "gender", "education", "occupation", "voice_timbre"]
    lines = ["Basic:"]
    for key in scalar_keys:
        lines.append(f"- {key}: {data.get(key, '未提供')}")

    relationships = data.get("Relationship", [])
    lines.append("Relationship:")
    if isinstance(relationships, list) and relationships:
        for idx, rel in enumerate(relationships, start=1):
            if not isinstance(rel, dict):
                continue
            lines.append(
                f"- {idx}. relation: {rel.get('relation', '未提供')}; "
                f"name: {rel.get('name', '未提供')}; "
                f"info: {rel.get('info', '未提供')}; "
                f"appearance: {rel.get('appearance', '未提供')}"
            )
    else:
        lines.append("- 无")

    pets = data.get("Pets", [])
    lines.append("Pets:")
    if isinstance(pets, list) and pets:
        for idx, pet in enumerate(pets, start=1):
            if not isinstance(pet, dict):
                continue
            lines.append(
                f"- {idx}. name: {pet.get('name', '未提供')}; "
                f"info: {pet.get('info', '未提供')}; "
                f"appearance: {pet.get('appearance', '未提供')}"
            )
    else:
        lines.append("- 无")

    return "\n".join(lines)


def load_existing_profiles(file_path: Path):
    if not file_path.exists():
        return []

    raw_text = file_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    try:
        loaded = json.loads(raw_text)
        if isinstance(loaded, list):
            return [item for item in loaded if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    profiles = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            profiles.append(item)
    return profiles


def normalize_profiles_by_id(profiles):
    profiles_by_id = {}
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        persona_id = profile.get("id")
        if isinstance(persona_id, int):
            profiles_by_id[persona_id] = profile
    return profiles_by_id


def save_profiles(file_path: Path, profiles_by_id: dict):
    ordered_profiles = sorted(profiles_by_id.values(), key=lambda x: x.get("id", 10**9))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(ordered_profiles, f, indent=4, ensure_ascii=False)


def count_preferences(response_dict):
    total_count = 0
    category_counts = {}
    for section in PREFERENCE_CATEGORY_KEYS:
        items = response_dict.get(section, [])
        count = len(items) if isinstance(items, list) else 0
        category_counts[section] = count
        total_count += count
    return total_count, category_counts


def validate_preference_count(response_dict, min_count, max_count):
    total_count, category_counts = count_preferences(response_dict)
    return {
        "valid": min_count <= total_count <= max_count,
        "total_count": total_count,
        "category_counts": category_counts,
        "min_count": min_count,
        "max_count": max_count,
    }


def count_preferences_with_evidence_source(response_dict, evidence_source):
    matched_count = 0
    for section in PREFERENCE_CATEGORY_KEYS:
        items = response_dict.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            evidence_sources = item.get("evidence_sources", [])
            if isinstance(evidence_sources, list) and evidence_source in evidence_sources:
                matched_count += 1
    return matched_count


def validate_visual_evidence_count(response_dict, min_visual_count):
    visual_count = count_preferences_with_evidence_source(response_dict, "visual")
    return {
        "valid": visual_count >= min_visual_count,
        "visual_count": visual_count,
        "min_visual_count": min_visual_count,
    }


def validate_generated_preferences(response_dict, min_count, max_count, min_visual_count):
    count_validation = validate_preference_count(response_dict, min_count, max_count)
    visual_validation = validate_visual_evidence_count(response_dict, min_visual_count)
    return {
        "valid": count_validation["valid"] and visual_validation["valid"],
        "count_validation": count_validation,
        "visual_validation": visual_validation,
    }


def extract_audio_keywords_from_response(response_dict):
    extracted = []
    for section in PREFERENCE_CATEGORY_KEYS:
        items = response_dict.get(section, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            evidence_sources = item.get("evidence_sources", [])
            analysis = item.get("analysis", [])
            if "audio" not in evidence_sources or not isinstance(analysis, list):
                continue
            for line in analysis:
                if not isinstance(line, str):
                    continue
                match = AUDIO_KEYWORD_PATTERN.match(line.strip())
                if match:
                    keyword = match.group(1).strip()
                    if keyword in VISUAL_ANALYSIS_LABELS:
                        continue
                    extracted.append(keyword)
    return extracted


def validate_audio_keywords(response_dict, assigned_keywords):
    used_keywords = extract_audio_keywords_from_response(response_dict)
    used_set_lower = {kw.lower() for kw in used_keywords}
    assigned_set_lower = {kw.lower() for kw in assigned_keywords}
    matched_assigned_keywords = [kw for kw in assigned_keywords if kw.lower() in used_set_lower]
    extra = [kw for kw in used_keywords if kw.lower() not in assigned_set_lower]
    required_keyword_count = math.ceil(len(assigned_keywords) / 2)
    return {
        "valid": len(matched_assigned_keywords) >= required_keyword_count and len(extra) == 0,
        "used_keywords": used_keywords,
        "matched_assigned_keywords": matched_assigned_keywords,
        "extra_keywords": extra,
        "matched_keyword_count": len(matched_assigned_keywords),
        "required_keyword_count": required_keyword_count,
    }


def format_prefs_for_prompt(prefs: dict, label: str) -> str:
    """将一组偏好格式化为易读文本，注入到另一步的 prompt 禁区中。"""
    category_labels = {
        "FoodAndDrink": "FoodAndDrink（饮食与饮品）",
        "HomeAndSpace": "HomeAndSpace（居家与空间）",
        "BodyAndHealth": "BodyAndHealth（身体与健康）",
        "HobbiesAndEntertainment": "HobbiesAndEntertainment（兴趣与娱乐）",
        "WorkAndLearning": "WorkAndLearning（工作与学习）",
        "MobilityAndTravel": "MobilityAndTravel（出行与旅行）",
    }
    lines = []
    for cat_key, cat_label in category_labels.items():
        items = prefs.get(cat_key, [])
        if not isinstance(items, list) or not items:
            continue
        lines.append(f"【{cat_label}】")
        for item in items:
            if not isinstance(item, dict):
                continue
            subcat = item.get("subcategory", "")
            pref = item.get("preference", "")
            lines.append(f"  - [{subcat}] {pref}")
    return "\n".join(lines) if lines else f"（暂无{label}）"


def session_implicit_prefs(persona_id, profile_str, mbti, assigned_keywords):
    """第一步：生成 6-8 条隐式偏好，使用定向音频关键词。"""
    prompt = PROMPT_IMPLICIT_PREFS.format(
        json_schema=_PREF_JSON_SCHEMA,
        categories_doc=_CATEGORIES_DOC,
        assigned_audio_keywords=", ".join(assigned_keywords),
        profile_str=profile_str,
        mbti=mbti,
    )
    for attempt in range(3):
        usage_info = None
        client = openai_client()
        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                reasoning_effort="high",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )
            result = ""
            for chunk in api_res:
                if chunk.choices and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            response = result.strip().replace("```json", "").replace("```", "")
            response = json.loads(repair_json(response))

            count_validation = validate_preference_count(response, 6, 8)
            if not count_validation["valid"]:
                print(
                    f"[Implicit] Persona {persona_id} preference count validation failed (attempt {attempt}). "
                    f"Expected {count_validation['min_count']}-{count_validation['max_count']}, "
                    f"got {count_validation['total_count']}. "
                    f"CategoryCounts={count_validation['category_counts']}. Retrying."
                )
                continue

            visual_validation = validate_visual_evidence_count(response, 4)
            if not visual_validation["valid"]:
                print(
                    f"[Implicit] Persona {persona_id} visual evidence validation failed (attempt {attempt}). "
                    f"Expected at least {visual_validation['min_visual_count']} visual items, "
                    f"got {visual_validation['visual_count']}. Retrying."
                )
                continue

            validation = validate_audio_keywords(response, assigned_keywords)
            if not validation["valid"]:
                print(
                    f"[Implicit] Persona {persona_id} audio keyword validation failed (attempt {attempt}). "
                    f"Matched={validation['matched_keyword_count']}/{validation['required_keyword_count']} "
                    f"AssignedMatched={validation['matched_assigned_keywords']} "
                    f"Extra={validation['extra_keywords']}. Retrying."
                )
                continue

            prompt_tokens = usage_info.prompt_tokens if usage_info else 0
            completion_tokens = usage_info.completion_tokens if usage_info else 0
            return response, prompt_tokens, completion_tokens

        except json.JSONDecodeError as e:
            print(f"[Implicit] Error decoding JSON for persona {persona_id} attempt {attempt}: {e}")

    print(f"Warning: Persona {persona_id} failed implicit pref generation after retries.")
    return {}, 0, 0


def session_explicit_prefs(persona_id, profile_str, mbti, implicit_prefs):
    """第二步：生成 6–8 条显式偏好，喂入隐式偏好作为"禁区"参照。"""
    implicit_prefs_str = format_prefs_for_prompt(implicit_prefs, "隐式偏好")
    prompt = PROMPT_EXPLICIT_PREFS.format(
        json_schema=_PREF_JSON_SCHEMA,
        categories_doc=_CATEGORIES_DOC,
        implicit_prefs_str=implicit_prefs_str,
        profile_str=profile_str,
        mbti=mbti,
    )
    for attempt in range(3):
        usage_info = None
        client = openai_client()
        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                reasoning_effort="high",
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                stream_options={"include_usage": True},
            )
            result = ""
            for chunk in api_res:
                if chunk.choices and chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            response = result.strip().replace("```json", "").replace("```", "")
            response = json.loads(repair_json(response))

            count_validation = validate_preference_count(response, 6, 8)
            if not count_validation["valid"]:
                print(
                    f"[Explicit] Persona {persona_id} preference count validation failed (attempt {attempt}). "
                    f"Expected {count_validation['min_count']}-{count_validation['max_count']}, "
                    f"got {count_validation['total_count']}. "
                    f"CategoryCounts={count_validation['category_counts']}. Retrying."
                )
                continue

            visual_validation = validate_visual_evidence_count(response, 4)
            if not visual_validation["valid"]:
                print(
                    f"[Explicit] Persona {persona_id} visual evidence validation failed (attempt {attempt}). "
                    f"Expected at least {visual_validation['min_visual_count']} visual items, "
                    f"got {visual_validation['visual_count']}. Retrying."
                )
                continue

            prompt_tokens = usage_info.prompt_tokens if usage_info else 0
            completion_tokens = usage_info.completion_tokens if usage_info else 0
            return response, prompt_tokens, completion_tokens

        except json.JSONDecodeError as e:
            print(f"[Explicit] Error decoding JSON for persona {persona_id} attempt {attempt}: {e}")

    print(f"Warning: Persona {persona_id} failed explicit pref generation after retries.")
    return {}, 0, 0


def merge_prefs(explicit_prefs: dict, implicit_prefs: dict) -> dict:
    """将两步生成的偏好合并：同类别列表拼接（explicit 在前，implicit 在后）。"""
    merged = {}
    for cat in PREFERENCE_CATEGORY_KEYS:
        exp_items = explicit_prefs.get(cat, [])
        imp_items = implicit_prefs.get(cat, [])
        if not isinstance(exp_items, list):
            exp_items = []
        if not isinstance(imp_items, list):
            imp_items = []
        merged[cat] = exp_items + imp_items
    return merged


def session(persona_id, persona, mbti):
    assigned_keywords = get_assigned_audio_keywords(persona_id)

    # ── 生成 Basic Profile ──────────────────────────────────────────────────
    profile, pt0, ct0 = session_profile(persona_id, persona, mbti)
    if profile == {}:
        return {}, 0, 0
    profile_str = get_profile_str(profile.get("Basic", {}))

    total_prompt_tokens = pt0
    total_completion_tokens = ct0

    # ── 第一步：隐式偏好 ────────────────────────────────────────────────────
    implicit_prefs, pt1, ct1 = session_implicit_prefs(
        persona_id, profile_str, mbti, assigned_keywords
    )
    total_prompt_tokens += pt1
    total_completion_tokens += ct1

    if not implicit_prefs:
        print(f"Warning: Persona {persona_id} has no implicit prefs; skipping explicit generation.")
        return profile, total_prompt_tokens, total_completion_tokens

    implicit_final_validation = validate_generated_preferences(implicit_prefs, 6, 8, 4)
    if not implicit_final_validation["valid"]:
        print(
            f"Warning: Persona {persona_id} implicit prefs failed final validation. "
            f"Count={implicit_final_validation['count_validation']['total_count']} "
            f"Visual={implicit_final_validation['visual_validation']['visual_count']}."
        )
        return profile, total_prompt_tokens, total_completion_tokens

    # ── 第二步：显式偏好（绕开隐式偏好） ────────────────────────────────────
    explicit_prefs, pt2, ct2 = session_explicit_prefs(
        persona_id, profile_str, mbti, implicit_prefs
    )
    total_prompt_tokens += pt2
    total_completion_tokens += ct2

    if not explicit_prefs:
        print(f"Warning: Persona {persona_id} has no explicit prefs; returning basic profile only.")
        return profile, total_prompt_tokens, total_completion_tokens

    explicit_final_validation = validate_generated_preferences(explicit_prefs, 6, 8, 4)
    if not explicit_final_validation["valid"]:
        print(
            f"Warning: Persona {persona_id} explicit prefs failed final validation. "
            f"Count={explicit_final_validation['count_validation']['total_count']} "
            f"Visual={explicit_final_validation['visual_validation']['visual_count']}."
        )
        return profile, total_prompt_tokens, total_completion_tokens

    # ── 合并并写回 profile ──────────────────────────────────────────────────
    merged = merge_prefs(explicit_prefs, implicit_prefs)
    profile.update(merged)

    return profile, total_prompt_tokens, total_completion_tokens


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate profiles: implicit preferences first, then explicit preferences that avoid overlap."
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=TARGET_PERSONA_COUNT,
        help=f"Number of personas to generate from the beginning of the input file. Default: {TARGET_PERSONA_COUNT}.",
    )
    parser.add_argument(
        "--input",
        default=str(PERSONA_FILE_PATH),
        help="Persona JSONL input, relative to the project root.",
    )
    parser.add_argument(
        "--output",
        default=str(SAVE_PROFILE_PATH),
        help="Output profile JSON/JSONL path, relative to the project root.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default=MODEL, help="LLM model name")
    return parser.parse_args()


def main():
    args = parse_args()
    global PERSONA_FILE_PATH, SAVE_PROFILE_PATH, MODEL
    PERSONA_FILE_PATH = resolve_path(args.input)
    SAVE_PROFILE_PATH = resolve_path(args.output)
    MODEL = args.model
    sample_count = args.sample

    if sample_count <= 0:
        raise ValueError(f"--sample must be a positive integer, got {sample_count}.")
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}.")
    if not PERSONA_FILE_PATH.exists():
        raise FileNotFoundError(f"Persona file not found: {PERSONA_FILE_PATH}")

    personas = [
        line.strip()
        for line in PERSONA_FILE_PATH.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    selected_personas = personas[:sample_count]

    if len(selected_personas) < sample_count:
        raise ValueError(
            f"personas.jsonl only has {len(selected_personas)} valid lines, "
            f"but --sample is {sample_count}."
        )

    missing_audio_keyword_ids = [
        persona_id for persona_id in range(sample_count)
        if persona_id not in PERSONA_AUDIO_KEYWORDS
    ]
    if missing_audio_keyword_ids:
        raise ValueError(
            "PERSONA_AUDIO_KEYWORDS is missing entries for persona ids: "
            f"{missing_audio_keyword_ids}."
        )

    existing_profiles = load_existing_profiles(SAVE_PROFILE_PATH)
    profiles_by_id = normalize_profiles_by_id(existing_profiles)
    completed_ids = {persona_id for persona_id in profiles_by_id if persona_id < sample_count}

    pending_personas = [
        (persona_id, persona)
        for persona_id, persona in enumerate(selected_personas)
        if persona_id not in completed_ids
    ]

    tasks = []
    mbti_lists = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]

    print(
        f"Loaded {len(completed_ids)} existing profiles within sample range 0-{sample_count - 1}. "
        f"Need to generate {len(pending_personas)} more."
    )

    if not pending_personas:
        save_profiles(SAVE_PROFILE_PATH, profiles_by_id)
        print("All requested personas already exist. Nothing to generate.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for persona_id, persona in tqdm(pending_personas, desc="Submitting personas"):
            mbti = random.sample(mbti_lists, 1)[0]
            tasks.append(executor.submit(session, persona_id, persona, mbti))

        total_prompt_tokens = 0
        total_completion_tokens = 0

        for future in tqdm(
            concurrent.futures.as_completed(tasks),
            total=len(tasks),
            desc="Completed generations",
        ):
            result, prompt_tokens, completion_tokens = future.result()
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            if result != {}:
                persona_id = result.get("id")
                if isinstance(persona_id, int):
                    profiles_by_id[persona_id] = result
                    save_profiles(SAVE_PROFILE_PATH, profiles_by_id)

    print(f"Total Prompt Tokens: {total_prompt_tokens}, Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Cost: ${(total_prompt_tokens * 1 * 0.000001 + total_completion_tokens * 3 * 0.000001):.3f}")
    save_profiles(SAVE_PROFILE_PATH, profiles_by_id)


if __name__ == "__main__":
    main()
