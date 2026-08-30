"""根据用户个人资料构建基于实体的单选（A/B/C/D）问答题。

系统支持三种实体来源，每种都会生成一批多项选择题：

  1. 关系（显式，来自 profile.Basic.Relationship）：针对每个命名
     人物，生成 3-4 道多项选择题，涵盖外貌 / 体型 / 职业 /
     性格 / 与用户的关系。
  2. 宠物（显式，源自 profile.Basic.Pets）：针对每只宠物，生成3-4道
     涵盖外貌/品种/年龄/性情的单选题。
  3. 物品（来自 profiles_*_with_images_entity.json 中的 Items 列表）：
     针对每个物品生成 1-3 道关于颜色/样式/用户使用习惯的单选题。

处理流程（针对每个实体的两阶段大语言模型调用）：
  - 问题生成：一次性为该实体生成若干单选题。
  - 答案筛选：针对每道生成的单选题，在 ``qa_formatted_data.json`` 中查找**提及
    该实体的对话**，将匹配的事件作为文本（附带以
    ``Dxx-NNN.png`` / ``Dxx-NNN.wav``作为索引），连同多项选择题一起输入到大语言模型（LLM）中，并要求其
    返回选定的字母以及一个指向
    支持该答案的回合和/或资源的``记忆线索``列表。
"""

import json
import os
import re
import argparse

import concurrent.futures
from tqdm import tqdm
from json_repair import repair_json

from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.qa.config import profile_path, qa_path

PROFILE_PATH = profile_path("profiles_with_anchors.jsonl")
ENTITY_PATH = profile_path("profiles_with_anchors_with_images_entity.json")
ITEMS_EVENT_PATH = profile_path("profiles_with_anchors_with_items.json")
FORMATTED_DIALOG_PATH = qa_path("qa_formatted_data_000_019.json")
OUTPUT_PATH = qa_path("qa_entity_mcq.json")
CHECKPOINT_EVERY = 10

MODEL = env_value("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")

MAX_WORKERS = 16
LLM_RETRIES = 3
TEMPERATURE_GEN = 0.9
TEMPERATURE_ANS = 0.2

prompt_relationship_qa = '''请根据给定的【用户社会关系】信息，构造 **3-4 道高难度单项选择题**，用于评估智能体能否**精确**记忆该人物的关键属性。

[输入信息]
人物角色: {relation}
人物姓名: {name}
人物简介: {info}
人物外貌: {appearance}

[出题维度（必须从以下维度中至少覆盖 3 个）]
1. **外表/穿着**：发型、发色、肤色、服饰偏好、配饰等可视化特征
2. **身材**：身高、体型描述（高/矮/瘦/微胖/匀称等）
3. **职业/身份**：职业、行业、教育背景、社会角色等
4. **性格/行为习惯**：性格关键词或代表性举动
5. **与用户的关系**：该人物与用户之间的关系类型

[出题要求]
1. 每道题为简短中文问句，**主语必须是该人物的明确姓名**（例如"郑可心"），不要使用"该人物"、"她"、"他"等代称。
   - 题干示例：
     - "下面哪一项最符合郑可心的发型与服饰？"
     - "下列哪一项最准确地描述了郑可心的职业？"
     - "下面哪一项最准确地概括了郑可心与用户之间的关系？"
2. 每题给出 **4 个选项 A/B/C/D**，每个选项是简短中文短语（≤25 汉字）。
3. **高迷惑性干扰项设计原则（核心）**：
   a. 4 个选项必须**结构平行**、**互斥**、**同维度同粒度**，长度和信息密度均衡。
   b. **有且仅有一个**选项与人物真实信息一致；其余 3 个为干扰项。
   c. **至少 2 个"强干扰项"**：与正确答案属于同一细分方向，仅在关键细节上不同。
      - 示例：若人物是"黑色齐肩短发"，强干扰项可以是"深棕色齐肩短发"或"黑色及腰长发"，而非"银白色爆炸头"。
      - 示例：若人物职业是"独立UI设计师"，强干扰项可以是"品牌视觉设计师"或"交互设计师"，而非"厨师"。
   d. 其余干扰项也必须在同领域合理可信，不能是显然荒谬的选项。
   e. 正确选项**不得**因为措辞更具体、更长而在形式上显得突出。
   f. 不要使用"以上都是/都不是""不确定""无法判断"等元选项。
   g. 不要在题面或选项中原样复述输入文本，可适当口语化改写。
4. 不同题之间维度不要重复（同一人不要出 3 道都是问外貌的）。
5. **正确答案位置随机化**：正确选项应均匀分布在 A/B/C/D 中。

[输出格式]
**严格输出**如下 JSON 列表（包含 3-4 个 MCQ），不要添加任何额外说明文字：
```json
[
    {{
        "dimension": "外表 / 身材 / 职业 / 性格 / 关系 中的一个",
        "Q": "题干文本（中文，主语为人物姓名）",
        "options": {{
            "A": "选项 A 文本",
            "B": "选项 B 文本",
            "C": "选项 C 文本",
            "D": "选项 D 文本"
        }},
        "answer": "A"
    }},
    ...
]
```
'''


prompt_pet_qa = '''请根据给定的【用户宠物】信息，构造 **3-4 道高难度单项选择题**，用于评估智能体能否**精确**记忆该宠物的关键属性。

[输入信息]
宠物姓名: {name}
宠物简介: {info}
宠物外貌: {appearance}

[出题维度（必须从以下维度中至少覆盖 3 个）]
1. **外表**：体型、被毛颜色与纹路、眼睛颜色、尾巴/爪子等可视化特征
2. **品种**：物种与具体品种（如美短、布偶、混血短毛等）
3. **年龄**：宠物的具体年龄
4. **性格/习惯**：性格关键词或代表性行为

[出题要求]
1. 每道题为简短中文问句，**主语必须是该宠物的明确名字**（例如"Pixel"或"Scribble（涂鸦）"），不要使用"该宠物"、"它"等代称。
   - 题干示例：
     - "下面哪一项最符合Pixel的外貌？"
     - "下列哪一项最准确地描述了Scribble的品种？"
     - "Pixel 的性格最贴近以下哪一种？"
2. 每题给出 **4 个选项 A/B/C/D**，每个选项是简短中文短语（≤25 汉字）。
3. **高迷惑性干扰项设计原则（核心）**：
   a. 4 个选项必须**结构平行**、**互斥**、**同维度同粒度**，长度和信息密度均衡。
   b. **有且仅有一个**选项与宠物真实信息一致；其余 3 个为干扰项。
   c. **至少 2 个"强干扰项"**：与正确答案属于同一细分方向，仅在关键细节上不同。
      - 示例：若宠物是"银灰色虎斑纹美国短毛猫"，强干扰项可以是"银灰色虎斑纹英国短毛猫"或"棕灰色虎斑纹美国短毛猫"，而非"金色拉布拉多"。
      - 示例：若宠物"喜欢趴在键盘上"，强干扰项可以是"喜欢趴在书桌上"或"喜欢趴在笔记本电脑旁"，而非"喜欢游泳"。
   d. 其余干扰项也必须在同领域合理可信，不能是显然荒谬的选项。
   e. 正确选项**不得**因为措辞更具体、更长而在形式上显得突出。
   f. 不要使用"以上都是/都不是""不确定""无法判断"等元选项。
   g. 不要原样复述输入文本，可适当改写。
4. 不同题之间维度尽量不重复。
5. **正确答案位置随机化**：正确选项应均匀分布在 A/B/C/D 中。

[输出格式]
**严格输出**如下 JSON 列表（包含 3-4 个 MCQ），不要添加任何额外说明文字：
```json
[
    {{
        "dimension": "外表 / 品种 / 年龄 / 性格 中的一个",
        "Q": "题干文本（中文，主语为宠物名字）",
        "options": {{
            "A": "选项 A 文本",
            "B": "选项 B 文本",
            "C": "选项 C 文本",
            "D": "选项 D 文本"
        }},
        "answer": "A"
    }},
    ...
]
```
'''


prompt_item_qa = '''请根据给定的【物品描述】，构造 **1-3 道单项选择题**，用于评估智能体能否记忆用户常用物品的视觉属性。

[输入信息]
物品描述（已含颜色/材质/形态等修饰词）: {description}
物品所属偏好子类目: {source_subcategory}

[该物品出现的事件]
以下是该物品在用户日常生活中实际出现的事件场景，帮助你理解此物品的使用场合和上下文（注意：出题时不得直接透露这些场景信息）：
{item_scenes}

[出题维度（每题选取一个维度，从物品描述中能提取多少维度就出多少题，最多 3 题）]
- **颜色**：例如"原木色"、"深蓝色"、"白色"、"哑光黑"、"浅米色"、"克莱因蓝"等
- **样式/材质**：例如"陶瓷"、"原木"、"哑光"、"皮质"、"金属"、"印花"、"简约造型"等
- **用户使用习惯**：例如"经常用来做菜"、"经常用来冲咖啡"、"经常用来画画"、"经常用来阅读"、"经常用来办公"等

[出题流程]
1. 首先从输入物品描述中识别出**物品本体（不带任何修饰的物品名词）**，例如：
   - "原木色厚实砧板" → 砧板
   - "白色陶瓷V60手冲滤杯" → V60手冲滤杯
   - "整齐排列的Pantone色卡" → Pantone色卡
   - "原木色简约造型猫爬架" → 猫爬架
   - "深蓝色翻开的手账本" → 手账本
2. 然后从颜色 / 样式 / 用户使用习惯 等维度中，**仅选取输入物品描述中明确提及的维度**出题。如果物品描述只包含 1 个可出题的维度，就只出 1 题；如果包含 2 个就出 2 题；最多 3 题。不要为物品描述中未提及的维度强行出题。
3. 题干主语必须是"用户常用的XXX"（XXX = 物品本体），不要透露答案，也不要在题面里使用任何修饰词。
   - 题干示例：
     - "下面哪一项最符合用户常用的砧板的颜色？"
     - "下列哪一项最贴近用户常用的猫爬架的样式与材质？"
     - "下面哪一项最符合用户使用Pantone色卡的习惯？"

[出题要求]
1. 每题给出 **4 个选项 A/B/C/D**，每个选项 ≤15 汉字。
2. 选项设计原则：
   - **有且仅有一个**选项与真实物品属性一致；其余 3 个为合理但错误的干扰项。
   - 颜色题的干扰项必须是**真实存在的、同领域常见的不同颜色**（例：砧板可选 原木色 / 黑色 / 白色 / 深竹色）。
   - 样式题的干扰项必须是同维度的不同样式（例：猫爬架可选 原木简约造型 / 塑料卡通造型 / 金属工业造型 / 仿草绒布造型）。
   - 4 个选项必须**结构平行**、**互斥**。
   - 不要使用"以上都是/都不是""不确定""无法判断"等元选项。
3. 不要在题面或选项中原样复述输入文本。

[输出格式]
**严格输出**如下 JSON 列表（包含 1-3 个 MCQ），不要添加任何额外说明文字：
```json
[
    {{
        "dimension": "颜色 / 样式 / 用户使用习惯 中的一个",
        "item_type": "物品本体名词（不含修饰）",
        "Q": "题干文本（'用户常用的XXX的YYY最可能是？'格式）",
        "options": {{
            "A": "选项 A 文本",
            "B": "选项 B 文本",
            "C": "选项 C 文本",
            "D": "选项 D 文本"
        }},
        "answer": "A"
    }},
    ...
]
```
'''


prompt_answer = '''你是一个精确的证据抽取系统。你会收到：
1. 【出题时的输入信息】—— 该实体的真实属性，用于理解为什么给定答案是正确的。
2. 【已知正确答案】—— 这道题的正确选项字母，以及对应的选项文本。
3. 【相关对话历史】—— 与该实体相关的对话记录（含文字、图像描述、音频描述），用于定位 memory clue。
4. 一道单项选择题（A/B/C/D）。

你的任务：
- 不需要重新判断答案，正确答案已经给出。
- 你只需要从【相关对话历史】中找出所有能支撑该正确答案的证据，作为 memory clue 返回。

[输入说明]
【相关对话历史】是与该题主题（某个人物 / 宠物 / 物品）相关的若干 session 拼接，每条 session 内部按轮次给出对话，并标注了同一轮次中出现的图像/音频证据；不同题目可能只有其中一种或两种线索，没有出现的线索类型不必强行寻找：
- 每条用户/助手消息以 [Dxx:NN] 表示其轮次编号，其中 Dxx 是 session_id，NN 是该 session 内的轮次序号；
- 用户在某轮次分享的图像描述以 "图像[Dxx-NNN.png]:" 出现；
- 该 session 的背景音频或用户语音消息描述以 "音频[Dxx-NNN.wav]:" 出现。

[出题时的输入信息]
{entity_info}

[已知正确答案]
正确选项: {answer_letter}
正确选项文本: {answer_text}

[提取原则]
1. 以【已知正确答案】为目标，从【相关对话历史】中找出所有能直接支撑该答案的证据。
2. 从三类线索中选用证据：(a) 对话文字、(b) 图像描述、(c) 音频/语音描述。任何一类都可独立支撑答案，不要求三类齐备。
3. **遍历**【相关对话历史】中的全部信息，把**所有**能够作为答案依据的证据都列入 `memory clue`，不要因为已有一条就提前停止；只要某条线索（轮次/图像/音频）能直接支持该正确选项，就必须收录。
4. **禁止**把与答案无直接关系或仅作为闲聊背景的轮次/图像/音频塞进 `memory clue`；只列真正能支撑结论的证据。
5. memory clue 元素格式：
   - 对话证据使用 "session_id:round_id" 格式（如 "D01:03"，必须严格匹配【相关对话历史】中实际出现的轮次编号）；
   - 图像证据使用 "Dxx-NNN.png" 格式；
   - 音频证据使用 "Dxx-NNN.wav" 格式；
   - 每一条 clue 都必须**真实出现**在【相关对话历史】中，禁止编造；
   - 如果【相关对话历史】中没有任何相关证据，可返回空列表 []。

[相关对话历史]
{dialog_str}

[选择题]
题目：{question}
A. {opt_a}
B. {opt_b}
C. {opt_c}
D. {opt_d}

[输出格式]
**严格输出**如下 JSON 结构，不要添加任何额外说明文字（注意 "memory clue" 中间是空格，与示例完全一致）：
```json
{{
    "memory clue": ["D01:03", "D01-001.png"]
}}
```
'''


def load_profiles(path: str) -> list:
    return load_records(path)


def load_entity_profiles(path: str) -> list:
    """Load the entity profile JSON (list of profiles with Items field)."""
    if not os.path.exists(path):
        return []
    data = load_records(path)
    return data if isinstance(data, list) else [data]


PREFERENCE_CATEGORIES = [
    "FoodAndDrink", "HomeAndSpace", "BodyAndHealth",
    "HobbiesAndEntertainment", "WorkAndLearning", "MobilityAndTravel", "Pets",
]


def build_item_event_index(items_profiles: list) -> dict:
    """Build {p_id: {item_description: [event_dicts]}} from the *_with_items profile.

    For each event across all preference categories, check its ``entity_anchors``
    list; if an Item's description appears as a substring of any anchor (or vice
    versa), that event is associated with the item.

    Returns a nested dict so ``index[p_id][description]`` gives the list of
    matching events (each carrying ``scene_description`` and
    ``user_shared_image_description``).
    """
    index: dict = {}
    for p in items_profiles:
        p_id = p.get("p_id", 0)
        all_events = []
        for cat in PREFERENCE_CATEGORIES:
            for rec in p.get(cat, []) or []:
                for ev in rec.get("events", []) or []:
                    anchors = ev.get("entity_anchors", []) or []
                    if anchors:
                        all_events.append((anchors, ev))

        items_list = p.get("Items", []) or []
        item_map: dict = {}
        for item in items_list:
            desc = (item.get("description") or "").strip()
            if not desc:
                continue
            matched = []
            for anchors, ev in all_events:
                if any(desc in a or a in desc for a in anchors):
                    matched.append(ev)
            item_map[desc] = matched
        index[p_id] = item_map
    return index


def format_item_scenes(events: list) -> str:
    """Render matching events into a compact scene context for item QA."""
    if not events:
        return "（该物品暂未在已记录的事件中出现）"
    parts = []
    for ev in events:
        task_id = ev.get("task_id", "?")
        scene = (ev.get("scene_description") or "").strip()
        img_desc = (ev.get("user_shared_image_description") or "").strip()
        block = [f"[事件 {task_id}]"]
        if scene:
            block.append(f"  场景: {scene}")
        if img_desc and img_desc.lower() != "none":
            block.append(f"  用户分享的图像: {img_desc}")
        parts.append("\n".join(block))
    return "\n\n".join(parts)


def load_formatted_dialog(path: str) -> list:
    """Load qa_formatted_data.json. Tolerates both JSON-array and line-delimited
    JSON formats. Returns [] if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    return load_records(path)


def build_pid_event_index(formatted_profiles: list) -> dict:
    """Return {p_id: [events]} sourced from qa_formatted_data.json."""
    index: dict = {}
    for fp in formatted_profiles:
        p_id = fp.get("p_id", 0)
        index[p_id] = list(fp.get("events", []) or [])
    return index


def build_pid_task_event_index(formatted_profiles: list) -> dict:
    """Return {p_id: {task_id: [events]}} for fallback item matching."""
    index: dict = {}
    for fp in formatted_profiles:
        p_id = fp.get("p_id", 0)
        task_map = {}
        for event in fp.get("events", []) or []:
            task_id = event.get("task_id")
            if task_id is None:
                continue
            task_map.setdefault(task_id, []).append(event)
        index[p_id] = task_map
    return index


def name_search_variants(name: str) -> list:
    """Return candidate substrings to match against ``scene_description``.

    Pet / person names sometimes carry a parenthesized alias (e.g.
    ``"Scribble（涂鸦）"``) while the dialogue scene only mentions one of the
    two forms. We therefore try the full name **and** any segment outside or
    inside the parentheses.
    """
    name = (name or "").strip()
    if not name:
        return []
    variants = {name}
    m = re.match(r"^(.*?)\s*[（(](.+?)[）)]\s*$", name)
    if m:
        head = m.group(1).strip()
        inner = m.group(2).strip()
        if head:
            variants.add(head)
        if inner:
            variants.add(inner)
    return [v for v in variants if v]


def find_events_by_name(events: list, name: str) -> list:
    """Return events whose ``scene_description`` contains the given entity
    name (or any of its parenthesized variants). Order is preserved.
    """
    variants = name_search_variants(name)
    if not variants:
        return []
    matched = []
    for event in events:
        scene = event.get("scene_description", "") or ""
        if any(v in scene for v in variants):
            matched.append(event)
    return matched


def find_events_by_anchor(events: list, anchor: str) -> list:
    """Return events whose item anchor list contains the given anchor.

    The formatted dialog data stores item anchors in ``entity_anchors`` (list),
    while some older intermediate files may use ``entity_anchor`` (string).
    Support both layouts here.
    """
    anchor = (anchor or "").strip()
    if not anchor or anchor.lower() == "none":
        return []
    matched = []
    for event in events:
        found = False

        ev_anchors = event.get("entity_anchors", []) or []
        for ev_anchor in ev_anchors:
            ev_anchor = str(ev_anchor or "").strip()
            if not ev_anchor or ev_anchor.lower() == "none":
                continue
            if anchor in ev_anchor or ev_anchor in anchor:
                found = True
                break

        if not found:
            ev_anchor = (event.get("entity_anchor") or "").strip()
            if ev_anchor and ev_anchor.lower() != "none":
                if anchor in ev_anchor or ev_anchor in anchor:
                    found = True

        if found:
            matched.append(event)
    return matched


def item_search_variants(description: str) -> list:
    """Build fallback suffix variants for fuzzy item lookup."""
    description = (description or "").strip()
    if not description:
        return []
    variants = {description}
    max_len = min(8, len(description))
    for n in range(2, max_len + 1):
        variants.add(description[-n:])
    return sorted(variants, key=len, reverse=True)


def find_events_by_item_text(events: list, description: str) -> list:
    """Fuzzy fallback for items when anchor-based matching misses.

    It checks whether a suffix variant of the item description appears in the
    event's anchors, scene description, or shared image description.
    """
    variants = item_search_variants(description)
    if not variants:
        return []

    matched = []
    for event in events:
        text_parts = []
        text_parts.extend(str(a or "") for a in (event.get("entity_anchors", []) or []))
        text_parts.append(str(event.get("scene_description") or ""))
        text_parts.append(str(event.get("user_shared_image_description") or ""))
        haystack = "\n".join(text_parts)
        if any(v and v in haystack for v in variants):
            matched.append(event)
    return matched


def find_item_events(
    p_id: int,
    pid_events: list,
    description: str,
    item_event_index: dict,
    pid_task_event_index: dict,
) -> list:
    """Find formatted-dialog events for an item.

    Priority:
    1. Direct match against formatted-dialog anchors.
    2. Fallback to item_event_index built from *_with_items profiles, then map
       those events back into qa_formatted_data by task_id.
    """
    direct = find_events_by_anchor(pid_events, description)
    if direct:
        return direct

    fallback_events = item_event_index.get(p_id, {}).get(description, []) or []
    if not fallback_events:
        return find_events_by_item_text(pid_events, description)

    task_map = pid_task_event_index.get(p_id, {}) or {}
    merged = []
    seen = set()
    for ev in fallback_events:
        task_id = ev.get("task_id")
        for fmt_ev in task_map.get(task_id, []) or []:
            session_id = fmt_ev.get("session_id")
            if session_id and session_id not in seen:
                seen.add(session_id)
                merged.append(fmt_ev)
    if merged:
        return merged
    return find_events_by_item_text(pid_events, description)


def format_events_for_prompt(events: list) -> str:
    """Render a list of events into a readable dialog context that includes
    image and audio descriptions keyed by ``Dxx-NNN.png`` / ``Dxx-NNN.wav``.
    """
    if not events:
        return "（无相关对话历史）"

    sections = []
    seen_session_ids = set()
    for event in events:
        session_id = event.get("session_id", "?")
        if session_id in seen_session_ids:
            continue
        seen_session_ids.add(session_id)

        scene = (event.get("scene_description") or "").strip()
        img_desc = (event.get("user_shared_image_description") or "").strip()
        bg_audio = (event.get("background_audio_info") or "").strip()
        speech = (event.get("human_speech_content") or "").strip()

        section = [f"=== Session {session_id} ==="]
        if scene:
            section.append(f"场景: {scene}")
        if img_desc and img_desc.lower() != "none":
            section.append(f"图像总览: {img_desc}")
        if bg_audio and bg_audio.lower() != "none":
            section.append(f"背景音频: {bg_audio}")
        if speech and speech.lower() != "none":
            section.append(f"用户语音: {speech}")

        bg_audio_map: dict = {}
        user_turn_idx = 0
        dialog_list = event.get("dialog_list", []) or []
        for dt in event.get("dialog", []) or []:
            if dt.get("role") != "user":
                continue
            if user_turn_idx < len(dialog_list):
                rid = dialog_list[user_turn_idx].get("round", "")
                ba = (dt.get("background_audio") or "").strip()
                if ba and ba.lower() != "none":
                    bg_audio_map[rid] = ba
            user_turn_idx += 1

        section.append("对话:")
        for turn in dialog_list:
            round_id = turn.get("round", "?")
            user = (turn.get("user") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            section.append(f"  [{round_id}] User: {user}")
            if round_id in bg_audio_map:
                section.append(f"    └─ 背景音频: {bg_audio_map[round_id]}")
            for k, v in turn.items():
                if k in ("round", "user", "assistant"):
                    continue
                if k.endswith(".png"):
                    image_text = img_desc if img_desc and img_desc.lower() != "none" else v
                    section.append(f"    └─ 图像[{k}]: {image_text}")
                elif k.endswith(".wav"):
                    section.append(f"    └─ 音频[{k}]: {v}")
            if assistant:
                section.append(f"  [{round_id}] Assistant: {assistant}")
        sections.append("\n".join(section))

    return "\n\n".join(sections)


def collect_valid_clue_keys(events: list) -> set:
    """Collect every legitimate clue identifier from the matched events.

    A clue identifier is either a round id (``Dxx:NN``) or an asset key
    (``Dxx-NNN.png`` / ``Dxx-NNN.wav``).
    """
    valid: set = set()
    for event in events:
        for turn in event.get("dialog_list", []) or []:
            round_id = turn.get("round")
            if isinstance(round_id, str) and round_id:
                valid.add(round_id)
            for k in turn:
                if k in ("round", "user", "assistant"):
                    continue
                if k.endswith(".png") or k.endswith(".wav"):
                    valid.add(k)
    return valid


def filter_memory_clues(clues, valid_keys: set) -> list:
    """Keep only LLM-returned clues that actually appear in the matched events."""
    if not isinstance(clues, list):
        return []
    seen = []
    for c in clues:
        if isinstance(c, str) and c.strip() and c.strip() in valid_keys and c.strip() not in seen:
            seen.append(c.strip())
    return seen


def call_llm(prompt: str, temperature: float = 0.7) -> tuple:
    client = openai_client(
        api_key_env="CUE_MEM_LLM_API_KEY",
        base_url_env="CUE_MEM_LLM_BASE_URL",
    )
    last_err = None
    last_text = None

    for _ in range(LLM_RETRIES):
        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            text = ""
            usage_info = None
            for chunk in api_res:
                if chunk.choices:
                    text += message_content_to_text(chunk.choices[0].delta.content)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            last_text = text
            cleaned = (
                text.strip().replace("```json", "").replace("```", "").strip()
            )
            data = json.loads(repair_json(cleaned))
            tokens_in = usage_value(usage_info, "prompt_tokens")
            tokens_out = usage_value(usage_info, "completion_tokens")
            return data, tokens_in, tokens_out
        except Exception as e:
            last_err = e
            print(f"[call_llm] Error: {e}; raw_tail={(last_text or '')[-200:]}")

    raise RuntimeError(f"call_llm failed after {LLM_RETRIES} attempts: {last_err}")


def get_relationship_info(person: dict) -> str:
    """Full entity info for the answer prompt (same as generation input)."""
    parts = ["实体类型: Relationship（人物）"]
    parts.append(f"人物角色: {person.get('relation', '')}")
    parts.append(f"人物姓名: {person.get('name', '')}")
    parts.append(f"人物简介: {person.get('info', '')}")
    parts.append(f"人物外貌: {person.get('appearance', '')}")
    return "\n".join(parts)


def get_pets_info(pet: dict) -> str:
    parts = ["实体类型: Pets（宠物）"]
    parts.append(f"宠物姓名: {pet.get('name', '')}")
    parts.append(f"宠物简介: {pet.get('info', '')}")
    parts.append(f"宠物外貌: {pet.get('appearance', '')}")
    return "\n".join(parts)


def get_item_info(description: str, source_subcategory: str) -> str:
    parts = ["实体类型: Items（物品）"]
    parts.append(f"物品描述: {description}")
    parts.append(f"所属偏好子类目: {source_subcategory}")
    return "\n".join(parts)


def find_relationship_profile(basic: dict, entity_name: str, entity_relation: str = "") -> dict:
    for person in basic.get("Relationship", []) or []:
        if (person.get("name") or "").strip() != (entity_name or "").strip():
            continue
        if entity_relation and (person.get("relation") or "").strip() != entity_relation.strip():
            continue
        return person
    return {}


def find_pet_profile(basic: dict, entity_name: str) -> dict:
    for pet in basic.get("Pets", []) or []:
        if (pet.get("name") or "").strip() == (entity_name or "").strip():
            return pet
    return {}


def answer_one_mcq(
    dialog_str: str,
    entity_info: str,
    question: str,
    options: dict,
    answer_letter: str,
    answer_text: str,
) -> tuple:
    prompt = prompt_answer.format(
        dialog_str=dialog_str,
        entity_info=entity_info,
        question=question,
        opt_a=options.get("A", ""),
        opt_b=options.get("B", ""),
        opt_c=options.get("C", ""),
        opt_d=options.get("D", ""),
        answer_letter=answer_letter,
        answer_text=answer_text,
    )
    return call_llm(prompt, temperature=TEMPERATURE_ANS)


def regenerate_clue_for_record(
    record: dict,
    profiles: list,
    entity_profiles: list,
    pid_event_index: dict,
    item_event_index: dict,
    pid_task_event_index: dict,
) -> tuple:
    """Reuse an existing MCQ and regenerate only the memory clue."""
    p_id = record.get("p_id", 0)
    entity_type = record.get("entity_type", "")
    pid_events = pid_event_index.get(p_id, [])

    if entity_type == "Relationship":
        basic = (profiles[p_id].get("Basic", {}) or {}) if p_id < len(profiles) else {}
        person = find_relationship_profile(
            basic,
            record.get("entity_name", ""),
            record.get("entity_relation", ""),
        )
        relevant_events = find_events_by_name(pid_events, record.get("entity_name", ""))
        entity_info = get_relationship_info(person) if person else "\n".join([
            "实体类型: Relationship（人物）",
            f"人物角色: {record.get('entity_relation', '')}",
            f"人物姓名: {record.get('entity_name', '')}",
        ])
    elif entity_type == "Pets":
        basic = (profiles[p_id].get("Basic", {}) or {}) if p_id < len(profiles) else {}
        pet = find_pet_profile(basic, record.get("entity_name", ""))
        relevant_events = find_events_by_name(pid_events, record.get("entity_name", ""))
        entity_info = get_pets_info(pet) if pet else "\n".join([
            "实体类型: Pets（宠物）",
            f"宠物姓名: {record.get('entity_name', '')}",
        ])
    elif entity_type == "Items":
        description = record.get("entity_description", "")
        relevant_events = find_item_events(
            p_id, pid_events, description, item_event_index, pid_task_event_index
        )
        entity_info = get_item_info(description, record.get("source_subcategory", ""))
    else:
        return record, 0, 0

    dialog_str = format_events_for_prompt(relevant_events)
    valid_keys = collect_valid_clue_keys(relevant_events)
    options = record.get("options", {}) or {}
    answer_letter = str(record.get("A", "")).strip().upper()
    answer_text = str(options.get(answer_letter, "")).strip()
    if answer_letter not in {"A", "B", "C", "D"} or not answer_text:
        return record, 0, 0
    ans, tin, tout = answer_one_mcq(
        dialog_str,
        entity_info,
        (record.get("Q") or "").strip(),
        {k: str(options.get(k, "")).strip() for k in ["A", "B", "C", "D"]},
        answer_letter,
        answer_text,
    )
    raw_clues = ans.get("memory clue")
    if raw_clues is None:
        raw_clues = ans.get("memory_clue")
    updated = dict(record)
    updated["memory clue"] = filter_memory_clues(raw_clues, valid_keys)
    updated["matched_session_ids"] = _matched_session_ids(relevant_events)
    return updated, tin, tout


def validate_mcq_block(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("Q"), str) or not item["Q"].strip():
        return False
    options = item.get("options")
    if not isinstance(options, dict):
        return False
    return all(k in options and str(options[k]).strip() for k in ["A", "B", "C", "D"])


def extract_answer_letter(item: dict) -> str:
    answer = str(item.get("answer", "")).strip().upper()
    return answer if answer in {"A", "B", "C", "D"} else ""


def _matched_session_ids(events: list) -> list:
    return sorted({e.get("session_id", "") for e in events if e.get("session_id")})


def build_relationship_qa(
    p_id: int,
    rel_idx: int,
    person: dict,
    pid_events: list,
) -> tuple:
    qa_prefix = f"{p_id}-Relationship-{rel_idx}"
    entity_name = person.get("name", "")
    relevant_events = find_events_by_name(pid_events, entity_name)
    dialog_str = format_events_for_prompt(relevant_events)
    valid_keys = collect_valid_clue_keys(relevant_events)
    entity_info = get_relationship_info(person)
    matched_sessions = _matched_session_ids(relevant_events)

    tokens_in, tokens_out = 0, 0
    records = []

    try:
        prompt_gen = prompt_relationship_qa.format(
            relation=person.get("relation", ""),
            name=entity_name,
            info=person.get("info", ""),
            appearance=person.get("appearance", ""),
        )
        mcq_list, tin, tout = call_llm(prompt_gen, temperature=TEMPERATURE_GEN)
        tokens_in += tin
        tokens_out += tout

        if not isinstance(mcq_list, list):
            print(f"[{qa_prefix}] Generation returned non-list: {mcq_list}")
            return [], tokens_in, tokens_out

        for q_idx, mcq in enumerate(mcq_list):
            if not validate_mcq_block(mcq):
                print(f"[{qa_prefix}-{q_idx}] invalid MCQ block: {mcq}")
                continue
            answer_letter = extract_answer_letter(mcq)
            if not answer_letter:
                print(f"[{qa_prefix}-{q_idx}] invalid answer in MCQ: {mcq}")
                continue

            options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
            answer_text = options[answer_letter]
            ans, tin2, tout2 = answer_one_mcq(
                dialog_str, entity_info, mcq["Q"].strip(), options,
                answer_letter, answer_text,
            )
            tokens_in += tin2
            tokens_out += tout2

            raw_clues = ans.get("memory clue")
            if raw_clues is None:
                raw_clues = ans.get("memory_clue")
            memory_clue = filter_memory_clues(raw_clues, valid_keys)

            records.append({
                "qa_id": f"{qa_prefix}-{q_idx}",
                "p_id": p_id,
                "entity_type": "Relationship",
                "entity_name": entity_name,
                "entity_relation": person.get("relation", ""),
                "dimension": mcq.get("dimension", ""),
                "Q": mcq["Q"].strip(),
                "options": options,
                "A": answer_letter,
                "memory clue": memory_clue,
                "matched_session_ids": matched_sessions,
                "type": "entity_mcq",
            })
    except Exception as e:
        print(f"[{qa_prefix}] build_relationship_qa failed: {e}")

    return records, tokens_in, tokens_out


def build_pet_qa(
    p_id: int,
    pet_idx: int,
    pet: dict,
    pid_events: list,
) -> tuple:
    qa_prefix = f"{p_id}-Pets-{pet_idx}"
    entity_name = pet.get("name", "")
    relevant_events = find_events_by_name(pid_events, entity_name)
    dialog_str = format_events_for_prompt(relevant_events)
    valid_keys = collect_valid_clue_keys(relevant_events)
    entity_info = get_pets_info(pet)
    matched_sessions = _matched_session_ids(relevant_events)

    tokens_in, tokens_out = 0, 0
    records = []

    try:
        prompt_gen = prompt_pet_qa.format(
            name=entity_name,
            info=pet.get("info", ""),
            appearance=pet.get("appearance", ""),
        )
        mcq_list, tin, tout = call_llm(prompt_gen, temperature=TEMPERATURE_GEN)
        tokens_in += tin
        tokens_out += tout

        if not isinstance(mcq_list, list):
            print(f"[{qa_prefix}] Generation returned non-list: {mcq_list}")
            return [], tokens_in, tokens_out

        for q_idx, mcq in enumerate(mcq_list):
            if not validate_mcq_block(mcq):
                print(f"[{qa_prefix}-{q_idx}] invalid MCQ block: {mcq}")
                continue
            answer_letter = extract_answer_letter(mcq)
            if not answer_letter:
                print(f"[{qa_prefix}-{q_idx}] invalid answer in MCQ: {mcq}")
                continue

            options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
            answer_text = options[answer_letter]
            ans, tin2, tout2 = answer_one_mcq(
                dialog_str, entity_info, mcq["Q"].strip(), options,
                answer_letter, answer_text,
            )
            tokens_in += tin2
            tokens_out += tout2

            raw_clues = ans.get("memory clue")
            if raw_clues is None:
                raw_clues = ans.get("memory_clue")
            memory_clue = filter_memory_clues(raw_clues, valid_keys)

            records.append({
                "qa_id": f"{qa_prefix}-{q_idx}",
                "p_id": p_id,
                "entity_type": "Pets",
                "entity_name": entity_name,
                "dimension": mcq.get("dimension", ""),
                "Q": mcq["Q"].strip(),
                "options": options,
                "A": answer_letter,
                "memory clue": memory_clue,
                "matched_session_ids": matched_sessions,
                "type": "entity_mcq",
            })
    except Exception as e:
        print(f"[{qa_prefix}] build_pet_qa failed: {e}")

    return records, tokens_in, tokens_out


def build_item_qa(
    p_id: int,
    item_idx: int,
    item: dict,
    pid_events: list,
    item_event_index: dict,
    pid_task_event_index: dict,
) -> tuple:
    description = (item.get("description") or "").strip()
    source_subcategory = (item.get("source_subcategory") or "").strip()
    qa_prefix = f"{p_id}-Items-{item_idx}"
    relevant_events = find_item_events(
        p_id, pid_events, description, item_event_index, pid_task_event_index
    )
    dialog_str = format_events_for_prompt(relevant_events)
    valid_keys = collect_valid_clue_keys(relevant_events)
    entity_info = get_item_info(description, source_subcategory)
    matched_sessions = _matched_session_ids(relevant_events)

    scene_events = item_event_index.get(p_id, {}).get(description, [])
    item_scenes = format_item_scenes(scene_events)

    tokens_in, tokens_out = 0, 0
    records = []

    try:
        prompt_gen = prompt_item_qa.format(
            description=description,
            source_subcategory=source_subcategory,
            item_scenes=item_scenes,
        )
        mcq_list, tin, tout = call_llm(prompt_gen, temperature=TEMPERATURE_GEN)
        tokens_in += tin
        tokens_out += tout

        if not isinstance(mcq_list, list):
            print(f"[{qa_prefix}] Generation returned non-list: {mcq_list}")
            return [], tokens_in, tokens_out

        for q_idx, mcq in enumerate(mcq_list):
            if not validate_mcq_block(mcq):
                print(f"[{qa_prefix}-{q_idx}] invalid MCQ block: {mcq}")
                continue
            answer_letter = extract_answer_letter(mcq)
            if not answer_letter:
                print(f"[{qa_prefix}-{q_idx}] invalid answer in MCQ: {mcq}")
                continue

            options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
            answer_text = options[answer_letter]
            ans, tin2, tout2 = answer_one_mcq(
                dialog_str, entity_info, mcq["Q"].strip(), options,
                answer_letter, answer_text,
            )
            tokens_in += tin2
            tokens_out += tout2

            raw_clues = ans.get("memory clue")
            if raw_clues is None:
                raw_clues = ans.get("memory_clue")
            memory_clue = filter_memory_clues(raw_clues, valid_keys)

            records.append({
                "qa_id": f"{qa_prefix}-{q_idx}",
                "p_id": p_id,
                "entity_type": "Items",
                "entity_description": description,
                "source_subcategory": source_subcategory,
                "item_type": mcq.get("item_type", ""),
                "dimension": mcq.get("dimension", ""),
                "Q": mcq["Q"].strip(),
                "options": options,
                "A": answer_letter,
                "memory clue": memory_clue,
                "matched_session_ids": matched_sessions,
                "type": "entity_mcq",
            })
    except Exception as e:
        print(f"[{qa_prefix}] build_item_qa failed: {e}")

    return records, tokens_in, tokens_out


TYPE_ORDER = {"Relationship": 0, "Pets": 1, "Items": 2}


def _sort_key(r):
    return (r["p_id"], TYPE_ORDER.get(r["entity_type"], 9), r["qa_id"])


def _save_checkpoint(records: list, path: str):
    sorted_records = sorted(records, key=_sort_key)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sorted_records, f, ensure_ascii=False, indent=4)
    try:
        os.replace(tmp, path)
    except PermissionError:
        import shutil
        shutil.copy2(tmp, path)
        try:
            os.remove(tmp)
        except OSError:
            pass


def _load_existing(path: str) -> dict:
    """Load existing output and return {qa_id: record} for resume."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["qa_id"]: r for r in records}
    except Exception as e:
        print(f"WARN: failed to load {path} for resume: {e}")
        return {}


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Generate entity MCQ QA and memory clues."
    )
    parser.add_argument(
        "--entity_types",
        nargs="+",
        default=["Relationship", "Pets", "Items"],
        choices=["Relationship", "Pets", "Items"],
        help="Only process the selected entity types.",
    )
    parser.add_argument(
        "--rerun_selected",
        action="store_true",
        help="Re-run selected entity types even if matching records already exist.",
    )
    parser.add_argument(
        "--clue_only",
        action="store_true",
        help="Reuse existing MCQ question/options and regenerate only memory clue for selected entity types.",
    )
    parser.add_argument(
        "--max-profiles",
        "--max_profiles",
        type=int,
        default=0,
        help="Only process profiles with p_id < N for small-batch debugging. Default: process all profiles.",
    )
    return parser


def main():
    import threading
    from collections import Counter
    args = build_argparser().parse_args()
    selected_types = set(args.entity_types)
    if args.max_profiles is not None and args.max_profiles < 0:
        raise SystemExit("--max-profiles must be >= 0")
    max_profiles = args.max_profiles or 0

    profiles = load_profiles(PROFILE_PATH)
    print(f"Loaded {len(profiles)} profile(s) from {PROFILE_PATH}.")
    if max_profiles:
        print(f"Max-profiles mode: only processing p_id < {max_profiles}.")
    print(f"Selected entity types: {', '.join(args.entity_types)}")
    if args.rerun_selected:
        print("Selected entity types will be regenerated even if records already exist.")
    if args.clue_only:
        print("Clue-only mode: keep existing MCQ stems/options and regenerate only memory clues.")

    entity_profiles = load_entity_profiles(ENTITY_PATH)
    if entity_profiles:
        total_items = sum(len(p.get("Items", []) or []) for p in entity_profiles)
        print(f"Loaded {len(entity_profiles)} entity profile(s) ({total_items} items) from {ENTITY_PATH}.")
    else:
        print(f"WARN: entity file not found at {ENTITY_PATH}; Items QA will be skipped.")

    items_event_profiles = load_entity_profiles(ITEMS_EVENT_PATH)
    if items_event_profiles:
        item_event_index = build_item_event_index(items_event_profiles)
        matched_count = sum(
            sum(1 for evs in pid_map.values() if evs)
            for pid_map in item_event_index.values()
        )
        print(f"Built item-event index from {ITEMS_EVENT_PATH} ({matched_count} item-event matches).")
    else:
        item_event_index = {}
        print(f"WARN: {ITEMS_EVENT_PATH} not found; item scene context will be empty.")

    formatted_profiles = load_formatted_dialog(FORMATTED_DIALOG_PATH)
    if formatted_profiles:
        total_events = sum(len(p.get("events", []) or []) for p in formatted_profiles)
        print(
            f"Loaded {len(formatted_profiles)} formatted profile(s) "
            f"({total_events} events) from {FORMATTED_DIALOG_PATH}."
        )
    else:
        print(
            f"WARN: {FORMATTED_DIALOG_PATH} is missing or empty; memory clues "
            f"will fall back to []."
        )
    pid_event_index = build_pid_event_index(formatted_profiles)
    pid_task_event_index = build_pid_task_event_index(formatted_profiles)

    existing = _load_existing(OUTPUT_PATH)
    if existing:
        print(f"[resume] 已有 {len(existing)} 条记录，将跳过已完成的 qa_id 前缀。")

    existing_prefixes = set()
    for qa_id in existing:
        parts = qa_id.rsplit("-", 1)
        if len(parts) == 2:
            existing_prefixes.add(parts[0])

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = 0
    total_out = 0
    new_since_checkpoint = 0
    tasks = []

    def _on_results(recs, tin, tout):
        nonlocal total_in, total_out, new_since_checkpoint
        total_in += tin
        total_out += tout
        if isinstance(recs, dict):
            recs = [recs]
        for r in recs:
            qa_map[r["qa_id"]] = r
        new_since_checkpoint += len(recs)
        if new_since_checkpoint >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_checkpoint = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        if args.clue_only:
            selected_existing = [
                record for record in existing.values()
                if record.get("entity_type") in selected_types
                and (not max_profiles or int(record.get("p_id", -1)) < max_profiles)
            ]
            if not selected_existing:
                print("clue_only 模式下未找到可复用的已有记录。")
                return
            for record in selected_existing:
                tasks.append(
                    executor.submit(
                        regenerate_clue_for_record,
                        record,
                        profiles,
                        entity_profiles,
                        pid_event_index,
                        item_event_index,
                        pid_task_event_index,
                    )
                )
        else:
            for p_id, profile in enumerate(profiles):
                if max_profiles and p_id >= max_profiles:
                    break
                basic = profile.get("Basic", {}) or {}
                pid_events = pid_event_index.get(p_id, [])

                if "Relationship" in selected_types:
                    for rel_idx, person in enumerate(basic.get("Relationship", []) or []):
                        prefix = f"{p_id}-Relationship-{rel_idx}"
                        if prefix in existing_prefixes and not args.rerun_selected:
                            continue
                        tasks.append(
                            executor.submit(
                                build_relationship_qa, p_id, rel_idx, person, pid_events
                            )
                        )

                if "Pets" in selected_types:
                    for pet_idx, pet in enumerate(basic.get("Pets", []) or []):
                        prefix = f"{p_id}-Pets-{pet_idx}"
                        if prefix in existing_prefixes and not args.rerun_selected:
                            continue
                        tasks.append(
                            executor.submit(build_pet_qa, p_id, pet_idx, pet, pid_events)
                        )

                if "Items" in selected_types:
                    entity_p = entity_profiles[p_id] if p_id < len(entity_profiles) else {}
                    items_list = entity_p.get("Items", []) or []
                    for item_idx, item in enumerate(items_list):
                        prefix = f"{p_id}-Items-{item_idx}"
                        if prefix in existing_prefixes and not args.rerun_selected:
                            continue
                        tasks.append(
                            executor.submit(
                                build_item_qa, p_id, item_idx, item, pid_events,
                                item_event_index,
                                pid_task_event_index,
                            )
                        )

        if not tasks:
            print("所有 entity QA 均已生成，无需重新运行。")
            return

        print(f"待生成: {len(tasks)} 个实体任务 (已有 {len(existing)} 条记录)")

        for future in tqdm(
            concurrent.futures.as_completed(tasks),
            total=len(tasks),
            desc="Building entity MCQs",
        ):
            recs, tin, tout = future.result()
            with lock:
                _on_results(recs, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)

    by_type = Counter(r["entity_type"] for r in qa_map.values())
    print(f"\n总记录数: {len(qa_map)} (本次新增 {len(qa_map) - len(existing)})")
    for t, c in by_type.items():
        print(f"  - {t}: {c}")
    print(f"Tokens — prompt: {total_in}, completion: {total_out}")
    print(f"Estimated cost (input $1/M, output $3/M): "
          f"${(total_in * 1e-6 + total_out * 3e-6):.4f}")
    print(f"Output -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
