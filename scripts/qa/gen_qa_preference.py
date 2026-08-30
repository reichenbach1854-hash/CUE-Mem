"""根据用户画像中的偏好生成 preference 类单选题（A/B/C/D）。

Pipeline（两阶段 LLM 调用）:
--------
1. 加载 profiles_000_002_with_anchors.jsonl，提取每个角色的 7 大类偏好
   （FoodAndDrink, HomeAndSpace, BodyAndHealth, HobbiesAndEntertainment,
   WorkAndLearning, MobilityAndTravel, Pets），每条偏好含
   subcategory / preference / expression_type / evidence_sources / analysis。
2. 对每条偏好调用 LLM（Stage 1），生成 1 道 MCQ（题干 + 4 个选项）。
3. 在 qa_formatted_data_000_002.json 中查找所有
   ``explicit_preferences_reflected`` 或 ``implicit_preferences_reflected``
   包含该偏好 code（如 "FoodAndDrink-0"）的事件，渲染为对话上下文。
4. 对每道 MCQ 调用 LLM（Stage 2），仅基于对话历史提取支撑已知正确答案的
   memory clue 列表。
5. 保存至 ./qa/qa_preference_mcq_000_002.json。
"""

import json
import os
import re
import concurrent.futures
import threading
import argparse
from collections import Counter

from tqdm import tqdm
from json_repair import repair_json

from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.qa.config import profile_path, qa_path

PROFILE_PATH = profile_path("profiles_with_anchors.jsonl")
FORMATTED_DIALOG_PATH = qa_path("qa_formatted_data_000_019.json")
OUTPUT_PATH = qa_path("qa_preference_mcq.json")

# 新增：P1 / P7 / P8 三类 QA 的独立输出文件
OUTPUT_PATH_P1 = qa_path("qa_preference_p1_temporal.json")
OUTPUT_PATH_P7 = qa_path("qa_preference_p7_trigger.json")
OUTPUT_PATH_P8 = qa_path("qa_preference_p8_ownership.json")

CHECKPOINT_EVERY = 10

# ---------------------------------------------------------------------------
# 时段 / 音频关键词字典（供 P1 / P7 使用）
# ---------------------------------------------------------------------------

DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")

# ---------------------------------------------------------------------------
# 时段关键词字典（P1 用）
#
# 【设计原则】
#   1. 6 桶覆盖一天：清晨 → 上午 → 午后 → 傍晚 → 晚上 → 深夜
#   2. 检查顺序有影响：先检查语义最独占的桶（深夜/傍晚），最后检查易混淆的
#      （早上兜底进清晨；下午与午后合并）。
#   3. 关键词均由 scan qa_formatted_data.json 中所有 scene_description 前缀
#      归纳得来，例如：
#          "傍晚六点半"、"周六傍晚"、"傍晚接近"、"夕阳的光"
#          "凌晨一点二十分"、"深夜十一点四十分"、"深夜十二点刚过"
#          "周三上午九点"、"上午十点"
#          "周日早上快十点"、"清晨七点二十分"、"早晨七点刚过"
#          "周五下午三点多"、"下午两点半"、"周六午后两点"、"中午十一点半"
#          "周三晚上七点多"、"晚上九点半"、"周一晚上十点四十分"
# ---------------------------------------------------------------------------
TIME_KEYWORDS = [
    # 深夜 — 优先检查，避免被 "晚上" 吞掉（如 "晚上十一点四十分"）
    ("深夜", ["深夜", "半夜", "凌晨", "后半夜", "三更", "夜深"]),
    # 傍晚 — 明确的黄昏/落日语义
    ("傍晚", ["傍晚", "黄昏", "夕阳", "夕照", "日落", "夜幕", "暮色"]),
    # 晚上 — 在深夜/傍晚之后匹配
    ("晚上", ["晚上", "夜晚", "夜里", "夜间", "晚间", "睡前", "入夜"]),
    # 清晨 — 含"鱼肚白/破晓"等意象；"早上"兜底放在最后
    ("清晨", ["清晨", "早晨", "破晓", "拂晓", "天刚亮", "天蒙蒙亮", "天亮", "大清早", "一早", "鱼肚白"]),
    # 上午 — 独占
    ("上午", ["上午"]),
    # 午后 — 中午 / 午后 / 下午 / 午休 合并（在 qa_formatted_data 中三者语义高度重叠）
    ("午后", ["中午", "午后", "下午", "响午", "午间", "午休"]),
    # 早上 — 兜底：默认进清晨；但 "早上十点/十一点" 由 parse_event_datetime 里的
    # post-processing 上调到上午
    ("清晨", ["早上"]),
]

# 【MCQ 选项池】—— time_of_day 轴的 4 选项从这里挑
TIME_BUCKET_OPTIONS = {
    "清晨": "清晨 5-9 点",
    "上午": "上午 9-12 点",
    "午后": "午后 12-17 点",
    "傍晚": "傍晚 17-19 点",
    "晚上": "晚上 19-22 点",
    "深夜": "深夜 22 点后",
}

# ---------------------------------------------------------------------------
# P7 · 音频关键词来源
#   ✅ 从每条 implicit + audio 偏好的 `analysis`（旧字段名 `rationale`）中
#      每个 bullet 开头的 (xxx声/xxx音/xxx响) 括号里抽取，如：
#        "(咖啡机声)清晰的蒸汽加压和萃取声..."   → "咖啡机声"
#        "(吸尘器声)吸尘器低频运转..."             → "吸尘器声"
#        "(猫叫声)远处隐约的猫叫..."              → "猫叫声"
#   ❌ 不再从 background_audio_info 用词典抽（已删除 AUDIO_LEXICON 相关逻辑）
# ---------------------------------------------------------------------------
# 匹配 analysis 项开头的 (xxx) 或 （xxx）
_AUDIO_KW_PAT = re.compile(r"^\s*[(（]\s*([^)）]+?)\s*[)）]")

MODEL = env_value("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")

MAX_WORKERS = 4
LLM_RETRIES = 3
TEMPERATURE_GEN = 0.9
TEMPERATURE_ANS = 0.2

CATEGORIES = [
    "FoodAndDrink",
    "HomeAndSpace",
    "BodyAndHealth",
    "HobbiesAndEntertainment",
    "WorkAndLearning",
    "MobilityAndTravel",
    "Pets",
]

# ---------------------------------------------------------------------------
# Prompt: Stage 1 — 根据单条偏好生成 1 道 MCQ
# ---------------------------------------------------------------------------
prompt_gen_mcq = '''请根据给定的【用户偏好信息】，设计 **1 道高难度单项选择题（MCQ）**，用于评估智能体能否准确记忆该用户的特定偏好。

[输入信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}（explicit = 用户在对话/图像主体/语音中直接表达；implicit = 通过图像背景/环境音等间接体现）
证据来源: {evidence_sources}
分析依据: {analysis}

[出题要求]
1. 题干必须是简短中文问句，询问用户在 "{subcategory}" 方面的偏好或习惯。
   - 题干中不得直接出现偏好描述的原文关键词，应适当改写。
   - 题干示例：
     - "该用户选餐馆时更倾向哪一项？"
     - "这位用户更偏向哪种起稿方式？"
     - "该用户更可能保留哪种日常习惯？"
2. 给出 **4 个选项 A/B/C/D**，每个选项是简短中文短语（≤25 汉字）。
3. **有且仅有一个**选项与偏好描述语义一致，为正确答案；其余 3 个为干扰项。
4. **高迷惑性干扰项设计原则（核心）**：
   a. 4 个选项必须**结构平行**、**互斥**、**同维度同粒度**，长度和信息密度均衡。
   b. **至少 2 个"强干扰项"**：与正确答案属于同一细分方向，仅在关键细节上不同。
      - 示例：若偏好是"喜欢新开的有设计感的小馆子"，强干扰项可以是"新开且口碑强的小馆"或"老牌且空间布置美的小馆"，而非"路边摊"。
      - 示例：若偏好是"习惯在家自制咖啡"，强干扰项可以是"开始工作前，外带一杯咖啡"或"午休时，手冲一杯咖啡"，而非"从不喝咖啡"。
   c. 其余干扰项也须在同领域内合理可信。
   d. 正确选项**不得**因措辞更具体或更长而在形式上显得突出。
   e. 不要使用"以上都是/都不是""不确定""无法判断"等元选项。
5. **正确答案位置随机化**：正确选项应随机分布在 A/B/C/D 中，不要总放在同一位置。
6. 不要在题面或选项中原样复述偏好描述文本，需适当改写。
7. 要求语言自然、流畅，符合日常用语。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "Q": "题干文本（中文）",
    "options": {{
        "A": "选项 A 文本",
        "B": "选项 B 文本",
        "C": "选项 C 文本",
        "D": "选项 D 文本"
    }},
    "answer": "A"
}}
```
'''

# ---------------------------------------------------------------------------
# Prompt: Stage 2 — 基于已知正确答案提取 memory clue
# ---------------------------------------------------------------------------
prompt_answer_clue = '''你是一个精确的答题系统。你会收到：
1. 【偏好信息】—— 该用户的真实偏好描述，用于理解为什么给定答案是正确的。
2. 【已知正确答案】—— 这道题的正确选项字母，以及对应的选项文本。
3. 【相关对话历史】—— 与该偏好相关的对话记录（含文字、图像描述、音频描述），用于定位 memory clue。
4. 一道单项选择题（A/B/C/D）。

你的任务：
- 不需要重新判断答案，正确答案已经给出。
- 你只需要从【相关对话历史】中找出所有能支撑该正确答案的证据，作为 memory clue 返回。

[输入说明]
【相关对话历史】是与该偏好相关的若干 session 拼接，每条 session 内部按轮次给出对话，并标注了同一轮次中出现的图像/音频证据：
- 每条用户/助手消息以 [Dxx:NN] 表示其轮次编号，其中 Dxx 是 session_id，NN 是该 session 内的轮次序号；
- 用户在某轮次分享的图像描述以 "图像[Dxx-NNN.png]:" 出现；
- 该 session 的背景音频或用户语音消息描述以 "音频[Dxx-NNN.wav]:" 出现。

[偏好信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}
证据来源: {evidence_sources}
分析依据: {analysis}

[已知正确答案]
正确选项: {answer_letter}
正确选项文本: {answer_text}

[提取原则]
1. 以【已知正确答案】为目标，从【相关对话历史】中找出所有能直接支撑该答案的证据。
2. 从三类线索中选用证据：(a) 对话文字、(b) 图像描述、(c) 音频/语音描述。任何一类都可独立支撑答案。
3. **遍历**【相关对话历史】中的全部信息，把**所有**能够作为答案依据的证据都列入 `memory clue`，不要因为已有一条就提前停止。
4. **禁止**把与答案无直接关系的轮次/图像/音频塞进 `memory clue`；只列真正能支撑结论的证据。
5. memory clue 元素格式：
   - 对话证据使用 "Dxx:NN" 格式（必须严格匹配实际出现的轮次编号）；
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
**严格输出**如下 JSON 结构，不要添加任何额外说明文字（注意 "memory clue" 中间是空格）：
```json
{{
    "memory clue": ["D01:03", "D01-001.png"]
}}
```
'''


# ---------------------------------------------------------------------------
# Prompt: P1 · 时段聚合题（time_of_day 轴）
# ---------------------------------------------------------------------------
prompt_gen_p1_temporal = '''请根据【用户偏好信息】以及该偏好在真实事件序列中的【时段分布统计】，
设计 **1 道时段聚合类单项选择题（MCQ）**，考察智能体是否理解用户在该偏好上
**通常发生在一天中的哪个时段**。

[偏好信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}

[事件时段分布]
时间跨度: {date_span}
事件总数: {n_events}
时段直方图: {time_hist}
**主导时段（正解桶）**: {dominant_time_bucket}
每个事件的时段摘要:
{event_time_summary}

[出题原则]
1. 题干含"什么时段" / "哪个时间段" / "通常在一天中的什么时候" 等表述，
   问的是**该用户长期习惯的时段**，而非某次具体事件；
   题干中不得提及具体日期 / session id / 某个具体事件。
2. 4 个选项**必须**从下列固定池中挑 4 个（保持互斥，覆盖不同时段）：
   {time_pool}
3. **正确答案 = 主导时段桶对应的选项文本**：
   例如 dominant_time_bucket 为 "傍晚" → 正解 = "傍晚 17-19 点"。
4. 3 个干扰项也从池中挑与正解**不同**的 3 个桶：
   - 至少 1 个要**时段接近**（如正解 "傍晚"，可放 "晚上" 或 "午后" 作强干扰）；
   - 不要 4 个选项都集中在早/晚一端。
5. **正解位置随机化**（不要总是 A）。
6. 不要在题干或选项中原样复述偏好描述关键词。
7. 不得使用 "以上都是 / 不确定 / 无法判断" 等 meta 选项。

[输出格式]
**严格输出**如下 JSON 对象，不要添加额外说明文字：
```json
{{
    "Q": "题干文本（中文）",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "reason": "1-2 句说明为什么根据时段直方图选此答案"
}}
```
'''

# ---------------------------------------------------------------------------
# Prompt: P7 · 上下文触发题（If-Then）
# ---------------------------------------------------------------------------
prompt_gen_p7_trigger = '''你需要为一个 **implicit + audio** 类偏好设计 1 道
**"排除法" 单项选择题（MCQ）**，考察智能体是否能把"用户在这类环境音下**真实发生过**
的事情" 与 "从未在这类场景中出现过的事情" 区分开。

[任务本质]
- 我们给你一组**该偏好的典型触发音频关键词**（例：吸尘器声 + 猫叫声）；
- 以及该 preference 关联的所有事件（含 scene / background_audio_info / 用户与 AI 的对话）；
- 你需要生成 1 道题，题干问：**"当这些声音出现时，该用户【最不可能】…"**
- 4 个选项：
  * **3 个错误选项**（对应问的是"最不可能"，所以这 3 项其实是"**真实发生过**的"）——
    必须能在下面【相关事件】的**对话内容**或**场景动作**中找到具体依据；
  * **1 个正确选项**（正解）——一个**虚构的**、这些 events **完全没出现过**的活动/话题，
    但仍要是"合理的日常活动"（不能是显然荒诞的东西），并且**与触发音频关联很弱**。

[偏好信息]
偏好类别: {category}
偏好子类: {subcategory}
偏好描述: {preference}
表达类型: {expression_type}

[触发音频关键词（该偏好在 analysis 中开头括号里的 audio 标签）]
{audio_keywords}

[相关事件（含场景、背景音描述、多轮对话）]
{events_summary}

[出题原则]
1. **题干**必须直接说出触发音频关键词（这是"触发条件"，不算答案泄露）。
   从下面二选一（若相关 events 的对话内容比场景动作更丰富，优先用 (b)）：
   (a) "当环境中同时（或依次）出现"X"与"Y"这类声音时，该用户【最不可能】正在做什么？"
   (b) "当环境中同时（或依次）出现"X"与"Y"这类声音时，该用户【最不可能】在和 AI 朋友聊什么话题？"
2. **3 个错误选项**必须能在【相关事件】的 dialog 或 scene 中找到直接依据
   （具体到某个话题/动作，不能是模糊改写）。
3. **1 个正确选项**（正解）：
   - **从未**在【相关事件】的对话或场景描述中出现过；
   - 是一个**合理的**日常活动/话题（例："周末登山路线规划"、"新款设计软件功能"），
     禁止用"骑独角兽上班"这类显然荒诞的内容；
   - **与触发音频关联很弱**：比如触发音是吸尘器 + 猫叫，正解应避开打扫/宠物话题。
4. **4 个选项结构平行**：
   - 若题干用 (b) → 都以"和 AI 聊 XXX" 或类似句式开头；
   - 若题干用 (a) → 都以"正在 XXX" 或类似动词句式开头；
   - 长度均衡（15-30 汉字），避免正解在字面上突出。
5. **正解位置随机化**（不要总是 A）。
6. 不得使用 "以上都是 / 都不是 / 不确定" 类 meta 选项。

[Good case（仅示范格式与思路，禁止照抄内容；实际的 3 个真实选项必须来自本次输入 events）]
音频关键词: 吸尘器声、猫叫声
偏好描述: 养猫且对居家清洁有较高要求，日常频繁使用吸尘器

相关 events 中真实聊过的话题（假设的）:
  - 家里两只猫掉毛严重，打扫时如何用吸尘器处理
  - 想再入一台扫地机器人配合手持吸尘器
  - 猫在打扫时总躲到卧室某个位置

期望输出（示范）:
{{
    "Q": "当环境中同时出现"吸尘器声"和"猫叫声"这类声音时，该用户【最不可能】在和 AI 朋友聊什么话题？",
    "options": {{
        "A": "家里两只猫掉毛时如何用吸尘器打理",
        "B": "考虑再入手一台扫地机器人配合手持吸尘器使用",
        "C": "自己下周准备去挑一双新的登山徒步鞋",
        "D": "猫在自己打扫时躲到卧室的固定位置"
    }},
    "answer": "C",
    "fictitious_reason": "'挑登山徒步鞋' 从未在该组 events 的对话或场景中出现，且与"打扫+猫" 语境弱相关"
}}

[输出格式]
**严格输出**如下 JSON 对象，不要添加额外说明文字：
```json
{{
    "Q": "题干（含触发音频关键词）",
    "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "answer": "A",
    "fictitious_reason": "1 句话说明为什么正解那个选项在本组 events 中未出现过"
}}
```
'''

# ---------------------------------------------------------------------------
# Prompt: P8 · 归属主体判定题
# ---------------------------------------------------------------------------
prompt_gen_p8_ownership = '''请根据【用户档案的多主体信息】以及对话历史中的相关事件片段，
设计 **1 道归属主体判定 MCQ**，考察智能体是否能分辨对话中某个行为/特征
究竟归属于哪个具体主体（用户本人 / 关系人 / 宠物）。

[所有可选主体（选项池）]
{subjects_list}

[目标主体（正解）]
标签: {target_label}
类型: {target_kind}
补充信息: {target_meta}

[与目标主体相关的事件摘要]
{target_events_summary}

[出题原则]
1. 从上面事件摘要中**提炼一个只属于目标主体**的显著行为/特征/属性（越具体越好），
   写入题干。行为示例："对显示器上移动的光标有强烈反应，会扑向屏幕"、
   "右手腕上戴着一条皮质编织手链"、"每天早晨用咖啡机制作现磨咖啡"。
2. **题干**必须描述该行为并询问"这一行为归属于以下哪个主体？"
3. **4 个选项**都是主体（人名或宠物名 + 一小段消歧信息），从"所有可选主体"里挑选：
   正解 = 目标主体；干扰 = 其他 3 个主体（人 or 宠物皆可）。
   若可选主体不足 4 个，可补充"路过的邻居"、"未在档案中出现的陌生人"等通用干扰项。
4. **题干中不得直接出现目标主体的名字**（否则题目就废了）；改用"这一行为"、
   "画面中这个个体" 等指代性说法。
5. 正解位置随机化。
6. 各选项须**结构平行**（同为"姓名 - 关系" 或 同为"宠物名 - 品种"），避免正解格式突出。

[输出格式]
```json
{{
    "Q": "题干（描述行为并问归属）",
    "options": {{"A": "主体标签1", "B": "主体标签2", "C": "主体标签3", "D": "主体标签4"}},
    "answer": "A",
    "behavior_description": "题干里所用的那个行为一句话概括"
}}
```
'''


# ========================= Utility functions =========================

def load_json_or_jsonl(path: str) -> list:
    return load_records(path)


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


def collect_preference_events(formatted_profile: dict, pref_code: str) -> list:
    """Return events whose explicit/implicit_preferences_reflected contains pref_code."""
    matched = []
    for event in formatted_profile.get("events", []) or []:
        codes = []
        codes += event.get("explicit_preferences_reflected", []) or []
        codes += event.get("implicit_preferences_reflected", []) or []
        if pref_code in codes:
            matched.append(event)
    return matched


def format_events_for_prompt(events: list) -> str:
    if not events:
        return "（无相关对话历史）"

    sections = []
    seen = set()
    for event in events:
        session_id = event.get("session_id", "?")
        if session_id in seen:
            continue
        seen.add(session_id)

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
                    section.append(f"    └─ 图像[{k}]: {v}")
                elif k.endswith(".wav"):
                    section.append(f"    └─ 音频[{k}]: {v}")
            if assistant:
                section.append(f"  [{round_id}] Assistant: {assistant}")
        sections.append("\n".join(section))

    return "\n\n".join(sections)


def collect_valid_clue_keys(events: list) -> set:
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
    if not isinstance(clues, list):
        return []
    seen = []
    for c in clues:
        if isinstance(c, str) and c.strip() and c.strip() in valid_keys and c.strip() not in seen:
            seen.append(c.strip())
    return seen


def validate_mcq(item: dict) -> bool:
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


# ========================= P1 / P7 / P8 helpers =========================

def parse_event_datetime(scene_description: str) -> tuple:
    """Return (date_MMDD, time_bucket) from `scene_description` prefix.

    scene 形如 "01/01/2025;林悦家中的客厅;傍晚六点半…"。
    - date_MMDD: 'MM/DD' 字符串或 None
    - time_bucket: 清晨/上午/午后/傍晚/晚上/深夜 之一，或 'unknown'

    【解析步骤】
    1. 正则抓 MM/DD/YYYY；
    2. 按 ';' / '；' 切分，只在**第 3 段的前 40 字**里搜时段关键词
       （避免把 scene 中段像 "如同深夜" 这类比喻错分）；
    3. 按 TIME_KEYWORDS 的顺序命中首个桶；
    4. **post-processing**（消歧规则）：
       - 清晨 + 十点/十一点   → 上午   （"早上快十点" 属上午）
       - 晚上 + 十一点/十二点 → 深夜   （"晚上十一点四十" 属深夜）
       - 午后 + 五点/六点      → 傍晚   （"下午五点左右" 属傍晚）
    """
    if not scene_description:
        return None, "unknown"

    m = DATE_RE.search(scene_description)
    date_mmdd = None
    if m:
        mm, dd, _ = m.groups()
        date_mmdd = f"{int(mm):02d}/{int(dd):02d}"

    # 时段短语通常在第 3 段（第 2 个分号之后）的开头
    # 只取前 15 字：避免像 "上午九点，遮光窗帘把卧室裹得如同深夜" 里
    # 后半段的 "如同深夜" 被误当作时段。数据里所有真实时段短语都 ≤ 12 字。
    parts = re.split(r"[;；]", scene_description, maxsplit=2)
    if len(parts) >= 3:
        head = parts[2][:15]
    else:
        head = scene_description[:25]

    bucket = "unknown"
    for label, kws in TIME_KEYWORDS:
        if any(kw in head for kw in kws):
            bucket = label
            break

    # ---- Post-processing 消歧 ----
    if bucket == "清晨" and any(h in head for h in ["十点", "十一点"]):
        bucket = "上午"
    if bucket == "晚上" and any(h in head for h in ["十一点", "十二点"]):
        bucket = "深夜"
    if bucket == "午后" and any(h in head for h in ["五点", "六点"]):
        bucket = "傍晚"

    return date_mmdd, bucket


def compute_time_stats(events: list) -> dict:
    """聚合事件的时间分布。

    输出的关键字段：
    - n_events / date_span / months_covered：描述性信息（供 prompt 展示）
    - time_hist：{桶名: 命中次数}
    - dominant_time_bucket：主导桶（若不明显则为 None）
    - is_confident：主导桶是否可靠（用于决定是否触发 P1 QA 生成）
    - per_event：每个 event 的 (session_id, date, time_bucket) 明细
    """
    dates: list = []
    buckets: list = []
    per_event: list = []
    for e in events:
        d, b = parse_event_datetime(e.get("scene_description", "") or "")
        if d:
            dates.append(d)
        if b and b != "unknown":
            buckets.append(b)
        per_event.append({"session_id": e.get("session_id", "?"), "date": d, "time_bucket": b})

    hist = Counter(buckets)
    dominant: str | None = None
    is_confident = False
    if hist:
        top = hist.most_common(2)
        dominant_bucket, dominant_count = top[0]
        second_count = top[1][1] if len(top) > 1 else 0
        # 主导桶「可靠」条件：至少命中 2 次，且严格多于第二名
        if dominant_count >= 2 and dominant_count > second_count:
            dominant = dominant_bucket
            is_confident = True

    months = {d[:2] for d in dates}

    return {
        "n_events": len(events),
        "date_span": f"{min(dates)} — {max(dates)}" if dates else "unknown",
        "time_hist": dict(hist),
        "dominant_time_bucket": dominant,
        "is_confident": is_confident,
        "months_covered": len(months),
        "per_event": per_event,
    }


def extract_audio_kws_from_analysis(analysis) -> list:
    """从 preference 的 `analysis` / `rationale` 列表中抽取音频关键词。

    每条 analysis 通常形如 `"(咖啡机声)清晰的蒸汽加压和萃取声..."`；
    我们只取开头 (xxx) 括号里的短标签，且只保留看起来像音频的标签
    （含 "声" / "音" / "响"），忽略 "(视觉)" / "(视频)" 这类其它模态标签。

    返回去重后的关键词列表，**至多 2 个**（多于 2 个只取前 2 个）。
    """
    if not analysis:
        return []
    kws: list = []
    for item in analysis:
        if not isinstance(item, str):
            continue
        m = _AUDIO_KW_PAT.match(item)
        if not m:
            continue
        kw = m.group(1).strip()
        if not kw:
            continue
        # 只保留音频类：含"声/音/响"
        if not any(c in kw for c in ["声", "音", "响"]):
            continue
        # 剔除明显非音频的标签（虽然含"音"字，但语义偏视觉）
        if any(bad in kw for bad in ["视觉", "画面", "图像"]):
            continue
        if kw not in kws:
            kws.append(kw)
        if len(kws) >= 2:      # 超过 2 个只取 2 个
            break
    return kws


def summarize_event_briefly(event: dict, max_chars: int = 120) -> str:
    """Compress an event into a one-line human summary for LLM prompt input."""
    sid = event.get("session_id", "?")
    scene = (event.get("scene_description") or "").strip().replace("\n", " ")
    bg = (event.get("background_audio_info") or "").strip().replace("\n", " ")
    parts = [f"[{sid}]"]
    if scene:
        parts.append(scene[:max_chars])
    if bg and bg.lower() != "none":
        parts.append(f"（背景音: {bg[:60]}）")
    return " ".join(parts)


def gather_subjects(profile: dict) -> list:
    basic = profile.get("Basic", {}) or {}
    subjects: list = []
    if basic.get("name"):
        subjects.append({
            "kind": "self",
            "label": basic["name"],
            "meta": basic.get("occupation", ""),
        })
    for r in basic.get("Relationship", []) or []:
        if r.get("name"):
            subjects.append({
                "kind": "relationship",
                "label": r["name"],
                "meta": r.get("relation", ""),
                "info": r.get("info", ""),
                "appearance": r.get("appearance", ""),
            })
    for p in basic.get("Pets", []) or []:
        if p.get("name"):
            subjects.append({
                "kind": "pet",
                "label": p["name"],
                "meta": p.get("info", ""),
                "appearance": p.get("appearance", ""),
            })
    return subjects


def find_events_mentioning(subject_label: str, formatted_profile: dict, max_events: int = 4) -> list:
    """Return events whose scene_description or dialog contains the subject's name."""
    if not subject_label:
        return []
    matched: list = []
    for event in formatted_profile.get("events", []) or []:
        scene = event.get("scene_description", "") or ""
        if subject_label in scene:
            matched.append(event)
            continue
        dialog_str = " ".join(
            (t.get("user") or "") + " " + (t.get("assistant") or "")
            for t in (event.get("dialog_list") or [])
        )
        if subject_label in dialog_str:
            matched.append(event)
    return matched[:max_events]


def _pref_summary_line(pref: dict) -> str:
    return f"- [{pref.get('subcategory','?')}] {pref.get('preference','')}"


# ========================= Core build function =========================

def build_preference_qa(
    p_id: int,
    category: str,
    pref_idx: int,
    pref: dict,
    formatted_profile: dict,
) -> tuple:
    """Two-stage LLM pipeline for one preference → one MCQ with memory clue."""
    pref_code = f"{category}-{pref_idx}"
    qa_id = f"{p_id}-{category}-{pref_idx}"

    subcategory = pref.get("subcategory", "")
    preference = pref.get("preference", "")
    expression_type = pref.get("expression_type", "")
    evidence_sources = pref.get("evidence_sources", [])
    analysis = pref.get("analysis", [])

    evidence_str = ", ".join(evidence_sources) if isinstance(evidence_sources, list) else str(evidence_sources)
    analysis_str = "\n".join(f"  - {a}" for a in analysis) if isinstance(analysis, list) else str(analysis)

    events = collect_preference_events(formatted_profile, pref_code)
    matched_sessions = sorted({e.get("session_id", "") for e in events if e.get("session_id")})

    tokens_in, tokens_out = 0, 0

    # ---- Stage 1: generate MCQ ----
    prompt1 = prompt_gen_mcq.format(
        category=category,
        subcategory=subcategory,
        preference=preference,
        expression_type=expression_type,
        evidence_sources=evidence_str,
        analysis=analysis_str,
    )

    try:
        mcq, tin, tout = call_llm(prompt1, temperature=TEMPERATURE_GEN)
        tokens_in += tin
        tokens_out += tout
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (gen) failed: {e}")
        return None, 0, 0

    if not validate_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid MCQ: {mcq}")
        return None, tokens_in, tokens_out
    answer_letter = extract_answer_letter(mcq)
    if not answer_letter:
        print(f"[{qa_id}] Stage 1 returned invalid answer: {mcq}")
        return None, tokens_in, tokens_out

    options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
    question = mcq["Q"].strip()
    answer_text = options[answer_letter]

    # ---- Stage 2: memory clue only ----
    dialog_str = format_events_for_prompt(events)
    valid_keys = collect_valid_clue_keys(events)

    prompt2 = prompt_answer_clue.format(
        category=category,
        subcategory=subcategory,
        preference=preference,
        expression_type=expression_type,
        evidence_sources=evidence_str,
        analysis=analysis_str,
        dialog_str=dialog_str,
        question=question,
        opt_a=options["A"],
        opt_b=options["B"],
        opt_c=options["C"],
        opt_d=options["D"],
        answer_letter=answer_letter,
        answer_text=answer_text,
    )

    try:
        ans, tin2, tout2 = call_llm(prompt2, temperature=TEMPERATURE_ANS)
        tokens_in += tin2
        tokens_out += tout2
    except Exception as e:
        print(f"[{qa_id}] Stage 2 (clue) failed: {e}")
        return None, tokens_in, tokens_out

    raw_clues = ans.get("memory clue")
    if raw_clues is None:
        raw_clues = ans.get("memory_clue")
    memory_clue = filter_memory_clues(raw_clues, valid_keys)

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "category": category,
        "subcategory": subcategory,
        "preference": preference,
        "expression_type": expression_type,
        "evidence_sources": evidence_sources,
        "Q": question,
        "options": options,
        "A": answer_letter,
        "memory clue": memory_clue,
        "matched_session_ids": matched_sessions,
        "type": "preference_mcq",
    }
    return record, tokens_in, tokens_out


# ========================= I/O helpers =========================

def _sort_key(r):
    # 兼容 P1/P7/P8 记录：P8 无 category 字段（按主体分类），P1/P7 与原 pref 记录一致有 category
    cat = r.get("category", "")
    cat_idx = CATEGORIES.index(cat) if cat in CATEGORIES else 99
    return (
        r.get("p_id", 0),
        cat_idx,
        r.get("qa_id", ""),
    )


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
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return {r["qa_id"]: r for r in records}
    except Exception as e:
        print(f"WARN: failed to load {path} for resume: {e}")
        return {}


# ========================= P1 · 时段/频率聚合 =========================

def build_p1_qa(
    p_id: int,
    category: str,
    pref_idx: int,
    pref: dict,
    formatted_profile: dict,
) -> tuple:
    pref_code = f"{category}-{pref_idx}"
    qa_id = f"{p_id}-{category}-{pref_idx}-p1"

    events = collect_preference_events(formatted_profile, pref_code)
    if len(events) < 3:
        return None, 0, 0  # 数据不足，静默跳过

    stats = compute_time_stats(events)
    # 主导时段不明显（如 hist={傍晚:2, 上午:2}）→ 无法出可靠的时段题，跳过
    if not stats["is_confident"] or not stats["dominant_time_bucket"]:
        return None, 0, 0

    event_time_summary = "\n".join(
        f"  - {p['session_id']}: date={p['date'] or '未知'}, 时段={p['time_bucket']}"
        for p in stats["per_event"]
    )

    prompt = prompt_gen_p1_temporal.format(
        category=category,
        subcategory=pref.get("subcategory", ""),
        preference=pref.get("preference", ""),
        expression_type=pref.get("expression_type", ""),
        date_span=stats["date_span"],
        n_events=stats["n_events"],
        time_hist=json.dumps(stats["time_hist"], ensure_ascii=False),
        dominant_time_bucket=stats["dominant_time_bucket"],
        event_time_summary=event_time_summary,
        time_pool=json.dumps(list(TIME_BUCKET_OPTIONS.values()), ensure_ascii=False),
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] P1 gen failed: {e}")
        return None, 0, 0

    if not validate_mcq(mcq):
        return None, tin, tout
    answer_letter = extract_answer_letter(mcq)
    if not answer_letter:
        return None, tin, tout

    options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
    matched_sessions = sorted({e.get("session_id", "") for e in events if e.get("session_id")})

    # 二次校验：正解选项文本应包含主导时段桶名（如 "傍晚"）
    dom = stats["dominant_time_bucket"]
    if dom and dom not in options.get(answer_letter, ""):
        print(f"[{qa_id}] P1 warn: 正解 {answer_letter}={options[answer_letter]!r} 未含主导桶 {dom!r}")

    # memory clue：所有涉及事件的轮次 + 相关音频/图像
    valid_keys = collect_valid_clue_keys(events)
    all_clues = []
    for e in events:
        for turn in e.get("dialog_list", []) or []:
            rid = turn.get("round")
            if isinstance(rid, str) and rid:
                all_clues.append(rid)
                # 至多取每 event 前 2 条媒体证据，避免 clue 泛滥
                for k in turn:
                    if k in ("round", "user", "assistant"):
                        continue
                    if k.endswith(".wav") or k.endswith(".png"):
                        all_clues.append(k)
    memory_clue = filter_memory_clues(all_clues, valid_keys)

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "qa_style": "p1_temporal",
        "category": category,
        "subcategory": pref.get("subcategory", ""),
        "preference": pref.get("preference", ""),
        "expression_type": pref.get("expression_type", ""),
        "evidence_sources": pref.get("evidence_sources", []),
        "axis": "time_of_day",
        "temporal_scope": "all",
        "time_stats": {k: v for k, v in stats.items() if k != "per_event"},
        "Q": mcq["Q"].strip(),
        "options": options,
        "A": answer_letter,
        "reason": str(mcq.get("reason", "")).strip(),
        "memory clue": memory_clue,
        "matched_session_ids": matched_sessions,
        "type": "preference_mcq",
    }
    return record, tin, tout


# ========================= P7 · 上下文触发（排除法）=========================

def build_p7_qa(
    p_id: int,
    category: str,
    pref_idx: int,
    pref: dict,
    formatted_profile: dict,
) -> tuple:
    """P7 v2（排除法）：
       - 只处理 implicit + audio 偏好；
       - 音频关键词直接从 pref["analysis"] 每条开头的 (xxx声) 抽取，最多 2 个；
       - 收集该偏好命中的所有 event（含 scene + dialog + bg_audio）；
       - 让 LLM 出题：3 个"真"选项（events 中出现过的话题/动作）+ 1 个"虚构"正解。
    """
    pref_code = f"{category}-{pref_idx}"
    qa_id = f"{p_id}-{category}-{pref_idx}-p7"

    # 1) 过滤：只处理 implicit + audio
    if str(pref.get("expression_type", "")).lower() != "implicit":
        return None, 0, 0
    evidence = [str(x).lower() for x in (pref.get("evidence_sources") or [])]
    if "audio" not in evidence:
        return None, 0, 0

    # 2) 从 analysis / rationale 抽音频关键词，最多 2 个
    analysis = pref.get("analysis") or pref.get("rationale") or []
    audio_kws = extract_audio_kws_from_analysis(analysis)
    if not audio_kws:
        return None, 0, 0

    # 3) 收集该 preference 命中的全部 events
    events = collect_preference_events(formatted_profile, pref_code)
    if len(events) < 2:
        return None, 0, 0  # 少于 2 个 event → 对话素材不够构造 3 个真实干扰项

    # 4) 用现有 format_events_for_prompt 拿到含 scene + dialog + 图/音引用的完整摘要
    events_summary = format_events_for_prompt(events)

    prompt = prompt_gen_p7_trigger.format(
        category=category,
        subcategory=pref.get("subcategory", ""),
        preference=pref.get("preference", ""),
        expression_type=pref.get("expression_type", ""),
        audio_keywords="、".join(audio_kws),
        events_summary=events_summary,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] P7 gen failed: {e}")
        return None, 0, 0

    if not validate_mcq(mcq):
        return None, tin, tout
    answer_letter = extract_answer_letter(mcq)
    if not answer_letter:
        return None, tin, tout

    options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}
    matched_sessions = sorted({e.get("session_id", "") for e in events if e.get("session_id")})

    # memory clue：由于 P7 是"排除法"——正解为虚构、无 clue 可支撑；
    # 这里的 clue 服务于"这 3 项确实发生过、所以第 4 项才是异常" 的推理路径，
    # 因此仍收集所有相关 event 的对话轮次 + .wav 键
    valid_keys = collect_valid_clue_keys(events)
    all_clues = []
    for e in events:
        for turn in e.get("dialog_list", []) or []:
            rid = turn.get("round")
            if isinstance(rid, str) and rid:
                all_clues.append(rid)
            for k in turn:
                if k.endswith(".wav"):
                    all_clues.append(k)
    memory_clue = filter_memory_clues(all_clues, valid_keys)

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "qa_style": "p7_trigger",
        "category": category,
        "subcategory": pref.get("subcategory", ""),
        "preference": pref.get("preference", ""),
        "expression_type": pref.get("expression_type", ""),
        "evidence_sources": pref.get("evidence_sources", []),
        "trigger_audio_keywords": audio_kws,
        "trigger_pref_code": pref_code,
        "Q": mcq["Q"].strip(),
        "options": options,
        "A": answer_letter,
        "fictitious_reason": str(mcq.get("fictitious_reason", "")).strip(),
        "memory clue": memory_clue,
        "matched_session_ids": matched_sessions,
        "type": "preference_mcq",
    }
    return record, tin, tout


# ========================= P8 · 归属主体 =========================

def build_p8_qa(
    p_id: int,
    subject_idx: int,
    subject: dict,
    all_subjects: list,
    profile: dict,
    formatted_profile: dict,
) -> tuple:
    target_label = subject.get("label", "")
    qa_id = f"{p_id}-subject{subject_idx:02d}-p8"

    # 目标主体相关的事件
    events = find_events_mentioning(target_label, formatted_profile, max_events=4)
    if not events:
        return None, 0, 0  # 找不到相关事件，跳过

    subjects_list_str = "\n".join(
        f"  - [{s['kind']}] {s['label']} — {s.get('meta','')[:60]}"
        for s in all_subjects
    )
    target_events_summary = "\n".join(summarize_event_briefly(e, max_chars=180) for e in events)

    prompt = prompt_gen_p8_ownership.format(
        subjects_list=subjects_list_str,
        target_label=target_label,
        target_kind=subject.get("kind", ""),
        target_meta=(subject.get("meta", "") + " " + subject.get("appearance", "")).strip(),
        target_events_summary=target_events_summary,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] P8 gen failed: {e}")
        return None, 0, 0

    if not validate_mcq(mcq):
        return None, tin, tout
    answer_letter = extract_answer_letter(mcq)
    if not answer_letter:
        return None, tin, tout

    options = {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]}

    # 校验：正确选项文本应等于/包含 target_label
    if target_label and target_label not in options.get(answer_letter, ""):
        print(f"[{qa_id}] P8 warn: 正解 {answer_letter}={options.get(answer_letter)} 未含 target={target_label}")

    matched_sessions = sorted({e.get("session_id", "") for e in events if e.get("session_id")})

    # memory clue：直接用相关事件的所有轮次 + 首张图
    valid_keys = collect_valid_clue_keys(events)
    all_clues = []
    for e in events:
        for turn in e.get("dialog_list", []) or []:
            rid = turn.get("round")
            if isinstance(rid, str) and rid:
                all_clues.append(rid)
            for k in turn:
                if k.endswith(".png") or k.endswith(".wav"):
                    all_clues.append(k)
    memory_clue = filter_memory_clues(all_clues, valid_keys)

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "qa_style": "p8_ownership",
        "subject_options": options,
        "target_subject": target_label,
        "target_kind": subject.get("kind", ""),
        "behavior_description": str(mcq.get("behavior_description", "")).strip(),
        "Q": mcq["Q"].strip(),
        "options": options,
        "A": answer_letter,
        "memory clue": memory_clue,
        "matched_session_ids": matched_sessions,
        "type": "preference_mcq",
    }
    return record, tin, tout


# ========================= P1 / P7 / P8 main 函数 =========================

def _run_generic_main(
    build_fn,
    task_iter_fn,
    output_path: str,
    style_label: str,
    profiles: list,
    formatted_by_pid: dict,
):
    """Shared driver for P1/P7/P8 main flows.

    - build_fn: (task_args) -> (record, tin, tout)
    - task_iter_fn: (profiles, formatted_by_pid, existing) -> list of task_args tuples
    """
    existing = _load_existing(output_path)
    tasks = task_iter_fn(profiles, formatted_by_pid, existing)

    if not tasks:
        print(f"[{style_label}] 无待生成任务（已全部完成或数据不满足触发条件）。")
        return

    print(f"[{style_label}] 待生成 {len(tasks)} 条 QA（已有 {len(existing)} 条）")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = 0
    total_out = 0
    new_since_cp = 0

    def _on_result(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin
        total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), output_path)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for args_tuple in tasks:
            fut = executor.submit(build_fn, *args_tuple)
            futures[fut] = None
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Building {style_label} QAs",
        ):
            record, tin, tout = future.result()
            with lock:
                _on_result(record, tin, tout)

    _save_checkpoint(list(qa_map.values()), output_path)

    print(f"\n[{style_label}] 总记录数: {len(qa_map)}")
    print(f"[{style_label}] Tokens — prompt: {total_in}, completion: {total_out}")
    print(f"[{style_label}] Output -> {output_path}")


def _load_profiles_common(max_profiles: int) -> tuple:
    profiles = load_json_or_jsonl(PROFILE_PATH)
    if not profiles:
        print(f"ERROR: cannot load {PROFILE_PATH}.")
        return None, None
    if max_profiles > 0:
        profiles = profiles[:max_profiles]

    formatted = load_json_or_jsonl(FORMATTED_DIALOG_PATH)
    if not formatted:
        print(f"ERROR: cannot load {FORMATTED_DIALOG_PATH}.")
        return None, None
    formatted_by_pid = {fp.get("p_id", i): fp for i, fp in enumerate(formatted)}
    return profiles, formatted_by_pid


def main_p1(max_profiles: int = 0):
    profiles, formatted_by_pid = _load_profiles_common(max_profiles)
    if profiles is None:
        return

    def _iter(profiles, fp_by_pid, existing):
        tasks = []
        for p_idx, profile in enumerate(profiles):
            p_id = profile.get("id", p_idx)
            fp = fp_by_pid.get(p_id)
            if fp is None:
                continue
            for category in CATEGORIES:
                for pref_idx, pref in enumerate(profile.get(category, []) or []):
                    qa_id = f"{p_id}-{category}-{pref_idx}-p1"
                    if qa_id in existing:
                        continue
                    tasks.append((p_id, category, pref_idx, pref, fp))
        return tasks

    _run_generic_main(build_p1_qa, _iter, OUTPUT_PATH_P1, "P1", profiles, formatted_by_pid)


def main_p7(max_profiles: int = 0):
    profiles, formatted_by_pid = _load_profiles_common(max_profiles)
    if profiles is None:
        return

    def _iter(profiles, fp_by_pid, existing):
        tasks = []
        for p_idx, profile in enumerate(profiles):
            p_id = profile.get("id", p_idx)
            fp = fp_by_pid.get(p_id)
            if fp is None:
                continue
            for category in CATEGORIES:
                for pref_idx, pref in enumerate(profile.get(category, []) or []):
                    qa_id = f"{p_id}-{category}-{pref_idx}-p7"
                    if qa_id in existing:
                        continue
                    tasks.append((p_id, category, pref_idx, pref, fp))
        return tasks

    _run_generic_main(build_p7_qa, _iter, OUTPUT_PATH_P7, "P7", profiles, formatted_by_pid)


def main_p8(max_profiles: int = 0):
    profiles, formatted_by_pid = _load_profiles_common(max_profiles)
    if profiles is None:
        return

    def _iter(profiles, fp_by_pid, existing):
        tasks = []
        for p_idx, profile in enumerate(profiles):
            p_id = profile.get("id", p_idx)
            fp = fp_by_pid.get(p_id)
            if fp is None:
                continue
            subjects = gather_subjects(profile)
            if len(subjects) < 2:
                continue
            # 每个主体生成 1 题（含 self，共 N 题）
            for sub_idx, subject in enumerate(subjects):
                qa_id = f"{p_id}-subject{sub_idx:02d}-p8"
                if qa_id in existing:
                    continue
                tasks.append((p_id, sub_idx, subject, subjects, profile, fp))
        return tasks

    _run_generic_main(build_p8_qa, _iter, OUTPUT_PATH_P8, "P8", profiles, formatted_by_pid)


# ========================= Main =========================

def main(max_profiles: int = 0):
    if max_profiles < 0:
        raise ValueError(f"--max-profiles 不能为负数: {max_profiles}")

    profiles = load_json_or_jsonl(PROFILE_PATH)
    if not profiles:
        print(f"ERROR: cannot load {PROFILE_PATH}.")
        return
    print(f"Loaded {len(profiles)} profile(s) from {PROFILE_PATH}.")
    if max_profiles > 0:
        profiles = profiles[:max_profiles]
        print(f"[max-profiles] 仅处理前 {max_profiles} 个 profile。")

    formatted_profiles = load_json_or_jsonl(FORMATTED_DIALOG_PATH)
    if not formatted_profiles:
        print(f"ERROR: cannot load {FORMATTED_DIALOG_PATH}.")
        return
    total_events = sum(len(p.get("events", []) or []) for p in formatted_profiles)
    print(
        f"Loaded {len(formatted_profiles)} formatted profile(s) "
        f"({total_events} events) from {FORMATTED_DIALOG_PATH}."
    )
    formatted_by_pid = {fp.get("p_id", i): fp for i, fp in enumerate(formatted_profiles)}

    existing = _load_existing(OUTPUT_PATH)
    if existing:
        print(f"[resume] 已有 {len(existing)} 条记录，将跳过已完成的 qa_id。")

    all_tasks = []
    total_prefs = 0
    for p_idx, profile in enumerate(profiles):
        p_id = profile.get("id", p_idx)
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            print(f"[p_id={p_id}] no formatted profile found, skip.")
            continue

        for category in CATEGORIES:
            pref_list = profile.get(category, []) or []
            for pref_idx, pref in enumerate(pref_list):
                qa_id = f"{p_id}-{category}-{pref_idx}"
                if qa_id in existing:
                    continue
                total_prefs += 1
                all_tasks.append((p_id, category, pref_idx, pref, fp))

    if not all_tasks:
        print("所有 preference QA 均已生成，无需重新运行。")
        return

    print(f"待生成: {len(all_tasks)} 条偏好 QA (已有 {len(existing)} 条记录)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = 0
    total_out = 0
    new_since_checkpoint = 0

    def _on_result(record, tin, tout):
        nonlocal total_in, total_out, new_since_checkpoint
        total_in += tin
        total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_checkpoint += 1
        if new_since_checkpoint >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_checkpoint = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for p_id, category, pref_idx, pref, fp in all_tasks:
            fut = executor.submit(
                build_preference_qa, p_id, category, pref_idx, pref, fp,
            )
            futures[fut] = None

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Building preference MCQs",
        ):
            record, tin, tout = future.result()
            with lock:
                _on_result(record, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)

    all_records = list(qa_map.values())
    by_cat = {}
    by_type = {"explicit": 0, "implicit": 0}
    for r in all_records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        et = r.get("expression_type", "")
        by_type[et] = by_type.get(et, 0) + 1

    print(f"\n总记录数: {len(all_records)} (本次新增 {len(all_records) - len(existing)})")
    print("By category:")
    for c in CATEGORIES:
        if c in by_cat:
            print(f"  - {c}: {by_cat[c]}")
    print(f"By expression_type: {by_type}")
    print(f"Tokens — prompt: {total_in}, completion: {total_out}")
    print(
        f"Estimated cost (input $1/M, output $3/M): "
        f"${(total_in * 1e-6 + total_out * 3e-6):.4f}"
    )
    print(f"Output -> {OUTPUT_PATH}")


def _parse_types(t: str) -> list:
    """Parse --type flag. Accept 'all' / 'original' / 'p1' / 'p7' / 'p8' or comma-separated combos."""
    t = (t or "").strip().lower()
    if t in ("", "original"):
        return ["original"]
    if t == "all":
        return ["original", "p1", "p7", "p8"]
    out = []
    for tok in t.split(","):
        tok = tok.strip()
        if tok in ("original", "p1", "p7", "p8") and tok not in out:
            out.append(tok)
    if not out:
        raise ValueError(f"--type 无法解析: {t!r}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 preference 类单选题 QA（含 P1/P7/P8）")
    parser.add_argument(
        "--max-profiles",
        type=int,
        default=0,
        help="只处理前 N 个 profile，用于小批量测试；0 表示处理全部。",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="original",
        help=(
            "QA 类型：original | p1 | p7 | p8 | all；"
            "或逗号分隔组合（如 'p1,p7,p8'）。默认 'original' 保持向后兼容。"
        ),
    )
    args = parser.parse_args()
    types_to_run = _parse_types(args.type)
    print(f"[gen_qa_preference] 将运行以下类型: {types_to_run}")

    if "original" in types_to_run:
        print("=" * 60)
        print("Running ORIGINAL preference MCQ pipeline")
        print("=" * 60)
        main(max_profiles=args.max_profiles)
    if "p1" in types_to_run:
        print("=" * 60)
        print("Running P1 (time_of_day aggregation)")
        print("=" * 60)
        main_p1(max_profiles=args.max_profiles)
    if "p7" in types_to_run:
        print("=" * 60)
        print("Running P7 (contextual trigger)")
        print("=" * 60)
        main_p7(max_profiles=args.max_profiles)
    if "p8" in types_to_run:
        print("=" * 60)
        print("Running P8 (ownership distinction)")
        print("=" * 60)
        main_p8(max_profiles=args.max_profiles)
