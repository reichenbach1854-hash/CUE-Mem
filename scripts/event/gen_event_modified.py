from pathlib import Path
import csv
import json
import os
import re
import random
import time
import threading
import concurrent.futures
import argparse
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    from json_repair import repair_json
except ImportError:
    def repair_json(text: str) -> str:
        return text

from scripts.common.llm import openai_client
from scripts.common.paths import project_path, resolve_path

random.seed(46689)

# 不同人物之间并行的 worker 数量
MAX_PROFILE_WORKERS = 10

_print_lock = threading.Lock()

def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)

PERSONA_FILE_PATH = str(project_path("profile", "profiles_with_anchors.jsonl"))
# 默认优先读取新的多-implicit 手工分组文件；若不存在，再回退到旧文件。
MANUAL_GROUPS_FILENAME = str(project_path("event", "manual_profiles_with_anchors_groups.json"))
SAVE_PROFILE_PATH = str(project_path("event", "events_with_anchors.jsonl"))
SAVE_GROUPS_PATH = str(project_path("event", "events_with_anchors_groups.json"))
GROUPS_WITH_DATES_PATH = str(project_path("event", "groups_with_dates.json"))
PREFERENCE_TIME_SPAN_REPORT_CSV = str(project_path("event", "preference_time_span_report.csv"))
ENTITY_ANCHOR_COVERAGE_REPORT_CSV = str(project_path("event", "entity_anchor_coverage_report.csv"))

# ── 手动重新生成模式相关路径 ──────────────────────────────────────────────────
MANUAL_REGEN_EVENTS_PATH = str(project_path("event", "events_with_anchors.jsonl"))

# manual groups 预筛查策略：
# - "skip_invalid": 打印告警并跳过非法 group
# - "error": 发现非法 group 直接抛错终止
MANUAL_GROUP_VALIDATION_MODE = "skip_invalid"

# user_shared_image_description 生成方式：
# - "two_step": 先生成显式前景，再生成隐式背景（旧逻辑，默认）
# - "combined": 一次性生成完整图片描述，但 prompt 中仍区分 foreground/background
IMAGE_DESCRIPTION_MODE = "two_step"

MODEL = os.environ.get("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")

prompt_combined = '''根据用户档案、指定偏好组合和推荐主场景，构建 1 个用户与AI朋友分享日常的对话事件。日期在 2025年1月1日 到 2025年12月31日 之间。

[本次推荐主场景]
{recommended_main_scene}

[隐式音频潜入说明]
{implicit_audio_integration_guidance}

[本次生成要求]
1. 必须优先围绕上面的"推荐主场景"组织事件，保持在**同一空间场景**中完成表达。
2. 如果推荐主场景里已经暗示了地点、人物关系或事件气氛，请优先沿用，不要随意切换到第二个空间。
3. 本次只允许围绕下面给出的 **1 个显式偏好** 和 **0 或 1 个隐式偏好** 进行表达，不要额外扩展到其它偏好类别。若[本次隐式偏好]为"无"，本次是 Relationship standalone 或显式-only 事件，不得编造任何隐式偏好线索。
4. 输出中的 explicit_preferences_reflected 和 implicit_preferences_reflected 必须只包含本次真正体现到的类别ID。
5. 如果上方提供了"隐式音频潜入说明"，只能用于构造 background_audio_info；不要把其中的隐式物品、动作、环境或音频线索写入 scene_description。
6. scene_description 必须严格使用格式：MM/DD/YYYY;地点;事件描述。日期、地点、事件描述之间必须使用英文分号`;`分隔，不要使用中文逗号、顿号或句号替代分隔符。

[核心规则]
1. **偏好表达方式**
- 显式偏好：dialogue（直接讨论）、visual（图像主体）、audio（人声内容）
- 隐式偏好：visual（图像边缘/背景）、audio（背景音，非人声）
- **隐式偏好绝对不能在 scene_description 中被提及！！！**
- 显式偏好要在 scene_description 中自然体现，但不要直接用"我喜欢......"这类硬表达。（visual 模态的图像描述将单独生成）

2. **精准覆盖原则（强制）**
- 所有显式偏好必须在 scene_description 中明确体现（visual 模态的图像描述将在后续单独生成，此处无需填写）
- 隐式偏好中的 audio 模态必须在 background_audio_info 中体现；（visual 模态的图像描述将在后续单独生成，此处无需填写）
- 如果某条隐式偏好的模态中含有 audio，必须在 background_audio_info 中体现其音频部分，且必须使用该隐式偏好中analysis字段中开头（）括号内的音频关键词，例如：
  e.g. "analysis": [
          "（铅笔书写声）沙沙的铅笔书写声持续，表明他正在手写笔记",
          "（翻动书页的声音）间歇性的翻页声提示他正在查阅纸质文献，而非电子屏幕",
          "（视觉）书桌一角摊开的笔记本上可见密布的铅笔字迹，与翻书声同步，说明他习惯在阅读时即时批注"
        ]
    则需要优先使用“铅笔书写声”和“翻动书页的声音”，用完括号内关键词后，可适当使用其他音频关键词。
- 当[隐式音频潜入说明]不是"无"时，background_audio_info 必须优先参考该说明；但 scene_description 仍然禁止泄露任何隐式偏好。
- 当[本次隐式偏好]为"无"时，implicit_preferences_reflected 必须输出 []，background_audio_info 填 none，不要为了丰富场景而添加隐式物品、动作或背景音。
- 如果你发现当前写法难以融入推荐主场景，请调整事件细节，但不要放弃任何一个给定偏好

3. **自然表达**
- 场景是真实生活切片，不是剧情梗概
- 人物、宠物、物品首次出现时可自然介绍，但不要重复冗长介绍
- 不要把隐式偏好说破，只能让它作为背景线索存在

4. **scene_description 写作重点（强制）**
- scene_description 的核心是“用户正在和 AI 朋友分享一段日常”，不是静态环境描写，也不是照片画面说明。
- 必须写出用户对 AI 朋友说的话，优先使用“她/他拿起手机跟AI朋友语音：'……'”“她/他对AI朋友说：'……'”这类自然表达。
- 环境、物品、人物外貌只能作为事件铺垫和对话上下文，不能成为主体；不要把 scene_description 写成纯画面描述、摄影描述或物品清单。
- 用户的表达要有具体生活细节、原因、感受、回忆或评价；避免只写一句泛泛的“今天很开心/这个很不错”。
- 当显式偏好是 Relationship-* 或 BasicPets-* 时，scene_description 不能只写用户与亲友/宠物互动；必须写用户把关于该亲友/宠物的事情分享给 AI 朋友。亲友/宠物可以在场，但叙事中心仍是用户对 AI 朋友的分享。
- 当显式偏好是物品、空间、活动或习惯时，scene_description 应写用户正在做这件事，并向 AI 朋友解释、吐槽、展示或回忆相关细节。
- scene_description 可以较长，建议 180-420 个汉字；要有具体动作和对话推进，但不要冗长堆砌环境细节。

5. **适当结合历史事件**
- 不重复已讨论话题，不重复介绍已经介绍过的人物/宠物/物品基本信息
- 新事件应体现新的内容或新的进展
- 可以适当 call back 之前发生的事，但不要复述整段历史

6. **Relationship中人物和Pets中宠物的使用**
- 当[本次显式偏好]中没有给出Relationship和Pets时，不能使用[此前发生的事件]、[已发生的事件]中出现过的人物和宠物。
- 只有当[本次显式偏好]中给出了Relationship和Pets时，才能使用给出的人物和宠物。

7. **模态一致原则**
- 如果遇到显式图像信息，可以在场景描述中提及"发了一张照片/随手拍了张图"
- 但是隐式图像信息、隐式音频信息不能在场景描述中被点明
- 保证模态一致性，不要随意更改偏好的表达模态

8. **Rationale 使用原则（强制）**
- 每条偏好下的 "Rationale" 是从原始语料里抽取出的**证据片段**，用于帮助你理解该偏好为何成立，以便把握语气、细节选择、物品/场景倾向。
- **严禁把 Rationale 中的句子原样照搬/整段复述到 scene_description / human_speech_content / background_audio_info 中任何字段。**
- 允许把 Rationale 的意思**改写、拆解、换一个新的具体切片**重新表达。

[scene_description Good Cases]
Good Case 1（普通显式偏好）：
"06/11/2025;林晓澜家中书架旁的阅读角;周六下午三点,林晓澜蹲在书架前整理新到的设计类书籍。她从纸箱里抽出一本厚重的字体年鉴,翻了翻扉页,又在已经很满的书架第三层腾出位置。她拿起手机跟AI朋友语音:'你看我这书架,上个月刚整理过,这个月又快满了。这套包装设计年鉴从柏林寄来,四本加起来快十公斤,快递员还问我是不是开书店的。'她往后退了两步,看着整面墙的书:'我知道该控制一下,但做项目卡住的时候,翻这些实体书真的比刷素材库有用。纸张质感和翻页节奏会让我慢下来,很多构图想法就是在这里冒出来的。'"

Good Case 2（Relationship 显式偏好）：
"07/03/2025;林晓澜家中客厅;周六下午四点,陈墨然斜靠在沙发扶手旁,复古相机包搁在茶几上,宽松工装裤膝盖处沾着一点灰。林晓澜翻着他带来的样片,指着其中一张黑白建筑照片说:'你这张右边窗框撑住了画面重心,但第三张角度还能再低一点。'陈墨然笑着说她眼睛还是这么毒。林晓澜拿起手机跟AI朋友语音:'墨然今天过来送上次合作项目的样片,我们俩又开始互相挑刺了。分手之后反而轻松很多,少了情侣之间的负担,聊创作的时候特别纯粹。他还是那种散漫的状态,但审美一直在线,所以每次讨论作品都挺有意思。'"

[输出格式]
```json
[
    {{
        "scene_description": "MM/DD/YYYY;地点;事件描述（必须用英文分号分隔；不可泄露隐式信息，重点突出显式信息）",
        "user_shared_image_description": "none",
        "background_audio_info": "背景音描述（仅当隐式偏好含 audio 模态时填写，否则填 none）",
        "human_speech_content": "none",
        "explicit_preferences_reflected": ["显式偏好类别列表，例如：[\"HobbiesAndEntertainment-0\"]"],
        "implicit_preferences_reflected": ["隐式偏好类别列表；若本次隐式偏好为无，则输出 []"]
    }}
]
```

[用户档案]
{profile_str}

[本次显式偏好]
{explicit_preferences_str}

[本次隐式偏好]
{implicit_preferences_str}

[此前发生的事件]
{previous_events_str}
'''


# ─────────────────────────────────────────────────────────────────────────────
# 图像描述两步生成 prompt
# ─────────────────────────────────────────────────────────────────────────────
PROMPT_IMAGE_FOREGROUND = (
    "你的任务是根据给定的显式偏好和场景描述，为一张第一视角照片生成**画面主体（前景）**的描述。\n\n"
    "## 【规则】\n"
    "1. **视角**：第一人称拍摄的照片，描述中不能出现拍摄者本人。描述时一定要出现\"第一人称视觉下......\"\"第一人称拍摄\"等类似表述。\n"
    "2. **必须点明具体场所/环境**：必须根据下方[场景描述]提取一个明确空间，例如客厅、厨房、卧室、玄关、书房、咖啡厅、餐厅、户外步道等；描述开头应类似\"第一人称视觉下的客厅里......\"，不能只写孤立物品。\n"
    "3. **画面主体/中心**：显式偏好所代表的内容必须占据画面主体，位于画面中央或视觉焦点位置。描述时一定要出现\"画面主体是......\"，\"画面中心是......\"等类似表述。\n"
    "4. **Entity Anchor 严格来源**：entity_anchors_selected 只能从下方显式偏好中明确列出的 entity_anchors 逐字复制，不能从 content、人物外貌、宠物特征或场景描述中自行创造新 anchor。若没有明确列出的 entity_anchors，必须返回空数组 []。\n"
    "5. 如果下方提供了【本次必须覆盖的 Entity Anchors】，必须优先使用这些 anchor：它们必须同时出现在 description 和 entity_anchors_selected 中；不要替换、改写或遗漏。\n"
    "6. 若显式偏好包含 Relationship（人物）或 Pet（宠物），相关人物/宠物必须在画面主体中出镜，至少露脸，尽量露出脸和上半身。\n"
    "7. 如果下方【必须逐字出现的人物/宠物名称】不是“无”，description 中必须逐字包含其中列出的每一个名称；不要只写“雪貂”“猫”“朋友”“孩子”等泛称。\n"
    "8. 如果下方【上一轮图片描述错误】不是“无”，必须逐条修正这些错误，尤其是缺少人物/宠物名的问题。\n"
    "9. 图像描述不要过于复杂，避免堆叠外貌、材质、光影等细节，不要像摄影说明书，描述控制在 **80 字以内**。\n\n"
    "## 【输出格式】\n"
    "必须输出合法 JSON，格式如下，不要输出任何 JSON 以外的内容：\n"
    "```json\n"
    "{{\n"
    '  \"description\": \"画面主体描述文本（70字以内）\",\n'
    '  \"entity_anchors_selected\": [\"挑选的锚点1\", \"挑选的锚点2\"]\n'
    "}}\n"
    "```\n\n"
    "[本次拍摄者的名字]\n{photographer_name}\n\n"
    "[场景描述]\n{scene_description}\n\n"
    "[本次必须覆盖的 Entity Anchors]\n{required_entity_anchors}\n\n"
    "[必须逐字出现的人物/宠物名称]\n{required_person_pet_names}\n\n"
    "[上一轮图片描述错误]\n{image_retry_feedback}\n\n"
    "[显式偏好及其 Entity Anchors]\n{explicit_prefs_with_anchors}\n"
)

PROMPT_IMAGE_BACKGROUND = (
    "你的任务是根据给定的隐式偏好，为一张已有画面主体描述的第一视角照片，补充**画面边缘和角落**的细节描述。\n\n"
    "## 【规则】\n"
    "1. **位置约束**：隐式偏好所代表的线索**只能出现在画面边缘、角落或背景中**，绝对不能出现在画面中央或主体位置。描述时一定要出现\"画面边缘......\"、\"画面角落......\"、\"背景处......\"、\"左/右下角......\"等类似表述。\n"
    "2. **与主体保持一致**：你的补充描述必须与下方提供的「画面主体描述」属于同一空间、同一场景，不能出现明显的场景矛盾。\n"
    "3. **隐式偏好归属**：下方隐式 entity anchors 是 profile user / 拍摄者生活空间中的边缘线索，不是 Relationship 人物、亲友或宠物的偏好；不要写成某位亲友拥有、使用、摆放或展示这些物品。\n"
    "4. **Entity Anchor 严格来源**：entity_anchors_selected 只能从下方隐式偏好中明确列出的 entity_anchors 逐字复制，不能从 content、rationale、环境描述或场景描述中自行创造新 anchor。若没有明确列出的 entity_anchors，必须返回空数组 []。\n"
    "5. 如果下方提供了【本次必须覆盖的 Entity Anchors】，必须优先使用这些 anchor：它们必须同时出现在 description 和 entity_anchors_selected 中；不要替换、改写或遗漏。\n"
    "6. **小面积局部露出**：隐式 Entity Anchor 只能露出一小部分、占画面很小比例，例如只露出一角、边缘、局部轮廓、半截或模糊一小块；不能完整清晰出镜，不能成为第二主体。\n"
    "7. 不要重复描述画面主体已经出现的内容，也不要在边缘区域引入显式偏好相关内容。\n"
    "8. 边缘描述只写一句，保持轻描淡写，不要展开解释。描述控制在 **35 字以内**。\n"
    "9. 若[隐式视觉潜入说明]出现了对background_audio_info说明的内容，请忽略；只使用[隐式视觉潜入说明]中对画面说明的部分。\n\n"
    "## 【输出格式】\n"
    "必须输出合法 JSON，格式如下，不要输出任何 JSON 以外的内容：\n"
    "```json\n"
    "{{\n"
    '  \"description\": \"画面边缘描述文本（30字以内）\",\n'
    '  \"entity_anchors_selected\": [\"挑选的锚点1\"]\n'
    "}}\n"
    "```\n\n"
    "[场景描述（仅供空间背景参考）]\n{scene_description}\n\n"
    "[画面主体描述（已生成，请保持与其场景一致）]\n{foreground_description}\n\n"
    "[隐式视觉潜入说明]\n{implicit_visual_integration_guidance}\n\n"
    "[本次必须覆盖的 Entity Anchors]\n{required_entity_anchors}\n\n"
    "[隐式偏好及其 Entity Anchors]\n{implicit_prefs_with_anchors}\n"
)

PROMPT_IMAGE_COMBINED = (
    "你的任务是根据给定的场景、显式偏好和隐式视觉偏好，一次性生成一张第一视角照片的 user_shared_image_description。\n"
    "虽然本次只调用一次，但你必须在构图上严格区分 foreground 和 background。\n\n"
    "## 【核心构图规则】\n"
    "1. **第一人称视角**：这是 profile user 第一人称拍摄或分享的照片，描述中不能出现拍摄者本人。描述必须以\"第一人称视觉下\"或\"第一人称拍摄\"开头。\n"
    "2. **必须点明具体场所/环境**：必须根据下方[场景描述]提取一个明确空间，例如客厅、厨房、卧室、玄关、书房、咖啡厅、餐厅、户外步道等；描述开头应类似\"第一人称视觉下的客厅里......\"，不能只写孤立物品。\n"
    "3. **foreground / 画面主体**：显式偏好必须作为画面主体或视觉焦点，出现在画面中心、近处或最清楚的位置。描述中必须明确写\"画面主体\"或\"画面中心\"。\n"
    "4. **background / 画面边缘**：隐式视觉偏好只能作为边缘、角落或背景里的轻微信息出现，不能成为主体，不能完整展开，不能抢占画面中心。\n"
    "5. **隐式偏好归属**：隐式 entity anchors 是 profile user / 拍摄者生活空间中的边缘线索，不是 Relationship 人物、亲友或宠物的偏好；不要写成某位亲友拥有、使用、摆放或展示这些物品。\n"
    "6. 如果存在隐式视觉 entity anchor，它只能在画面边缘、角落、背景处露出一小部分，占画面很小比例，例如只露出一角、边缘、局部轮廓、半截或模糊一小块；描述时必须使用\"画面边缘\"、\"角落\"、\"背景处\"、\"左/右下角\"、\"一小部分\"、\"隐约能看到\"、\"一小截\"等词。\n"
    "7. 显式主体和隐式边缘线索必须属于同一空间场景，不能出现空间矛盾或两个场景拼贴。\n"
    "8. entity_anchors_selected 只能从下方明确列出的 foreground/background Entity Anchors 或偏好 entity_anchors 中逐字复制，不能从 Relationship/Pets 外貌、宠物特征、content、rationale 或场景描述中自行创造新 anchor；若没有明确列出的 entity_anchors，必须返回空数组 []。\n"
    "9. 如果下方提供了【本次必须覆盖的 foreground Entity Anchors】，必须让这些 anchor 作为显式主体出现在 description 和 entity_anchors_selected 中，不要改写或遗漏。\n"
    "10. 如果下方提供了【本次必须覆盖的 background Entity Anchors】，必须让这些 anchor 在画面边缘/角落/背景处出现，并写入 description 和 entity_anchors_selected 中，不要改写或遗漏。\n"
    "11. 若显式偏好包含 Relationship（人物）或 Pet（宠物），相关人物/宠物必须作为 foreground 主体出镜，至少露脸，尽量露出脸和上半身；但人物衣着、手链、发型、宠物花纹、尾巴等特征不能写入 entity_anchors_selected，除非它们本身明确出现在 entity_anchors 列表中。\n"
    "12. 如果下方【必须逐字出现的人物/宠物名称】不是“无”，description 中必须逐字包含其中列出的每一个名称；不要只写“雪貂”“猫”“朋友”“孩子”等泛称。\n"
    "13. 如果下方【上一轮图片描述错误】不是“无”，必须逐条修正这些错误，尤其是缺少人物/宠物名的问题。\n"
    "14. 不要把隐式偏好写成用户主动展示的对象，不要让隐式物品出现在画面中心，不要解释隐式偏好的含义。\n"
    "15. 描述控制在 100 字以内，语言自然，不要像摄影说明书。\n\n"
    "## 【输出格式】\n"
    "必须输出合法 JSON，格式如下，不要输出任何 JSON 以外的内容：\n"
    "```json\n"
    "{{\n"
    '  \"description\": \"完整图片描述文本（90字以内）\",\n'
    '  \"entity_anchors_selected\": [\"foreground锚点\", \"background锚点\"]\n'
    "}}\n"
    "```\n\n"
    "[本次拍摄者的名字]\n{photographer_name}\n\n"
    "[场景描述]\n{scene_description}\n\n"
    "[隐式视觉潜入说明]\n{implicit_visual_integration_guidance}\n\n"
    "[本次必须覆盖的 foreground Entity Anchors]\n{required_explicit_entity_anchors}\n\n"
    "[本次必须覆盖的 background Entity Anchors]\n{required_implicit_entity_anchors}\n\n"
    "[必须逐字出现的人物/宠物名称]\n{required_person_pet_names}\n\n"
    "[上一轮图片描述错误]\n{image_retry_feedback}\n\n"
    "[显式偏好及其 Entity Anchors（foreground 主体来源）]\n{explicit_prefs_with_anchors}\n\n"
    "[隐式视觉偏好及其 Entity Anchors（background 边缘线索来源）]\n{implicit_prefs_with_anchors}\n"
)


def get_preference_str_with_anchors(preferences: List[Dict[str, Any]]) -> str:
    """格式化偏好列表，附带 entity_anchors，供图像生成 prompt 使用。"""
    lines = []
    for i, pref in enumerate(preferences):
        line = f"{i + 1}. [{pref.get('category', '')}] {pref.get('content', '')}"
        line += f" | Modality: {', '.join(pref.get('sources', []))}"
        lines.append(line)
        anchors = pref.get('entity_anchors', [])
        if isinstance(anchors, list) and anchors:
            lines.append(f"   Entity Anchors: {', '.join(anchors)}")
        rationale_items = _format_rationale(pref.get('rationale'))
        if rationale_items:
            lines.append("   Rationale（仅用于理解偏好成因，不得原样照搬）:")
            for item in rationale_items:
                lines.append(f"     - {item}")
    return "\n".join(lines)


def _call_llm_text(prompt_text: str, task_id: str, label: str) -> Tuple[str, int, int]:
    """简单 LLM 调用，返回纯文本，重试 3 次。"""
    for attempt in range(3):
        client = openai_client()
        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt_text}],
                temperature=0.8,
                top_p=0.9,
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
            pt = usage_info.prompt_tokens if usage_info else 0
            ct = usage_info.completion_tokens if usage_info else 0
            return result.strip(), pt, ct
        except Exception as e:
            _safe_print(f"[{label}] task={task_id} attempt {attempt + 1}/3 error: {e}")
            time.sleep(attempt + 1)
    return "", 0, 0


def _parse_llm_image_json(raw: str, task_id: str, label: str):
    """解析 LLM 返回的 {description, entity_anchors_selected} JSON；失败时退回到纯文本。"""
    import json as _json
    text = raw.strip()
    # 剥除可能的 ```json ... ``` 包裹
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()
    try:
        obj = _json.loads(text)
        desc = str(obj.get("description") or "").strip()
        anchors = obj.get("entity_anchors_selected")
        if not isinstance(anchors, list):
            anchors = []
        anchors = [str(a).strip() for a in anchors if str(a).strip()]
        return desc, anchors
    except Exception:
        _safe_print(f"[{task_id}] [{label}] JSON 解析失败，退回纯文本: {text[:60]}")
        return text, []


def _generate_image_description(
    task_id: str,
    scene_description: str,
    explicit_prefs: List[Dict[str, Any]],
    implicit_prefs: List[Dict[str, Any]],
    photographer_name: str = "",
    required_explicit_anchors: Optional[List[str]] = None,
    required_implicit_anchors: Optional[List[str]] = None,
    required_person_pet_names: Optional[List[str]] = None,
    image_retry_feedback: Optional[List[str]] = None,
    implicit_visual_integration_guidance: str = "无",
    image_description_mode: str = IMAGE_DESCRIPTION_MODE,
) -> Tuple[str, List[str], int, int]:
    """
    生成 user_shared_image_description。
      two_step: 显式偏好 → 画面主体（前景），隐式偏好 → 画面边缘/角落（背景）
      combined: 一次性生成完整图片描述，但 prompt 中仍区分 foreground/background
    返回 (merged_description, entity_anchors_selected_list, prompt_tokens, completion_tokens)

    Args:
        photographer_name: 拍摄者（profile 主人）的名字，用于 prompt 中指明第一人称视角的拍摄者
    """
    has_exp_visual = has_explicit_visual_subject_in_prefs(explicit_prefs)
    has_imp_visual = has_visual_modality_in_prefs(implicit_prefs)

    if not has_exp_visual and not has_imp_visual:
        return 'none', [], 0, 0

    total_pt, total_ct = 0, 0
    foreground = ""
    background = ""
    all_selected_anchors: List[str] = []
    mode = str(image_description_mode or IMAGE_DESCRIPTION_MODE).strip().lower().replace("-", "_")

    try:
        if mode in {"combined", "single", "one_step", "merged"}:
            visual_exp = [p for p in explicit_prefs if is_explicit_visual_subject(p)]
            visual_imp = [p for p in implicit_prefs if has_visual_modality(p)]
            prompt = PROMPT_IMAGE_COMBINED.format(
                scene_description=scene_description,
                photographer_name=photographer_name,
                implicit_visual_integration_guidance=implicit_visual_integration_guidance or "无",
                required_explicit_entity_anchors=_format_required_entity_anchors(required_explicit_anchors),
                required_implicit_entity_anchors=_format_required_entity_anchors(required_implicit_anchors),
                required_person_pet_names=_format_required_person_pet_names(required_person_pet_names),
                image_retry_feedback=_format_image_retry_feedback(image_retry_feedback),
                explicit_prefs_with_anchors=get_preference_str_with_anchors(visual_exp),
                implicit_prefs_with_anchors=get_preference_str_with_anchors(visual_imp),
            )
            raw_img, pt, ct = _call_llm_text(prompt, task_id, "image-combined")
            total_pt += pt
            total_ct += ct
            description, selected_anchors = _parse_llm_image_json(raw_img, task_id, "image-combined")
            _safe_print(f"[{task_id}] [image-combined] {description[:80]}  anchors={selected_anchors}")
            seen = set()
            unique_anchors = [a for a in selected_anchors if a not in seen and not seen.add(a)]
            return (description or 'none'), unique_anchors, total_pt, total_ct

        if mode not in {"two_step", "split", "foreground_background"}:
            _safe_print(
                f"[{task_id}] unknown image_description_mode={image_description_mode!r}; "
                "fallback to two_step"
            )

        if has_exp_visual:
            visual_exp = [p for p in explicit_prefs if is_explicit_visual_subject(p)]
            prompt = PROMPT_IMAGE_FOREGROUND.format(
                scene_description=scene_description,
                explicit_prefs_with_anchors=get_preference_str_with_anchors(visual_exp),
                photographer_name=photographer_name,
                required_entity_anchors=_format_required_entity_anchors(required_explicit_anchors),
                required_person_pet_names=_format_required_person_pet_names(required_person_pet_names),
                image_retry_feedback=_format_image_retry_feedback(image_retry_feedback),
            )
            raw_fg, pt, ct = _call_llm_text(prompt, task_id, "image-foreground")
            total_pt += pt
            total_ct += ct
            foreground, fg_anchors = _parse_llm_image_json(raw_fg, task_id, "image-foreground")
            all_selected_anchors.extend(fg_anchors)
            _safe_print(f"[{task_id}] [image-foreground] {foreground[:80]}  anchors={fg_anchors}")

        if has_imp_visual:
            visual_imp = [p for p in implicit_prefs if has_visual_modality(p)]
            prompt = PROMPT_IMAGE_BACKGROUND.format(
                scene_description=scene_description,
                foreground_description=foreground if foreground else "无主体描述",
                implicit_visual_integration_guidance=implicit_visual_integration_guidance or "无",
                implicit_prefs_with_anchors=get_preference_str_with_anchors(visual_imp),
                required_entity_anchors=_format_required_entity_anchors(required_implicit_anchors),
            )
            raw_bg, pt, ct = _call_llm_text(prompt, task_id, "image-background")
            total_pt += pt
            total_ct += ct
            background, bg_anchors = _parse_llm_image_json(raw_bg, task_id, "image-background")
            all_selected_anchors.extend(bg_anchors)
            _safe_print(f"[{task_id}] [image-background] {background[:80]}  anchors={bg_anchors}")

        parts = [s for s in (foreground, background) if s]
        if not parts:
            return 'none', [], total_pt, total_ct

        merged = "".join(
            p if p.endswith(("。", ".", "！", "!")) else p + "。"
            for p in parts
        )
        # 去重保序
        seen = set()
        unique_anchors = [a for a in all_selected_anchors if a not in seen and not seen.add(a)]
        return merged, unique_anchors, total_pt, total_ct
    except Exception as e:
        import traceback
        _safe_print(f"[{task_id}] [_generate_image_description] 异常: {e}")
        _safe_print(f"[{task_id}] [_generate_image_description] traceback: {traceback.format_exc()}")
        return 'none', [], total_pt, total_ct


def resolve_existing_path(candidates: List[str]) -> Path:
    for candidate in candidates:
        path = resolve_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these paths exists: {candidates}")


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == '[':
        return json.loads(text)

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def get_manual_groups_path() -> Path:
    candidates = [
        MANUAL_GROUPS_FILENAME,
        "event/manual_v1_events_000_010_groups.jsonl",
    ]
    return resolve_existing_path(candidates)


def get_persona_path() -> Path:
    candidates = [
        PERSONA_FILE_PATH,
    ]
    return resolve_existing_path(candidates)


def ensure_parent_dir(path_str: str) -> Path:
    path = resolve_path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_event_key(p_id: Any, group_id: Any) -> Tuple[str, str]:
    return str(p_id), str(group_id)


def load_existing_event_results(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """读取已保存结果，用 (p_id, group_id) 作为断点续跑的唯一键。"""
    if not path.exists():
        return {}

    records = load_json_or_jsonl(path)
    if not isinstance(records, list):
        raise ValueError(f"Existing event result file must be a list/jsonl records: {path}")

    existing: Dict[Tuple[str, str], Dict[str, Any]] = {}
    duplicate_keys: List[Tuple[str, str]] = []
    skipped = 0

    for record in records:
        if not isinstance(record, dict):
            skipped += 1
            continue
        if record.get('p_id') is None or record.get('group_id') is None:
            skipped += 1
            continue
        if not isinstance(record.get('event'), dict) or not record.get('event'):
            skipped += 1
            continue

        key = make_event_key(record.get('p_id'), record.get('group_id'))
        if key in existing:
            duplicate_keys.append(key)
            continue
        existing[key] = record

    if skipped:
        print(f"[Resume] Skipped {skipped} malformed existing record(s).")
    if duplicate_keys:
        print(f"[Resume] Found {len(duplicate_keys)} duplicate existing event key(s); keeping first occurrence.")

    return existing


def get_dict_str(data: Dict[str, Any], mbti: str) -> str:
    data_str = ""
    for k in ['name', 'age', 'gender', 'education', 'occupation']:
        if k in data:
            data_str += f"{k}: {data[k]}\n"
    data_str += f"MBTI: {mbti}\n"

    return data_str


def _format_rationale(raw: Any) -> List[str]:
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        items = [seg.strip() for seg in raw.splitlines() if seg.strip()]
    else:
        items = []
    return items


def get_preference_str(preferences: List[Dict[str, Any]]) -> str:
    if not preferences:
        return "无"
    lines = []
    for i, pref in enumerate(preferences):
        line = f"{i + 1}. [{pref['category']}] {pref.get('content', '')}"
        line += f" | Modality: {', '.join(pref.get('sources', []))}"
        lines.append(line)

        rationale_items = _format_rationale(pref.get('rationale'))
        if rationale_items:
            lines.append("   Rationale（仅用于理解偏好成因，不得原样照搬进事件文本）:")
            for item in rationale_items:
                lines.append(f"     - {item}")
    return "\n".join(lines)


def has_audio_modality(pref: Dict[str, Any]) -> bool:
    return 'audio' in pref.get('sources', [])


def has_visual_modality(pref: Dict[str, Any]) -> bool:
    return 'visual' in pref.get('sources', [])


def is_basic_visual_subject(pref: Dict[str, Any]) -> bool:
    """Relationship/BasicPets can be used as explicit visual subjects even with sources=['basic']."""
    category = str(pref.get('category', '') or '')
    return category.startswith('Relationship-') or category.startswith('BasicPets-') or category.startswith('Pet-')


def is_explicit_visual_subject(pref: Dict[str, Any]) -> bool:
    return has_visual_modality(pref) or is_basic_visual_subject(pref)


def has_audio_modality_in_prefs(prefs: List[Dict[str, Any]]) -> bool:
    return any(has_audio_modality(p) for p in prefs)


def has_visual_modality_in_prefs(prefs: List[Dict[str, Any]]) -> bool:
    return any(has_visual_modality(p) for p in prefs)


def has_explicit_visual_subject_in_prefs(prefs: List[Dict[str, Any]]) -> bool:
    return any(is_explicit_visual_subject(p) for p in prefs)


def guidance_for_modality(
    implicit_prefs: List[Dict[str, Any]],
    implicit_integration_guidance: str,
    modality: str,
) -> str:
    """Return guidance only when the current implicit prefs contain the modality."""
    guidance = str(implicit_integration_guidance or "").strip()
    if not guidance:
        return "无"
    modality = modality.lower()
    if any(modality in [str(src).lower() for src in pref.get('sources', [])] for pref in implicit_prefs):
        return guidance
    return "无"


def _normalize_entity_anchors(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def allowed_entity_anchors_from_preferences(
    explicit_prefs: List[Dict[str, Any]],
    implicit_prefs: List[Dict[str, Any]],
) -> List[str]:
    """Return source-authoritative entity anchors allowed for event["entity_anchors"]."""
    allowed: List[str] = []
    seen: Set[str] = set()
    for pref in explicit_prefs + implicit_prefs:
        if not isinstance(pref, dict):
            continue
        anchors = _normalize_entity_anchors(pref.get('entity_anchors'))
        anchors.extend(_normalize_entity_anchors(pref.get('entity_anchor')))
        for anchor in anchors:
            if anchor not in seen:
                allowed.append(anchor)
                seen.add(anchor)
    return allowed


def filter_generated_entity_anchors(
    generated_anchors: List[str],
    explicit_prefs: List[Dict[str, Any]],
    implicit_prefs: List[Dict[str, Any]],
    task_id: str = "",
) -> List[str]:
    """Drop LLM-created anchors that are not present in the current group prefs."""
    allowed = set(allowed_entity_anchors_from_preferences(explicit_prefs, implicit_prefs))
    if not allowed:
        if generated_anchors:
            _safe_print(
                f"[{task_id}] dropped generated entity_anchors because current group has no "
                f"source anchors: {generated_anchors}"
            )
        return []

    filtered: List[str] = []
    dropped: List[str] = []
    seen: Set[str] = set()
    for anchor in generated_anchors:
        anchor = str(anchor or "").strip()
        if not anchor:
            continue
        if anchor in allowed:
            if anchor not in seen:
                filtered.append(anchor)
                seen.add(anchor)
        else:
            dropped.append(anchor)

    if dropped:
        _safe_print(f"[{task_id}] dropped non-source entity_anchors: {dropped}")
    return filtered


def get_visual_pref_anchors(preferences: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    anchors: List[Tuple[str, str]] = []
    for pref in preferences:
        if not isinstance(pref, dict) or not has_visual_modality(pref):
            continue
        category = str(pref.get('category', '')).strip()
        if not category:
            continue
        for anchor in _normalize_entity_anchors(pref.get('entity_anchors')):
            anchors.append((category, anchor))
    return anchors


def select_required_anchors(
    preferences: List[Dict[str, Any]],
    anchor_usage: Dict[Tuple[str, str], int],
    max_count: int,
    min_count_if_available: int = 1,
) -> List[str]:
    candidates = get_visual_pref_anchors(preferences)
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda item: (anchor_usage.get(item, 0), item[0], item[1]))
    selected: List[str] = []
    seen: Set[str] = set()
    target_count = min(max_count, len(candidates))
    if target_count < min_count_if_available:
        target_count = min_count_if_available

    for _category, anchor in candidates:
        if anchor in seen:
            continue
        selected.append(anchor)
        seen.add(anchor)
        if len(selected) >= target_count:
            break
    return selected


def _format_required_entity_anchors(anchors: Optional[List[str]]) -> str:
    anchors = [str(anchor).strip() for anchor in (anchors or []) if str(anchor).strip()]
    if not anchors:
        return "无。本次可从下方全部 entity_anchors 中自由挑选 1-2 个。"
    return "\n".join(f"- {anchor}" for anchor in anchors)


def _format_required_person_pet_names(names: Optional[List[str]]) -> str:
    names = [str(name).strip() for name in (names or []) if str(name).strip()]
    if not names:
        return "无"
    return "\n".join(f"- {name}" for name in dict.fromkeys(names))


def _format_image_retry_feedback(feedback: Optional[List[str]]) -> str:
    items = [str(item).strip() for item in (feedback or []) if str(item).strip()]
    if not items:
        return "无"
    return "\n".join(f"- {item}" for item in items[-8:])


def update_anchor_usage_from_event(
    anchor_usage: Dict[Tuple[str, str], int],
    explicit_prefs: List[Dict[str, Any]],
    implicit_prefs: List[Dict[str, Any]],
    event: Dict[str, Any],
) -> None:
    selected = set(_normalize_entity_anchors(event.get('entity_anchors')))
    if not selected:
        return
    for category, anchor in get_visual_pref_anchors(explicit_prefs + implicit_prefs):
        if anchor in selected:
            anchor_usage[(category, anchor)] = anchor_usage.get((category, anchor), 0) + 1


def unwrap_event_object(obj: Any) -> Dict[str, Any]:
    """
    兼容以下返回结构：
    - {...}
    - [{...}]
    - [[{...}]]
    - 列表里包含多个对象时，优先取第一个 dict
    """
    current = obj
    for _ in range(10):
        if isinstance(current, dict):
            return current
        if isinstance(current, list):
            if not current:
                break
            if len(current) == 1:
                current = current[0]
                continue
            first_dict = next((x for x in current if isinstance(x, dict)), None)
            if first_dict is not None:
                return first_dict
            current = current[0]
            continue
        break
    raise ValueError(f"Unexpected response structure: {type(current).__name__} | value={repr(current)[:500]}")


def _split_pet_aliases(raw_name: str) -> List[str]:
    """把 'Pixel（像素）' / 'Pixel(像素)' 这种带括号的名字拆成多个别名。"""
    raw = raw_name.strip()
    if not raw:
        return []
    aliases = [raw]
    m = re.match(r'^([^（(]+)[（(]([^）)]+)[)）]\s*$', raw)
    if m:
        for part in (m.group(1), m.group(2)):
            part = part.strip()
            if part and part not in aliases:
                aliases.append(part)
    return aliases


def _collect_person_names(profile: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for item in profile.get('Basic', {}).get('Relationship', []) or []:
        if not isinstance(item, dict):
            continue
        for key in ('name', 'person', 'relation'):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                names.add(v.strip())
    return names


def _collect_pet_names(profile: Dict[str, Any]) -> Set[str]:
    names: Set[str] = set()
    for item in profile.get('Basic', {}).get('Pets', []) or []:
        if not isinstance(item, dict):
            continue
        v = item.get('name')
        if isinstance(v, str) and v.strip():
            names.update(_split_pet_aliases(v))
    return names


def _allowed_person_names(profile: Dict[str, Any], explicit_prefs: List[Dict[str, Any]]) -> Set[str]:
    """显式偏好里 Relationship-i 指向的那些人，事件中允许出现。"""
    allowed: Set[str] = set()
    rels = profile.get('Basic', {}).get('Relationship', []) or []
    for p in explicit_prefs:
        cat = str(p.get('category', ''))
        if not cat.startswith('Relationship-'):
            continue
        try:
            idx = int(cat.split('-', 1)[1])
        except (ValueError, IndexError):
            continue
        if 0 <= idx < len(rels) and isinstance(rels[idx], dict):
            for key in ('name', 'person', 'relation'):
                v = rels[idx].get(key)
                if isinstance(v, str) and v.strip():
                    allowed.add(v.strip())
    return allowed


def _allowed_pet_names(profile: Dict[str, Any], explicit_prefs: List[Dict[str, Any]]) -> Set[str]:
    """显式偏好里 BasicPets-i / Pet-i 指向的那些宠物，事件中允许出现。"""
    allowed: Set[str] = set()
    pets = profile.get('Basic', {}).get('Pets', []) or []
    for p in explicit_prefs:
        cat = str(p.get('category', ''))
        if not (cat.startswith('BasicPets-') or cat.startswith('Pet-')):
            continue
        try:
            idx = int(cat.split('-', 1)[1])
        except (ValueError, IndexError):
            continue
        if 0 <= idx < len(pets) and isinstance(pets[idx], dict):
            v = pets[idx].get('name')
            if isinstance(v, str) and v.strip():
                allowed.update(_split_pet_aliases(v))
    return allowed


def _allowed_person_names_for_img(profile: Dict[str, Any], explicit_prefs: List[Dict[str, Any]]) -> Set[str]:
    """只取 name 字段（不含 relation 别名），用于图像描述中的姓名校验。"""
    allowed: Set[str] = set()
    rels = profile.get('Basic', {}).get('Relationship', []) or []
    for p in explicit_prefs:
        cat = str(p.get('category', ''))
        if not cat.startswith('Relationship-'):
            continue
        try:
            idx = int(cat.split('-', 1)[1])
        except (ValueError, IndexError):
            continue
        if 0 <= idx < len(rels) and isinstance(rels[idx], dict):
            v = rels[idx].get('name')
            if isinstance(v, str) and v.strip():
                allowed.add(v.strip())
    return allowed


def validate_person_in_image_desc(
    event: Dict[str, Any],
    profile: Dict[str, Any],
    explicit_prefs: List[Dict[str, Any]],
) -> List[str]:
    """
    硬性检查：当 explicit_prefs 包含 Relationship-* 或 Pet-*/BasicPets-* 时，
    对应人物/宠物姓名必须出现在 user_shared_image_description 中。
    若图像描述为 'none' 则跳过（has_visual 已由 clean_event_fields 保障）。
    返回 issue 列表；非空表示校验失败，需重新生成。
    """
    img_desc = str(event.get('user_shared_image_description', '') or '').strip()
    if not img_desc or img_desc.lower() == 'none':
        return []

    issues: List[str] = []

    # ── Relationship 检查 ─────────────────────────────────────────────────────
    has_rel_pref = any(
        str(p.get('category', '')).startswith('Relationship-')
        for p in explicit_prefs
    )
    if has_rel_pref:
        required_names = _allowed_person_names_for_img(profile, explicit_prefs)
        missing = [n for n in sorted(required_names) if n not in img_desc]
        if missing:
            issues.append(
                f"user_shared_image_description 缺少 Relationship 人物姓名 {missing}；"
                f"描述片段：{img_desc[:100]!r}"
            )

    # ── Pets 检查 ─────────────────────────────────────────────────────────────
    has_pet_pref = any(
        str(p.get('category', '')).startswith('BasicPets-') or
        str(p.get('category', '')).startswith('Pet-')
        for p in explicit_prefs
    )
    if has_pet_pref:
        allowed_pets = _allowed_pet_names(profile, explicit_prefs)
        # 宠物可能有别名，任意一个别名出现即通过
        if allowed_pets and not any(n in img_desc for n in allowed_pets):
            issues.append(
                f"user_shared_image_description 缺少 Pets 宠物名 {sorted(allowed_pets)} 中的任意一个；"
                f"描述片段：{img_desc[:100]!r}"
            )

    return issues


def validate_persons_and_pets_usage(
    event: Dict[str, Any],
    profile: Dict[str, Any],
    explicit_prefs: List[Dict[str, Any]],
) -> List[str]:
    """
    强制规则：事件文本中出现的人物/宠物，必须在本次显式偏好的 Relationship-* / BasicPets-* / Pet-* 中被指明。
    返回 issue 列表；非空表示有违规。
    """
    text_fields = [
        str(event.get('scene_description', '')),
        str(event.get('user_shared_image_description', '')),
        str(event.get('human_speech_content', '')),
        str(event.get('background_audio_info', '')),
    ]
    full_text = ' '.join(t for t in text_fields if t and t.lower() != 'none')

    issues: List[str] = []

    all_persons = _collect_person_names(profile)
    allowed_persons = _allowed_person_names(profile, explicit_prefs)
    forbidden_persons = all_persons - allowed_persons
    appeared_persons = sorted({n for n in forbidden_persons if n and n in full_text})
    if appeared_persons:
        if not allowed_persons:
            issues.append(
                f"事件文本中出现了 Relationship 中的人物 {appeared_persons}，"
                f"但本次显式偏好未包含 Relationship-* 类目。"
            )
        else:
            issues.append(
                f"事件文本中出现了未被显式偏好允许的人物 {appeared_persons}（"
                f"允许：{sorted(allowed_persons)}）。"
            )

    all_pets = _collect_pet_names(profile)
    allowed_pets = _allowed_pet_names(profile, explicit_prefs)
    forbidden_pets = all_pets - allowed_pets
    appeared_pets = sorted({n for n in forbidden_pets if n and n in full_text})
    if appeared_pets:
        if not allowed_pets:
            issues.append(
                f"事件文本中出现了 Pets 中的宠物 {appeared_pets}，"
                f"但本次显式偏好未包含 BasicPets-*/Pet-* 类目。"
            )
        else:
            issues.append(
                f"事件文本中出现了未被显式偏好允许的宠物 {appeared_pets}（"
                f"允许：{sorted(allowed_pets)}）。"
            )

    return issues


def clean_event_fields(
    event: Dict[str, Any],
    explicit_prefs: List[Dict[str, Any]],
    implicit_prefs: List[Dict[str, Any]],
    issues: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    issues = issues if issues is not None else []
    has_implicit_audio = has_audio_modality_in_prefs(implicit_prefs)
    has_explicit_audio = has_audio_modality_in_prefs(explicit_prefs)
    has_visual = has_explicit_visual_subject_in_prefs(explicit_prefs) or has_visual_modality_in_prefs(implicit_prefs)

    if has_implicit_audio and str(event.get('background_audio_info', 'none')).strip().lower() == 'none':
        message = "Implicit audio preference exists but no background_audio_info generated"
        issues.append(message)
        print(f"Warning: {message}")
        return None

    if has_visual and str(event.get('user_shared_image_description', 'none')).strip().lower() == 'none':
        message = "Visual preference exists but no user_shared_image_description generated"
        issues.append(message)
        print(f"Warning: {message}")
        return None

    if not has_implicit_audio:
        event['background_audio_info'] = 'none'

    if not has_explicit_audio:
        event['human_speech_content'] = 'none'

    if not has_visual:
        event['user_shared_image_description'] = 'none'

    # entity anchors are stored on preference objects as `entity_anchors`.
    # Drop the legacy event-level singular field if the model still returns it.
    event.pop('entity_anchor', None)

    required_explicit = {p['category'] for p in explicit_prefs}
    required_implicit = {p['category'] for p in implicit_prefs}

    raw_reflected_explicit = event.get('explicit_preferences_reflected', []) or []
    raw_reflected_implicit = event.get('implicit_preferences_reflected', []) or []
    if not isinstance(raw_reflected_explicit, list):
        raw_reflected_explicit = []
    if not isinstance(raw_reflected_implicit, list):
        raw_reflected_implicit = []

    reflected_explicit = {c for c in raw_reflected_explicit if c in required_explicit}
    reflected_implicit = {c for c in raw_reflected_implicit if c in required_implicit}

    missing_explicit = required_explicit - reflected_explicit
    missing_implicit = required_implicit - reflected_implicit
    if missing_explicit or missing_implicit:
        message = (
            f"preference coverage incomplete - missing explicit: {sorted(missing_explicit)}, "
            f"missing implicit: {sorted(missing_implicit)}; raw reflected explicit={raw_reflected_explicit}, "
            f"raw reflected implicit={raw_reflected_implicit}"
        )
        issues.append(message)
        print(f"Warning: {message}")
        return None

    event['explicit_preferences_reflected'] = sorted(reflected_explicit)
    event['implicit_preferences_reflected'] = sorted(reflected_implicit)

    ordered_keys = [
        'scene_description',
        'user_shared_image_description',
        'background_audio_info',
        'human_speech_content',
        'explicit_preferences_reflected',
        'implicit_preferences_reflected',
    ]
    ordered_event: Dict[str, Any] = {k: event[k] for k in ordered_keys if k in event}
    for k, v in event.items():
        if k not in ordered_event:
            ordered_event[k] = v
    return ordered_event


def build_profile_preference_map(profile: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    pref_map: Dict[str, Dict[str, Any]] = {}

    for k in ['FoodAndDrink', 'HomeAndSpace', 'BodyAndHealth', 'HobbiesAndEntertainment', 'WorkAndLearning', 'MobilityAndTravel', 'Pets']:
        for item_id, item in enumerate(profile.get(k, [])):
            pref_map[f"{k}-{item_id}"] = {
                'category': f"{k}-{item_id}",
                'subcategory': item.get('subcategory', ''),
                'content': item.get('preference', ''),
                'expression_type': item.get('expression_type', 'explicit'),
                'sources': item.get('evidence_sources', []),
                'rationale': item.get('analysis', []),
                'entity_anchors': item.get('entity_anchors', []),
            }

    for item_id, item in enumerate(profile.get('Basic', {}).get('Relationship', [])):
        record = {
            'subcategory': item.get('relation', ''),
            'content': f"{item.get('relation', '')};{item.get('name', item.get('person', ''))};{item.get('appearance', '').strip('.').strip(';')};{item.get('info', '').strip('.').strip(';')};",
            'expression_type': item.get('expression_type', 'explicit') or 'explicit',
            'sources': item.get('evidence_sources', ['visual', 'dialogue']),
            'rationale': ['Person appearance in visual', 'Person details in dialogue'],
        }
        pref_map[f"Relationship-{item_id}"] = {'category': f"Relationship-{item_id}", **record}

    for item_id, item in enumerate(profile.get('Basic', {}).get('Pets', [])):
        base_record = {
            'subcategory': 'BasicPet',
            'content': f"{item.get('name', '')};{item.get('appearance', '').strip('.').strip(';')};{item.get('info', '').strip('.').strip(';')};",
            'expression_type': 'explicit',
            'sources': item.get('evidence_sources', ['visual', 'dialogue']),
            'rationale': ['Pet appearance in visual', 'Pet details in dialogue'],
        }
        pref_map[f"BasicPets-{item_id}"] = {'category': f"BasicPets-{item_id}", **base_record}
        # 兼容旧脚本里写错/写混的别名
        pref_map[f"Pet-{item_id}"] = {'category': f"Pet-{item_id}", **base_record}

    return pref_map


def normalize_pref_list(
    pref_list: List[Dict[str, Any]],
    pref_map: Dict[str, Dict[str, Any]],
    default_expression_type: str,
    category_hints: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    category_hints = category_hints or []

    for idx, pref in enumerate(pref_list):
        category = pref.get('category') or (category_hints[idx] if idx < len(category_hints) else None)
        if category in pref_map:
            base = dict(pref_map[category])
            base.update(pref)
            # Always use the authoritative 'content' from pref_map so that
            # any newly added fields (e.g. 'info' in Relationship items) are
            # not silently dropped by stale content stored in manual groups.
            base['content'] = pref_map[category]['content']
        else:
            base = dict(pref)
            if category is not None:
                base.setdefault('category', category)
            base.setdefault('subcategory', '')
            base.setdefault('content', '')
            base.setdefault('sources', [])
            base.setdefault('rationale', [])
        base.setdefault('expression_type', default_expression_type)
        # Keep only the canonical plural field. Some older intermediate files
        # carried a duplicate `entity_anchor` key alongside `entity_anchors`.
        base.pop('entity_anchor', None)
        normalized.append(base)

    return normalized


def validate_one_manual_group(group: Dict[str, Any], pref_map: Dict[str, Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    group_id = group.get('group_id', 'unknown')

    explicit_categories = group.get('explicit_categories', []) or []
    implicit_categories = group.get('implicit_categories', []) or []
    explicit_preferences = group.get('explicit_preferences', []) or []
    implicit_preferences = group.get('implicit_preferences', []) or []
    recommended_main_scene = (group.get('recommended_main_scene', '') or '').strip()
    is_relationship_group = (
        len(explicit_categories) == 1
        and str(explicit_categories[0]).startswith('Relationship-')
    )

    if len(explicit_categories) != 1:
        issues.append(f"group {group_id}: explicit_categories 长度应为 1，当前为 {len(explicit_categories)}")
    if is_relationship_group:
        if len(implicit_categories) != 0:
            issues.append(f"group {group_id}: Relationship-* standalone group 的 implicit_categories 应为 0，当前为 {len(implicit_categories)}")
    elif not (1 <= len(implicit_categories) <= 3):
        issues.append(f"group {group_id}: implicit_categories 长度应为 1-3，当前为 {len(implicit_categories)}")
    if len(explicit_preferences) != 1:
        issues.append(f"group {group_id}: explicit_preferences 长度应为 1，当前为 {len(explicit_preferences)}")
    if is_relationship_group:
        if len(implicit_preferences) != 0:
            issues.append(f"group {group_id}: Relationship-* standalone group 的 implicit_preferences 应为 0，当前为 {len(implicit_preferences)}")
    elif not (1 <= len(implicit_preferences) <= 3):
        issues.append(f"group {group_id}: implicit_preferences 长度应为 1-3，当前为 {len(implicit_preferences)}")

    if len(implicit_categories) != len(implicit_preferences):
        issues.append(
            f"group {group_id}: implicit_categories 数量({len(implicit_categories)}) 与 implicit_preferences 数量({len(implicit_preferences)}) 不一致"
        )

    if len(set(explicit_categories)) != len(explicit_categories):
        issues.append(f"group {group_id}: explicit_categories 中存在重复")
    if len(set(implicit_categories)) != len(implicit_categories):
        issues.append(f"group {group_id}: implicit_categories 中存在重复")

    if not recommended_main_scene:
        issues.append(f"group {group_id}: recommended_main_scene 为空")

    if explicit_categories and explicit_preferences:
        pref_cat = explicit_preferences[0].get('category')
        if explicit_categories[0] != pref_cat:
            issues.append(
                f"group {group_id}: explicit_categories[0]={explicit_categories[0]} 与 explicit_preferences[0].category={pref_cat} 不一致"
            )

    implicit_pref_cats = [p.get('category') for p in implicit_preferences]
    if implicit_categories and implicit_preferences:
        if set(implicit_categories) != set(implicit_pref_cats):
            issues.append(
                f"group {group_id}: implicit_categories={implicit_categories} 与 implicit_preferences categories={implicit_pref_cats} 不一致"
            )

    for pref in explicit_preferences:
        cat = pref.get('category')
        expression_type = pref.get('expression_type')
        if cat in pref_map:
            expression_type = pref_map[cat].get('expression_type', expression_type)
        if expression_type == 'implicit':
            issues.append(f"group {group_id}: {cat} 被放在 explicit_preferences 中，但其 expression_type=implicit")

    for pref in implicit_preferences:
        cat = pref.get('category')
        expression_type = pref.get('expression_type')
        if cat in pref_map:
            expression_type = pref_map[cat].get('expression_type', expression_type)
        if expression_type != 'implicit':
            issues.append(f"group {group_id}: {cat} 被放在 implicit_preferences 中，但其 expression_type 不是 implicit")

    has_explicit_visual = has_explicit_visual_subject_in_prefs(explicit_preferences)
    has_implicit_visual = any(has_visual_modality(p) for p in implicit_preferences)
    if has_implicit_visual and not has_explicit_visual:
        issues.append(
            f"group {group_id}: 含 implicit visual，但同组 explicit 不含 visual；"
            f"explicit={explicit_categories}, implicit={implicit_categories}"
        )

    for pref in explicit_preferences + implicit_preferences:
        if not pref.get('content'):
            issues.append(f"group {group_id}: {pref.get('category')} 缺少 content")
        if not isinstance(pref.get('sources', []), list):
            issues.append(f"group {group_id}: {pref.get('category')} 的 sources 不是 list")

    return issues


def validate_and_filter_manual_groups(
    profile_id: int,
    normalized_groups: List[Dict[str, Any]],
    pref_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid_groups: List[Dict[str, Any]] = []
    invalid_groups: List[Dict[str, Any]] = []

    for group in normalized_groups:
        issues = validate_one_manual_group(group, pref_map)
        if issues:
            invalid_groups.append({
                'group_id': group.get('group_id'),
                'explicit_categories': group.get('explicit_categories', []),
                'implicit_categories': group.get('implicit_categories', []),
                'issues': issues,
            })
        else:
            valid_groups.append(group)

    report = {
        'profile_id': profile_id,
        'total_groups_before_filter': len(normalized_groups),
        'valid_group_count': len(valid_groups),
        'invalid_group_count': len(invalid_groups),
        'invalid_groups': invalid_groups,
    }

    print("\n=== Manual Group Validation Report ===")
    print(f"Profile {profile_id}: total={len(normalized_groups)}, valid={len(valid_groups)}, invalid={len(invalid_groups)}")
    for item in invalid_groups:
        print(
            f"[INVALID] group_id={item['group_id']} | explicit={item['explicit_categories']} | "
            f"implicit={item['implicit_categories']}"
        )
        for issue in item['issues']:
            print(f"  - {issue}")
    print("=====================================\n")

    if invalid_groups and MANUAL_GROUP_VALIDATION_MODE == 'error':
        raise ValueError(
            f"Profile {profile_id} has {len(invalid_groups)} invalid manual groups. "
            f"Set MANUAL_GROUP_VALIDATION_MODE='skip_invalid' to skip them."
        )

    return valid_groups, report


def _candidate_dates_for_year(year: int) -> List[datetime]:
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 28)
    total_days = (end - start).days
    return [start + timedelta(days=offset) for offset in range(total_days + 1)]


def _parse_planned_date(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return None


def _group_explicit_categories(group: Dict[str, Any]) -> List[str]:
    categories = group.get('explicit_categories') or []
    if not categories:
        categories = [
            pref.get('category')
            for pref in group.get('explicit_preferences', [])
            if isinstance(pref, dict) and pref.get('category')
        ]
    return [str(category) for category in categories if category]


def _group_implicit_categories(group: Dict[str, Any]) -> List[str]:
    categories = group.get('implicit_categories') or []
    if not categories:
        categories = [
            pref.get('category')
            for pref in group.get('implicit_preferences', [])
            if isinstance(pref, dict) and pref.get('category')
        ]
    return [str(category) for category in categories if category]


def _group_preference_keys(group: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    explicit = _group_explicit_categories(group)
    implicit = _group_implicit_categories(group)
    pairs = [f"{exp}||{imp}" for exp in explicit for imp in implicit]
    return explicit, implicit, pairs


def _score_date_assignment(
    groups: List[Dict[str, Any]],
    date_by_index: Dict[int, datetime],
    implicit_weight: float = 2.0,
    explicit_weight: float = 1.0,
    pair_weight: float = 0.5,
) -> float:
    buckets: Dict[Tuple[str, str], List[int]] = {}
    used_day_counts: Dict[str, int] = {}

    for index, group in enumerate(groups):
        date = date_by_index.get(index)
        if date is None:
            continue
        ordinal = date.toordinal()
        used_day_counts[date.strftime("%Y-%m-%d")] = used_day_counts.get(date.strftime("%Y-%m-%d"), 0) + 1
        explicit, implicit, pairs = _group_preference_keys(group)
        for category in explicit:
            buckets.setdefault(('explicit', category), []).append(ordinal)
        for category in implicit:
            buckets.setdefault(('implicit', category), []).append(ordinal)
        for pair in pairs:
            buckets.setdefault(('mixed', pair), []).append(ordinal)

    score = 0.0
    for (pref_type, _key), ordinals in buckets.items():
        if len(ordinals) <= 1:
            continue
        ordinals = sorted(ordinals)
        gaps = [b - a for a, b in zip(ordinals, ordinals[1:])]
        span_days = ordinals[-1] - ordinals[0]
        min_gap_days = min(gaps) if gaps else 0
        avg_gap_days = sum(gaps) / len(gaps) if gaps else 0.0
        if pref_type == 'implicit':
            weight = implicit_weight
        elif pref_type == 'explicit':
            weight = explicit_weight
        else:
            weight = pair_weight
        score += weight * (span_days + min_gap_days * 2.0 + avg_gap_days * 0.25)

    duplicate_dates = sum(max(0, count - 1) for count in used_day_counts.values())
    score -= duplicate_dates * 1000.0
    return score


def plan_group_dates_spaced(
    groups: List[Dict[str, Any]],
    year: int = 2025,
    local_search_steps: int = 5000,
) -> List[Dict[str, Any]]:
    """
    Add planned_date to one profile's groups while spreading repeated preferences over the year.

    The function only copies group dictionaries and adds/updates planned_date. It does not alter
    group_id, preferences, recommended_main_scene, or coverage-related fields.
    """
    planned_groups = [dict(group) for group in groups]
    if not planned_groups:
        return planned_groups

    candidate_dates = _candidate_dates_for_year(year)
    if not candidate_dates:
        raise ValueError(f"No candidate dates generated for year={year}")

    category_frequency: Dict[str, int] = {}
    for group in planned_groups:
        explicit, implicit, pairs = _group_preference_keys(group)
        for category in explicit + implicit + pairs:
            category_frequency[category] = category_frequency.get(category, 0) + 1

    ordered_indices = sorted(
        range(len(planned_groups)),
        key=lambda index: (
            -sum(category_frequency.get(key, 0) for keys in _group_preference_keys(planned_groups[index]) for key in keys),
            planned_groups[index].get('group_id', index),
        ),
    )

    date_by_index: Dict[int, datetime] = {}
    used_dates: Set[str] = set()

    for index in ordered_indices:
        best_date: Optional[datetime] = None
        best_score: Optional[float] = None

        for candidate in candidate_dates:
            candidate_str = candidate.strftime("%Y-%m-%d")
            trial = dict(date_by_index)
            trial[index] = candidate
            score = _score_date_assignment(planned_groups, trial)
            if candidate_str in used_dates:
                score -= 10000.0
            if best_score is None or score > best_score:
                best_score = score
                best_date = candidate

        if best_date is None:
            best_date = random.choice(candidate_dates)
        date_by_index[index] = best_date
        used_dates.add(best_date.strftime("%Y-%m-%d"))

    current_score = _score_date_assignment(planned_groups, date_by_index)
    if len(planned_groups) >= 2 and local_search_steps > 0:
        indices = list(range(len(planned_groups)))
        for _ in range(local_search_steps):
            left, right = random.sample(indices, 2)
            if date_by_index[left] == date_by_index[right]:
                continue
            trial = dict(date_by_index)
            trial[left], trial[right] = trial[right], trial[left]
            trial_score = _score_date_assignment(planned_groups, trial)
            if trial_score > current_score:
                date_by_index = trial
                current_score = trial_score

    for index, group in enumerate(planned_groups):
        group['planned_date'] = date_by_index[index].strftime("%Y-%m-%d")

    planned_groups.sort(key=lambda group: group.get('group_id', 0))
    return planned_groups


def build_preference_time_span_report_rows(
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    p_id = record.get('p_id')
    profile_name = record.get('profile_name', '')
    buckets: Dict[Tuple[str, str], List[str]] = {}

    for group in record.get('groups', []):
        date_str = group.get('planned_date')
        if not _parse_planned_date(date_str):
            continue
        explicit, implicit, pairs = _group_preference_keys(group)
        for category in explicit:
            buckets.setdefault(('explicit', category), []).append(date_str)
        for category in implicit:
            buckets.setdefault(('implicit', category), []).append(date_str)
        for pair in pairs:
            buckets.setdefault(('mixed', pair), []).append(date_str)

    rows: List[Dict[str, Any]] = []
    for (pref_type, category), date_strings in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        parsed_dates = [_parse_planned_date(value) for value in date_strings]
        parsed_dates = sorted(value for value in parsed_dates if value is not None)
        if not parsed_dates:
            continue
        gaps = [(b - a).days for a, b in zip(parsed_dates, parsed_dates[1:])]
        first_date = parsed_dates[0]
        last_date = parsed_dates[-1]
        rows.append({
            'p_id': p_id,
            'profile_name': profile_name,
            'preference_category': category,
            'pref_type': pref_type,
            'frequency': len(parsed_dates),
            'first_date': first_date.strftime("%Y-%m-%d"),
            'last_date': last_date.strftime("%Y-%m-%d"),
            'span_days': (last_date - first_date).days,
            'min_gap_days': min(gaps) if gaps else '',
            'avg_gap_days': round(sum(gaps) / len(gaps), 2) if gaps else '',
            'dates': ";".join(date.strftime("%Y-%m-%d") for date in parsed_dates),
        })
    return rows


def write_preference_time_span_report(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'p_id',
        'profile_name',
        'preference_category',
        'pref_type',
        'frequency',
        'first_date',
        'last_date',
        'span_days',
        'min_gap_days',
        'avg_gap_days',
        'dates',
    ]
    rows: List[Dict[str, Any]] = []
    for record in records:
        rows.extend(build_preference_time_span_report_rows(record))

    with output_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_entity_anchor_coverage_report(
    groups_records: List[Dict[str, Any]],
    event_results: List[Dict[str, Any]],
    output_csv_path: Path,
) -> None:
    expected: Dict[Tuple[Any, str, str, str], Dict[str, Any]] = {}
    category_stats: Dict[Tuple[Any, str, str], Dict[str, Any]] = {}

    for record in groups_records:
        p_id = record.get('p_id')
        profile_name = record.get('profile_name', '')
        for group in record.get('groups', []):
            group_id = group.get('group_id')
            pref_specs = [
                ('explicit', group.get('explicit_preferences', [])),
                ('implicit', group.get('implicit_preferences', [])),
            ]
            for pref_type, prefs in pref_specs:
                for pref in prefs:
                    if not isinstance(pref, dict) or not has_visual_modality(pref):
                        continue
                    category = str(pref.get('category', '')).strip()
                    anchors = _normalize_entity_anchors(pref.get('entity_anchors'))
                    if not category or not anchors:
                        continue
                    stat_key = (p_id, pref_type, category)
                    stats = category_stats.setdefault(
                        stat_key,
                        {'visual_group_ids': set(), 'anchors': set()},
                    )
                    stats['visual_group_ids'].add(group_id)
                    stats['anchors'].update(anchors)

                    for anchor in anchors:
                        key = (p_id, pref_type, category, anchor)
                        item = expected.setdefault(
                            key,
                            {
                                'p_id': p_id,
                                'profile_name': profile_name,
                                'preference_category': category,
                                'pref_type': pref_type,
                                'anchor': anchor,
                                'occurrence_count_in_groups': 0,
                                'appeared_group_ids': set(),
                            },
                        )
                        item['occurrence_count_in_groups'] += 1
                        item['appeared_group_ids'].add(group_id)

    events_by_pid: Dict[Any, List[Dict[str, Any]]] = {}
    for result in event_results:
        events_by_pid.setdefault(result.get('p_id'), []).append(result)

    rows: List[Dict[str, Any]] = []
    for key, item in sorted(expected.items(), key=lambda kv: (str(kv[0][0]), kv[0][1], kv[0][2], kv[0][3])):
        p_id, pref_type, category, anchor = key
        selected_count = 0
        text_hit_count = 0
        for result in events_by_pid.get(p_id, []):
            event = result.get('event', {}) if isinstance(result, dict) else {}
            selected_anchors = []
            if isinstance(result, dict):
                selected_anchors.extend(_normalize_entity_anchors(result.get('entity_anchors')))
            if isinstance(event, dict):
                selected_anchors.extend(_normalize_entity_anchors(event.get('entity_anchors')))
                image_description = str(event.get('user_shared_image_description', '') or '')
            else:
                image_description = ''
            selected_count += sum(1 for selected in selected_anchors if selected == anchor)
            if anchor and image_description and anchor in image_description:
                text_hit_count += 1

        stat = category_stats.get((p_id, pref_type, category), {})
        visual_occurrences = len(stat.get('visual_group_ids', set()))
        unique_anchor_count = len(stat.get('anchors', set()))
        coverage_possible = (visual_occurrences * 2) >= unique_anchor_count if unique_anchor_count else True
        is_covered = selected_count > 0 or text_hit_count > 0
        if is_covered:
            missing_reason = ''
        elif not coverage_possible:
            missing_reason = 'not_enough_visual_occurrences'
        else:
            missing_reason = 'not_selected_or_text_missing'

        group_ids = sorted(
            [gid for gid in item['appeared_group_ids'] if gid is not None],
            key=lambda value: (0, int(value)) if str(value).isdigit() else (1, str(value)),
        )
        rows.append({
            'p_id': item['p_id'],
            'profile_name': item['profile_name'],
            'preference_category': item['preference_category'],
            'pref_type': item['pref_type'],
            'anchor': item['anchor'],
            'occurrence_count_in_groups': item['occurrence_count_in_groups'],
            'selected_count_in_event_entity_anchors': selected_count,
            'text_hit_count_in_user_shared_image_description': text_hit_count,
            'is_covered': is_covered,
            'coverage_possible': coverage_possible,
            'appeared_group_ids': ";".join(str(gid) for gid in group_ids),
            'missing_reason': missing_reason,
        })

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'p_id',
        'profile_name',
        'preference_category',
        'pref_type',
        'anchor',
        'occurrence_count_in_groups',
        'selected_count_in_event_entity_anchors',
        'text_hit_count_in_user_shared_image_description',
        'is_covered',
        'coverage_possible',
        'appeared_group_ids',
        'missing_reason',
    ]
    with output_csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plan_events_for_groups(groups: List[Dict[str, Any]], used_dates_global: Optional[set] = None) -> List[Dict[str, Any]]:
    event_plans: List[Dict[str, Any]] = []
    used_dates = used_dates_global if used_dates_global is not None else set()

    for group in groups:
        planned_date = _parse_planned_date(group.get('planned_date'))
        if planned_date is not None:
            date = planned_date
            date_str = date.strftime("%Y-%m-%d")
            used_dates.add(date_str)
        else:
            while True:
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                date = datetime(2025, month, day)
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in used_dates:
                    used_dates.add(date_str)
                    break

        event_plans.append({
            'group_id': group['group_id'],
            'exp_prefs': group['explicit_preferences'],
            'imp_prefs': group['implicit_preferences'],
            'recommended_main_scene': group.get('recommended_main_scene', ''),
            'implicit_integration_guidance': group.get('implicit_integration_guidance', ''),
            'date': date,
            'date_str': date_str,
        })

    event_plans.sort(key=lambda x: x['date'])
    print("\n=== Event Plan ===")
    print(f"Total events planned: {len(event_plans)}")
    if event_plans:
        print(f"Date range: {event_plans[0]['date_str']} to {event_plans[-1]['date_str']}")
    print("==================\n")
    return event_plans


def build_retry_feedback_block(retry_feedback: List[str]) -> str:
    """Format local validation failures so the next LLM attempt can repair them."""
    cleaned_feedback = [str(item).strip() for item in retry_feedback if str(item).strip()]
    if not cleaned_feedback:
        return ""
    feedback_json = json.dumps(cleaned_feedback, ensure_ascii=False, indent=2)
    return (
        "\n\n[上一轮输出未通过本地检查]\n"
        "下面是上一轮被拒绝的具体原因。请在本轮生成中逐条修正这些问题；"
        "仍然只输出目标 JSON，不要解释，不要输出 markdown。\n"
        f"{feedback_json}\n"
    )


def image_retry_feedback_items(retry_feedback: List[str]) -> List[str]:
    """Keep only retry feedback that is useful for image-description regeneration."""
    keywords = (
        "image description",
        "user_shared_image_description",
        "Relationship",
        "Pets",
        "宠物名",
        "人物姓名",
        "图像描述",
        "图片描述",
    )
    selected: List[str] = []
    for item in retry_feedback:
        text = str(item).strip()
        if text and any(keyword in text for keyword in keywords):
            selected.append(text)
    return selected


def required_person_pet_names_for_image(
    profile: Optional[Dict[str, Any]],
    explicit_prefs: List[Dict[str, Any]],
) -> List[str]:
    """Return exact Relationship/Pets names that image descriptions must contain."""
    if profile is None:
        return []
    names: List[str] = []
    names.extend(sorted(_allowed_person_names_for_img(profile, explicit_prefs)))
    pet_aliases = sorted(_allowed_pet_names(profile, explicit_prefs))
    # For pets, the validator accepts any alias, but the Basic name is usually
    # the most stable visual-reference key. Keep all aliases to avoid overfitting.
    names.extend(pet_aliases)
    return list(dict.fromkeys(name for name in names if name))


def session(
    task_id: str,
    profile_id: int,
    group_id: int,
    persona: str,
    explicit_prefs: List[Dict[str, Any]],
    implicit_prefs: List[Dict[str, Any]],
    event_date: datetime,
    recommended_main_scene: str = "",
    implicit_integration_guidance: str = "",
    previous_events: Optional[List[Dict[str, Any]]] = None,
    profile: Optional[Dict[str, Any]] = None,
    required_explicit_anchors: Optional[List[str]] = None,
    required_implicit_anchors: Optional[List[str]] = None,
    image_description_mode: str = IMAGE_DESCRIPTION_MODE,
) -> Tuple[Dict[str, Any], int, int]:
    date_str = event_date.strftime("%m/%d/%Y")
    implicit_audio_integration_guidance = guidance_for_modality(
        implicit_prefs,
        implicit_integration_guidance,
        "audio",
    )
    implicit_visual_integration_guidance = guidance_for_modality(
        implicit_prefs,
        implicit_integration_guidance,
        "visual",
    )

    previous_events_str = "none"
    if previous_events:
        recent_events = previous_events[-5:]
        events_summary = []
        for e in recent_events:
            scene = e.get('scene_description', '')
            explicit = e.get('explicit_preferences_reflected', [])
            events_summary.append(f"- {scene} (显式偏好: {explicit})")
        previous_events_str = (
            "[已发生的事件]\n"
            + "\n".join(events_summary)
            + "\n\n**重要**：\n"
            + "- 可以继续讨论相同话题，但不要机械重复\n"
            + "- 不要重复介绍已介绍过的人物、宠物、物品的基本信息\n"
            + "- 新事件应该展现新的内容或进展，而非重复旧信息\n"
        )

    base_prompt = prompt_combined.format(
        recommended_main_scene=recommended_main_scene or "无；请自行在同一空间内构造自然场景",
        implicit_audio_integration_guidance=implicit_audio_integration_guidance,
        profile_str=persona,
        explicit_preferences_str=get_preference_str(explicit_prefs),
        implicit_preferences_str=get_preference_str(implicit_prefs),
        previous_events_str=previous_events_str,
    )

    _safe_print(
        f"\n[task={task_id} | group={group_id}]\n"
        f"recommended_main_scene: {recommended_main_scene}\n"
        f"implicit_audio_integration_guidance: {implicit_audio_integration_guidance}\n"
        f"implicit_visual_integration_guidance: {implicit_visual_integration_guidance}\n"
        f"[explicit]\n{get_preference_str(explicit_prefs)}\n"
        f"[implicit]\n{get_preference_str(implicit_prefs)}"
    )

    retry_feedback: List[str] = []

    for attempt in range(3):
        usage_info = None
        prompt_tokens = 0
        completion_tokens = 0
        client = openai_client()
        raw_response_text = ""
        prompt = base_prompt + build_retry_feedback_block(retry_feedback)
        if retry_feedback:
            _safe_print(
                f"[task={task_id}] retry attempt {attempt + 1}/3 with local validation feedback:\n"
                + "\n".join(f"  - {item}" for item in retry_feedback)
            )

        try:
            api_res = client.chat.completions.create(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.8,
                top_p=0.9,
                stream=True,
                stream_options={"include_usage": True},
            )

            for chunk in api_res:
                if chunk.choices and chunk.choices[0].delta.content:
                    raw_response_text += chunk.choices[0].delta.content
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    usage_info = chunk.usage

            response_text = raw_response_text.strip().replace("```json", "").replace("```", "")
            if not response_text:
                raise ValueError("Model returned empty response text")

            parsed = json.loads(repair_json(response_text))
            event = unwrap_event_object(parsed)

            scene = str(event.get('scene_description', '')).strip()
            if re.match(r'^\d{2}/\d{2}/\d{4}', scene):
                scene = re.sub(r'^\d{2}/\d{2}/\d{4}', date_str, scene)
            else:
                scene = f"{date_str};{scene.lstrip(';； ')}"
            event['scene_description'] = scene

            # ── 生成 user_shared_image_description ───────────────────────────
            # 从 profile 中提取拍摄者名字
            photographer_name = ""
            if profile is not None:
                photographer_name = str((profile.get("Basic") or {}).get("name", "") or "").strip()
            required_person_pet_names = required_person_pet_names_for_image(profile, explicit_prefs)
            image_feedback = image_retry_feedback_items(retry_feedback)

            image_desc, image_anchors, img_pt, img_ct = _generate_image_description(
                task_id=task_id,
                scene_description=scene,
                explicit_prefs=explicit_prefs,
                implicit_prefs=implicit_prefs,
                photographer_name=photographer_name,
                required_explicit_anchors=required_explicit_anchors,
                required_implicit_anchors=required_implicit_anchors,
                required_person_pet_names=required_person_pet_names,
                image_retry_feedback=image_feedback,
                implicit_visual_integration_guidance=implicit_visual_integration_guidance,
                image_description_mode=image_description_mode,
            )
            event['user_shared_image_description'] = image_desc
            event['entity_anchors'] = filter_generated_entity_anchors(
                image_anchors,
                explicit_prefs,
                implicit_prefs,
                task_id=task_id,
            )
            prompt_tokens += img_pt
            completion_tokens += img_ct

            clean_issues: List[str] = []
            cleaned_event = clean_event_fields(event, explicit_prefs, implicit_prefs, issues=clean_issues)
            if cleaned_event is None:
                print(f"Retrying {task_id}: clean_event_fields rejected the event")
                retry_feedback = [
                    "clean_event_fields rejected the previous event.",
                    *clean_issues,
                ]
                time.sleep(1)
                continue

            if profile is not None:
                person_pet_issues = validate_persons_and_pets_usage(cleaned_event, profile, explicit_prefs)
                if person_pet_issues:
                    print(f"Retrying {task_id}: persons/pets usage violations:")
                    for issue in person_pet_issues:
                        print(f"  - {issue}")
                    retry_feedback = [
                        "The previous event used Relationship/Pets entities incorrectly.",
                        *person_pet_issues,
                    ]
                    time.sleep(1)
                    continue

                img_desc_issues = validate_person_in_image_desc(cleaned_event, profile, explicit_prefs)
                if img_desc_issues:
                    print(f"Retrying {task_id}: person/pet name missing in image description:")
                    for issue in img_desc_issues:
                        print(f"  - {issue}")
                    retry_feedback = [
                        "The previous image description failed Relationship/Pets name requirements.",
                        *img_desc_issues,
                    ]
                    time.sleep(1)
                    continue

            result = {
                'p_id': profile_id,
                'task_id': task_id,
                'group_id': group_id,
                'recommended_main_scene': recommended_main_scene,
                'implicit_integration_guidance': implicit_integration_guidance,
                'explicit_preferences': explicit_prefs,
                'implicit_preferences': implicit_prefs,
                'event': cleaned_event,
            }

            if usage_info is not None:
                prompt_tokens += usage_info.prompt_tokens
                completion_tokens += usage_info.completion_tokens

            return result, prompt_tokens, completion_tokens

        except Exception as e:
            err_msg = str(e)
            print(f"[session error] task={task_id} | attempt={attempt + 1}/3 | error={err_msg}")
            if raw_response_text:
                print(f"[raw response snippet] {raw_response_text[:1000]}")
            retry_feedback = [f"The previous attempt failed with an exception or invalid JSON: {err_msg}"]
            if raw_response_text:
                retry_feedback.append(f"Previous raw response snippet: {raw_response_text[:800]}")
            if "503" in err_msg or "SERVICE_UNAVAILABLE" in err_msg:
                sleep_s = 2 * (attempt + 1)
            else:
                sleep_s = attempt + 1
            time.sleep(sleep_s)

    return {}, 0, 0


def _run_profile_events(
    p_id: int,
    profile: Dict[str, Any],
    profile_str: str,
    event_plans: List[Dict[str, Any]],
    existing_results: Dict[Tuple[str, str], Dict[str, Any]],
    image_description_mode: str = IMAGE_DESCRIPTION_MODE,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """顺序执行单个人物的所有事件，返回 (results, prompt_tokens, completion_tokens)。
    不同人物之间通过 ThreadPoolExecutor 并发调用此函数。
    人物内部事件必须串行，因为 previous_events 存在依赖。
    """
    profile_events: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    reused_count = 0
    generated_count = 0
    anchor_usage: Dict[Tuple[str, str], int] = {}

    _safe_print(f"\n{'=' * 60}\nStarting Profile {p_id} — {len(event_plans)} events\n{'=' * 60}")

    for i, plan in enumerate(event_plans):
        task_id = f"{p_id}-{plan['group_id']}-0"
        event_key = make_event_key(p_id, plan['group_id'])
        existing_result = existing_results.get(event_key)
        if existing_result:
            event = existing_result.get('event', {})
            if event:
                profile_events.append(event)
                update_anchor_usage_from_event(
                    anchor_usage,
                    plan['exp_prefs'],
                    plan['imp_prefs'],
                    event,
                )
            results.append(existing_result)
            reused_count += 1
            _safe_print(
                f"\n[Profile {p_id}] --- Skipping event {i + 1}/{len(event_plans)} "
                f"(group_id={plan['group_id']}) — already generated ---"
            )
            continue

        _safe_print(f"\n[Profile {p_id}] --- Generating event {i + 1}/{len(event_plans)}: {plan['date_str']} ---")
        required_explicit_anchors = select_required_anchors(
            plan['exp_prefs'],
            anchor_usage,
            max_count=2,
            min_count_if_available=1,
        )
        required_implicit_anchors = select_required_anchors(
            plan['imp_prefs'],
            anchor_usage,
            max_count=2,
            min_count_if_available=1,
        )
        if required_explicit_anchors or required_implicit_anchors:
            _safe_print(
                f"[Profile {p_id}] required anchors for group {plan['group_id']}: "
                f"foreground={required_explicit_anchors}, background={required_implicit_anchors}"
            )

        result, prompt_tokens, completion_tokens = session(
            task_id=task_id,
            profile_id=p_id,
            group_id=plan['group_id'],
            persona=profile_str,
            explicit_prefs=plan['exp_prefs'],
            implicit_prefs=plan['imp_prefs'],
            event_date=plan['date'],
            recommended_main_scene=plan.get('recommended_main_scene', ''),
            implicit_integration_guidance=plan.get('implicit_integration_guidance', ''),
            previous_events=profile_events,
            profile=profile,
            required_explicit_anchors=required_explicit_anchors,
            required_implicit_anchors=required_implicit_anchors,
            image_description_mode=image_description_mode,
        )

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens

        if result:
            event = result.get('event', {})
            if event:
                profile_events.append(event)
                update_anchor_usage_from_event(
                    anchor_usage,
                    plan['exp_prefs'],
                    plan['imp_prefs'],
                    event,
                )
            results.append(result)
            generated_count += 1

    _safe_print(
        f"\n[Profile {p_id}] Done — {len(results)}/{len(event_plans)} events available "
        f"({reused_count} reused, {generated_count} newly generated)."
    )
    return results, total_prompt_tokens, total_completion_tokens


def parse_regen_targets(target_strs: List[str]) -> List[Tuple[int, int]]:
    """
    解析命令行传入的重新生成目标。
    支持两种格式：
    - task_id 格式: "0-35-0" -> (p_id=0, group_id=35)
    - p_id+group_id 格式: "0:35" 或 "0-35" -> (p_id=0, group_id=35)

    返回: [(p_id, group_id), ...] 列表
    """
    targets: List[Tuple[int, int]] = []
    for s in target_strs:
        s = s.strip()
        if not s:
            continue

        # 尝试 task_id 格式: "p_id-group_id-turn" (如 "0-35-0")
        task_match = re.match(r'^(\d+)-(\d+)-\d+$', s)
        if task_match:
            p_id = int(task_match.group(1))
            group_id = int(task_match.group(2))
            targets.append((p_id, group_id))
            continue

        # 尝试 p_id:group_id 格式
        colon_match = re.match(r'^(\d+):(\d+)$', s)
        if colon_match:
            p_id = int(colon_match.group(1))
            group_id = int(colon_match.group(2))
            targets.append((p_id, group_id))
            continue

        # 尝试 p_id-group_id 格式（两个数字）
        dash_match = re.match(r'^(\d+)-(\d+)$', s)
        if dash_match:
            p_id = int(dash_match.group(1))
            group_id = int(dash_match.group(2))
            targets.append((p_id, group_id))
            continue

        raise ValueError(f"无法解析目标: '{s}'。支持格式: task_id (如 '0-35-0') 或 p_id:group_id (如 '0:35')")

    # 去重
    seen = set()
    unique_targets = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    return unique_targets


def find_group_info(
    manual_group_records: List[Dict[str, Any]],
    p_id: int,
    group_id: int,
) -> Optional[Dict[str, Any]]:
    """
    从 manual_groups 中查找指定 (p_id, group_id) 的分组信息。
    返回该 group 的完整信息 dict，若未找到返回 None。
    """
    for record in manual_group_records:
        if record.get('p_id') != p_id:
            continue
        for group in record.get('groups', []):
            if group.get('group_id') == group_id:
                return group
    return None


def run_manual_regen(
    regen_targets: List[Tuple[int, int]],
    profiles: List[Dict[str, Any]],
    manual_group_records: List[Dict[str, Any]],
    existing_events: List[Dict[str, Any]],
    save_path: Path,
    image_description_mode: str = IMAGE_DESCRIPTION_MODE,
) -> None:
    """
    手动重新生成指定的 events。

    Args:
        regen_targets: [(p_id, group_id), ...] 需要重新生成的目标列表
        profiles: 所有用户档案列表
        manual_group_records: manual_groups 文件内容
        existing_events: 已有的 events 列表
        save_path: events 输出路径
    """
    print(f"\n{'=' * 60}")
    print("手动重新生成模式")
    print(f"目标数量: {len(regen_targets)}")
    print(f"{'=' * 60}\n")

    # 构建 (p_id, group_id) -> existing_event 的映射
    existing_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for event in existing_events:
        p_id = event.get('p_id')
        group_id = event.get('group_id')
        if p_id is not None and group_id is not None:
            existing_map[(p_id, group_id)] = event

    # 按 p_id 分组目标，方便按人物顺序处理
    targets_by_pid: Dict[int, List[int]] = {}
    for p_id, group_id in regen_targets:
        targets_by_pid.setdefault(p_id, []).append(group_id)

    # 收集该人物之前已生成的所有 events（用于 previous_events）
    def get_previous_events(p_id: int, up_to_group_id: int) -> List[Dict[str, Any]]:
        """获取指定人物在某个 group_id 之前的所有 events（按 group_id 排序）"""
        prev_events = []
        for (pid, gid), event in existing_map.items():
            if pid == p_id and gid < up_to_group_id:
                prev_events.append((gid, event.get('event', {})))
        prev_events.sort(key=lambda x: x[0])
        return [e for _, e in prev_events]

    new_results: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for p_id in sorted(targets_by_pid.keys()):
        group_ids = sorted(targets_by_pid[p_id])

        if p_id >= len(profiles):
            print(f"[Warning] p_id={p_id} 超出 profiles 范围，跳过")
            continue

        profile = profiles[p_id]
        profile_str = get_dict_str(profile['Basic'], profile['mbti'])
        pref_map = build_profile_preference_map(profile)

        # 获取该人物的 manual_record
        manual_record = None
        for record in manual_group_records:
            if record.get('p_id') == p_id:
                manual_record = record
                break

        if manual_record is None:
            print(f"[Warning] p_id={p_id} 在 manual_groups 中未找到，跳过")
            continue

        print(f"\n{'=' * 60}")
        print(f"处理 Profile {p_id} — 需重新生成 {len(group_ids)} 个 events")
        print(f"{'=' * 60}")

        # 收集该人物之前所有 events 用于 previous_events
        profile_events: List[Dict[str, Any]] = get_previous_events(p_id, min(group_ids))

        for group_id in group_ids:
            task_id = f"{p_id}-{group_id}-0"

            # 查找分组信息
            group_info = find_group_info(manual_group_records, p_id, group_id)
            if group_info is None:
                print(f"[Warning] (p_id={p_id}, group_id={group_id}) 在 manual_groups 中未找到，跳过")
                continue

            # 规范化偏好
            explicit_prefs = normalize_pref_list(
                group_info.get('explicit_preferences', []),
                pref_map,
                default_expression_type='explicit',
                category_hints=group_info.get('explicit_categories', []),
            )
            implicit_prefs = normalize_pref_list(
                group_info.get('implicit_preferences', []),
                pref_map,
                default_expression_type='implicit',
                category_hints=group_info.get('implicit_categories', []),
            )

            recommended_main_scene = group_info.get('recommended_main_scene', '')
            implicit_integration_guidance = group_info.get('implicit_integration_guidance', '')

            planned_date = _parse_planned_date(group_info.get('planned_date'))
            if planned_date is not None:
                event_date = planned_date
            else:
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                event_date = datetime(2025, month, day)

            print(f"\n[Regenerating] task_id={task_id}")
            print(f"  date: {event_date.strftime('%Y-%m-%d')}")
            print(f"  recommended_main_scene: {recommended_main_scene[:60]}...")
            print(f"  explicit: {[p.get('category') for p in explicit_prefs]}")
            print(f"  implicit: {[p.get('category') for p in implicit_prefs]}")

            # 调用 session 生成
            result, pt, ct = session(
                task_id=task_id,
                profile_id=p_id,
                group_id=group_id,
                persona=profile_str,
                explicit_prefs=explicit_prefs,
                implicit_prefs=implicit_prefs,
                event_date=event_date,
                recommended_main_scene=recommended_main_scene,
                implicit_integration_guidance=implicit_integration_guidance,
                previous_events=profile_events,
                profile=profile,
                image_description_mode=image_description_mode,
            )

            total_prompt_tokens += pt
            total_completion_tokens += ct

            if result:
                new_results.append(result)
                # 更新 profile_events 供后续 events 使用
                event = result.get('event', {})
                if event:
                    profile_events.append(event)
                print(f"[OK] task_id={task_id} 生成成功")
            else:
                print(f"[Failed] task_id={task_id} 生成失败")

    # 合并结果：新结果替换旧结果
    print(f"\n{'=' * 60}")
    print("合并结果")
    print(f"{'=' * 60}")

    # 构建最终输出
    regen_keys = set(regen_targets)
    final_events: List[Dict[str, Any]] = []

    # 保留未被重新生成的旧 events
    for event in existing_events:
        p_id = event.get('p_id')
        group_id = event.get('group_id')
        if (p_id, group_id) not in regen_keys:
            final_events.append(event)

    # 添加新生成的 events
    final_events.extend(new_results)

    # 按 (p_id, group_id) 排序
    final_events.sort(key=lambda e: (e.get('p_id', 0), e.get('group_id', 0)))

    # 写回文件
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(final_events, indent=4, ensure_ascii=False), encoding='utf-8')
    build_entity_anchor_coverage_report(
        manual_group_records,
        final_events,
        ensure_parent_dir(ENTITY_ANCHOR_COVERAGE_REPORT_CSV),
    )

    print(f"\n结果已保存到: {save_path}")
    print(f"Entity anchor coverage report saved to: {ENTITY_ANCHOR_COVERAGE_REPORT_CSV}")
    print(f"  - 保留旧 events: {len(existing_events) - len(regen_targets)}")
    print(f"  - 新生成 events: {len(new_results)}")
    print(f"  - 总计 events: {len(final_events)}")
    print("\nToken 统计:")
    print(f"  - Prompt Tokens: {total_prompt_tokens}")
    print(f"  - Completion Tokens: {total_completion_tokens}")
    print(f"  - 估算费用: ${(total_prompt_tokens * 1 * 0.000001 + total_completion_tokens * 3 * 0.000001):.3f}")


def run_image_only_regen(
    regen_targets: List[Tuple[int, int]],
    profiles: List[Dict[str, Any]],
    manual_group_records: List[Dict[str, Any]],
    existing_events: List[Dict[str, Any]],
    save_path: Path,
    image_description_mode: str = IMAGE_DESCRIPTION_MODE,
) -> None:
    """
    只重新生成指定 events 的 user_shared_image_description / entity_anchors。

    这个模式要求目标 event 已经存在；它不会重新生成 scene_description、
    background_audio_info、human_speech_content 等正文信息。
    """
    print(f"\n{'=' * 60}")
    print("只重新生成图片描述模式")
    print(f"目标数量: {len(regen_targets)}")
    print(f"{'=' * 60}\n")

    existing_map: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for event_record in existing_events:
        p_id = event_record.get('p_id')
        group_id = event_record.get('group_id')
        if p_id is not None and group_id is not None:
            existing_map[(p_id, group_id)] = event_record

    updated_keys: Set[Tuple[int, int]] = set()
    missing_existing: List[Tuple[int, int]] = []
    failed_targets: List[Tuple[int, int]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for p_id, group_id in regen_targets:
        task_id = f"{p_id}-{group_id}-0"
        existing_record = existing_map.get((p_id, group_id))
        if existing_record is None:
            print(f"[Skip] {task_id}: 现有 events 中没有这条记录；请用 --regen 生成整条 event")
            missing_existing.append((p_id, group_id))
            continue

        if p_id >= len(profiles):
            print(f"[Skip] {task_id}: p_id 超出 profiles 范围")
            failed_targets.append((p_id, group_id))
            continue

        group_info = find_group_info(manual_group_records, p_id, group_id)
        if group_info is None:
            print(f"[Skip] {task_id}: manual groups 中找不到对应 group")
            failed_targets.append((p_id, group_id))
            continue

        profile = profiles[p_id]
        pref_map = build_profile_preference_map(profile)
        explicit_prefs = normalize_pref_list(
            group_info.get('explicit_preferences', []),
            pref_map,
            default_expression_type='explicit',
            category_hints=group_info.get('explicit_categories', []),
        )
        implicit_prefs = normalize_pref_list(
            group_info.get('implicit_preferences', []),
            pref_map,
            default_expression_type='implicit',
            category_hints=group_info.get('implicit_categories', []),
        )

        has_visual = has_explicit_visual_subject_in_prefs(explicit_prefs) or has_visual_modality_in_prefs(implicit_prefs)
        updated_record = deepcopy(existing_record)
        event_obj = updated_record.get('event')
        if not isinstance(event_obj, dict):
            print(f"[Skip] {task_id}: event 字段不是对象，无法只更新图片描述")
            failed_targets.append((p_id, group_id))
            continue

        if not has_visual:
            event_obj['user_shared_image_description'] = 'none'
            event_obj['entity_anchors'] = []
            existing_map[(p_id, group_id)] = updated_record
            updated_keys.add((p_id, group_id))
            print(f"[OK] {task_id}: 当前 group 无 visual，已设置 image_description=none")
            continue

        scene_description = str(event_obj.get('scene_description', '') or '').strip()
        if not scene_description:
            scene_description = str(group_info.get('recommended_main_scene', '') or '').strip()
        if not scene_description:
            print(f"[Skip] {task_id}: 缺少 scene_description / recommended_main_scene")
            failed_targets.append((p_id, group_id))
            continue

        photographer_name = str((profile.get("Basic") or {}).get("name", "") or "").strip()
        required_person_pet_names = required_person_pet_names_for_image(profile, explicit_prefs)
        required_explicit_anchors = select_required_anchors(
            explicit_prefs,
            anchor_usage={},
            max_count=2,
            min_count_if_available=1,
        )
        required_implicit_anchors = select_required_anchors(
            implicit_prefs,
            anchor_usage={},
            max_count=2,
            min_count_if_available=1,
        )
        implicit_visual_integration_guidance = guidance_for_modality(
            implicit_prefs,
            str(group_info.get('implicit_integration_guidance', '') or ''),
            "visual",
        )

        print(f"\n[ImageOnly] task_id={task_id}")
        print(f"  explicit: {[p.get('category') for p in explicit_prefs]}")
        print(f"  implicit: {[p.get('category') for p in implicit_prefs]}")
        print(f"  required names: {required_person_pet_names or []}")
        print(f"  required anchors: foreground={required_explicit_anchors}, background={required_implicit_anchors}")

        retry_feedback: List[str] = []
        success = False
        for attempt in range(3):
            image_feedback = image_retry_feedback_items(retry_feedback)
            if retry_feedback:
                print(
                    f"[ImageOnly] {task_id} retry attempt {attempt + 1}/3:\n"
                    + "\n".join(f"  - {item}" for item in retry_feedback)
                )

            image_desc, image_anchors, img_pt, img_ct = _generate_image_description(
                task_id=task_id,
                scene_description=scene_description,
                explicit_prefs=explicit_prefs,
                implicit_prefs=implicit_prefs,
                photographer_name=photographer_name,
                required_explicit_anchors=required_explicit_anchors,
                required_implicit_anchors=required_implicit_anchors,
                required_person_pet_names=required_person_pet_names,
                image_retry_feedback=image_feedback,
                implicit_visual_integration_guidance=implicit_visual_integration_guidance,
                image_description_mode=image_description_mode,
            )
            total_prompt_tokens += img_pt
            total_completion_tokens += img_ct

            trial_event = deepcopy(event_obj)
            trial_event['user_shared_image_description'] = image_desc
            trial_event['entity_anchors'] = filter_generated_entity_anchors(
                image_anchors,
                explicit_prefs,
                implicit_prefs,
                task_id=task_id,
            )

            issues: List[str] = []
            if str(image_desc or '').strip().lower() == 'none':
                issues.append("Visual preference exists but user_shared_image_description is none")
            if profile is not None:
                issues.extend(validate_person_in_image_desc(trial_event, profile, explicit_prefs))

            if issues:
                retry_feedback = [
                    "The previous image description failed local validation.",
                    *issues,
                ]
                print(f"[ImageOnly] {task_id} validation failed:")
                for issue in issues:
                    print(f"  - {issue}")
                time.sleep(1)
                continue

            event_obj['user_shared_image_description'] = trial_event['user_shared_image_description']
            event_obj['entity_anchors'] = trial_event['entity_anchors']
            existing_map[(p_id, group_id)] = updated_record
            updated_keys.add((p_id, group_id))
            success = True
            print(f"[OK] {task_id}: image description regenerated")
            break

        if not success:
            failed_targets.append((p_id, group_id))
            print(f"[Failed] {task_id}: 3 次尝试后仍未生成有效图片描述，保留原 event 不变")

    final_events: List[Dict[str, Any]] = []
    for event_record in existing_events:
        key = (event_record.get('p_id'), event_record.get('group_id'))
        final_events.append(existing_map.get(key, event_record))
    final_events.sort(key=lambda e: (e.get('p_id', 0), e.get('group_id', 0)))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(final_events, indent=4, ensure_ascii=False), encoding='utf-8')
    build_entity_anchor_coverage_report(
        manual_group_records,
        final_events,
        ensure_parent_dir(ENTITY_ANCHOR_COVERAGE_REPORT_CSV),
    )

    print(f"\n结果已保存到: {save_path}")
    print(f"Entity anchor coverage report saved to: {ENTITY_ANCHOR_COVERAGE_REPORT_CSV}")
    print(f"  - 成功更新 image description: {len(updated_keys)}")
    print(f"  - 目标 event 不存在: {len(missing_existing)}")
    print(f"  - 失败/跳过: {len(failed_targets)}")
    print(f"  - 总计 events: {len(final_events)}")
    print("Token 统计:")
    print(f"  - Prompt Tokens: {total_prompt_tokens}")
    print(f"  - Completion Tokens: {total_completion_tokens}")


def parse_profile_id_filter(raw_values: Optional[List[str]]) -> Optional[Set[int]]:
    if not raw_values:
        return None
    result: Set[int] = set()
    for raw in raw_values:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                result.add(int(part))
    return result


def select_profiles_for_generation(
    profiles: List[Dict[str, Any]],
    sample: Optional[int] = None,
    only_profile_ids: Optional[Set[int]] = None,
) -> List[Tuple[int, Dict[str, Any]]]:
    selected: List[Tuple[int, Dict[str, Any]]] = []
    for p_id, profile in enumerate(profiles):
        if only_profile_ids is not None and p_id not in only_profile_ids:
            continue
        selected.append((p_id, profile))
    if sample is not None:
        selected = selected[:sample]
    return selected


def merge_profile_results(
    profile_results: Dict[int, List[Dict[str, Any]]],
    existing_results: Dict[Tuple[str, str], Dict[str, Any]],
    preserve_existing: bool,
) -> List[Dict[str, Any]]:
    total_results: List[Dict[str, Any]] = []
    included_keys: Set[Tuple[str, str]] = set()

    for p_id in sorted(profile_results.keys()):
        for result in profile_results[p_id]:
            included_keys.add(make_event_key(result.get('p_id'), result.get('group_id')))
            total_results.append(result)

    if preserve_existing:
        for key, result in existing_results.items():
            if key not in included_keys:
                total_results.append(result)

    total_results.sort(key=lambda r: (int(r.get('p_id', 10**9)), int(r.get('group_id', 10**9))))
    return total_results


def save_generation_checkpoint(
    *,
    save_profile_path: Path,
    save_groups_path: Path,
    groups_with_dates_path: Path,
    preference_time_span_report_path: Path,
    entity_anchor_coverage_report_path: Path,
    total_groups_output: List[Dict[str, Any]],
    profile_results: Dict[int, List[Dict[str, Any]]],
    existing_results: Dict[Tuple[str, str], Dict[str, Any]],
    preserve_existing: bool,
) -> List[Dict[str, Any]]:
    total_results = merge_profile_results(profile_results, existing_results, preserve_existing)
    save_profile_path.write_text(json.dumps(total_results, indent=4, ensure_ascii=False), encoding='utf-8')
    save_groups_path.write_text(json.dumps(total_groups_output, indent=4, ensure_ascii=False), encoding='utf-8')
    groups_with_dates_path.write_text(json.dumps(total_groups_output, indent=4, ensure_ascii=False), encoding='utf-8')
    write_preference_time_span_report(total_groups_output, preference_time_span_report_path)
    build_entity_anchor_coverage_report(total_groups_output, total_results, entity_anchor_coverage_report_path)
    return total_results


def main() -> None:
    global PERSONA_FILE_PATH, MANUAL_GROUPS_FILENAME, SAVE_PROFILE_PATH
    global SAVE_GROUPS_PATH, GROUPS_WITH_DATES_PATH, PREFERENCE_TIME_SPAN_REPORT_CSV
    global ENTITY_ANCHOR_COVERAGE_REPORT_CSV, MANUAL_REGEN_EVENTS_PATH, MODEL
    # ── 解析命令行参数 ───────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="生成用户事件。默认全量生成，传入 --regen 后进入手动重新生成模式。"
    )
    parser.add_argument(
        "--regen",
        nargs="+",
        metavar="TARGET",
        help=(
            "手动重新生成指定的 events。TARGET 支持两种格式：\n"
            "  - task_id 格式: '0-35-0' (会提取 p_id=0, group_id=35)\n"
            "  - p_id:group_id 格式: '0:35' 或 '0-35'"
        ),
    )
    parser.add_argument(
        "--regen_image_only",
        nargs="+",
        metavar="TARGET",
        help=(
            "只重新生成指定 events 的 user_shared_image_description 和 entity_anchors，"
            "保留原 scene_description/background_audio_info/dialogue。"
            "TARGET 格式同 --regen，例如 '0-35-0'、'0:35' 或 '0-35'。"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="只处理筛选后的前 N 个 profile，用于小批量测试。",
    )
    parser.add_argument(
        "--only_profile_ids",
        nargs="*",
        default=None,
        help="只处理指定 p_id，支持空格或逗号形式，例如 --only_profile_ids 0 1 或 --only_profile_ids 0,1。",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="复用输出文件中已存在的 event，跳过已生成的 (p_id, group_id)。默认开启。",
    )
    parser.add_argument(
        "--no_resume",
        dest="resume",
        action="store_false",
        help="不复用已有 event，从当前 sample/筛选范围重新生成输出。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_PROFILE_WORKERS,
        help=f"profile 级并发数，默认 {MAX_PROFILE_WORKERS}。",
    )
    parser.add_argument(
        "--image_description_mode",
        "--image_mode",
        choices=["two_step", "combined"],
        default=IMAGE_DESCRIPTION_MODE,
        help=(
            "user_shared_image_description 生成方式："
            "two_step=foreground/background 两步生成；"
            "combined=一次性生成完整图片描述但保留前景/背景约束。"
        ),
    )
    parser.add_argument("--profiles", default=None, help="profile JSON/JSONL input")
    parser.add_argument("--groups", default=None, help="manual group JSON/JSONL input")
    parser.add_argument("--output", default=None, help="generated event output")
    parser.add_argument("--groups-output", default=None, help="normalized group output")
    parser.add_argument("--dates-output", default=None, help="group/date output")
    parser.add_argument("--time-report", default=None, help="preference time-span CSV")
    parser.add_argument("--anchor-report", default=None, help="entity-anchor coverage CSV")
    parser.add_argument("--regen-events", default=None, help="event file used by --regen")
    parser.add_argument("--model", default=MODEL, help="LLM model name")
    args = parser.parse_args()
    if args.profiles:
        PERSONA_FILE_PATH = str(resolve_path(args.profiles))
    if args.groups:
        MANUAL_GROUPS_FILENAME = str(resolve_path(args.groups))
    if args.output:
        SAVE_PROFILE_PATH = str(resolve_path(args.output))
    if args.groups_output:
        SAVE_GROUPS_PATH = str(resolve_path(args.groups_output))
    if args.dates_output:
        GROUPS_WITH_DATES_PATH = str(resolve_path(args.dates_output))
    if args.time_report:
        PREFERENCE_TIME_SPAN_REPORT_CSV = str(resolve_path(args.time_report))
    if args.anchor_report:
        ENTITY_ANCHOR_COVERAGE_REPORT_CSV = str(resolve_path(args.anchor_report))
    if args.regen_events:
        MANUAL_REGEN_EVENTS_PATH = str(resolve_path(args.regen_events))
    MODEL = args.model
    if args.sample is not None and args.sample < 1:
        raise ValueError("--sample must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    persona_path = get_persona_path()
    manual_groups_path = get_manual_groups_path()
    print(f"Using persona file: {persona_path}")
    print(f"Using manual groups file: {manual_groups_path}")
    print(f"Manual group validation mode: {MANUAL_GROUP_VALIDATION_MODE}")
    print(f"Max parallel profiles: {args.workers}")
    print(f"Image description mode: {args.image_description_mode}")
    if args.sample is not None:
        print(f"Sample mode: selected first {args.sample} profile(s) after filtering.")
    if args.only_profile_ids:
        print(f"Only profile ids: {sorted(parse_profile_id_filter(args.only_profile_ids) or [])}")
    print(f"Resume mode: {'on' if args.resume else 'off'}")

    profiles = load_json_or_jsonl(persona_path)
    manual_group_records = load_json_or_jsonl(manual_groups_path)
    manual_group_map = {record['p_id']: record for record in manual_group_records}

    if args.regen and args.regen_image_only:
        print("[Error] --regen 和 --regen_image_only 不能同时使用")
        return

    # ── 手动重新生成模式 ───────────────────────────────────────────────────────
    if args.regen:
        regen_targets = parse_regen_targets(args.regen)
        if not regen_targets:
            print("[Error] 未指定有效的重新生成目标")
            return

        # 解析 events 输出路径
        events_path = Path(MANUAL_REGEN_EVENTS_PATH)
        if not events_path.exists():
            # 尝试其他候选路径
            candidates = [
                MANUAL_REGEN_EVENTS_PATH,
                "event/events_000_002_with_anchors.jsonl",
            ]
            for c in candidates:
                candidate_path = resolve_path(c)
                if candidate_path.exists():
                    events_path = candidate_path
                    break

        existing_events = []
        if events_path.exists():
            existing_events = load_json_or_jsonl(events_path)
            print(f"[Manual Regen] 已加载现有 events: {len(existing_events)} 条")
        else:
            print("[Manual Regen] 未找到现有 events 文件，将创建新文件")

        run_manual_regen(
            regen_targets=regen_targets,
            profiles=profiles,
            manual_group_records=manual_group_records,
            existing_events=existing_events,
            save_path=events_path,
            image_description_mode=args.image_description_mode,
        )
        return

    # ── 只重新生成图片描述模式 ─────────────────────────────────────────────────
    if args.regen_image_only:
        regen_targets = parse_regen_targets(args.regen_image_only)
        if not regen_targets:
            print("[Error] 未指定有效的图片描述重新生成目标")
            return

        events_path = Path(SAVE_PROFILE_PATH)
        if not events_path.exists():
            print(f"[Error] 找不到现有 events 文件: {events_path}")
            return

        existing_events = load_json_or_jsonl(events_path)
        print(f"[ImageOnly Regen] 已加载现有 events: {len(existing_events)} 条")
        run_image_only_regen(
            regen_targets=regen_targets,
            profiles=profiles,
            manual_group_records=manual_group_records,
            existing_events=existing_events,
            save_path=events_path,
            image_description_mode=args.image_description_mode,
        )
        return

    # ── 默认全量生成模式 ─────────────────────────────────────────────────────────
    save_profile_path = ensure_parent_dir(SAVE_PROFILE_PATH)
    save_groups_path = ensure_parent_dir(SAVE_GROUPS_PATH)
    groups_with_dates_path = ensure_parent_dir(GROUPS_WITH_DATES_PATH)
    preference_time_span_report_path = ensure_parent_dir(PREFERENCE_TIME_SPAN_REPORT_CSV)
    entity_anchor_coverage_report_path = ensure_parent_dir(ENTITY_ANCHOR_COVERAGE_REPORT_CSV)
    existing_results = load_existing_event_results(save_profile_path) if args.resume else {}
    if existing_results:
        existing_by_pid: Dict[str, int] = {}
        for p_id, _group_id in existing_results.keys():
            existing_by_pid[p_id] = existing_by_pid.get(p_id, 0) + 1
        print(
            "[Resume] Loaded existing events: "
            + ", ".join(f"p_id={p_id}: {count}" for p_id, count in sorted(existing_by_pid.items()))
        )
    else:
        if args.resume:
            print("[Resume] No existing event results found; generating from scratch.")
        else:
            print("[Resume] Disabled; generating selected profiles without loading existing events.")

    total_groups_output: List[Dict[str, Any]] = []

    # ── Phase 1（串行）: 验证 + 日期规划 ─────────────────────────────────────
    profile_tasks: List[Tuple[int, Dict[str, Any], str, List[Dict[str, Any]]]] = []
    only_profile_ids = parse_profile_id_filter(args.only_profile_ids)
    selected_profiles = select_profiles_for_generation(
        profiles,
        sample=args.sample,
        only_profile_ids=only_profile_ids,
    )
    selected_profile_ids = {p_id for p_id, _profile in selected_profiles}
    print(
        f"[Selection] Loaded {len(profiles)} profile(s); "
        f"selected {len(selected_profiles)} profile(s): {sorted(selected_profile_ids)}"
    )

    for p_id, profile in selected_profiles:
        print(f"\n{'=' * 60}")
        print(f"[Phase 1] Validating Profile {p_id}")
        print(f"{'=' * 60}")

        if p_id not in manual_group_map:
            raise ValueError(f"No manual groups found for profile p_id={p_id}")

        manual_record = manual_group_map[p_id]
        pref_map = build_profile_preference_map(profile)

        normalized_groups = []
        for group in manual_record.get('groups', []):
            normalized_group = dict(group)
            normalized_group['explicit_preferences'] = normalize_pref_list(
                group.get('explicit_preferences', []),
                pref_map,
                default_expression_type='explicit',
                category_hints=group.get('explicit_categories', []),
            )
            normalized_group['implicit_preferences'] = normalize_pref_list(
                group.get('implicit_preferences', []),
                pref_map,
                default_expression_type='implicit',
                category_hints=group.get('implicit_categories', []),
            )
            normalized_groups.append(normalized_group)

        valid_groups, validation_report = validate_and_filter_manual_groups(
            profile_id=p_id,
            normalized_groups=normalized_groups,
            pref_map=pref_map,
        )

        if not valid_groups:
            manual_record_to_save = dict(manual_record)
            manual_record_to_save['groups'] = valid_groups
            manual_record_to_save['num_groups'] = len(valid_groups)
            manual_record_to_save['validation_report'] = validation_report
            total_groups_output.append(manual_record_to_save)
            print(f"Profile {p_id}: no valid groups remain after validation, skipping event generation.")
            continue

        if all(_parse_planned_date(group.get('planned_date')) is not None for group in valid_groups):
            groups_with_dates = valid_groups
            print(f"Profile {p_id}: using existing planned_date values from group file.")
        else:
            groups_with_dates = plan_group_dates_spaced(valid_groups, year=2025, local_search_steps=5000)
            print(f"Profile {p_id}: planned dates for {len(groups_with_dates)} group(s).")

        manual_record_to_save = dict(manual_record)
        manual_record_to_save['groups'] = groups_with_dates
        manual_record_to_save['num_groups'] = len(groups_with_dates)
        manual_record_to_save['validation_report'] = validation_report
        total_groups_output.append(manual_record_to_save)

        profile_str = get_dict_str(profile['Basic'], profile['mbti'])
        event_plans = plan_events_for_groups(groups_with_dates)
        profile_tasks.append((p_id, profile, profile_str, event_plans))

    groups_with_dates_path.write_text(json.dumps(total_groups_output, indent=4, ensure_ascii=False), encoding='utf-8')
    write_preference_time_span_report(total_groups_output, preference_time_span_report_path)
    print(f"Groups with planned dates saved to {groups_with_dates_path}")
    print(f"Preference time span report saved to {preference_time_span_report_path}")

    print(f"\n[Phase 1 done] {len(profile_tasks)} profile(s) ready for event generation.")

    # ── Phase 2（并行）: 不同人物并发生成事件 ────────────────────────────────
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 用 list 暂存每个人物的结果，最后按 p_id 排序后合并，保证输出顺序稳定
    profile_results: Dict[int, List[Dict[str, Any]]] = {}
    event_plans_by_pid: Dict[int, List[Dict[str, Any]]] = {
        p_id: event_plans for p_id, _profile, _profile_str, event_plans in profile_tasks
    }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pid = {
            executor.submit(
                _run_profile_events,
                p_id,
                profile,
                profile_str,
                event_plans,
                existing_results,
                args.image_description_mode,
            ): p_id
            for p_id, profile, profile_str, event_plans in profile_tasks
        }
        for future in concurrent.futures.as_completed(future_to_pid):
            p_id = future_to_pid[future]
            try:
                results, prompt_tokens, completion_tokens = future.result()
                profile_results[p_id] = results
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                save_generation_checkpoint(
                    save_profile_path=save_profile_path,
                    save_groups_path=save_groups_path,
                    groups_with_dates_path=groups_with_dates_path,
                    preference_time_span_report_path=preference_time_span_report_path,
                    entity_anchor_coverage_report_path=entity_anchor_coverage_report_path,
                    total_groups_output=total_groups_output,
                    profile_results=profile_results,
                    existing_results=existing_results,
                    preserve_existing=args.resume,
                )
                _safe_print(f"[Checkpoint] Saved after Profile {p_id} -> {save_profile_path}")
            except Exception as exc:
                _safe_print(f"[ERROR] Profile {p_id} raised an exception: {exc}")
                fallback_results = []
                for plan in event_plans_by_pid.get(p_id, []):
                    existing_result = existing_results.get(make_event_key(p_id, plan['group_id']))
                    if existing_result:
                        fallback_results.append(existing_result)
                if fallback_results:
                    profile_results[p_id] = fallback_results
                    _safe_print(f"[Resume] Preserved {len(fallback_results)} existing event(s) for Profile {p_id}.")
                    save_generation_checkpoint(
                        save_profile_path=save_profile_path,
                        save_groups_path=save_groups_path,
                        groups_with_dates_path=groups_with_dates_path,
                        preference_time_span_report_path=preference_time_span_report_path,
                        entity_anchor_coverage_report_path=entity_anchor_coverage_report_path,
                        total_groups_output=total_groups_output,
                        profile_results=profile_results,
                        existing_results=existing_results,
                        preserve_existing=args.resume,
                    )
                    _safe_print(f"[Checkpoint] Saved fallback results after Profile {p_id} -> {save_profile_path}")

    # 按 p_id 顺序合并结果
    save_generation_checkpoint(
        save_profile_path=save_profile_path,
        save_groups_path=save_groups_path,
        groups_with_dates_path=groups_with_dates_path,
        preference_time_span_report_path=preference_time_span_report_path,
        entity_anchor_coverage_report_path=entity_anchor_coverage_report_path,
        total_groups_output=total_groups_output,
        profile_results=profile_results,
        existing_results=existing_results,
        preserve_existing=args.resume,
    )
    if args.resume:
        selected_result_keys = {
            make_event_key(result.get('p_id'), result.get('group_id'))
            for results in profile_results.values()
            for result in results
        }
        preserved_count = sum(1 for key in existing_results if key not in selected_result_keys)
        if preserved_count:
            print(f"[Resume] Preserved {preserved_count} existing event(s) outside current generated results.")

    print(f"\nEvents saved to {save_profile_path}")
    print(f"Groups saved to {save_groups_path}")
    print(f"Groups with planned dates saved to {groups_with_dates_path}")
    print(f"Preference time span report saved to {preference_time_span_report_path}")
    print(f"Entity anchor coverage report saved to {entity_anchor_coverage_report_path}")
    print(f"\nTotal Prompt Tokens: {total_prompt_tokens}, Total Completion Tokens: {total_completion_tokens}")
    print(f"Total Cost: ${(total_prompt_tokens * 1 * 0.000001 + total_completion_tokens * 3 * 0.000001):.3f}")


if __name__ == "__main__":
    main()
