"""为 Relationship / Pets / Items 实体生成图片选择题。

Pipeline（三阶段，同 gen_qa_pref_image.py）:
--------
1. **Stage 1（LLM·描述生成）**：
   Relationship:
   - appearance_image: 外表/发型/穿搭类
   - profession_image: 职业类
   - identify_portrait: 判断哪张照片是该人物
   Pets:
   - pet_identify_portrait: 判断哪张照片是该宠物
   - pet_personality_image: 性格类（生图时喂入定妆照）
   Items:
   - item_identify: 判断哪张图片是用户常用的
2. **Stage 2（图像生成）**：调用 AIFast Gemini 生成选项图片。
3. **Stage 3（memory clue）**：从对话历史中提取 memory clue。

用法:
    python qa/gen_qa_entity_image.py                  # 运行全部阶段
    python qa/gen_qa_entity_image.py --stage 1        # 仅 Stage 1
    python qa/gen_qa_entity_image.py --stage 2        # 仅 Stage 2
    python qa/gen_qa_entity_image.py --stage 3        # 仅 Stage 3
    python qa/gen_qa_entity_image.py --rerun-empty-clues   # 仅对 memory clue 为空的记录重跑 Stage 3
    python qa/gen_qa_entity_image.py --refresh-noun-extraction  # 只刷新 item_identify 题干的核心名词
    python qa/gen_qa_entity_image.py --sample 3       # 只处理前 N 条
    python qa/gen_qa_entity_image.py --sub-type pet_personality_image  # 只处理一种子类型
    python qa/gen_qa_entity_image.py --stage 2 --image-model openrouter_gemini_3_1_flash_image_preview
"""

import base64
import json
import os
import re
import time
import concurrent.futures
import threading
from pathlib import Path

from tqdm import tqdm
from json_repair import repair_json

from scripts.common.images import generate_aifast_image_to_path
from scripts.common.io import load_json_or_jsonl as load_records
from scripts.common.llm import env_value, message_content_to_text, openai_client, usage_value
from scripts.qa.config import profile_path, qa_path

# ========================= 路径 & 模型配置 =========================

PROFILE_PATH = profile_path("profiles_with_anchors_with_images_entity.json")
FORMATTED_DIALOG_PATH = qa_path("qa_formatted_data_000_019.json")
EXISTING_ENTITY_QA_PATH = qa_path("qa_entity_mcq.json")
OUTPUT_PATH = qa_path("qa_entity_image_mcq.json")
IMAGE_DIR = qa_path("entity_images")

CHECKPOINT_EVERY = 5

# Credentials and optional endpoints are supplied only at runtime.
LLM_MODEL = env_value("CUE_MEM_LLM_MODEL", "gpt-5.5")
NOUN_LLM_MODEL = env_value("CUE_MEM_QA_NOUN_LLM_MODEL", "deepseek-v4-pro")

MODEL_AIFAST_GEMINI_IMAGE = "aifast_gemini_3_pro_image_preview"
MODEL_OPENROUTER_GEMINI_IMAGE = "openrouter_gemini_3_1_flash_image_preview"
IMAGE_MODEL = env_value("CUE_MEM_QA_ENTITY_IMAGE_MODEL", MODEL_AIFAST_GEMINI_IMAGE)
AIFAST_IMAGE_MODEL = env_value("CUE_MEM_IMAGE_AIFAST_MODEL", "gemini-3-pro-image-preview")
OPENROUTER_IMAGE_MODEL = env_value(
    "CUE_MEM_IMAGE_OPENROUTER_MODEL", "google/gemini-3.1-flash-image-preview"
)


MAX_WORKERS_LLM = 8
MAX_WORKERS_IMG = 4
LLM_RETRIES = 3
IMG_RETRIES = 4
TEMPERATURE_GEN = 0.9
TEMPERATURE_ANS = 0.2

SUB_TYPES = [
    "appearance_image", "profession_image", "identify_portrait",
    "pet_identify_portrait", "pet_personality_image",
    "item_identify",
]

# ========================= Prompts =========================

# ---------------------------------------------------------------------------
# Sub-type 1: 外表/发型/穿搭类
# ---------------------------------------------------------------------------
prompt_appearance_image_mcq = '''你需要根据一个人物的外貌特征，设计 **1 道图片选择题**。该题将用于评估智能体能否识别出该人物的真实外表。

**关键：你需要同时输出一道中文题干，以及 A/B/C/D 四个选项各自对应的"肖像生成描述"（中文），以便后续调用图像生成模型分别画出四张肖像作为四个选项。**

[输入信息]
人物姓名: {entity_name}
与用户关系: {entity_relation}
人物信息: {entity_info}
外貌描述: {appearance}
参考文本题目: {reference_question}
参考文本选项: {reference_options}

[设计原则]

1. **题干 (Q)**：
   - 简短中文问句，询问哪张肖像最符合该人物的外表。
   - 例如："以下哪张肖像最符合{entity_name}的发型与穿搭？"
   - **不要**在题干中泄露外貌的关键细节。

2. **四个选项的 image_prompt（核心）**：
   - 每个 image_prompt 是一段中文肖像描述（50-80 词），用于生成白底写实半身照。
   - **格式前缀**（每个 prompt 都必须以此开头）：
     "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。"
   - **正确选项**的 image_prompt 必须**忠实**还原外貌描述中的所有关键细节（发型、发色、发长、配饰、肤色、体型、穿搭风格等）。
   - **3 个干扰选项**的 image_prompt 必须：
     a. 与正确选项**同性别、相近年龄段**；
     b. 每个干扰选项需在 **2-3 个关键外貌要素**上与正确选项**明显不同**（如发型+穿搭同时不同、眼镜+发色+服装风格同时不同），差异要足够大，使生成的图片一眼即可区分；
     c. 3 个干扰选项之间的差异方向也应各不相同（例如：一个改发型+穿搭，一个改配饰+体型，一个改发色+服装风格），避免三个干扰看起来雷同。
   - 四个 image_prompt 在长度和结构上保持**平行**。

3. **正确答案位置随机化**：正确选项应随机分布在 A/B/C/D 中。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "Q": "中文题干",
    "options": {{
        "A": "选项 A 的 image_prompt",
        "B": "选项 B 的 image_prompt",
        "C": "选项 C 的 image_prompt",
        "D": "选项 D 的 image_prompt"
    }},
    "A": "正确答案字母",
    "A_false_reason": "",
    "B_false_reason": "",
    "C_false_reason": "",
    "D_false_reason": ""
}}
```
其中 3 个错误选项的 false_reason 填写一句简短中文说明其与真实外表的关键差异；正确选项的 false_reason 填空字符串 ""。

以下是一个good example（注意每个干扰选项同时改了 2-3 个要素，确保生成图片一眼可辨）:
{{
"Q": "以下哪张肖像最符合苏雨桐的真实外表？",
"options": {{
"A": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一位身材纤细的年轻女性，皮肤白皙，眼神温和，留着齐肩黑色微卷发，不戴眼镜，身穿宽松的深灰色卫衣，整体气质慵懒随性。",
"B": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一位身材微胖的年轻女性，皮肤白皙，眼神温和，留着齐肩棕色直发，佩戴方框粗边眼镜，身穿剪裁利落的深色商务西装上衣，整体气质干练沉稳。",
"C": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一位身高约165厘米、身材纤细的年轻女性，皮肤白皙，眼神温和，留着齐肩黑色直发，佩戴圆框细边眼镜，身穿简约素雅的棉麻风格上衣，整体气质温柔沉稳，带有安静知性的书卷气息。",
"D": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一位身材纤细的年轻女性，皮肤偏小麦色，眼神锐利，留着过肩长黑色直发，佩戴圆框细边眼镜，身穿色彩鲜艳的印花衬衫，整体气质明快张扬。"
}},
"A": "C",
"A_false_reason": "发型为微卷发，不戴眼镜，穿搭为灰色卫衣，与真实的齐肩直发+圆框眼镜+棉麻风格均不符。",
"B_false_reason": "发色为棕色，眼镜为方框粗边，穿搭为商务西装，体型偏胖，多处与真实外貌不符。",
"C_false_reason": "",
"D_false_reason": "肤色偏小麦色，发长过肩，穿搭为印花衬衫，气质明快张扬，与真实的白皙肤色+齐肩发+棉麻风格不符。"
}}

'''

# ---------------------------------------------------------------------------
# Sub-type 2: 职业类
# ---------------------------------------------------------------------------
prompt_profession_image_mcq = '''你需要根据一个人物的职业信息，设计 **1 道图片选择题**。该题将用于评估智能体能否从图像中判断该人物的真实职业。

**关键：你需要同时输出一道中文题干，以及 A/B/C/D 四个选项各自对应的"职业场景描述"（中文），以便后续调用图像生成模型分别画出四张照片作为四个选项。**

[输入信息]
人物姓名: {entity_name}
与用户关系: {entity_relation}
人物信息: {entity_info}
外貌描述: {appearance}
参考文本题目: {reference_question}
参考文本选项: {reference_options}

[设计原则]

1. **题干 (Q)**：
   - 简短中文问句，询问哪张图最符合该人物的职业场景。
   - 例如："以下哪张图最符合{entity_name}的职业工作场景？"
   - **不要**在题干中泄露具体职业名称。

2. **四个选项的 image_prompt（核心）**：
   - 每个 image_prompt 是一段中文场景描述（60-100 词），描绘该人物在某个职业场景中工作的样子。
   - **正确选项**：人物外表忠实还原 appearance，在其**真实职业**的典型工作场景中。
   - **3 个干扰选项**：同一人物外表（基于 appearance），但在**相关但可区分的职业**场景中。
     a. 干扰职业应与正确职业有一定关联性或处于**相邻领域**，具备迷惑性（例如正确为"独立书店经营者"，干扰可为"咖啡馆店长"、"花艺工作室主理人"、"文创市集摊主"），但每个干扰的**工作环境、核心工具或服装**必须与正确选项存在明确的视觉差异，确保生成的图片能够区分；
     b. 避免干扰职业与正确职业完全相同领域内仅岗位名称不同（如"书店店员"vs"书店经营者"），也避免跨到完全无关的行业（如"汽修工"、"外科医生"）——保持"相关但不同"的平衡。
   - 每个 image_prompt 都必须包含：
     a. 人物外表特征（基于 appearance，保持一致）
     b. 职业环境/工具/服装
     c. 工作中的典型动作或姿态
   - 四个 image_prompt 在长度和结构上保持**平行**。
   - **禁止**出现文字标签、品牌 logo、选项字母。

3. **正确答案位置随机化**。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "Q": "中文题干",
    "options": {{
        "A": "选项 A 的 image_prompt",
        "B": "选项 B 的 image_prompt",
        "C": "选项 C 的 image_prompt",
        "D": "选项 D 的 image_prompt"
    }},
    "A": "正确答案字母",
    "A_false_reason": "",
    "B_false_reason": "",
    "C_false_reason": "",
    "D_false_reason": ""
}}
```
其中 3 个错误选项的 false_reason 填写一句简短中文说明该职业与真实职业的区别；正确选项的 false_reason 填空字符串 ""。

以下是一个good example（注意干扰职业与正确职业相关但可区分，环境/工具有明确视觉差异）:
{{
"Q": "以下哪张图最符合郑凯宇的职业工作场景？",
"options": {{
"A": "一名身高约176厘米、体型匀称的男性，寸头干净利落，戴黑框方形眼镜，面部棱角分明，穿黑白灰系列商务休闲装，整体形象精致而强势。他坐在现代化设计工作室的会议桌前，桌上摆着品牌视觉方案、色彩板和包装打样，正用手势向团队阐述创意方向，背景有电脑屏幕和设计稿，不出现文字标签、品牌logo或选项字母。",
"B": "一名身高约176厘米、体型匀称的男性，寸头干净利落，戴黑框方形眼镜，面部棱角分明，穿黑白灰系列商务休闲装，整体形象精致而强势。他站在广告拍摄现场的监视器旁，周围有大型摄影灯、反光板、摄像机和布景道具，正认真观察画面构图并用对讲机指挥模特站位，呈现商业广告导演的工作状态，不出现文字标签、品牌logo或选项字母。",
"C": "一名身高约176厘米、体型匀称的男性，寸头干净利落，戴黑框方形眼镜，面部棱角分明，穿黑白灰系列商务休闲装，整体形象精致而强势。他坐在建筑事务所的大型制图桌前，桌上摊开着建筑蓝图和3D建筑模型，面前有比例尺和工程图纸，正拿着铅笔在蓝图上标注结构细节，不出现文字标签、品牌logo或选项字母。",
"D": "一名身高约176厘米、体型匀称的男性，寸头干净利落，戴黑框方形眼镜，面部棱角分明，穿黑白灰系列商务休闲装，整体形象精致而强势。他站在时装秀后台的衣架区，周围挂满了当季成衣和面料样本，他正手持量尺检查一件西装的肩线剪裁，旁边有人台模特和缝纫工具，不出现文字标签、品牌logo或选项字母。"
}},
"A": "A",
"A_false_reason": "",
"B_false_reason": "该场景为广告拍摄导演，虽同属创意领域但核心工具是摄影器材而非设计稿。",
"C_false_reason": "该场景为建筑设计师，工作对象是建筑蓝图与模型，而非品牌视觉方案。",
"D_false_reason": "该场景为时装设计师，工作对象是成衣与面料，与品牌视觉设计不同。"
}}

'''

# ---------------------------------------------------------------------------
# Sub-type 3: 判断人物照片
# ---------------------------------------------------------------------------
prompt_identify_portrait_mcq = '''你需要为一道"判断哪张照片是某人物"的选择题，生成 **3 张干扰肖像**的生成描述。

**背景**：正确选项已有现成肖像图，你只需要设计 3 张与正确肖像**非常相似但存在关键差异**的干扰肖像描述。

[输入信息]
人物姓名: {entity_name}
与用户关系: {entity_relation}
人物信息: {entity_info}
真实外貌描述: {appearance}

[设计原则]

1. **3 个干扰 image_prompt**：
   - 每个 image_prompt 是一段中文肖像描述（50-80 词），用于生成白底写实半身照。
   - **格式前缀**（每个 prompt 都必须以此开头）：
     "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。"
   - 每张干扰肖像必须：
     a. 与真实人物**同性别、相近年龄段**；
     b. 整体气质相似，但在 **1-2 个关键外貌要素**上明显不同（如发型、发色、配饰、穿搭风格、体型等）；
     c. 差异点应足够让仔细观察的评测模型区分，但对粗略一看足以造成混淆。
   - 三个干扰之间也应彼此不同。

2. **正确选项位置随机化**：随机选择 A/B/C/D 中的一个作为正确答案位置，其余三个位置放干扰。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "correct_position": "正确答案字母（A/B/C/D 之一）",
    "distractors": {{
        "1": "第 1 张干扰肖像的 image_prompt",
        "2": "第 2 张干扰肖像的 image_prompt",
        "3": "第 3 张干扰肖像的 image_prompt"
    }},
    "distractor_reasons": {{
        "1": "第 1 张干扰与真实外貌的关键差异（中文）",
        "2": "第 2 张干扰与真实外貌的关键差异（中文）",
        "3": "第 3 张干扰与真实外貌的关键差异（中文）"
    }}
}}
```
'''

# ---------------------------------------------------------------------------
# Stage 3 — memory clue 提取（实体类通用）
# ---------------------------------------------------------------------------
prompt_entity_memory_clue = '''你是一个精确的答题系统。你会收到：
1. 【实体信息】—— 某个与用户相关的人物的基本信息。
2. 【相关对话历史】—— 与该人物相关的对话记录（含文字、图像描述、音频描述），用于定位 memory clue。
3. 一道已经确定了正确答案的选择题信息。

你的任务：
- 从【相关对话历史】中找出所有能支撑正确答案的证据，作为 memory clue 返回。

[输入说明]
【相关对话历史】是与该人物相关的若干 session 拼接：
- 每条用户/助手消息以 [Dxx:NN] 表示其轮次编号；
- 用户在某轮次分享的图像描述以 "图像[Dxx-NNN.png]:" 出现；
- 该 session 的背景音频或用户语音消息描述以 "音频[Dxx-NNN.wav]:" 出现。

[实体信息]
人物姓名: {entity_name}
与用户关系: {entity_relation}
人物信息: {entity_info}
外貌描述: {appearance}

[已确定的正确答案]
题干: {question}
正确答案: {answer_letter}
正确选项描述: {answer_desc}
题目维度: {dimension}

[判定原则]
1. **遍历**【相关对话历史】中的全部信息，把**所有**能够作为依据的证据都列入 `memory clue`。
2. 从三类线索中选用证据：(a) 对话文字、(b) 图像描述、(c) 音频/语音描述。**任何一类**都可独立支撑。
3. **禁止**把与答案无直接关系的轮次/图像/音频塞进 `memory clue`。
4. memory clue 元素格式：
   - 对话证据: "Dxx:NN"
   - 图像证据: "Dxx-NNN.png"
   - 音频证据: "Dxx-NNN.wav"
   - 每一条 clue 都必须**真实出现**在【相关对话历史】中。
   - 如果无相关证据，返回空列表 []。

[相关对话历史]
{dialog_str}

[输出格式]
**严格输出**如下 JSON 结构，不要添加任何额外说明文字：
```json
{{
    "memory clue": ["D01:03", "D01-001.png"]
}}
```
'''

# ---------------------------------------------------------------------------
# Pets Sub-type 1: 判断哪张照片是该宠物
# ---------------------------------------------------------------------------
prompt_pet_identify_portrait_mcq = '''你需要为一道"判断哪张照片是某宠物"的选择题，生成 **3 张干扰宠物肖像**的生成描述。

**背景**：正确选项已有现成宠物肖像图，你只需要设计 3 张与正确肖像**同品种、整体相似但可区分**的干扰宠物描述。干扰项应具有迷惑性（同品种、相近气质），但在特定细节上存在足以区分的差异。

[输入信息]
宠物名: {pet_name}
宠物种类/品种: {pet_breed}
宠物信息: {pet_info}
真实外貌描述: {appearance}

[设计原则]

1. **3 个干扰 image_prompt**：
   - 每个 image_prompt 是一段中文肖像描述（50-80 词），用于生成白底写实宠物照。
   - **格式前缀**（每个 prompt 都必须以此开头）：
     "生成一张逼真的写实宠物肖像。要求：纯白色背景（#FFFFFF），一只宠物，正面或微侧，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。"
   - 每张干扰肖像必须：
     a. 与真实宠物**同品种或极相近品种**，**整体毛色保持一致**；
     b. 在 **2-3 个外貌要素**上与真实宠物不同，差异方向应选择**品种、体型/胖瘦、斑纹颜色、斑纹分布位置、眼睛颜色**等；
     c. 差异要足够让仔细观察的模型区分出来。
   - 3 个干扰之间的差异方向也应各不相同。

2. **正确选项位置随机化**：随机选择 A/B/C/D 中的一个作为正确答案位置。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "correct_position": "正确答案字母（A/B/C/D 之一）",
    "distractors": {{
        "1": "第 1 张干扰肖像的 image_prompt",
        "2": "第 2 张干扰肖像的 image_prompt",
        "3": "第 3 张干扰肖像的 image_prompt"
    }},
    "distractor_reasons": {{
        "1": "第 1 张干扰与真实外貌的关键差异（中文）",
        "2": "第 2 张干扰与真实外貌的关键差异（中文）",
        "3": "第 3 张干扰与真实外貌的关键差异（中文）"
    }}
}}
以下是一个good example:
{{
"correct_position": "B",
"distractors": {{
"1": "生成一张逼真的写实宠物肖像。要求：纯白色背景（#FFFFFF），一只宠物，正面或微侧，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一只黑白双色长毛缅因猫，体型明显更大，毛发蓬松浓密，整体以黑色为主，鼻子周围、下颌、胸口和四只爪子为白色，眼睛为绿色略带蓝调，胡须很长且洁白，尾巴粗长蓬松，末端为白色。",
"2": "生成一张逼真的写实宠物肖像。要求：纯白色背景（#FFFFFF），一只宠物，正面或微侧，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一只黑白双色混血短毛猫，体型偏修长，全身大面积黑色，鼻子周围、下颌、胸口和四只爪子为白色，整体像穿了燕尾服，眼睛呈金绿色，胡须极长且洁白，尾巴细长，末端为白色。",
"3": "生成一张逼真的写实宠物肖像。要求：纯白色背景（#FFFFFF），一只宠物，正面或微侧，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。一只黑白双色卷耳猫，体型偏修长，整体以黑色为主，鼻子周围、下颌、胸口和四只爪子为白色，眼睛为绿色略带蓝调，胡须极长且洁白，耳朵明显向后卷曲，尾巴细长，末端为白色。"
}},
"distractor_reasons": {{
"1": "第1张干扰改成了长毛缅因猫，体型更大且毛发蓬松浓密，与真实的混血短毛猫差异明显。",
"2": "第2张干扰的眼睛呈金绿色，与真实外貌中绿色略带蓝调的眼睛不符。",
"3": "第3张干扰改成了卷耳猫，耳朵明显向后卷曲，与真实的普通混血短毛猫不符。"
}}
}}
```
'''

# ---------------------------------------------------------------------------
# Pets Sub-type 2: 性格类
# ---------------------------------------------------------------------------
prompt_pet_personality_image_mcq = '''你需要根据一只宠物的性格特征，设计 **1 道图片选择题**。该题将用于评估智能体能否从图片中判断该宠物的真实性格。

**关键：你需要同时输出一道中文题干，以及 A/B/C/D 四个选项各自对应的"宠物场景描述"（中文），以便后续调用图像生成模型分别画出四张宠物照片作为四个选项。图像生成时会以该宠物的定妆照作为参考，因此 image_prompt 中不必详细描述宠物外貌，只需描述宠物的姿态、表情和场景即可。**

[输入信息]
宠物名: {pet_name}
宠物种类/品种: {pet_breed}
宠物信息: {pet_info}
外貌描述: {appearance}
参考文本题目: {reference_question}
参考文本选项: {reference_options}

[设计原则]

1. **题干 (Q)**：
   - 简短中文问句，询问哪张图最符合该宠物的性格/习惯。
   - 例如："以下哪张图最符合{pet_name}的性格与日常习惯？"
   - **不要**在题干中泄露具体性格关键词。

2. **四个选项的 image_prompt（核心）**：
   - 每个 image_prompt 是一段中文场景描述（50-80 词），描绘该宠物在某个场景中的姿态与表情。
   - 由于生图时会喂入定妆照作为参考，image_prompt 只需侧重**姿态、表情、动作、所处环境**来体现性格，无需重复外貌细节。
   - **正确选项**：image_prompt 要精准体现该宠物的真实性格特征（如慵懒→蜷缩在垫子上打盹；好奇活跃→跳上桌子探头看东西）。
   - **3 个干扰选项**：体现**不同但合理的性格特征**（如把"慵懒黏人"换成"活泼好动"、"高冷独立"、"胆小警觉"），通过不同的姿态/表情/场景来表达。
   - 四个 image_prompt 在长度和结构上保持**平行**。
   - **禁止**出现文字标签、品牌 logo。

3. **正确答案位置随机化**。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "Q": "中文题干",
    "options": {{
        "A": "选项 A 的 image_prompt",
        "B": "选项 B 的 image_prompt",
        "C": "选项 C 的 image_prompt",
        "D": "选项 D 的 image_prompt"
    }},
    "A": "正确答案字母",
    "A_false_reason": "",
    "B_false_reason": "",
    "C_false_reason": "",
    "D_false_reason": ""
}}
```
其中 3 个错误选项的 false_reason 填写一句简短中文说明该性格描述与真实性格的差异；正确选项的 false_reason 填空字符串 ""。

以下是一个good example:
{{
"Q": "以下哪张图最符合 Scribble 的性格与日常习惯？",
"options": {{
"A": "一只猫正在明亮的画室里快速奔跑，跃起追逐工作台旁垂下的丝带。它的身体在空中舒展开来，眼睛因兴奋而睁大，尾巴高高竖起。周围散落着软垫和画具，整体场景表现出它精力旺盛、喜欢追逐移动物体的活泼性格。",
"B": "一只猫独自坐在安静房间里的高书架上，与附近的人保持明显距离。它姿态端正而克制，表情平静但略显疏离，尾巴整齐地环绕在前爪旁。整体场景表现出它独立高冷，更喜欢远远观察，而不是主动亲近陪伴。",
"C": "一只猫蜷缩在画架旁的柔软坐垫上，安静地睡在主人工作的地方附近。它的身体放松而舒适，一只耳朵却微微转向窗外传来的轻微声响。整体场景温暖而熟悉，表现出它喜欢黏在主人身边休息，同时对周围声音保持敏感。",
"D": "一只猫自信地站在热闹客厅中央，主动靠近几位陌生访客，尾巴高高竖起。它神情好奇而毫不畏惧，伸出前爪仿佛在邀请他人关注。整体场景表现出它外向大胆、喜欢热闹聚会，并愿意主动探索陌生人与新环境。"
}},
"A": "C",
"A_false_reason": "该场景表现出活泼好动、热衷追逐的性格，与 Scribble 慵懒爱睡的习惯不符。",
"B_false_reason": "该场景表现出高冷独立、刻意远离人的性格，与 Scribble 黏人的特点不符。",
"C_false_reason": "",
"D_false_reason": "该场景表现出大胆外向、喜欢热闹的性格，与 Scribble 对外界声音敏感的特点不符。"
}}

'''

# ---------------------------------------------------------------------------
# Items: 判断哪张图片是用户常用的
# ---------------------------------------------------------------------------
prompt_item_identify_mcq = '''你需要为一道"判断哪张图片是用户常用的某物品"的选择题，生成 **3 张干扰物品**的生成描述。

**背景**：正确选项已有现成物品照片，你只需要设计 3 张与正确物品**相似但存在关键差异**的干扰物品描述。

[输入信息]
物品描述: {item_description}
所属类别: {source_subcategory}

[设计原则]

1. **3 个干扰 image_prompt**：
   - 每个 image_prompt 是一段中文物品描述（40-70 词），用于生成白底写实物品照。
   - **格式前缀**（每个 prompt 都必须以此开头）：
     "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无水印。"
   - 每张干扰物品必须：
     a. 与正确物品属于**同类别**（同样是杯子、同样是书、同样是背包等）；
     b. 在 **2-3 个关键视觉要素**上明显不同（如颜色、材质、形状、尺寸等）；
     c. 差异点要足够区分。
   - 三个干扰之间也应彼此不同。

2. **正确选项位置随机化**。

[输出格式]
**严格输出**如下 JSON 对象，不要添加任何额外说明文字：
```json
{{
    "correct_position": "正确答案字母（A/B/C/D 之一）",
    "distractors": {{
        "1": "第 1 张干扰物品的 image_prompt",
        "2": "第 2 张干扰物品的 image_prompt",
        "3": "第 3 张干扰物品的 image_prompt"
    }},
    "distractor_reasons": {{
        "1": "第 1 张干扰与真实物品的关键差异（中文）",
        "2": "第 2 张干扰与真实物品的关键差异（中文）",
        "3": "第 3 张干扰与真实物品的关键差异（中文）"
    }}
}}

以下是一个good example:
{{
    "correct_position": "C",
    "distractors": {{
        "1": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。画面主体为一只磨砂质感的细长玻璃杯，杯身修长直筒，半透明表面带柔和雾感，杯口平整，整体简约干净，白底产品写真风格。",
        "2": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。画面主体为一只透明玻璃杯，但杯身较矮且更宽，呈厚底圆筒造型，整体结实敦厚，清晰反光明显，属于常见矮款水杯产品照。",
        "3": "生成一张逼真的写真肖像。要求：纯白色背景（#FFFFFF），单人，亚洲人面孔，正面半身，主体居中，构图简洁，光线柔和均匀，无明显阴影，无环境元素，无道具，无文字，无水印。画面主体为一只透明细长玻璃杯，但杯口略微外扩，底部微微收窄，形成轻度郁金香形轮廓，线条更有弧度，通透清亮，极简静物写实风格。"
    }},
    "distractor_reasons": {{
        "1": "与真实物品相比，材质观感不同：真实物品是透明玻璃杯，这一项是磨砂半透明玻璃杯。",
        "2": "与真实物品相比，形状和比例不同：真实物品细长，这一项是矮而宽的厚底玻璃杯。",
        "3": "与真实物品相比，杯身轮廓不同：真实物品通常为直筒细长，这一项为杯口外扩、底部微收的弧形杯。"
    }}
}}
```
'''


# ========================= Utility functions =========================

def load_json(path: str):
    return load_records(path)


def load_json_or_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
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
                model=LLM_MODEL,
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


# ========================= Image generation =========================

def _save_from_data_url(data_url: str, save_path: Path) -> bool:
    try:
        if "," not in data_url or not data_url.strip().lower().startswith("data:"):
            return False
        _, b64_part = data_url.split(",", 1)
        raw = base64.b64decode(b64_part)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(raw)
        return True
    except Exception as exc:
        print(f"    [DATA_URL SAVE ERR] {exc}")
        return False


def _download_image(url: str, save_path: Path) -> bool:
    try:
        import requests
        save_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    [DOWNLOAD ERR] {exc}")
        return False


def _save_image_from_url_or_dataurl(src: str, save_path: Path) -> bool:
    s = (src or "").strip()
    if not s:
        return False
    if s.lower().startswith("data:"):
        return _save_from_data_url(s, save_path)
    if s.startswith("http://") or s.startswith("https://"):
        return _download_image(s, save_path)
    print(f"    [SAVE ERR] unsupported image source: {s[:80]!r}...")
    return False


def generate_openrouter_option_image(
    image_prompt: str,
    save_path: Path,
    ref_image_paths: list[Path] | None = None,
) -> bool:
    key = env_value("CUE_MEM_IMAGE_OPENROUTER_API_KEY")
    if not key:
        print("    [OPENROUTER ERR] CUE_MEM_IMAGE_OPENROUTER_API_KEY 未设置")
        return False

    valid_refs = []
    for ref_path in ref_image_paths or []:
        data_url = _encode_image_to_data_url(str(ref_path))
        if data_url:
            valid_refs.append(data_url)

    if valid_refs:
        content = [
            {"type": "image_url", "image_url": {"url": data_url}}
            for data_url in valid_refs
        ] + [{"type": "text", "text": image_prompt}]
        print(f"    [OpenRouter] attached {len(valid_refs)} reference image(s) via base64")
    else:
        content = image_prompt

    for attempt in range(1, IMG_RETRIES + 1):
        try:
            client = openai_client(
                api_key=key,
                base_url_env="CUE_MEM_IMAGE_OPENROUTER_BASE_URL",
            )
            completion = client.chat.completions.create(
                model=OPENROUTER_IMAGE_MODEL,
                messages=[{"role": "user", "content": content}],
                extra_body={"modalities": ["image", "text"]},
            )
            if not completion.choices:
                raise RuntimeError("empty choices")
            message = completion.choices[0].message
            urls = _extract_urls_from_message(message)
            if not urls:
                text = getattr(message, "content", "") or ""
                raise RuntimeError(f"no images in response; text response: {text[:300]}")
            return _save_image_from_url_or_dataurl(urls[0], save_path)
        except Exception as exc:
            print(f"    [OPENROUTER ERR] attempt {attempt}/{IMG_RETRIES}: {exc}")
            if attempt < IMG_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))

    return False


def generate_option_image(image_prompt: str, save_path: Path) -> bool:
    if IMAGE_MODEL == MODEL_OPENROUTER_GEMINI_IMAGE:
        return generate_openrouter_option_image(image_prompt, save_path)
    return generate_aifast_image_to_path(
        image_prompt,
        save_path,
        model=AIFAST_IMAGE_MODEL,
        retries=IMG_RETRIES,
    )


def _encode_image_to_data_url(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        ext = Path(image_path).suffix.lower()
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext.lstrip("."), "image/png")
        return f"data:{mime};base64,{b64}"
    except Exception as exc:
        print(f"    [ENCODE ERR] {exc}")
        return None


def generate_option_image_with_ref(image_prompt: str, ref_image_path: str, save_path: Path) -> bool:
    """Generate an image with a reference image (pet portrait) fed alongside the prompt."""
    ref_path = Path(ref_image_path)
    if not ref_path.is_file():
        print(f"    [ERR] missing reference image: {ref_image_path}")
        return generate_option_image(image_prompt, save_path)

    full_prompt = (
        f"Based on the reference photo of the pet below, generate a new image of this SAME pet "
        f"(keep its exact appearance, breed, coat pattern, and colors) in the following scene: "
        f"{image_prompt} "
        f"Output one high-quality, sharp, clear image."
    )
    if IMAGE_MODEL == MODEL_OPENROUTER_GEMINI_IMAGE:
        return generate_openrouter_option_image(full_prompt, save_path, ref_image_paths=[ref_path])
    return generate_aifast_image_to_path(
        full_prompt,
        save_path,
        model=AIFAST_IMAGE_MODEL,
        ref_image_paths=[ref_path],
        retries=IMG_RETRIES,
    )


def _extract_urls_from_message(message) -> list:
    urls = []
    if message is None:
        return urls
    if hasattr(message, "model_dump"):
        try:
            data = message.model_dump()
        except Exception:
            data = {}
    elif isinstance(message, dict):
        data = message
    else:
        data = {}

    images = data.get("images")
    if images is None:
        images = getattr(message, "images", None)
    if not images:
        return urls

    for img in images:
        if isinstance(img, dict):
            iu = img.get("image_url")
            if isinstance(iu, dict):
                u = iu.get("url")
            elif isinstance(iu, str):
                u = iu
            else:
                u = None
            if not u:
                u = img.get("url")
        else:
            iu = getattr(img, "image_url", None)
            u = getattr(iu, "url", None) if iu is not None else getattr(img, "url", None)
        if isinstance(u, str) and u.strip():
            urls.append(u.strip())
    return urls


# ========================= Entity event helpers =========================

def collect_entity_events(formatted_profile: dict, entity_code: str) -> list:
    """Collect events matching entity_code (e.g. 'Relationship-0')."""
    matched = []
    for event in formatted_profile.get("events", []) or []:
        codes = []
        codes += event.get("explicit_preferences_reflected", []) or []
        codes += event.get("implicit_preferences_reflected", []) or []
        if entity_code in codes:
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
                    image_text = img_desc if img_desc and img_desc.lower() != "none" else v
                    section.append(f"    └─ 图像[{k}]: {image_text}")
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


# ========================= Validation =========================

def validate_image_mcq(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    if not isinstance(item.get("Q"), str) or not item["Q"].strip():
        return False
    options = item.get("options")
    if not isinstance(options, dict):
        return False
    if not all(k in options and str(options[k]).strip() for k in ["A", "B", "C", "D"]):
        return False
    answer = (item.get("A") or "").strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        return False
    return True


def validate_identify_mcq(item: dict) -> bool:
    if not isinstance(item, dict):
        return False
    pos = (item.get("correct_position") or "").strip().upper()
    if pos not in {"A", "B", "C", "D"}:
        return False
    distractors = item.get("distractors")
    if not isinstance(distractors, dict):
        return False
    if not all(str(k) in distractors and str(distractors[str(k)]).strip() for k in [1, 2, 3]):
        if not all(str(k) in distractors and str(distractors[str(k)]).strip() for k in ["1", "2", "3"]):
            return False
    return True


# ========================= Build existing QA lookup =========================

def build_existing_qa_lookup(entity_qas: list) -> dict:
    """Build lookup: (p_id, entity_type, rel_idx, dimension) -> qa record."""
    lookup = {}
    for qa in entity_qas:
        qa_id = qa.get("qa_id", "")
        parts = qa_id.split("-")
        if len(parts) >= 4:
            p_id = int(parts[0])
            entity_type = parts[1]
            rel_idx = int(parts[2])
            dim = qa.get("dimension", "")
            lookup[(p_id, entity_type, rel_idx, dim)] = qa
    return lookup


# ========================= Stage 1: LLM gen descriptions =========================

def run_stage1_appearance(
    p_id: int, rel_idx: int, rel: dict,
    existing_text_qa: dict | None,
) -> tuple:
    qa_id = f"{p_id}-Relationship-{rel_idx}-img_appearance"
    entity_name = rel.get("name", "")
    entity_relation = rel.get("relation", "")
    entity_info = rel.get("info", "")
    appearance = rel.get("appearance", "")

    ref_q = existing_text_qa.get("Q", "") if existing_text_qa else ""
    ref_opts = json.dumps(existing_text_qa.get("options", {}), ensure_ascii=False) if existing_text_qa else "{}"

    prompt = prompt_appearance_image_mcq.format(
        entity_name=entity_name,
        entity_relation=entity_relation,
        entity_info=entity_info,
        appearance=appearance,
        reference_question=ref_q,
        reference_options=ref_opts,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (appearance) failed: {e}")
        return None, 0, 0

    if not validate_image_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid MCQ: {mcq}")
        return None, tin, tout

    answer = mcq["A"].strip().upper()
    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "entity_type": "Relationship",
        "entity_name": entity_name,
        "entity_relation": entity_relation,
        "rel_idx": rel_idx,
        "dimension": "外表_image",
        "sub_type": "appearance_image",
        "ref_image_path": rel.get("img_path", ""),
        "Q": mcq["Q"].strip(),
        "question_image_descriptions": {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]},
        "option_images": {"A": "", "B": "", "C": "", "D": ""},
        "A": answer,
        "A_false_reason": str(mcq.get("A_false_reason", "")).strip(),
        "B_false_reason": str(mcq.get("B_false_reason", "")).strip(),
        "C_false_reason": str(mcq.get("C_false_reason", "")).strip(),
        "D_false_reason": str(mcq.get("D_false_reason", "")).strip(),
        "memory clue": [],
        "matched_session_ids": [],
        "type": "entity_image_mcq",
    }
    return record, tin, tout


def run_stage1_profession(
    p_id: int, rel_idx: int, rel: dict,
    existing_text_qa: dict | None,
) -> tuple:
    qa_id = f"{p_id}-Relationship-{rel_idx}-img_profession"
    entity_name = rel.get("name", "")
    entity_relation = rel.get("relation", "")
    entity_info = rel.get("info", "")
    appearance = rel.get("appearance", "")

    ref_q = existing_text_qa.get("Q", "") if existing_text_qa else ""
    ref_opts = json.dumps(existing_text_qa.get("options", {}), ensure_ascii=False) if existing_text_qa else "{}"

    prompt = prompt_profession_image_mcq.format(
        entity_name=entity_name,
        entity_relation=entity_relation,
        entity_info=entity_info,
        appearance=appearance,
        reference_question=ref_q,
        reference_options=ref_opts,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (profession) failed: {e}")
        return None, 0, 0

    if not validate_image_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid MCQ: {mcq}")
        return None, tin, tout

    answer = mcq["A"].strip().upper()
    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "entity_type": "Relationship",
        "entity_name": entity_name,
        "entity_relation": entity_relation,
        "rel_idx": rel_idx,
        "dimension": "职业_image",
        "sub_type": "profession_image",
        "ref_image_path": rel.get("img_path", ""),
        "Q": mcq["Q"].strip(),
        "question_image_descriptions": {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]},
        "option_images": {"A": "", "B": "", "C": "", "D": ""},
        "A": answer,
        "A_false_reason": str(mcq.get("A_false_reason", "")).strip(),
        "B_false_reason": str(mcq.get("B_false_reason", "")).strip(),
        "C_false_reason": str(mcq.get("C_false_reason", "")).strip(),
        "D_false_reason": str(mcq.get("D_false_reason", "")).strip(),
        "memory clue": [],
        "matched_session_ids": [],
        "type": "entity_image_mcq",
    }
    return record, tin, tout


def run_stage1_identify(
    p_id: int, rel_idx: int, rel: dict,
) -> tuple:
    qa_id = f"{p_id}-Relationship-{rel_idx}-img_identify"
    entity_name = rel.get("name", "")
    entity_relation = rel.get("relation", "")
    entity_info = rel.get("info", "")
    appearance = rel.get("appearance", "")
    img_path = rel.get("img_path", "")

    if not img_path or not os.path.exists(img_path):
        print(f"[{qa_id}] Stage 1 (identify) skipped: no img_path or file missing: {img_path}")
        return None, 0, 0

    prompt = prompt_identify_portrait_mcq.format(
        entity_name=entity_name,
        entity_relation=entity_relation,
        entity_info=entity_info,
        appearance=appearance,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (identify) failed: {e}")
        return None, 0, 0

    if not validate_identify_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid identify MCQ: {mcq}")
        return None, tin, tout

    correct_pos = mcq["correct_position"].strip().upper()
    distractors = mcq["distractors"]
    distractor_reasons = mcq.get("distractor_reasons", {})

    all_letters = ["A", "B", "C", "D"]
    wrong_letters = [l for l in all_letters if l != correct_pos]

    question_image_descriptions = {}
    option_images = {}
    false_reasons = {}

    question_image_descriptions[correct_pos] = f"(existing portrait: {entity_name})"
    option_images[correct_pos] = img_path
    false_reasons[correct_pos] = ""

    for i, letter in enumerate(wrong_letters):
        d_key = str(i + 1)
        question_image_descriptions[letter] = str(distractors.get(d_key, "")).strip()
        option_images[letter] = ""
        false_reasons[letter] = str(distractor_reasons.get(d_key, "")).strip()

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "entity_type": "Relationship",
        "entity_name": entity_name,
        "entity_relation": entity_relation,
        "rel_idx": rel_idx,
        "dimension": "identify_portrait",
        "sub_type": "identify_portrait",
        "Q": f"以下哪张照片是{entity_name}？",
        "question_image_descriptions": question_image_descriptions,
        "option_images": option_images,
        "A": correct_pos,
        "A_false_reason": false_reasons.get("A", ""),
        "B_false_reason": false_reasons.get("B", ""),
        "C_false_reason": false_reasons.get("C", ""),
        "D_false_reason": false_reasons.get("D", ""),
        "memory clue": [],
        "matched_session_ids": [],
        "type": "entity_image_mcq",
    }
    return record, tin, tout


# ========================= Stage 1: Pets =========================

def _parse_pet_breed(info: str) -> str:
    """Extract breed from pet info string like '猫;美国短毛猫;4岁;...'."""
    parts = [p.strip() for p in info.split(";") if p.strip()]
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else ""


def run_stage1_pet_identify(
    p_id: int, pet_idx: int, pet: dict,
) -> tuple:
    qa_id = f"{p_id}-Pets-{pet_idx}-img_identify"
    pet_name = pet.get("name", "")
    pet_info = pet.get("info", "")
    appearance = pet.get("appearance", "")
    img_path = pet.get("img_path", "")
    pet_breed = _parse_pet_breed(pet_info)

    if not img_path or not os.path.exists(img_path):
        print(f"[{qa_id}] Stage 1 (pet identify) skipped: no img_path or file missing")
        return None, 0, 0

    prompt = prompt_pet_identify_portrait_mcq.format(
        pet_name=pet_name,
        pet_breed=pet_breed,
        pet_info=pet_info,
        appearance=appearance,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (pet identify) failed: {e}")
        return None, 0, 0

    if not validate_identify_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid identify MCQ: {mcq}")
        return None, tin, tout

    correct_pos = mcq["correct_position"].strip().upper()
    distractors = mcq["distractors"]
    distractor_reasons = mcq.get("distractor_reasons", {})

    all_letters = ["A", "B", "C", "D"]
    wrong_letters = [l for l in all_letters if l != correct_pos]

    question_image_descriptions = {}
    option_images = {}
    false_reasons = {}

    question_image_descriptions[correct_pos] = f"(existing portrait: {pet_name})"
    option_images[correct_pos] = img_path
    false_reasons[correct_pos] = ""

    for i, letter in enumerate(wrong_letters):
        d_key = str(i + 1)
        question_image_descriptions[letter] = str(distractors.get(d_key, "")).strip()
        option_images[letter] = ""
        false_reasons[letter] = str(distractor_reasons.get(d_key, "")).strip()

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "entity_type": "Pets",
        "entity_name": pet_name,
        "entity_relation": "",
        "rel_idx": pet_idx,
        "dimension": "identify_portrait",
        "sub_type": "pet_identify_portrait",
        "Q": f"以下哪张照片是{pet_name}？",
        "question_image_descriptions": question_image_descriptions,
        "option_images": option_images,
        "A": correct_pos,
        "A_false_reason": false_reasons.get("A", ""),
        "B_false_reason": false_reasons.get("B", ""),
        "C_false_reason": false_reasons.get("C", ""),
        "D_false_reason": false_reasons.get("D", ""),
        "memory clue": [],
        "matched_session_ids": [],
        "type": "entity_image_mcq",
    }
    return record, tin, tout


def run_stage1_pet_personality(
    p_id: int, pet_idx: int, pet: dict,
    existing_text_qa: dict | None,
) -> tuple:
    qa_id = f"{p_id}-Pets-{pet_idx}-img_personality"
    pet_name = pet.get("name", "")
    pet_info = pet.get("info", "")
    appearance = pet.get("appearance", "")
    pet_breed = _parse_pet_breed(pet_info)

    ref_q = existing_text_qa.get("Q", "") if existing_text_qa else ""
    ref_opts = json.dumps(existing_text_qa.get("options", {}), ensure_ascii=False) if existing_text_qa else "{}"

    prompt = prompt_pet_personality_image_mcq.format(
        pet_name=pet_name,
        pet_breed=pet_breed,
        pet_info=pet_info,
        appearance=appearance,
        reference_question=ref_q,
        reference_options=ref_opts,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (pet personality) failed: {e}")
        return None, 0, 0

    if not validate_image_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid MCQ: {mcq}")
        return None, tin, tout

    answer = mcq["A"].strip().upper()
    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "entity_type": "Pets",
        "entity_name": pet_name,
        "entity_relation": "",
        "rel_idx": pet_idx,
        "dimension": "性格_image",
        "sub_type": "pet_personality_image",
        "ref_image_path": pet.get("img_path", ""),
        "Q": mcq["Q"].strip(),
        "question_image_descriptions": {k: str(mcq["options"][k]).strip() for k in ["A", "B", "C", "D"]},
        "option_images": {"A": "", "B": "", "C": "", "D": ""},
        "A": answer,
        "A_false_reason": str(mcq.get("A_false_reason", "")).strip(),
        "B_false_reason": str(mcq.get("B_false_reason", "")).strip(),
        "C_false_reason": str(mcq.get("C_false_reason", "")).strip(),
        "D_false_reason": str(mcq.get("D_false_reason", "")).strip(),
        "memory clue": [],
        "matched_session_ids": [],
        "type": "entity_image_mcq",
    }
    return record, tin, tout


# ========================= Stage 1: Items =========================

_NOUN_EXTRACT_PROMPT = '''你是中文物品名词归一化器。请从“物品描述”中提取最核心、最通用的物品名词，用于选择题题干“以下哪张图片是用户常用的{{noun}}？”。

硬性规则：
1. 去掉颜色、材质、尺寸、形状、风格、位置、状态、品牌、装饰、磨损痕迹、用途细节等修饰词。
2. 只保留物体类别名词，不要原样复制整段输入。
3. 输出通常应明显短于输入；除非输入本身已经是最简物体名词。
4. 如果是复合描述，选择最适合做题干的主物体类别。
5. 如果你的输出和输入完全一样，说明你没有完成任务，必须继续压缩。
6. 严格只输出 JSON，不要解释，不要 markdown。

示例：
透明细长玻璃杯 -> {{"noun":"玻璃杯"}}
陶瓷小圆花盆 -> {{"noun":"花盆"}}
绿色多肉植物 -> {{"noun":"多肉植物"}}
大开本设计精装画册 -> {{"noun":"画册"}}
黑色有线耳机 -> {{"noun":"耳机"}}
牛皮纸旅行目的地手账本 -> {{"noun":"手账本"}}
卷起的紫色TPE瑜伽垫 -> {{"noun":"瑜伽垫"}}
叠放的薄册剧本手稿 -> {{"noun":"剧本手稿"}}
粉蓝色多肉植物白色陶瓷小圆盆 -> {{"noun":"花盆"}}
边角翻折的《Wallpaper*》设计杂志 -> {{"noun":"设计杂志"}}
手写标注路线的折叠纸质地图 -> {{"noun":"纸质地图"}}
红色袋装重庆麻辣火锅底料 -> {{"noun":"火锅底料"}}
深红色玻璃瓶装辣椒酱 -> {{"noun":"辣椒酱"}}
贴有小批次IPA标签的棕色精酿啤酒瓶 -> {{"noun":"啤酒瓶"}}
不锈钢双节波士顿摇酒壶 -> {{"noun":"摇酒壶"}}
银色长柄吧台调酒勺 -> {{"noun":"调酒勺"}}
灰色布面恒温睡眠眼罩 -> {{"noun":"眼罩"}}

物品描述：{desc}

输出要求：
- 只能输出一个 JSON object。
- 必须包含且只包含 noun 字段。
- noun 必须是最核心、最通用的物品类别名词。
- noun 不能等于原始物品描述。
- 不要输出解释、注释、markdown 或任何额外文字。

输出 JSON 格式：
{{"noun":"<最核心、最通用的物品类别名词>"}}'''

NOUN_LLM_RETRIES = int(os.environ.get("NOUN_LLM_RETRIES", str(LLM_RETRIES)))

_noun_cache: dict[str, str] = {}
_noun_api_failures: set[str] = set()
_noun_same_as_input: set[str] = set()
_noun_fallback_used: dict[str, str] = {}
_noun_status_cache: dict[str, dict] = {}


_NOUN_SUFFIX_PATTERNS = [
    "白噪音机", "火锅底料", "辣椒酱", "啤酒瓶", "啤酒杯", "摇酒壶", "调酒勺",
    "浇水壶", "睡眠眼罩", "眼罩", "多肉拼盘圆盆", "圆盆", "花盆", "盆栽",
    "精装画册", "设计杂志", "杂志", "纸质地图", "地图", "路线图", "明信片",
    "冰箱贴", "照片", "合影照片", "相框", "展示板", "奖状", "证书", "胸牌",
    "流程卡", "面料样卡", "打样纸袋", "收纳盒", "收纳袋", "收纳托盘",
    "水杯", "玻璃杯", "搪瓷杯", "茶叶盒", "茶叶罐", "茶壶", "咖啡杯",
    "书籍", "旧书", "画册", "笔记本", "记事本", "手账本", "便利贴",
    "键盘", "显示器", "电脑屏幕", "鼠标", "文件夹", "文件收纳架",
    "台灯", "耳塞盒", "窗帘", "胶布", "桌面电话机", "车钥匙", "加油小票",
    "车锁", "车灯", "自行车铃铛", "骑行路线牌", "登山鞋", "登山杖",
    "瑜伽垫", "瑜伽砖", "壶铃", "护手霜", "防晒霜", "遮阳帽",
    "泳镜", "泳帽", "鼓棒", "鼓棒收纳盒", "吸音棉墙板", "低频陷阱",
    "消防头盔", "消防员徽章",
]

_NOUN_MODIFIER_RE = re.compile(
    r"(红色|绿色|蓝色|白色|黑色|灰色|银色|金色|棕色|深红色|深棕色|"
    r"透明|玻璃|陶瓷|金属|不锈钢|布面|木质|原木色|皮质|塑料|硅胶|"
    r"贴有|印有|写有|带有|磨损|折角|袋装|瓶装|长柄|细嘴|圆柱形|双节|"
    r"小批次|IPA|重庆|麻辣|恒温|睡眠|吧台|精酿|多肉|拼盘)"
)


def _normalize_item_noun_local(description: str) -> str:
    """Local fallback for item nouns when the LLM returns the full anchor unchanged."""
    desc = (description or "").strip()
    if not desc:
        return desc

    # Prefer the longest matching known object-category suffix.
    for pattern in sorted(_NOUN_SUFFIX_PATTERNS, key=len, reverse=True):
        if pattern and pattern in desc:
            if pattern == "睡眠眼罩":
                return "眼罩"
            if pattern == "多肉拼盘圆盆":
                return "花盆"
            if pattern in {"合影照片"}:
                return "照片"
            return pattern

    # Generic suffix extraction for common Chinese object endings.
    suffix_match = re.search(
        r"([\u4e00-\u9fff]{1,8}(?:瓶|杯|壶|勺|盆|盒|袋|罐|垫|砖|帽|镜|杖|鞋|灯|贴|票|图|板|架|夹|本|书|册|帘|机|器|钟|锁|钥匙|奖状|证书|胸牌|流程卡|样卡|纸袋))$",
        desc,
    )
    if suffix_match:
        candidate = suffix_match.group(1)
        candidate = _NOUN_MODIFIER_RE.sub("", candidate).strip()
        if candidate:
            return candidate

    return desc


def _is_bad_extracted_noun(noun: str, description: str) -> bool:
    noun = (noun or "").strip()
    desc = (description or "").strip()
    if not noun:
        return True
    if any(ch in noun for ch in "{}[]\"'`<>:：") or "noun" in noun.lower():
        return True
    if noun == desc:
        return True
    if len(desc) >= 6 and len(noun) / max(len(desc), 1) > 0.75:
        return True
    return bool(len(noun) >= 6 and _NOUN_MODIFIER_RE.search(noun))


def _clean_extracted_noun(text: str, fallback: str) -> str:
    raw = (text or "").strip()
    noun = ""
    if raw:
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        json_start = cleaned.find("{")
        json_end = cleaned.rfind("}")
        json_text = cleaned[json_start:json_end + 1] if json_start >= 0 and json_end > json_start else cleaned
        try:
            obj = json.loads(json_text)
        except Exception:
            try:
                obj = json.loads(repair_json(json_text))
            except Exception:
                obj = None
        if isinstance(obj, dict):
            noun = str(obj.get("noun") or "").strip()
        elif isinstance(obj, str):
            noun = obj.strip()

    if not noun:
        # Malformed JSON fragments should be retried/fallback, not accepted as nouns.
        if re.search(r"[{}\[\]\"`]|noun\s*[:：]", raw, flags=re.IGNORECASE):
            noun = ""
        else:
            noun = raw.splitlines()[0].strip() if raw else ""
    noun = re.sub(r"^(输出|核心名词短语|核心名词|名词)[:：]\s*", "", noun).strip()
    noun = noun.strip("\"'`“”‘’。，、 \t\n")
    return noun


def _extract_item_noun_with_status(description: str) -> tuple[str, dict]:
    description = (description or "").strip()
    if description in _noun_cache:
        return _noun_cache[description], dict(_noun_status_cache.get(description, {}))

    noun = ""
    attempts: list[dict] = []
    api_failed = False
    last_error = ""
    client = openai_client(
        api_key_env="CUE_MEM_LLM_API_KEY",
        base_url_env="CUE_MEM_LLM_BASE_URL",
    )
    for attempt in range(1, NOUN_LLM_RETRIES + 1):
        try:
            res = client.chat.completions.create(
                model=NOUN_LLM_MODEL,
                messages=[{"role": "user", "content": _NOUN_EXTRACT_PROMPT.format(desc=description)}],
                temperature=0.0,
                top_p=0.9,
                max_tokens=80,
                stream=True,
                stream_options={"include_usage": True},
            )
            raw_content = ""
            for chunk in res:
                if chunk.choices:
                    raw_content += message_content_to_text(chunk.choices[0].delta.content)
            noun = _clean_extracted_noun(raw_content, description)
            bad = _is_bad_extracted_noun(noun, description)
            attempts.append({
                "attempt": attempt,
                "raw": raw_content,
                "noun": noun,
                "is_same_as_input": noun == description,
                "is_bad": bad,
                "error": "",
            })
            if not bad:
                break
            if attempt < NOUN_LLM_RETRIES:
                time.sleep(min(2 ** (attempt - 1), 8))
        except Exception as e:
            last_error = str(e)
            attempts.append({
                "attempt": attempt,
                "raw": "",
                "noun": "",
                "is_same_as_input": False,
                "is_bad": True,
                "error": last_error,
            })
            if attempt >= NOUN_LLM_RETRIES:
                print(f"  [WARN] noun extraction failed for '{description}': {e}")
                _noun_api_failures.add(description)
                api_failed = True
            else:
                time.sleep(min(2 ** (attempt - 1), 8))

    method = "llm_success"
    fallback_noun = ""
    if _is_bad_extracted_noun(noun, description):
        fallback_noun = _normalize_item_noun_local(description)
        if fallback_noun != noun:
            _noun_fallback_used[description] = fallback_noun
        noun = fallback_noun
        method = "local_fallback"

    if noun == description:
        _noun_same_as_input.add(description)
        method = "same_as_input" if not api_failed else "api_failed_same_as_input"
    elif api_failed:
        method = "api_failed_local_fallback" if method == "local_fallback" else "api_failed"

    status = {
        "description": description,
        "noun": noun,
        "method": method,
        "model": NOUN_LLM_MODEL,
        "base_url_configured_at_runtime": bool(env_value("CUE_MEM_LLM_BASE_URL")),
        "retries": NOUN_LLM_RETRIES,
        "attempt_count": len(attempts),
        "api_failed": api_failed,
        "last_error": last_error,
        "fallback_noun": fallback_noun,
        "attempts": attempts,
    }
    _noun_cache[description] = noun
    _noun_status_cache[description] = status
    return noun, dict(status)


def _extract_item_noun(description: str) -> str:
    noun, _ = _extract_item_noun_with_status(description)
    return noun


def _noun_from_item_identify_question(question: str) -> str:
    question = (question or "").strip()
    m = re.match(r"^以下哪张图片是用户常用的(.+?)？?$", question)
    if m:
        return m.group(1).strip()
    return ""


def _record_matches_refresh_noun_target(rec: dict, target: str) -> bool:
    if target == "all":
        return True
    description = (rec.get("entity_name") or rec.get("item_description") or "").strip()
    current_noun = _noun_from_item_identify_question(str(rec.get("Q") or ""))
    noun_meta = rec.get("noun_extraction") if isinstance(rec.get("noun_extraction"), dict) else {}
    noun_method = str(noun_meta.get("method") or "")
    has_noun_meta = bool(noun_method)
    if not description or not current_noun:
        return target in {"bad", "same_or_fallback"}

    is_same_as_input = current_noun == description
    fallback_methods = {"local_fallback", "api_failed_local_fallback"}
    same_methods = {"same_as_input", "api_failed_same_as_input"}
    is_local_fallback = has_noun_meta and noun_method in fallback_methods
    is_recorded_same = has_noun_meta and noun_method in same_methods

    if target == "same_as_input":
        return is_recorded_same or is_same_as_input
    if target == "fallback":
        return is_local_fallback
    if target in {"bad", "same_or_fallback"}:
        return is_recorded_same or is_same_as_input or is_local_fallback
    return True


def run_stage1_item_identify(
    p_id: int, item_idx: int, item: dict,
) -> tuple:
    qa_id = f"{p_id}-Items-{item_idx}-img_identify"
    description = item.get("description", "")
    source_subcategory = item.get("source_subcategory", "")
    img_path = item.get("img_path", "")

    if not img_path or not os.path.exists(img_path):
        print(f"[{qa_id}] Stage 1 (item identify) skipped: no img_path or file missing")
        return None, 0, 0

    prompt = prompt_item_identify_mcq.format(
        item_description=description,
        source_subcategory=source_subcategory,
    )

    try:
        mcq, tin, tout = call_llm(prompt, temperature=TEMPERATURE_GEN)
    except Exception as e:
        print(f"[{qa_id}] Stage 1 (item identify) failed: {e}")
        return None, 0, 0

    if not validate_identify_mcq(mcq):
        print(f"[{qa_id}] Stage 1 returned invalid identify MCQ: {mcq}")
        return None, tin, tout

    correct_pos = mcq["correct_position"].strip().upper()
    distractors = mcq["distractors"]
    distractor_reasons = mcq.get("distractor_reasons", {})

    all_letters = ["A", "B", "C", "D"]
    wrong_letters = [l for l in all_letters if l != correct_pos]

    question_image_descriptions = {}
    option_images = {}
    false_reasons = {}

    question_image_descriptions[correct_pos] = f"(existing photo: {description})"
    option_images[correct_pos] = img_path
    false_reasons[correct_pos] = ""

    for i, letter in enumerate(wrong_letters):
        d_key = str(i + 1)
        question_image_descriptions[letter] = str(distractors.get(d_key, "")).strip()
        option_images[letter] = ""
        false_reasons[letter] = str(distractor_reasons.get(d_key, "")).strip()

    record = {
        "qa_id": qa_id,
        "p_id": p_id,
        "entity_type": "Items",
        "entity_name": description,
        "entity_relation": "",
        "rel_idx": item_idx,
        "dimension": "identify_item",
        "sub_type": "item_identify",
        "Q": f"以下哪张图片是用户常用的{_extract_item_noun(description)}？",
        "question_image_descriptions": question_image_descriptions,
        "option_images": option_images,
        "A": correct_pos,
        "A_false_reason": false_reasons.get("A", ""),
        "B_false_reason": false_reasons.get("B", ""),
        "C_false_reason": false_reasons.get("C", ""),
        "D_false_reason": false_reasons.get("D", ""),
        "memory clue": [],
        "matched_session_ids": [],
        "type": "entity_image_mcq",
    }
    return record, tin, tout


# ========================= Stage 2: generate images =========================

def run_stage2_one(record: dict) -> tuple:
    qa_id = record["qa_id"]
    sub_type = record.get("sub_type", "")
    image_prompts = record.get("question_image_descriptions", {})
    correct_letter = record.get("A", "")

    identify_types = {"identify_portrait", "pet_identify_portrait", "item_identify"}
    ref_sub_types = {"appearance_image", "profession_image", "pet_personality_image"}
    use_ref_image = (sub_type in ref_sub_types)
    ref_image_path = record.get("ref_image_path", "") if use_ref_image else ""

    img_dir = Path(IMAGE_DIR)
    img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = dict(record.get("option_images", {}))
    all_ok = True

    for letter in ["A", "B", "C", "D"]:
        if image_paths.get(letter) and os.path.exists(image_paths[letter]):
            continue

        if sub_type in identify_types and letter == correct_letter:
            continue

        prompt_text = image_prompts.get(letter, "")
        if not prompt_text or prompt_text.startswith("(existing portrait") or prompt_text.startswith("(existing photo"):
            continue

        safe_qid = re.sub(r"[^A-Za-z0-9_\-]", "_", qa_id)
        img_path = img_dir / f"{safe_qid}_{letter}.png"

        if img_path.exists():
            image_paths[letter] = str(img_path)
            continue

        if sub_type == "item_identify":
            prompt_text = prompt_text.rstrip() + " 注意：生成的图片中不要出现人物，只有该物品。"
        elif sub_type == "identify_portrait":
            prompt_text = prompt_text.rstrip() + " 注意：若无特殊说明，生成的人物一定要是亚洲人面孔。"

        print(f"  [{qa_id}] generating image {letter} ...")
        if use_ref_image and ref_image_path and os.path.exists(ref_image_path):
            ok = generate_option_image_with_ref(prompt_text, ref_image_path, img_path)
        else:
            ok = generate_option_image(prompt_text, img_path)
        if ok:
            image_paths[letter] = str(img_path)
        else:
            print(f"  [{qa_id}] FAILED to generate image {letter}")
            image_paths[letter] = ""
            all_ok = False

    if not all_ok:
        print(f"[{qa_id}] some images failed, record saved with partial paths")

    record["option_images"] = image_paths
    return record, 0, 0


# ========================= Stage 3: memory clue =========================

def _collect_item_events_by_anchor(formatted_profile: dict, item_description: str) -> list:
    """Collect events where entity_anchors contains the item description."""
    matched = []
    for event in formatted_profile.get("events", []) or []:
        anchors = event.get("entity_anchors", []) or []
        if item_description in anchors:
            matched.append(event)
    return matched


def run_stage3_one(record: dict, formatted_profile: dict) -> tuple:
    qa_id = record["qa_id"]
    ent_idx = record.get("rel_idx", 0)
    entity_type = record.get("entity_type", "Relationship")

    entity_name = record.get("entity_name", "")
    entity_relation = record.get("entity_relation", "")
    entity_info = ""
    appearance = ""

    if entity_type == "Items":
        events = _collect_item_events_by_anchor(formatted_profile, entity_name)
    else:
        # Pets uses both "Pets-{idx}" and "BasicPets-{idx}" codes
        if entity_type == "Pets":
            entity_code = f"Pets-{ent_idx}"
            events = collect_entity_events(formatted_profile, entity_code)
            basic_code = f"BasicPets-{ent_idx}"
            events += collect_entity_events(formatted_profile, basic_code)
        else:
            entity_code = f"Relationship-{ent_idx}"
            events = collect_entity_events(formatted_profile, entity_code)
    dialog_str = format_events_for_prompt(events)
    valid_keys = collect_valid_clue_keys(events)

    record["matched_session_ids"] = sorted({
        e.get("session_id", "") for e in events if e.get("session_id")
    })

    question = record.get("Q", "")
    answer_letter = record.get("A", "")
    image_prompts = record.get("question_image_descriptions", {})
    answer_desc = image_prompts.get(answer_letter, "")
    dimension = record.get("dimension", "")

    prompt3 = prompt_entity_memory_clue.format(
        entity_name=entity_name,
        entity_relation=entity_relation,
        entity_info=entity_info,
        appearance=appearance,
        question=question,
        answer_letter=answer_letter,
        answer_desc=answer_desc,
        dimension=dimension,
        dialog_str=dialog_str,
    )

    tokens_in, tokens_out = 0, 0
    memory_clue = []
    try:
        ans, tin, tout = call_llm(prompt3, temperature=TEMPERATURE_ANS)
        tokens_in += tin
        tokens_out += tout
        raw_clues = ans.get("memory clue")
        if raw_clues is None:
            raw_clues = ans.get("memory_clue")
        memory_clue = filter_memory_clues(raw_clues, valid_keys)
    except Exception as e:
        print(f"[{qa_id}] Stage 3 (memory clue) failed: {e}")

    record["memory clue"] = memory_clue
    return record, tokens_in, tokens_out


# ========================= I/O helpers =========================

def _sort_key(r):
    sub_order = {"appearance_image": 0, "profession_image": 1, "identify_portrait": 2}
    return (
        r.get("p_id", 0),
        r.get("rel_idx", 0),
        sub_order.get(r.get("sub_type", ""), 9),
    )


def _sample_per_sub_type(items: list, n: int, sub_type_index: int = 3) -> list:
    """Take at most n items per sub_type from a task list.

    sub_type_index: the position of sub_type in each tuple element.
    For plain dicts (Stage 2/3), pass sub_type_index=-1 to use dict key.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for item in items:
        if sub_type_index == -1:
            st = item.get("sub_type", "") if isinstance(item, dict) else (item[0].get("sub_type", "") if isinstance(item[0], dict) else "")
        else:
            st = item[sub_type_index] if isinstance(item, tuple) and len(item) > sub_type_index else ""
        buckets[st].append(item)
    result = []
    for st in buckets:
        result.extend(buckets[st][:n])
    return result


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


def _is_record_complete(record: dict) -> bool:
    sub_type = record.get("sub_type", "")
    correct_letter = record.get("A", "")
    identify_types = {"identify_portrait", "pet_identify_portrait", "item_identify"}
    paths = record.get("option_images", {})
    for k in ["A", "B", "C", "D"]:
        p = paths.get(k, "")
        if sub_type in identify_types and k == correct_letter:
            if not p or not os.path.exists(p):
                return False
            continue
        if not isinstance(p, str) or not p or not os.path.exists(p):
            return False
    return True


# ========================= Task collection =========================

def _make_qa_id(p_id: int, ent_idx: int, sub_type: str) -> str:
    """Derive qa_id from sub_type."""
    if sub_type == "appearance_image":
        return f"{p_id}-Relationship-{ent_idx}-img_appearance"
    elif sub_type == "profession_image":
        return f"{p_id}-Relationship-{ent_idx}-img_profession"
    elif sub_type == "identify_portrait":
        return f"{p_id}-Relationship-{ent_idx}-img_identify"
    elif sub_type == "pet_identify_portrait":
        return f"{p_id}-Pets-{ent_idx}-img_identify"
    elif sub_type == "pet_personality_image":
        return f"{p_id}-Pets-{ent_idx}-img_personality"
    elif sub_type == "item_identify":
        return f"{p_id}-Items-{ent_idx}-img_identify"
    return f"{p_id}-Unknown-{ent_idx}-{sub_type}"


def collect_all_tasks(
    profiles: list,
    existing_entity_qas: list,
    sub_type_filter: str | None = None,
    skip_img_check: bool = False,
) -> list:
    """Collect all (p_id, entity_idx, entity_dict, sub_type, existing_text_qa) tasks.

    skip_img_check: if True, don't require img_path to exist on disk
                    (used by --regen mode).
    """
    qa_lookup = build_existing_qa_lookup(existing_entity_qas)
    tasks = []

    def _has_img(entity):
        path = entity.get("img_path", "")
        return bool(path) and (skip_img_check or os.path.exists(path))

    for p_idx, profile in enumerate(profiles):
        p_id = p_idx
        basic = profile.get("Basic", {})

        # --- Relationship ---
        rels = basic.get("Relationship", []) or []
        for rel_idx, rel in enumerate(rels):
            if not rel.get("appearance"):
                continue

            if sub_type_filter is None or sub_type_filter == "appearance_image":
                text_qa = qa_lookup.get((p_id, "Relationship", rel_idx, "外表"))
                tasks.append((p_id, rel_idx, rel, "appearance_image", text_qa))

            if sub_type_filter is None or sub_type_filter == "profession_image":
                text_qa = qa_lookup.get((p_id, "Relationship", rel_idx, "职业"))
                if text_qa:
                    tasks.append((p_id, rel_idx, rel, "profession_image", text_qa))

            if sub_type_filter is None or sub_type_filter == "identify_portrait":
                if _has_img(rel):
                    tasks.append((p_id, rel_idx, rel, "identify_portrait", None))

        # --- Pets ---
        pets = basic.get("Pets", []) or []
        for pet_idx, pet in enumerate(pets):
            if not pet.get("appearance"):
                continue

            if sub_type_filter is None or sub_type_filter == "pet_identify_portrait":
                if _has_img(pet):
                    tasks.append((p_id, pet_idx, pet, "pet_identify_portrait", None))

            if sub_type_filter is None or sub_type_filter == "pet_personality_image":
                text_qa = qa_lookup.get((p_id, "Pets", pet_idx, "性格"))
                tasks.append((p_id, pet_idx, pet, "pet_personality_image", text_qa))

        # --- Items ---
        items = profile.get("Items", []) or []
        for item_idx, item in enumerate(items):
            if sub_type_filter is None or sub_type_filter == "item_identify":
                if _has_img(item):
                    tasks.append((p_id, item_idx, item, "item_identify", None))

    return tasks


def parse_profile_id_filter(raw_values: list[str] | None) -> set[int] | None:
    if not raw_values:
        return None
    ids: set[int] = set()
    for raw in raw_values:
        for part in str(raw).replace(",", " ").split():
            if part.strip():
                ids.add(int(part))
    return ids


def profile_allowed(p_id: int, max_profiles: int | None = None, only_profile_ids: set[int] | None = None) -> bool:
    if max_profiles is not None and max_profiles > 0 and p_id >= max_profiles:
        return False
    if only_profile_ids is not None and p_id not in only_profile_ids:
        return False
    return True


def filter_tasks_by_profile(
    tasks: list,
    max_profiles: int | None = None,
    only_profile_ids: set[int] | None = None,
) -> list:
    if not max_profiles and only_profile_ids is None:
        return tasks
    return [
        task for task in tasks
        if task and profile_allowed(int(task[0]), max_profiles, only_profile_ids)
    ]


def filter_records_by_profile(
    records: list,
    max_profiles: int | None = None,
    only_profile_ids: set[int] | None = None,
) -> list:
    if not max_profiles and only_profile_ids is None:
        return records
    out = []
    for rec in records:
        try:
            p_id = int(rec.get("p_id", -1))
        except Exception:
            p_id = -1
        if profile_allowed(p_id, max_profiles, only_profile_ids):
            out.append(rec)
    return out


# ========================= Stage runners =========================

def _load_common():
    profiles = load_json(PROFILE_PATH)
    if not profiles:
        print(f"ERROR: cannot load {PROFILE_PATH}.")
        return None, None, None, None

    print(f"Loaded {len(profiles)} profile(s) from {PROFILE_PATH}.")

    formatted_profiles = load_json_or_jsonl(FORMATTED_DIALOG_PATH)
    if not formatted_profiles:
        print(f"ERROR: cannot load {FORMATTED_DIALOG_PATH}.")
        return None, None, None, None
    print(f"Loaded {len(formatted_profiles)} formatted profile(s) from {FORMATTED_DIALOG_PATH}.")
    formatted_by_pid = {fp.get("p_id", i): fp for i, fp in enumerate(formatted_profiles)}

    existing_entity_qas = load_json(EXISTING_ENTITY_QA_PATH) if os.path.exists(EXISTING_ENTITY_QA_PATH) else []
    print(f"Loaded {len(existing_entity_qas)} existing entity text QAs.")

    existing = _load_existing(OUTPUT_PATH)
    if existing:
        print(f"[resume] 已有 {len(existing)} 条图片 QA 记录。")

    return profiles, formatted_by_pid, existing, existing_entity_qas


def _print_summary(qa_map: dict, n_existing: int, total_in: int, total_out: int):
    all_records = list(qa_map.values())
    by_sub: dict = {}
    complete = 0
    for r in all_records:
        st = r.get("sub_type", "unknown")
        by_sub[st] = by_sub.get(st, 0) + 1
        if _is_record_complete(r):
            complete += 1

    print(f"\n总记录数: {len(all_records)} (本次新增/更新 {len(all_records) - n_existing})")
    print(f"图片完整: {complete} / {len(all_records)}")
    print("By sub_type:")
    for st in SUB_TYPES:
        if st in by_sub:
            print(f"  - {st}: {by_sub[st]}")
    if total_in or total_out:
        print(f"Tokens — prompt: {total_in}, completion: {total_out}")
    print(f"Output -> {OUTPUT_PATH}")
    print(f"Images -> {IMAGE_DIR}/")


# ---------------------------------------------------------------------------
# Stage 1 only
# ---------------------------------------------------------------------------
def main_stage1(sample=None, sub_type_filter=None, max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, existing_entity_qas = _load_common()
    if profiles is None:
        return

    all_tasks = collect_all_tasks(profiles, existing_entity_qas, sub_type_filter)
    all_tasks = filter_tasks_by_profile(all_tasks, max_profiles, only_profile_ids)

    pending = []
    identify_types = {"identify_portrait", "pet_identify_portrait", "item_identify"}
    for p_id, ent_idx, ent, sub_type, text_qa in all_tasks:
        qa_id = _make_qa_id(p_id, ent_idx, sub_type)

        rec = existing.get(qa_id)
        if rec and rec.get("Q") and rec.get("question_image_descriptions"):
            descs = rec["question_image_descriptions"]
            if sub_type in identify_types:
                correct = rec.get("A", "")
                wrong_letters = [l for l in ["A", "B", "C", "D"] if l != correct]
                if all(descs.get(l) for l in wrong_letters):
                    continue
            else:
                if all(descs.get(k) for k in ["A", "B", "C", "D"]):
                    continue
        pending.append((p_id, ent_idx, ent, sub_type, text_qa))

    if sample is not None:
        pending = _sample_per_sub_type(pending, sample, sub_type_index=3)

    if not pending:
        print("Stage 1: 所有 image_prompt 均已生成，无需重新运行。")
        return
    print(f"Stage 1: 待生成 {len(pending)} 条 (已有 {len(existing)} 条记录)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _process_one(task):
        p_id, ent_idx, ent, sub_type, text_qa = task
        if sub_type == "appearance_image":
            return run_stage1_appearance(p_id, ent_idx, ent, text_qa)
        elif sub_type == "profession_image":
            return run_stage1_profession(p_id, ent_idx, ent, text_qa)
        elif sub_type == "identify_portrait":
            return run_stage1_identify(p_id, ent_idx, ent)
        elif sub_type == "pet_identify_portrait":
            return run_stage1_pet_identify(p_id, ent_idx, ent)
        elif sub_type == "pet_personality_image":
            return run_stage1_pet_personality(p_id, ent_idx, ent, text_qa)
        elif sub_type == "item_identify":
            return run_stage1_item_identify(p_id, ent_idx, ent)
        return None, 0, 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM) as executor:
        for task in pending:
            futures[executor.submit(_process_one, task)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 1: gen descriptions"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ---------------------------------------------------------------------------
# Stage 2 only
# ---------------------------------------------------------------------------
def main_stage2(sample=None, sub_type_filter=None, max_profiles=None, only_profile_ids=None):
    _, _, existing, _ = _load_common()
    if existing is None:
        return

    tasks = []
    for qa_id, rec in existing.items():
        if sub_type_filter and rec.get("sub_type") != sub_type_filter:
            continue
        if not filter_records_by_profile([rec], max_profiles, only_profile_ids):
            continue
        descs = rec.get("question_image_descriptions", {})
        sub_type = rec.get("sub_type", "")
        correct = rec.get("A", "")

        identify_types = {"identify_portrait", "pet_identify_portrait", "item_identify"}
        if sub_type in identify_types:
            wrong_letters = [l for l in ["A", "B", "C", "D"] if l != correct]
            if not all(descs.get(l) for l in wrong_letters):
                continue
        else:
            if not all(descs.get(k) for k in ["A", "B", "C", "D"]):
                continue

        if _is_record_complete(rec):
            continue
        tasks.append(rec)

    if sample is not None:
        tasks = _sample_per_sub_type(tasks, sample, sub_type_index=-1)

    if not tasks:
        print("Stage 2: 所有图片均已生成，无需重新运行。")
        return
    print(f"Stage 2: 待生成图片 {len(tasks)} 条记录")

    qa_map = dict(existing)
    lock = threading.Lock()
    new_since_cp = 0

    def _on(record):
        nonlocal new_since_cp
        qa_map[record["qa_id"]] = record
        new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_IMG) as executor:
        for rec in tasks:
            futures[executor.submit(run_stage2_one, rec)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 2: gen images"):
            rec, _, _ = future.result()
            with lock:
                _on(rec)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), 0, 0)


# ---------------------------------------------------------------------------
# Stage 3 only
# ---------------------------------------------------------------------------
def main_stage3(sample=None, sub_type_filter=None, max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, _ = _load_common()
    if profiles is None:
        return

    tasks = []
    for qa_id, rec in existing.items():
        if sub_type_filter and rec.get("sub_type") != sub_type_filter:
            continue
        if not filter_records_by_profile([rec], max_profiles, only_profile_ids):
            continue
        if not rec.get("Q") or not rec.get("A"):
            continue
        if rec.get("memory clue"):
            continue
        p_id = rec.get("p_id", 0)
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            continue
        tasks.append((rec, fp))

    if sample is not None:
        tasks = _sample_per_sub_type(tasks, sample, sub_type_index=-1)

    if not tasks:
        print("Stage 3: 所有 memory clue 均已生成，无需重新运行。")
        return
    print(f"Stage 3: 待提取 memory clue {len(tasks)} 条记录")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        qa_map[record["qa_id"]] = record
        new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM) as executor:
        for rec, fp in tasks:
            futures[executor.submit(run_stage3_one, rec, fp)] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Stage 3: memory clue"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ---------------------------------------------------------------------------
# Refresh mode: 只重新做 item_identify 题干中的 noun extraction
# ---------------------------------------------------------------------------
def main_refresh_noun_extraction(
    sample=None,
    max_profiles=None,
    only_profile_ids=None,
    refresh_noun_targets: str = "all",
    noun_workers: int = 1,
):
    existing = _load_existing(OUTPUT_PATH)
    if not existing:
        print(f"ERROR: cannot load existing QA records from {OUTPUT_PATH}.")
        return

    candidate_records = filter_records_by_profile(
        list(existing.values()),
        max_profiles=max_profiles,
        only_profile_ids=only_profile_ids,
    )
    targets = [
        rec for rec in candidate_records
        if rec.get("sub_type") == "item_identify"
        and (rec.get("entity_name") or rec.get("item_description"))
    ]
    targets = [rec for rec in targets if _record_matches_refresh_noun_target(rec, refresh_noun_targets)]
    if sample is not None:
        targets = targets[:sample]

    if not targets:
        print("Refresh noun extraction: 没有找到需要刷新的 item_identify QA。")
        return

    print(
        f"Refresh noun extraction: model={NOUN_LLM_MODEL} "
        f"retries={NOUN_LLM_RETRIES} targets={refresh_noun_targets} workers={noun_workers}"
    )
    print(f"Refresh noun extraction: 待刷新 {len(targets)} 条 item_identify 题干")

    updated = unchanged = 0
    new_since_cp = 0
    qa_map = dict(existing)
    method_counts: dict[str, int] = {}

    def refresh_one(rec: dict) -> tuple[dict, bool, str]:
        description = rec.get("entity_name") or rec.get("item_description") or ""
        old_q = rec.get("Q", "")
        noun, noun_status = _extract_item_noun_with_status(description)
        new_q = f"以下哪张图片是用户常用的{noun}？"
        noun_status["old_question"] = old_q
        noun_status["new_question"] = new_q
        rec["noun_extraction"] = noun_status
        method = str(noun_status.get("method") or "unknown")
        changed = rec.get("Q") != new_q

        if changed:
            rec["Q"] = new_q
        return rec, changed, method

    def on_done(rec: dict, changed: bool, method: str) -> None:
        nonlocal updated, unchanged, new_since_cp
        qa_id = rec.get("qa_id", "")
        method_counts[method] = method_counts.get(method, 0) + 1
        if changed:
            updated += 1
        else:
            unchanged += 1
        qa_map[qa_id] = rec
        new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    noun_workers = max(1, int(noun_workers or 1))
    if noun_workers == 1:
        for rec in tqdm(targets, desc="Refresh item nouns"):
            on_done(*refresh_one(rec))
    else:
        lock = threading.Lock()
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=noun_workers) as executor:
            for rec in targets:
                futures[executor.submit(refresh_one, rec)] = rec.get("qa_id", "")
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Refresh item nouns",
            ):
                try:
                    rec, changed, method = future.result()
                except Exception as exc:
                    qa_id = futures[future]
                    print(f"  [WARN] noun extraction worker failed for {qa_id}: {exc}")
                    continue
                with lock:
                    on_done(rec, changed, method)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    print(
        "Refresh noun extraction done: "
        f"updated={updated}, unchanged={unchanged}, "
        f"api_failures={len(_noun_api_failures)}, "
        f"same_as_input={len(_noun_same_as_input)}, "
        f"fallback_used={len(_noun_fallback_used)}, output={OUTPUT_PATH}"
    )
    print(f"  Method counts: {method_counts}")
    if _noun_api_failures:
        print("  API failure examples:")
        for desc in sorted(_noun_api_failures)[:10]:
            print(f"  - {desc}")
    if _noun_same_as_input:
        print("  Same-as-input examples:")
        for desc in sorted(_noun_same_as_input)[:10]:
            print(f"  - {desc}")
    if _noun_fallback_used:
        print("  Local fallback examples:")
        for desc, noun in sorted(_noun_fallback_used.items())[:10]:
            print(f"  - {desc} -> {noun}")


# ---------------------------------------------------------------------------
# All stages: 1 → 2 → 3 sequentially per record
# ---------------------------------------------------------------------------
def build_entity_image_qa(
    p_id: int, ent_idx: int, ent: dict,
    sub_type: str, text_qa: dict | None,
    formatted_profile: dict,
) -> tuple:
    if sub_type == "appearance_image":
        record, tin1, tout1 = run_stage1_appearance(p_id, ent_idx, ent, text_qa)
    elif sub_type == "profession_image":
        record, tin1, tout1 = run_stage1_profession(p_id, ent_idx, ent, text_qa)
    elif sub_type == "identify_portrait":
        record, tin1, tout1 = run_stage1_identify(p_id, ent_idx, ent)
    elif sub_type == "pet_identify_portrait":
        record, tin1, tout1 = run_stage1_pet_identify(p_id, ent_idx, ent)
    elif sub_type == "pet_personality_image":
        record, tin1, tout1 = run_stage1_pet_personality(p_id, ent_idx, ent, text_qa)
    elif sub_type == "item_identify":
        record, tin1, tout1 = run_stage1_item_identify(p_id, ent_idx, ent)
    else:
        return None, 0, 0

    if record is None:
        return None, tin1, tout1
    tokens_in, tokens_out = tin1, tout1

    record, _, _ = run_stage2_one(record)

    record, tin3, tout3 = run_stage3_one(record, formatted_profile)
    tokens_in += tin3
    tokens_out += tout3

    return record, tokens_in, tokens_out


def main_all(sample=None, sub_type_filter=None, max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, existing_entity_qas = _load_common()
    if profiles is None:
        return

    all_tasks = collect_all_tasks(profiles, existing_entity_qas, sub_type_filter)
    all_tasks = filter_tasks_by_profile(all_tasks, max_profiles, only_profile_ids)

    pending = []
    for p_id, ent_idx, ent, sub_type, text_qa in all_tasks:
        qa_id = _make_qa_id(p_id, ent_idx, sub_type)

        if qa_id in existing and _is_record_complete(existing[qa_id]):
            continue

        fp = formatted_by_pid.get(p_id)
        if fp is None:
            continue
        pending.append((p_id, ent_idx, ent, sub_type, text_qa, fp))

    if sample is not None:
        pending = _sample_per_sub_type(pending, sample, sub_type_index=3)

    if not pending:
        print("所有 entity image QA 均已生成，无需重新运行。")
        return
    print(f"待生成: {len(pending)} 条 entity image QA (已有 {len(existing)} 条)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_IMG) as executor:
        for p_id, ent_idx, ent, sub_type, text_qa, fp in pending:
            futures[executor.submit(
                build_entity_image_qa, p_id, ent_idx, ent, sub_type, text_qa, fp
            )] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Building entity image MCQs"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ---------------------------------------------------------------------------
# Regen mode: 删除指定 qa_id 的记录和图片，然后从 Stage 1 重跑
# ---------------------------------------------------------------------------
def main_regen(regen_ids: list[str], max_profiles=None, only_profile_ids=None):
    profiles, formatted_by_pid, existing, existing_entity_qas = _load_common()
    if profiles is None:
        return

    # 删除旧记录及其生成的图片（保留定妆照等原始资源）
    identify_types = {"identify_portrait", "pet_identify_portrait", "item_identify"}
    for qid in regen_ids:
        old = existing.pop(qid, None)
        if old:
            print(f"[REGEN] 已删除记录: {qid}")
            correct_letter = old.get("A", "")
            sub_type = old.get("sub_type", "")
            for letter in ["A", "B", "C", "D"]:
                # identify 类的正确选项是定妆照原始文件，不能删
                if sub_type in identify_types and letter == correct_letter:
                    continue
                img = (old.get("option_images") or {}).get(letter, "")
                if img and os.path.exists(img):
                    os.remove(img)
                    print(f"  已删除图片: {img}")
        else:
            print(f"[REGEN] 记录不存在（将新建）: {qid}")

    _save_checkpoint(list(existing.values()), OUTPUT_PATH)

    # 构建任务
    regen_set = set(regen_ids)
    all_tasks = collect_all_tasks(profiles, existing_entity_qas, skip_img_check=True)
    all_tasks = filter_tasks_by_profile(all_tasks, max_profiles, only_profile_ids)

    pending = []
    for p_id, ent_idx, ent, sub_type, text_qa in all_tasks:
        qa_id = _make_qa_id(p_id, ent_idx, sub_type)
        if qa_id not in regen_set:
            continue
        fp = formatted_by_pid.get(p_id)
        if fp is None:
            continue
        pending.append((p_id, ent_idx, ent, sub_type, text_qa, fp))

    not_found = regen_set - {_make_qa_id(t[0], t[1], t[3]) for t in all_tasks}
    if not_found:
        print(f"[WARN] 以下 qa_id 无法匹配到任何任务: {not_found}")

    if not pending:
        print("没有需要重新生成的任务。")
        return
    print(f"[REGEN] 将重新生成 {len(pending)} 条 (Stage 1→2→3)")

    qa_map = dict(existing)
    lock = threading.Lock()
    total_in = total_out = new_since_cp = 0

    def _on(record, tin, tout):
        nonlocal total_in, total_out, new_since_cp
        total_in += tin; total_out += tout
        if record is not None:
            qa_map[record["qa_id"]] = record
            new_since_cp += 1
        if new_since_cp >= CHECKPOINT_EVERY:
            _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
            new_since_cp = 0

    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_IMG) as executor:
        for p_id, ent_idx, ent, sub_type, text_qa, fp in pending:
            futures[executor.submit(
                build_entity_image_qa, p_id, ent_idx, ent, sub_type, text_qa, fp
            )] = None
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="REGEN: building entity image MCQs"):
            rec, tin, tout = future.result()
            with lock:
                _on(rec, tin, tout)

    _save_checkpoint(list(qa_map.values()), OUTPUT_PATH)
    _print_summary(qa_map, len(existing), total_in, total_out)


# ========================= Entry point =========================

def main():
    import argparse
    global IMAGE_MODEL, NOUN_LLM_RETRIES
    parser = argparse.ArgumentParser(
        description="Relationship 实体图片选择题生成（支持分阶段运行）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help=(
            "只运行指定阶段:\n"
            "  1 = LLM 生成题干 + image_prompt\n"
            "  2 = 根据已有 image_prompt 生成图片\n"
            "  3 = 从对话历史提取 memory clue\n"
            "不指定则按 1→2→3 顺序全部运行"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="只处理前 N 条任务，用于小批量测试",
    )
    parser.add_argument(
        "--max-profiles",
        "--max_profiles",
        type=int,
        default=None,
        help="只处理 p_id < N 的 profile，例如 --max-profiles 5 只处理 p_id=0..4",
    )
    parser.add_argument(
        "--only_profile_ids",
        "--only-profile-ids",
        nargs="*",
        default=None,
        metavar="P_ID",
        help="只处理指定 p_id，支持空格或逗号分隔，例如 --only_profile_ids 0 或 --only_profile_ids 0,1,2",
    )
    parser.add_argument(
        "--sub-type",
        choices=SUB_TYPES,
        default=None,
        help="只处理指定子类型",
    )
    parser.add_argument(
        "--image-model",
        choices=[MODEL_AIFAST_GEMINI_IMAGE, MODEL_OPENROUTER_GEMINI_IMAGE],
        default=IMAGE_MODEL,
        help=(
            "Stage 2 生图后端：\n"
            f"  {MODEL_AIFAST_GEMINI_IMAGE} = AIFast Gemini 原生 generateContent（默认）\n"
            f"  {MODEL_OPENROUTER_GEMINI_IMAGE} = OpenRouter google/gemini-3.1-flash-image-preview"
        ),
    )
    parser.add_argument(
        "--regen",
        type=str,
        default=None,
        help=(
            "指定要从 Stage 1 重新生成的 qa_id，逗号分隔。\n"
            "会删除已有记录和图片后重跑全部 3 个阶段。\n"
            "例如: --regen 0-Relationship-0-img_appearance,0-Pets-0-img_identify"
        ),
    )
    parser.add_argument(
        "--rerun-empty-clues",
        action="store_true",
        help=(
            "直接运行 Stage 3，并且只处理 memory clue 为空的记录。"
            "可与 --sub-type 和 --sample 组合使用。"
        ),
    )
    parser.add_argument(
        "--refresh-noun-extraction",
        "--refresh_item_nouns",
        action="store_true",
        help=(
            "只重新调用 DeepSeek 刷新 item_identify 题干中的核心名词，"
            "不重跑 Stage 1/2/3，不重新生成图片。"
        ),
    )
    parser.add_argument(
        "--refresh-noun-targets",
        choices=["all", "same_as_input", "fallback", "same_or_fallback", "bad"],
        default="all",
        help=(
            "配合 --refresh-noun-extraction 使用，限制刷新范围：\n"
            "  all = 刷新所有 item_identify\n"
            "  same_as_input = 只刷新当前题干名词等于完整 entity anchor 的记录\n"
            "  fallback = 只刷新当前题干名词看起来使用了本地 fallback 的记录\n"
            "  same_or_fallback/bad = 刷新以上两类"
        ),
    )
    parser.add_argument(
        "--noun-retries",
        type=int,
        default=NOUN_LLM_RETRIES,
        help=f"noun extraction API 最大重试次数，默认 {NOUN_LLM_RETRIES}",
    )
    parser.add_argument(
        "--noun-workers",
        type=int,
        default=1,
        help="noun extraction 并发数；默认 1 为串行，建议 2-4 起步，避免触发限流",
    )
    args = parser.parse_args()
    IMAGE_MODEL = args.image_model
    NOUN_LLM_RETRIES = max(1, int(args.noun_retries))
    only_profile_ids = parse_profile_id_filter(args.only_profile_ids)

    if args.max_profiles is not None and args.max_profiles < 1:
        print("ERROR: --max-profiles 必须为正整数")
        return
    if args.max_profiles:
        print(f"[filter] max_profiles={args.max_profiles} -> only p_id < {args.max_profiles}")
    if only_profile_ids is not None:
        print(f"[filter] only_profile_ids={sorted(only_profile_ids)}")
    print(f"[image] backend={IMAGE_MODEL}")
    if IMAGE_MODEL == MODEL_OPENROUTER_GEMINI_IMAGE and not env_value("CUE_MEM_IMAGE_OPENROUTER_API_KEY"):
        print("WARN: OpenRouter backend selected but CUE_MEM_IMAGE_OPENROUTER_API_KEY is empty.")

    if args.refresh_noun_extraction:
        print("=" * 60)
        print("Refresh noun extraction only")
        print("=" * 60)
        main_refresh_noun_extraction(
            sample=args.sample,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
            refresh_noun_targets=args.refresh_noun_targets,
            noun_workers=args.noun_workers,
        )
    elif args.regen:
        regen_ids = [x.strip() for x in args.regen.split(",") if x.strip()]
        print("=" * 60)
        print(f"REGEN mode: regenerating {len(regen_ids)} qa_id(s) from Stage 1")
        print("=" * 60)
        main_regen(regen_ids, max_profiles=args.max_profiles, only_profile_ids=only_profile_ids)
    elif args.rerun_empty_clues:
        print("=" * 60)
        print("Running Stage 3 only: regenerate memory clue for records with empty clue")
        print("=" * 60)
        main_stage3(
            sample=args.sample,
            sub_type_filter=args.sub_type,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    elif args.stage == 1:
        print("=" * 60)
        print("Running Stage 1 only: LLM gen Q + image_prompts")
        print("=" * 60)
        main_stage1(
            sample=args.sample,
            sub_type_filter=args.sub_type,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    elif args.stage == 2:
        print("=" * 60)
        print("Running Stage 2 only: generate option images")
        print("=" * 60)
        main_stage2(
            sample=args.sample,
            sub_type_filter=args.sub_type,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    elif args.stage == 3:
        print("=" * 60)
        print("Running Stage 3 only: LLM extract memory clue")
        print("=" * 60)
        main_stage3(
            sample=args.sample,
            sub_type_filter=args.sub_type,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )
    else:
        print("=" * 60)
        print("Running all stages: 1 → 2 → 3")
        print("=" * 60)
        main_all(
            sample=args.sample,
            sub_type_filter=args.sub_type,
            max_profiles=args.max_profiles,
            only_profile_ids=only_profile_ids,
        )


if __name__ == "__main__":
    main()
