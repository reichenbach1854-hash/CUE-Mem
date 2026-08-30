"""生成 entity_image_qa_viewer.html，用于可视化检查实体图片选择题。

展示内容：
  - 题目信息与选项图片
  - 正确/错误标记
  - Memory clue（轮次文本、图片、音频）

用法:
    python qa/entity_image_qa_reviewer.py
    python qa/entity_image_qa_reviewer.py --input qa/qa_entity_image_mcq.json
    python qa/entity_image_qa_reviewer.py --output qa/my_viewer.html
"""

import argparse
import base64
import html
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts.common.io import load_json_or_jsonl
from scripts.common.paths import resolve_path
from scripts.qa.config import profile_path, qa_path

INPUT_PATH = qa_path("qa_entity_image_mcq.json")
OUTPUT_PATH = qa_path("entity_image_qa_viewer.html")
FORMATTED_DATA_PATH = qa_path("qa_formatted_data.json")
PROFILE_PATH = profile_path("profiles_with_anchors_with_images_entity.json")
LFS_POINTER_MARKER = b"git-lfs.github.com/spec/v1"

SUB_TYPE_LABELS = {
    "appearance_image": "外表/穿搭",
    "profession_image": "职业场景",
    "identify_portrait": "人物辨识",
    "pet_identify_portrait": "宠物辨识",
    "pet_personality_image": "宠物性格",
    "item_identify": "物品辨识",
}

ENTITY_TYPE_LABELS = {
    "Relationship": "人物",
    "Pets": "宠物",
    "Items": "物品",
}


def esc(text: str) -> str:
    return html.escape(str(text))


def load_json_file(path: str):
    if not path or not os.path.exists(path):
        return None
    return load_json_or_jsonl(path)


def _replace_text(value, old: str, new: str):
    if not old or old == new:
        return value
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {k: _replace_text(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_text(v, old, new) for v in value]
    return value


def sync_records_with_current_profiles(records: list, profiles: list | None) -> int:
    """Use current profile file to correct stale entity names/img paths in viewer records.

    This only mutates in-memory records for HTML display; it does not rewrite the
    QA JSON. It protects the viewer from older entity_image QA files whose
    `entity_name`/`entity_relation` still came from a previous profile version.
    """
    if not profiles:
        return 0

    changed = 0
    identify_types = {"identify_portrait", "pet_identify_portrait", "item_identify"}

    for record in records:
        try:
            p_id = int(record.get("p_id", -1))
            ent_idx = int(record.get("rel_idx", -1))
        except Exception:
            continue
        if p_id < 0 or ent_idx < 0 or p_id >= len(profiles):
            continue

        profile = profiles[p_id] or {}
        basic = profile.get("Basic", {}) or {}
        entity_type = record.get("entity_type", "")
        sub_type = record.get("sub_type", "")
        correct = record.get("A", "")

        current_name = ""
        current_relation = ""
        current_img = ""

        if entity_type == "Relationship":
            rels = basic.get("Relationship", []) or []
            if ent_idx >= len(rels):
                continue
            ent = rels[ent_idx]
            current_name = ent.get("name", "")
            current_relation = ent.get("relation", "")
            current_img = ent.get("img_path", "")
        elif entity_type == "Pets":
            pets = basic.get("Pets", []) or []
            if ent_idx >= len(pets):
                continue
            ent = pets[ent_idx]
            current_name = ent.get("name", "")
            current_relation = ""
            current_img = ent.get("img_path", "")
        elif entity_type == "Items":
            items = profile.get("Items", []) or []
            if ent_idx >= len(items):
                continue
            ent = items[ent_idx]
            current_name = ent.get("description", "")
            current_relation = ent.get("source_subcategory", "")
            current_img = ent.get("img_path", "")
        else:
            continue

        old_name = record.get("entity_name", "")
        if current_name and old_name != current_name:
            record["Q"] = _replace_text(record.get("Q", ""), old_name, current_name)
            record["question_image_descriptions"] = _replace_text(
                record.get("question_image_descriptions", {}), old_name, current_name
            )
            record["entity_name"] = current_name
            changed += 1

        if record.get("entity_relation", "") != current_relation:
            record["entity_relation"] = current_relation
            changed += 1

        if sub_type in {"appearance_image", "profession_image", "pet_personality_image"}:
            if current_img and record.get("ref_image_path", "") != current_img:
                record["ref_image_path"] = current_img
                changed += 1

        if sub_type in identify_types and correct in {"A", "B", "C", "D"} and current_img:
            option_images = dict(record.get("option_images", {}) or {})
            if option_images.get(correct, "") != current_img:
                option_images[correct] = current_img
                record["option_images"] = option_images
                changed += 1

    return changed


def resolve_local_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return resolve_path(candidate)


def is_existing_media_path(value: str) -> bool:
    return bool(value and resolve_local_path(value).exists())


def scene_date(scene_description: str) -> str:
    scene = (scene_description or "").strip()
    if ";" in scene:
        return scene.split(";", 1)[0].strip()
    return ""


def detect_image_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    return ""


def local_file_url(path: str) -> str:
    if not path:
        return ""
    resolved = resolve_local_path(path)
    if not resolved.exists():
        return ""
    with open(resolved, "rb") as f:
        prefix = f.read(200)
    if prefix.startswith(b"version ") and LFS_POINTER_MARKER in prefix:
        raise RuntimeError(
            f"{resolved} 是 Git LFS 指针，不是真实媒体文件。请先拉取 Git LFS 对象。"
        )
    return resolved.as_uri()


def img_to_data_url(path: str) -> str:
    if not path:
        return ""
    resolved = resolve_local_path(path)
    if not resolved.exists():
        return ""
    try:
        with open(resolved, "rb") as f:
            raw = f.read()
        if raw.startswith(b"version ") and LFS_POINTER_MARKER in raw[:200]:
            raise RuntimeError(
                f"{resolved} 是 Git LFS 指针，不是真实图片。"
                "请先运行 `git lfs pull` 或 `git lfs checkout qa/entity_images`。"
            )
        mime = detect_image_mime(raw)
        if not mime:
            raise RuntimeError(f"{resolved} 不是受支持的图片文件。")
        return f"data:{mime};base64,{base64.b64encode(raw).decode()}"
    except RuntimeError:
        raise
    except Exception:
        return ""


def file_to_data_uri(filepath: str, mime: str) -> str:
    try:
        resolved = resolve_local_path(filepath)
        with open(resolved, "rb") as f:
            raw = f.read()
        if raw.startswith(b"version ") and LFS_POINTER_MARKER in raw[:200]:
            raise RuntimeError(
                f"{resolved} 是 Git LFS 指针，不是真实媒体文件。请先拉取 Git LFS 对象。"
            )
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except RuntimeError:
        raise
    except Exception:
        return ""


def build_clue_maps(formatted_data_path: str):
    """从 qa_formatted_data.json 构建 memory clue 的媒体与轮次索引。"""
    voice_map = {}
    image_map = {}
    round_map = {}

    if not os.path.exists(formatted_data_path):
        print(f"WARN: formatted data not found: {formatted_data_path}")
        return voice_map, image_map, round_map

    with open(formatted_data_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    if isinstance(profiles, dict):
        profiles = [profiles]

    for fallback_pid, profile in enumerate(profiles):
        try:
            pid = int(profile.get("p_id", fallback_pid))
        except Exception:
            pid = fallback_pid

        for event in profile.get("events", []) or []:
            sid = event.get("session_id", "")
            sess_date = scene_date(event.get("scene_description", ""))
            img_desc = (event.get("user_shared_image_description") or "").strip()
            bg_audio = (event.get("background_audio_info") or "").strip()
            human_speech = (event.get("human_speech_content") or "").strip()

            for d in event.get("dialog_list", []) or []:
                rnd = d.get("round", "")
                round_info = {
                    "session_id": sid,
                    "date": sess_date,
                    "round": rnd,
                    "p_id": pid,
                    "task_id": event.get("task_id", ""),
                    "scene_description": event.get("scene_description", ""),
                }
                if d.get("user"):
                    round_info["user"] = d["user"]
                if d.get("assistant"):
                    round_info["assistant"] = d["assistant"]
                round_map[(pid, rnd)] = round_info

                for key, value in d.items():
                    if not isinstance(key, str):
                        continue
                    value = str(value or "").strip()
                    if key.lower().endswith((".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg")):
                        cid = key.rsplit(".", 1)[0]
                        voice_map[(pid, cid)] = {
                            "path": value if is_existing_media_path(value) else "",
                            "caption": "" if is_existing_media_path(value) else value,
                            "background_audio_info": bg_audio if bg_audio.lower() != "none" else "",
                            "human_speech_content": human_speech if human_speech.lower() != "none" else "",
                        }
                    elif key.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        cid = key.rsplit(".", 1)[0]
                        image_map[(pid, cid)] = {
                            "path": value if is_existing_media_path(value) else "",
                            "caption": img_desc if img_desc.lower() != "none" else (
                                "" if is_existing_media_path(value) else value
                            ),
                        }

    return voice_map, image_map, round_map


def classify_clue(clue_str: str):
    if clue_str.endswith(".wav"):
        return "voice", clue_str.replace(".wav", "")
    if clue_str.endswith(".png") or clue_str.endswith(".jpg"):
        ext = clue_str.rsplit(".", 1)[1]
        return "image", clue_str.replace(f".{ext}", "")
    if ":" in clue_str:
        return "round", clue_str
    return "unknown", clue_str


def build_clue_html(record: dict, voice_map, image_map, round_map, embed_media: bool):
    clues = record.get("memory clue", [])
    if not clues:
        return '<div class="clue-empty">无 Memory Clue</div>'

    p_id = record.get("p_id", 0)
    matched_sessions = record.get("matched_session_ids", [])

    by_session = defaultdict(list)
    for clue_str in clues:
        ctype, cid = classify_clue(clue_str)
        sess_id = cid.split(":")[0] if ctype == "round" else cid.split("-")[0]
        by_session[sess_id].append((clue_str, ctype, cid))

    parts = []
    for sess_id in sorted(by_session.keys(), key=lambda x: (x[0] != "D", x)):
        items = by_session[sess_id]
        sess_parts = []

        for clue_str, ctype, cid in items:
            if ctype == "round":
                info = round_map.get((p_id, cid), {})
                user_text = info.get("user", "")
                voice_cap = info.get("voice_caption", "")
                assistant_text = info.get("assistant", "")

                content = ""
                if voice_cap:
                    content += f'<div class="clue-voice-cap">🎤 <em>{esc(voice_cap)}</em></div>'
                if user_text:
                    content += f'<div class="clue-user">👤 {esc(user_text)}</div>'
                if assistant_text:
                    content += f'<div class="clue-assistant">🤖 {esc(assistant_text)}</div>'
                if not content:
                    content = f'<div class="clue-text">{esc(clue_str)}</div>'

                sess_parts.append(
                    f'''
                <div class="clue-item clue-round">
                    <span class="clue-tag tag-round">{esc(cid)}</span>
                    {content}
                </div>'''
                )
            elif ctype == "voice":
                info = voice_map.get((p_id, cid), {})
                audio_path = info.get("path", "")
                caption = info.get("caption", "")
                bg_audio = info.get("background_audio_info", "")
                human_speech = info.get("human_speech_content", "")
                audio_html = ""
                if audio_path:
                    if embed_media:
                        media_src = file_to_data_uri(audio_path, "audio/wav")
                    else:
                        media_src = local_file_url(audio_path)
                    if media_src:
                        audio_html = f'<audio controls preload="none" src="{esc(media_src)}"></audio>'
                    if not audio_html:
                        audio_html = f'<div class="clue-missing">音频文件: {esc(audio_path)}</div>'

                sess_parts.append(
                    f'''
                <div class="clue-item clue-audio">
                    <span class="clue-tag tag-voice">🔊 {esc(clue_str)}</span>
                    {audio_html}
                    {f'<div class="clue-caption">{esc(caption)}</div>' if caption else ''}
                    {f'<div class="clue-caption">背景音：{esc(bg_audio)}</div>' if bg_audio else ''}
                    {f'<div class="clue-caption">语音：{esc(human_speech)}</div>' if human_speech else ''}
                </div>'''
                )
            elif ctype == "image":
                info = image_map.get((p_id, cid), {})
                img_path = info.get("path", "")
                caption = info.get("caption", "")
                img_html = ""
                if img_path:
                    if embed_media:
                        media_src = img_to_data_url(img_path)
                    else:
                        media_src = local_file_url(img_path)
                    if media_src:
                        img_html = f'<img class="clue-img" src="{esc(media_src)}" loading="lazy">'
                    if not img_html:
                        img_html = f'<div class="clue-missing">图片文件: {esc(img_path)}</div>'

                sess_parts.append(
                    f'''
                <div class="clue-item clue-image">
                    <span class="clue-tag tag-image">🖼 {esc(clue_str)}</span>
                    {img_html}
                    {f'<div class="clue-caption">{esc(caption)}</div>' if caption else ''}
                </div>'''
                )
            else:
                sess_parts.append(
                    f'''
                <div class="clue-item">
                    <span class="clue-tag">{esc(clue_str)}</span>
                </div>'''
                )

        sess_header_date = ""
        first_round = round_map.get((p_id, f"{sess_id}:00"), {})
        if first_round.get("date"):
            sess_header_date = f' <span class="sess-date">({esc(first_round["date"])})</span>'

        parts.append(
            f'''
        <div class="clue-session">
            <div class="clue-session-header">
                <span class="sess-badge">Session {esc(sess_id)}</span>{sess_header_date}
                <span class="clue-count">{len(items)} clues</span>
            </div>
            {"".join(sess_parts)}
        </div>'''
        )

    return f'''
    <div class="clue-section">
        <div class="clue-title" onclick="this.parentElement.classList.toggle('collapsed')">
            ▶ Memory Clue ({len(clues)} 条，涉及 Session: {", ".join(matched_sessions)})
        </div>
        <div class="clue-body">
            {"".join(parts)}
        </div>
    </div>'''


def build_html(records: list, voice_map, image_map, round_map, embed_images: bool = True) -> str:
    by_sub = Counter(r.get("sub_type", "?") for r in records)
    by_etype = Counter(r.get("entity_type", "?") for r in records)
    total = len(records)

    stats_sub = "".join(
        f"<li>{SUB_TYPE_LABELS.get(st, st)}: {c}</li>"
        for st, c in sorted(by_sub.items(), key=lambda x: x[0])
    )
    stats_etype = "".join(
        f"<li>{ENTITY_TYPE_LABELS.get(et, et)}: {c}</li>"
        for et, c in sorted(by_etype.items(), key=lambda x: x[0])
    )

    sub_options = "".join(
        f'<option value="{st}">{SUB_TYPE_LABELS.get(st, st)}</option>'
        for st in sorted(by_sub.keys())
    )
    etype_options = "".join(
        f'<option value="{et}">{ENTITY_TYPE_LABELS.get(et, et)}</option>'
        for et in sorted(by_etype.keys())
    )

    cards = []
    for idx, r in enumerate(records):
        qa_id = r.get("qa_id", "")
        sub_type = r.get("sub_type", "")
        entity_type = r.get("entity_type", "")
        entity_name = r.get("entity_name", "")
        entity_relation = r.get("entity_relation", "")
        dimension = r.get("dimension", "")
        question = r.get("Q", "")
        correct = r.get("A", "")
        descs = r.get("question_image_descriptions", {})
        imgs = r.get("option_images", {})
        ref_img = r.get("ref_image_path", "")

        sub_label = SUB_TYPE_LABELS.get(sub_type, sub_type)
        etype_label = ENTITY_TYPE_LABELS.get(entity_type, entity_type)
        relation_str = f" ({esc(entity_relation)})" if entity_relation else ""
        dimension_str = f'<div class="dimension-row"><strong>考察维度：</strong>{esc(dimension)}</div>' if dimension else ""

        ref_html = ""
        if ref_img:
            if embed_images:
                du = img_to_data_url(ref_img)
                if du:
                    ref_html = f'<div class="ref-portrait"><strong>定妆照：</strong><img src="{du}" alt="ref"></div>'
            else:
                ref_src = local_file_url(ref_img)
                if ref_src:
                    ref_html = f'<div class="ref-portrait"><strong>定妆照：</strong><img src="{esc(ref_src)}" alt="ref" onerror="this.style.display=\'none\'"></div>'

        opt_cards = []
        for letter in ["A", "B", "C", "D"]:
            is_correct = letter == correct
            cls = "correct" if is_correct else "wrong"
            label_cls = "label-correct" if is_correct else "label-wrong"

            img_path = imgs.get(letter, "")
            if embed_images:
                du = img_to_data_url(img_path)
                img_tag = f'<img src="{du}" alt="Option {letter}" loading="lazy">' if du else '<div class="img-missing" style="display:flex;">图片缺失</div>'
            else:
                img_src = local_file_url(img_path)
                img_tag = (
                    f'<img src="{esc(img_src)}" alt="Option {letter}" loading="lazy"'
                    f' onerror="this.style.display=\'none\';this.nextElementSibling.style.display=\'flex\';">'
                    f'<div class="img-missing" style="display:none;">图片缺失</div>'
                ) if img_src else '<div class="img-missing" style="display:flex;">图片缺失</div>'

            reason_key = f"{letter}_false_reason"
            reason = r.get(reason_key, "")
            if is_correct:
                reason_html = '<div class="reason correct-reason">&#10003; 正确答案</div>'
            elif reason:
                reason_html = f'<div class="reason wrong-reason">&#10007; {esc(reason)}</div>'
            else:
                reason_html = '<div class="reason wrong-reason">&#10007; 干扰选项</div>'

            desc = descs.get(letter, "")
            desc_html = ""
            if desc and not desc.startswith("(existing"):
                short = desc[:80] + "..." if len(desc) > 80 else desc
                desc_html = f'<div class="desc-text" title="{esc(desc)}">{esc(short)}</div>'

            opt_cards.append(
                f'''
        <div class="option-card {cls}">
            <div class="option-label {label_cls}">{letter}</div>
            {img_tag}
            {reason_html}
            {desc_html}
        </div>'''
            )

        clue_html = build_clue_html(r, voice_map, image_map, round_map, embed_images)
        card = f'''
    <div class="qa-card" data-sub="{sub_type}" data-etype="{entity_type}" data-qid="{qa_id}" id="q-{idx}">
        <div class="card-header">
            <span class="qa-index">#{idx + 1}</span>
            <span class="qa-id">{esc(qa_id)}</span>
            <span class="badge badge-{entity_type.lower()}">{esc(etype_label)}</span>
            <span class="category-tag">{esc(sub_label)}</span>
        </div>
        <div class="entity-row">
            <strong>{esc(etype_label)}：</strong>{esc(entity_name)}{relation_str}
        </div>
        {dimension_str}
        {ref_html}
        <div class="question-row">
            <strong>Q：</strong>{esc(question)}
        </div>
        <div class="options-grid">
            {"".join(opt_cards)}
        </div>
        {clue_html}
    </div>'''
        cards.append(card)

    cards_html = "\n".join(cards)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>实体图片选择题检查</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f0f2f5; color: #333; padding: 20px;
}}
.header {{
    max-width: 1200px; margin: 0 auto 24px; background: #fff;
    border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.header h1 {{ font-size: 22px; margin-bottom: 12px; }}
.stats {{ display: flex; gap: 32px; flex-wrap: wrap; font-size: 14px; color: #666; }}
.stats ul {{ list-style: none; }}
.stats li {{ margin: 2px 0; }}

.filter-bar {{
    max-width: 1200px; margin: 0 auto 20px; background: #fff;
    border-radius: 12px; padding: 16px 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
}}
.filter-bar label {{ font-size: 14px; font-weight: 500; }}
.filter-bar select, .filter-bar input {{
    padding: 6px 12px; border: 1px solid #d9d9d9; border-radius: 6px;
    font-size: 14px; outline: none;
}}
.filter-bar select:focus, .filter-bar input:focus {{ border-color: #4096ff; }}

.qa-card {{
    max-width: 1200px; margin: 0 auto 28px; background: #fff;
    border-radius: 12px; padding: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.card-header {{
    display: flex; align-items: center; gap: 10px; margin-bottom: 12px; flex-wrap: wrap;
}}
.qa-index {{ font-weight: 700; font-size: 16px; color: #1677ff; }}
.qa-id {{ font-family: monospace; font-size: 13px; color: #999; }}
.badge {{
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 12px; font-weight: 500;
}}
.badge-relationship {{ background: #e6f4ff; color: #1677ff; }}
.badge-pets {{ background: #fff7e6; color: #d48806; }}
.badge-items {{ background: #f6ffed; color: #389e0d; }}
.category-tag {{
    font-size: 13px; color: #8c8c8c;
    border: 1px solid #d9d9d9; border-radius: 10px; padding: 2px 10px;
}}

.entity-row, .dimension-row {{
    font-size: 14px; color: #555; margin-bottom: 8px;
    padding: 8px 12px; background: #fafafa; border-radius: 6px;
    border-left: 3px solid #1677ff;
}}
.ref-portrait {{
    margin-bottom: 12px; font-size: 14px;
}}
.ref-portrait img {{
    display: block; margin-top: 6px; max-width: 120px; max-height: 120px;
    border-radius: 8px; border: 2px solid #d9d9d9; object-fit: cover;
}}
.question-row {{
    font-size: 15px; margin-bottom: 16px; line-height: 1.6;
}}

.options-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
    margin-bottom: 16px;
}}
@media (max-width: 900px) {{
    .options-grid {{ grid-template-columns: repeat(2, 1fr); }}
}}

.option-card {{
    border: 3px solid #e8e8e8; border-radius: 10px; overflow: hidden;
    display: flex; flex-direction: column; position: relative;
    transition: transform 0.15s;
}}
.option-card:hover {{ transform: scale(1.02); }}
.option-card.correct {{ border-color: #52c41a; }}
.option-card.wrong {{ border-color: #e8e8e8; }}

.option-label {{
    position: absolute; top: 8px; left: 8px; z-index: 2;
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 14px; color: #fff;
}}
.label-correct {{ background: #52c41a; }}
.label-wrong {{ background: #8c8c8c; }}

.option-card img {{
    width: 100%; aspect-ratio: 1; object-fit: cover; display: block;
}}
.img-missing {{
    width: 100%; aspect-ratio: 1; background: #fafafa;
    align-items: center; justify-content: center;
    color: #bbb; font-size: 14px;
}}
.reason {{
    padding: 8px 10px; font-size: 12px; line-height: 1.5;
}}
.correct-reason {{ background: #f6ffed; color: #389e0d; font-weight: 500; }}
.wrong-reason {{ background: #fff2f0; color: #cf1322; }}
.desc-text {{
    padding: 4px 10px 8px; font-size: 11px; color: #999; line-height: 1.4;
    border-top: 1px solid #f0f0f0; cursor: help;
}}

.clue-section {{
    border: 1px solid #e8e8e8; border-radius: 8px; overflow: hidden;
}}
.clue-section.collapsed .clue-body {{ display: none; }}
.clue-title {{
    padding: 12px 16px; background: #fafafa; font-size: 14px; font-weight: 600;
    cursor: pointer; user-select: none; color: #555; border-bottom: 1px solid #e8e8e8;
}}
.clue-title:hover {{ background: #f0f0f0; }}
.clue-section:not(.collapsed) .clue-title {{ background: #e6f4ff; color: #1677ff; }}
.clue-body {{ padding: 12px 16px; }}
.clue-empty {{ padding: 12px; color: #bbb; font-size: 13px; }}

.clue-session {{
    margin-bottom: 16px; border: 1px solid #f0f0f0; border-radius: 8px; overflow: hidden;
}}
.clue-session:last-child {{ margin-bottom: 0; }}
.clue-session-header {{
    padding: 8px 12px; background: #f7f8fa;
    display: flex; align-items: center; gap: 8px; border-bottom: 1px solid #f0f0f0;
}}
.sess-badge {{
    font-weight: 600; font-size: 13px; color: #1677ff;
    background: #e6f4ff; padding: 2px 10px; border-radius: 10px;
}}
.sess-date {{ font-size: 12px; color: #999; }}
.clue-count {{ font-size: 12px; color: #bbb; margin-left: auto; }}

.clue-item {{
    padding: 10px 12px; border-bottom: 1px solid #f5f5f5;
    font-size: 13px; line-height: 1.6;
}}
.clue-item:last-child {{ border-bottom: none; }}
.clue-tag {{
    display: inline-block; padding: 1px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 500; margin-bottom: 4px; font-family: monospace;
}}
.tag-round {{ background: #f0f0f0; color: #666; }}
.tag-voice {{ background: #fff7e6; color: #d48806; }}
.tag-image {{ background: #f6ffed; color: #389e0d; }}

.clue-user {{ color: #333; margin: 4px 0; }}
.clue-assistant {{ color: #888; margin: 4px 0; font-size: 12px; }}
.clue-voice-cap {{ color: #d48806; margin: 4px 0; font-style: italic; }}
.clue-caption {{ color: #999; font-size: 12px; margin-top: 4px; }}
.clue-missing {{ color: #cf1322; font-size: 12px; }}
.clue-img {{
    max-width: 280px; max-height: 200px; border-radius: 6px;
    margin: 6px 0; border: 1px solid #e8e8e8; cursor: pointer;
}}
audio {{
    display: block; margin: 6px 0; height: 36px; width: 100%; max-width: 360px;
}}
</style>
</head>
<body>

<div class="header">
    <h1>实体图片选择题 — 可视化检查</h1>
    <div class="stats">
        <div><strong>总题数：</strong>{total}</div>
        <div><strong>按子类型：</strong><ul>{stats_sub}</ul></div>
        <div><strong>按实体类型：</strong><ul>{stats_etype}</ul></div>
    </div>
</div>

<div class="filter-bar">
    <label>子类型：</label>
    <select id="filterSub">
        <option value="">全部</option>
        {sub_options}
    </select>
    <label>实体类型：</label>
    <select id="filterEtype">
        <option value="">全部</option>
        {etype_options}
    </select>
    <label>搜索：</label>
    <input id="filterSearch" type="text" placeholder="qa_id / 实体名" style="width:220px;">
    <button id="expandAll" style="padding:6px 14px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;font-size:13px;">展开全部 Clue</button>
    <button id="collapseAll" style="padding:6px 14px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;font-size:13px;">折叠全部 Clue</button>
    <span id="filterCount" style="font-size:13px;color:#999;margin-left:auto;"></span>
</div>

{cards_html}

<script>
const cards = document.querySelectorAll('.qa-card');
const filterSub = document.getElementById('filterSub');
const filterEtype = document.getElementById('filterEtype');
const filterSearch = document.getElementById('filterSearch');
const filterCount = document.getElementById('filterCount');

function applyFilters() {{
    const sub = filterSub.value;
    const etype = filterEtype.value;
    const q = filterSearch.value.toLowerCase();
    let shown = 0;
    cards.forEach(c => {{
        const matchSub = !sub || c.dataset.sub === sub;
        const matchEtype = !etype || c.dataset.etype === etype;
        const matchSearch = !q || c.dataset.qid.toLowerCase().includes(q) || c.textContent.toLowerCase().includes(q);
        const vis = matchSub && matchEtype && matchSearch;
        c.style.display = vis ? '' : 'none';
        if (vis) shown++;
    }});
    filterCount.textContent = shown + ' / ' + cards.length;
}}
filterSub.addEventListener('change', applyFilters);
filterEtype.addEventListener('change', applyFilters);
filterSearch.addEventListener('input', applyFilters);
document.getElementById('expandAll').addEventListener('click', () => {{
    document.querySelectorAll('.clue-section').forEach(s => s.classList.remove('collapsed'));
}});
document.getElementById('collapseAll').addEventListener('click', () => {{
    document.querySelectorAll('.clue-section').forEach(s => s.classList.add('collapsed'));
}});
applyFilters();
</script>
</body>
</html>'''


def main():
    parser = argparse.ArgumentParser(description="生成实体图片选择题 HTML 检查页面")
    parser.add_argument("--input", default=INPUT_PATH, help="输入 JSON 路径")
    parser.add_argument("--output", default=OUTPUT_PATH, help="输出 HTML 路径")
    parser.add_argument("--profiles", default=PROFILE_PATH, help="当前 profile JSON，用于同步实体名称/定妆照路径")
    parser.add_argument("--formatted-data", default=FORMATTED_DATA_PATH, help="用于构建 memory clue 索引的 formatted data")
    parser.add_argument(
        "--embed",
        action="store_true",
        help="将图片和音频以内嵌 base64 写入 HTML（文件会非常大）",
    )
    parser.add_argument("--no-embed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.input = resolve_path(args.input)
    args.output = resolve_path(args.output)
    args.profiles = resolve_path(args.profiles)
    formatted_data_path = resolve_path(args.formatted_data)

    if not os.path.exists(args.input):
        msg = (
            f"输入文件不存在: {args.input}<br>"
            f"请先运行 qa/gen_qa_entity_image.py 生成该文件，或用 --input 指定正确的 entity image QA JSON。"
        )
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(
                "<!doctype html><meta charset='utf-8'>"
                "<title>Entity Image QA Viewer</title>"
                "<body style='font-family:Segoe UI,Microsoft YaHei,sans-serif;padding:32px;'>"
                f"<h2>无法生成页面</h2><p>{msg}</p></body>"
            )
        print(msg.replace("<br>", "\n"))
        print(f"Written warning page to {args.output}")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Loaded {len(records)} records from {args.input}")
    profiles = load_json_file(args.profiles)
    changed = sync_records_with_current_profiles(records, profiles if isinstance(profiles, list) else None)
    if changed:
        print(f"Synced {changed} stale display field(s) from current profiles: {args.profiles}")
    print("Building clue maps from qa_formatted_data.json...")
    voice_map, image_map, round_map = build_clue_maps(str(formatted_data_path))
    print(f"  voice: {len(voice_map)}, image: {len(image_map)}, rounds: {len(round_map)}")

    html_text = build_html(
        records,
        voice_map=voice_map,
        image_map=image_map,
        round_map=round_map,
        embed_images=args.embed and not args.no_embed,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
