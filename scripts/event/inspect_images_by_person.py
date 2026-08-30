"""
按人物分组检查生成图片效果。

Section 1 — 事件图片：读取 events_with_anchors.jsonl，展示每张事件图片及
           scene_description、user_shared_image_description、偏好信息。

Section 2 — 定妆照：读取 profile/generated_portraits/manifest.json 及
           profile/profiles_with_anchors_with_items.json，按
           Relationship / Pets / Items 三类展示定妆照及来源描述。

用法：
    python event/inspect_images_by_person.py
    # 然后用浏览器打开 event/inspect_images_by_person.html
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

from scripts.common.io import load_json_or_jsonl
from scripts.common.paths import project_path, resolve_path

EVENTS_PATH = project_path("event", "events_with_anchors.jsonl")
DIALOGUE_PATH = project_path("event", "dialogue_000_019_with_anchors.jsonl")
QA_FORMATTED_PATH = project_path("qa", "qa_formatted_data.json")
IMAGES_DIR = project_path("event", "images")
VOICE_MIXED_DIR = project_path("event", "voice_mixed_000_002")
VOICE_MESSAGE_DIR = project_path("event", "voice_message_000_002")
MANIFEST_PATH = project_path("profile", "generated_portraits", "manifest.json")
PROFILES_PATH = project_path("profile", "profiles_with_anchors_with_images_entity.json")
OUTPUT_HTML = project_path("event", "inspect_images_by_person.html")

# HTML 所在目录（所有相对路径以此为基准）
_HTML_DIR = OUTPUT_HTML.parent.resolve()


def _build_preference_maps(profiles: list) -> list[dict[str, dict]]:
    """
    为每个 p_id 构建 category_id → pref_obj 的映射，用于从 reflected id 还原偏好详情。
    category_id 规则与生成侧保持一致：
      - 顶层分类：{CategoryName}-{idx}
      - Basic.Relationship / Basic.Pets：Relationship-{idx} / Pets-{idx}
    """
    top_cats = ['FoodAndDrink', 'HomeAndSpace', 'BodyAndHealth',
                'HobbiesAndEntertainment', 'WorkAndLearning', 'MobilityAndTravel']
    maps: list[dict[str, dict]] = []
    for profile in profiles:
        mp: dict[str, dict] = {}
        if not isinstance(profile, dict):
            maps.append(mp)
            continue

        # 顶层偏好（结构：[{subcategory, preference, expression_type, evidence_sources, analysis, entity_anchors, ...}, ...]）
        for cat in top_cats:
            arr = profile.get(cat) or []
            if not isinstance(arr, list):
                continue
            for idx, rec in enumerate(arr):
                if not isinstance(rec, dict):
                    continue
                cid = f"{cat}-{idx}"
                mp[cid] = {
                    "category": cid,
                    "subcategory": rec.get("subcategory", ""),
                    "content": rec.get("preference", ""),
                    "sources": rec.get("evidence_sources") or rec.get("sources") or [],
                    "rationale": rec.get("analysis") or rec.get("rationale") or [],
                    "entity_anchors": rec.get("entity_anchors") or [],
                }

        # Basic 偏好
        basic = profile.get("Basic") or {}
        for bcat in ("Relationship", "Pets"):
            arr = (basic.get(bcat) or [])
            if not isinstance(arr, list):
                continue
            for idx, rec in enumerate(arr):
                if not isinstance(rec, dict):
                    continue
                cid = f"{bcat}-{idx}"
                # Relationship / Pets 字段结构不同，尽量兼容
                content = rec.get("info") or rec.get("name") or rec.get("relation") or ""
                mp[cid] = {
                    "category": cid,
                    "subcategory": rec.get("relation", "") if bcat == "Relationship" else "",
                    "content": content,
                    "sources": rec.get("evidence_sources") or rec.get("sources") or [],
                    "rationale": rec.get("analysis") or rec.get("rationale") or [],
                    "entity_anchors": rec.get("entity_anchors") or [],
                }
        maps.append(mp)
    return maps


def _collect_events_from_profiles(entity_profiles: list) -> list[dict]:
    """
    从 profile/..._with_events.json 收集事件，按 (p_id, task_id) 去重。
    遍历顺序与 gen_images_from_descriptions.py 保持一致：TOP_LEVEL_CATEGORIES 在前，BASIC 在后；
    同一个 (p_id, task_id) 首次出现的事件优先。
    """
    top_cats = ['FoodAndDrink', 'HomeAndSpace', 'BodyAndHealth',
                'HobbiesAndEntertainment', 'WorkAndLearning', 'MobilityAndTravel']
    basic_cats = ['Relationship', 'Pets']

    pref_maps = _build_preference_maps(entity_profiles)

    out: list[dict] = []
    seen: set[tuple[int, str]] = set()

    for p_id, profile in enumerate(entity_profiles):
        if not isinstance(profile, dict):
            continue
        mp = pref_maps[p_id] if p_id < len(pref_maps) else {}

        def _iter_records(cat: str):
            if cat in basic_cats:
                return ((profile.get("Basic") or {}).get(cat) or [])
            return (profile.get(cat) or [])

        for cat in top_cats + basic_cats:
            records = _iter_records(cat)
            if not isinstance(records, list):
                continue
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                for evt in (rec.get("events") or []):
                    if not isinstance(evt, dict):
                        continue
                    task_id = str(evt.get("task_id") or "").strip()
                    if not task_id:
                        continue
                    key = (p_id, task_id)
                    if key in seen:
                        continue
                    seen.add(key)

                    event_obj = {
                        "scene_description": evt.get("scene_description", ""),
                        "user_shared_image_description": evt.get("user_shared_image_description", "none"),
                        "background_audio_info": evt.get("background_audio_info", "none"),
                        "human_speech_content": evt.get("human_speech_content", "none"),
                        "entity_anchor": evt.get("entity_anchor", "none"),
                        "explicit_preferences_reflected": evt.get("explicit_preferences_reflected") or [],
                        "implicit_preferences_reflected": evt.get("implicit_preferences_reflected") or [],
                        "entity_anchors": evt.get("entity_anchors") or [],
                    }

                    exp_ids = list(event_obj.get("explicit_preferences_reflected") or [])
                    imp_ids = list(event_obj.get("implicit_preferences_reflected") or [])
                    explicit_prefs = [mp[i] for i in exp_ids if i in mp]
                    implicit_prefs = [mp[i] for i in imp_ids if i in mp]

                    out.append({
                        "p_id": p_id,
                        "task_id": task_id,
                        "event": event_obj,
                        "explicit_preferences": explicit_prefs,
                        "implicit_preferences": implicit_prefs,
                        # 兼容旧逻辑（tab 标题用）
                        "profile_str": f"name: {((profile.get('Basic') or {}).get('name') or '').strip()}",
                    })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# 通用工具
# ─────────────────────────────────────────────────────────────────────────────

def img_to_rel_src(img_path: Path) -> str:
    """返回相对于 HTML 目录的路径字符串（用斜杠分隔，便于浏览器识别）。"""
    if not img_path or not img_path.exists():
        return ""
    try:
        rel = Path(img_path).resolve().relative_to(_HTML_DIR)
        return rel.as_posix()          # 统一用正斜杠
    except ValueError:
        # 无法取相对路径时回退到绝对路径（file:/// 仍可访问）
        return Path(img_path).resolve().as_posix()


def asset_path_to_rel_src(path_text: str) -> str:
    """把 qa_formatted_data 中保存的相对/绝对资源路径转成 HTML 可播放路径。"""
    raw = str(path_text or "").strip()
    if not raw:
        return ""
    p = resolve_path(raw)
    return img_to_rel_src(p)


def find_event_image(p_id: int, task_id: str) -> Path:
    safe_tid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(task_id))
    return IMAGES_DIR / f"pid_{p_id:04d}_task_{safe_tid}.png"


def escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def safe_stem(text: str) -> str:
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-\u4e00-\u9fa5]", "", text)
    return text[:40] if text else "unknown"


def row(label: str, value: str, cls: str = "") -> str:
    if not value or value == "none":
        return ""
    return f'<tr class="{cls}"><th>{label}</th><td>{value}</td></tr>'


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — 对话 & 音频
# ─────────────────────────────────────────────────────────────────────────────

def find_audio(p_id: int, task_id: str, turn_idx: int) -> str:
    """
    优先返回混音文件（TTS + bgm）；若不存在则返回纯 TTS 文件。
    路径均相对于 HTML 所在目录（event/）。
    voice_mixed_000_002 子目录格式：{p_id}-{task_id}/
    """
    mixed_sub  = VOICE_MIXED_DIR / f"{p_id}-{task_id}" / f"{task_id}_turn{turn_idx}.wav"
    voice_only = VOICE_MESSAGE_DIR / f"{task_id}_turn{turn_idx}.wav"
    if mixed_sub.exists():
        return img_to_rel_src(mixed_sub)
    if voice_only.exists():
        return img_to_rel_src(voice_only)
    return ""


def build_dialogue_html(dialogue_entry: dict, p_id: int, task_id: str) -> str:
    """把一条 dialogue 记录渲染成 HTML 对话气泡列表。"""
    event_data    = dialogue_entry.get("event") or {}
    dialog        = event_data.get("dialog") or []
    image_indices = set(event_data.get("image_turn_indices") or [])

    if not dialog:
        return '<p class="dlg-empty">（暂无对话数据）</p>'

    turns_html = []
    for idx, turn in enumerate(dialog):
        role       = turn.get("role", "")
        content    = escape(turn.get("content", ""))
        bg_audio   = turn.get("background_audio", "")

        role_cls   = "turn-user" if role == "user" else "turn-ai"
        role_label = "用户" if role == "user" else "AI"

        img_mark   = '<span class="turn-img-mark" title="图像轮次">📷</span>' \
                     if idx in image_indices else ""
        audio_badge = (
            f'<span class="turn-audio-badge">🎵 {escape(bg_audio)}</span>'
            if bg_audio else ""
        )

        # 优先使用 qa_formatted_data.json / dialogue_with_assets 中已经写好的 audio_path；
        # 若没有，再按旧规则从 voice_mixed/voice_message 目录推断。
        audio_tag = ""
        audio_path = str(turn.get("audio_path") or "").strip()
        if role == "user" and (bg_audio or audio_path):
            src = asset_path_to_rel_src(audio_path) if audio_path else ""
            if not src and bg_audio:
                src = find_audio(p_id, task_id, idx)
            if src:
                audio_tag = (
                    f'<audio class="turn-audio" controls preload="none">'
                    f'<source src="{src}" type="audio/wav"></audio>'
                )

        turns_html.append(f"""
    <div class="turn {role_cls}">
      <div class="turn-meta">
        <span class="turn-role-badge">{role_label}</span>
        <span class="turn-idx">#{idx}</span>
        {img_mark}{audio_badge}
      </div>
      <div class="turn-content">{content}</div>
      {audio_tag}
    </div>""")

    return "\n".join(turns_html)


def load_qa_formatted_dialogues(path: Path) -> dict[str, dict]:
    """从 qa_formatted_data.json 读取带 audio_path/image_path 的 event dialog。"""
    if not path.exists():
        return {}
    records = load_json_or_jsonl(path)
    out: dict[str, dict] = {}
    for profile in records:
        if not isinstance(profile, dict):
            continue
        p_id = profile.get("p_id")
        for event in profile.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            task_id = str(event.get("task_id") or "").strip()
            if not task_id:
                continue
            out[task_id] = {
                "p_id": p_id,
                "task_id": task_id,
                "event": event,
            }
    return out


def enrich_dialogue_audio_paths(dialogue_entry: dict, qa_entry: dict) -> bool:
    """Only copy audio metadata from qa_formatted_data; keep dialogue text unchanged."""
    dialog = (dialogue_entry.get("event") or {}).get("dialog") or []
    qa_dialog = (qa_entry.get("event") or {}).get("dialog") or []
    if not dialog or not qa_dialog:
        return False
    changed = False
    for turn, qa_turn in zip(dialog, qa_dialog):
        if not isinstance(turn, dict) or not isinstance(qa_turn, dict):
            continue
        if turn.get("role") != qa_turn.get("role"):
            continue
        for key in ("audio_path", "audio_source"):
            if not turn.get(key) and qa_turn.get(key):
                turn[key] = qa_turn[key]
                changed = True
    return changed


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — 事件图片
# ─────────────────────────────────────────────────────────────────────────────

def event_status_badge(img_path: Path) -> str:
    if img_path and img_path.exists():
        return '<span class="badge ok">✓ 已生成</span>'
    elif img_path:
        return '<span class="badge missing">⚠ 文件缺失</span>'
    return '<span class="badge noimg">— 无图事件</span>'


def pref_list_html(prefs: list) -> str:
    if not prefs:
        return ""
    rows = []
    for p in prefs:
        cat     = escape(p.get("category", ""))
        subcat  = escape(p.get("subcategory", ""))
        content = escape(p.get("content", ""))
        sources = escape(", ".join(str(x) for x in (p.get("sources") or [])))
        anchors = p.get("entity_anchors")
        if anchors is None:
            anchors = p.get("entity_anchor")
        if isinstance(anchors, list):
            anchors_text = "、".join(str(x) for x in anchors if str(x).strip())
        elif anchors:
            anchors_text = str(anchors)
        else:
            anchors_text = ""
        anchors_html = (
            f'<div class="pref-anchors"><b>entity anchors:</b> {escape(anchors_text)}</div>'
            if anchors_text else ""
        )
        rat_items = p.get("rationale") or []
        rat_html = "".join(f'<li>{escape(r)}</li>' for r in rat_items)
        rows.append(f"""
          <div class="pref-item">
            <div class="pref-header">
              <span class="pref-cat">{cat}</span>
              <span class="pref-sub">{subcat}</span>
              <span class="pref-src">[{sources}]</span>
            </div>
            <div class="pref-content">{content}</div>
            {anchors_html}
            {"<ul class='rationale'>" + rat_html + "</ul>" if rat_html else ""}
          </div>""")
    return "\n".join(rows)


def build_event_card(entry: dict, dialogue_entry: dict = None,
                     img_desc_override: str = None) -> str:
    p_id    = entry["p_id"]
    task_id = entry.get("task_id", "")
    group_id = entry.get("group_id", "")
    event   = entry.get("event") or {}

    scene    = escape(event.get("scene_description", ""))
    # 优先使用从 entity profiles 中读取的描述，fallback 到 events 自身的字段
    _raw_img_desc = img_desc_override if img_desc_override is not None \
        else event.get("user_shared_image_description", "none")
    img_desc = escape(_raw_img_desc)
    bg_audio = escape(event.get("background_audio_info", "none"))
    speech   = escape(event.get("human_speech_content", "none"))
    anchor   = escape(event.get("entity_anchor", "none"))
    recommended_scene = escape(entry.get("recommended_main_scene", ""))
    integration_guidance = escape(entry.get("implicit_integration_guidance", ""))
    selected_anchors_raw = event.get("entity_anchors") or entry.get("entity_anchors") or []
    if isinstance(selected_anchors_raw, list):
        selected_anchors = escape("、".join(str(x) for x in selected_anchors_raw if str(x).strip()))
    else:
        selected_anchors = escape(str(selected_anchors_raw))

    explicit_prefs = entry.get("explicit_preferences") or []
    implicit_prefs = entry.get("implicit_preferences") or []
    reflected_exp  = set(event.get("explicit_preferences_reflected") or [])
    reflected_imp  = set(event.get("implicit_preferences_reflected") or [])

    img_path   = find_event_image(p_id, task_id)
    rel_src    = img_to_rel_src(img_path)
    has_visual = img_desc and img_desc != "none"

    img_tag = (
        f'<img src="{rel_src}" alt="{escape(task_id)}">'
        if rel_src
        else ('<div class="no-img">无图片</div>' if has_visual
              else '<div class="no-img no-visual">无图像事件</div>')
    )

    def pills(ids):
        return " ".join(f'<span class="pill">{escape(i)}</span>' for i in sorted(ids)) or "—"

    table = f"""
    <table class="meta">
      {row("task_id", f'<code>{escape(task_id)}</code>')}
      {row("group_id", f'<code>{escape(str(group_id))}</code>' if group_id != "" else "")}
      {row("推荐主场景", recommended_scene, "scene")}
      {row("隐式潜入指导", integration_guidance, "guidance")}
      {row("场景描述", scene, "scene")}
      {row("图像描述", img_desc, "imgdesc")}
      {row("背景音", bg_audio, "audio")}
      {row("语音内容", speech)}
      {row("实体锚", anchor, "anchor")}
      {row("选中 anchors", selected_anchors, "anchor")}
    </table>"""

    prefs_block = f"""
    <div class="pref-section">
      <div class="pref-col">
        <div class="pref-title">显式偏好
          <span class="reflected-label">体现：{pills(reflected_exp)}</span>
        </div>
        {pref_list_html(explicit_prefs) or '<span class="empty-prefs">—</span>'}
      </div>
      <div class="pref-col">
        <div class="pref-title">隐式偏好
          <span class="reflected-label">体现：{pills(reflected_imp)}</span>
        </div>
        {pref_list_html(implicit_prefs) or '<span class="empty-prefs">—</span>'}
      </div>
    </div>"""

    badge = event_status_badge(img_path if has_visual else None)

    # ── 对话区块 ──────────────────────────────────────────────────────────────
    if dialogue_entry:
        dlg_turns = (dialogue_entry.get("event") or {}).get("dialog") or []
        n_turns   = len(dlg_turns)
        n_audio   = sum(
            1 for i, t in enumerate(dlg_turns)
            if t.get("role") == "user" and t.get("background_audio")
        )
        dlg_body_id  = f"dlg-{escape(task_id)}"
        dlg_arrow_id = f"dlg-arrow-{escape(task_id)}"
        dlg_content  = build_dialogue_html(dialogue_entry, p_id, task_id)
        dlg_counts   = f"{n_turns} 轮"
        if n_audio:
            dlg_counts += f"，{n_audio} 条语音"
        dialogue_block = f"""
    <div class="dlg-section">
      <div class="dlg-header" onclick="toggleSection('{dlg_body_id}','{dlg_arrow_id}')">
        <span>💬 对话记录</span>
        <span class="section-count">{dlg_counts}</span>
        <span class="arrow closed" id="{dlg_arrow_id}">▼</span>
      </div>
      <div class="dlg-body" id="{dlg_body_id}" style="display:none">
        {dlg_content}
      </div>
    </div>"""
    else:
        dialogue_block = ""

    return f"""
  <div class="card" id="card-{escape(task_id)}">
    <div class="card-header">
      <span class="card-title">{escape(task_id)}</span>
      {badge}
    </div>
    <div class="card-body">
      <div class="img-col"><div class="img-box">{img_tag}</div></div>
      <div class="info-col">{table}{prefs_block}</div>
    </div>
    {dialogue_block}
  </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — 定妆照
# ─────────────────────────────────────────────────────────────────────────────

PORTRAIT_TYPE_LABELS = {
    "Relationship": "人物关系",
    "Pets":         "宠物",
    "Items":        "物品",
}

# 不同类型的卡片背景色
PORTRAIT_TYPE_COLORS = {
    "Relationship": "#f0f4ff",
    "Pets":         "#f0fdf4",
    "Items":        "#fffbeb",
}
PORTRAIT_TYPE_BORDER = {
    "Relationship": "#c7d2fe",
    "Pets":         "#bbf7d0",
    "Items":        "#fde68a",
}


def portrait_badge(img_path: Path) -> str:
    if img_path and img_path.exists():
        return '<span class="badge ok">✓ 已生成</span>'
    return '<span class="badge missing">⚠ 文件缺失</span>'


def build_portrait_file_index() -> dict[tuple[int, str, str], list[Path]]:
    """Index generated portrait files by (p_id, type, label_stem)."""
    portraits_dir = MANIFEST_PATH.parent
    index: dict[tuple[int, str, str], list[Path]] = defaultdict(list)
    pattern = re.compile(
        r"^profile_(\d+)_(.+?)_(Relationship|Pets|Items)_(.+)_(\d+)\.png$",
        re.IGNORECASE,
    )
    if not portraits_dir.exists():
        return index
    for img_file in sorted(portraits_dir.glob("*.png")):
        match = pattern.match(img_file.name)
        if not match:
            continue
        p_id = int(match.group(1))
        ptype = match.group(3)
        label_stem = safe_stem(match.group(4))
        index[(p_id, ptype, label_stem)].append(img_file)
    return index


def collect_portraits_from_profiles(profiles: list) -> dict[int, list]:
    """Build portrait cards from profile objects' own img_path fields.

    This avoids pairing images and text by filename index, which can become
    stale after Items are regenerated or reordered.
    """
    portrait_by_pid: dict[int, list] = defaultdict(list)
    file_index = build_portrait_file_index()
    for p_id, profile in enumerate(profiles):
        if not isinstance(profile, dict):
            continue
        real_pid = profile.get("p_id", p_id)
        basic = profile.get("Basic") or {}

        for ptype, records in [
            ("Relationship", basic.get("Relationship") or []),
            ("Pets", basic.get("Pets") or []),
            ("Items", profile.get("Items") or []),
        ]:
            if not isinstance(records, list):
                continue
            for idx, rec in enumerate(records):
                if not isinstance(rec, dict):
                    continue
                if ptype == "Relationship":
                    source_name = rec.get("name") or rec.get("relation") or f"{ptype}-{idx}"
                elif ptype == "Pets":
                    source_name = rec.get("name") or f"{ptype}-{idx}"
                else:
                    source_name = rec.get("description") or f"{ptype}-{idx}"
                label_stem = safe_stem(source_name)
                matched_files = file_index.get((int(real_pid), ptype, label_stem), [])
                # Prefer filename-label matching. The img_path field can be stale after Items
                # are regenerated/reordered and old paths are copied by index.
                if matched_files:
                    img_path = str(matched_files[0])
                else:
                    img_path = str(rec.get("img_path") or "").strip()
                if not img_path:
                    continue
                portrait_by_pid[int(real_pid)].append({
                    "profile_id": int(real_pid),
                    "type": ptype,
                    "index": idx,
                    "file": img_path,
                    "status": "ok" if img_path and Path(img_path).exists() else "missing",
                    "source_name": source_name,
                })

    _TYPE_ORDER = {"Relationship": 0, "Pets": 1, "Items": 2}
    for pid in portrait_by_pid:
        portrait_by_pid[pid].sort(
            key=lambda e: (_TYPE_ORDER.get(e["type"], 9), e["index"])
        )
    return portrait_by_pid


def build_portrait_card(manifest_entry: dict, profile: dict) -> str:
    """
    根据 manifest 条目和对应 profile，构建定妆照卡片。
    """
    ptype    = manifest_entry.get("type", "")
    idx      = manifest_entry.get("index", 0)
    file_raw = manifest_entry.get("file", "")
    name     = escape(manifest_entry.get("source_name", "") or "")

    # file 可能是绝对路径（扫目录所得）或相对路径（manifest 旧格式），统一处理
    file_path = Path(file_raw)
    rel_src   = img_to_rel_src(file_path)

    img_tag = (
        f'<img src="{rel_src}" alt="{name}">'
        if rel_src
        else '<div class="no-img">无图片</div>'
    )

    # 从 profile 数据提取详细信息
    basic = profile.get("Basic", {}) or {}
    detail_html = ""

    if ptype == "Relationship":
        rels = basic.get("Relationship", []) or []
        if idx < len(rels):
            rel = rels[idx]
            relation   = escape(rel.get("relation", ""))
            rel_name   = escape(rel.get("name", ""))
            info       = escape(rel.get("info", ""))
            appearance = escape(rel.get("appearance", ""))
            evidence   = ", ".join(rel.get("evidence_sources") or rel.get("sources") or [])
            detail_html = f"""
            <table class="meta">
              {row("关系", relation)}
              {row("姓名", rel_name)}
              {row("外貌", appearance, "scene")}
              {row("信息", info)}
              {row("来源", evidence)}
            </table>"""

    elif ptype == "Pets":
        pets = basic.get("Pets", []) or []
        if idx < len(pets):
            pet = pets[idx]
            pet_name   = escape(pet.get("name", ""))
            info       = escape(pet.get("info", ""))
            appearance = escape(pet.get("appearance", ""))
            detail_html = f"""
            <table class="meta">
              {row("名字", pet_name)}
              {row("外貌", appearance, "scene")}
              {row("信息", info)}
            </table>"""

    elif ptype == "Items":
        items = profile.get("Items", []) or []
        if idx < len(items):
            item = items[idx]
            desc      = escape(item.get("description", ""))
            item_events = item.get("events") or []
            first_event = item_events[0] if item_events and isinstance(item_events[0], dict) else {}
            src_tid_raw = item.get("source_task_id") or first_event.get("task_id") or ""
            scene_raw = item.get("event") or first_event.get("scene_description") or ""
            src_tid   = escape(str(src_tid_raw))
            evt_scene = escape(str(scene_raw)[:200] + ("…" if len(str(scene_raw)) > 200 else ""))
            detail_html = f"""
            <table class="meta">
              {row("描述", desc, "scene")}
              {row("来源事件", f'<code>{src_tid}</code>' if src_tid else "")}
              {row("事件场景", evt_scene)}
            </table>"""

    type_label = PORTRAIT_TYPE_LABELS.get(ptype, ptype)
    bg_color   = PORTRAIT_TYPE_COLORS.get(ptype, "#fff")
    bd_color   = PORTRAIT_TYPE_BORDER.get(ptype, "#eee")
    card_id    = f"portrait-p{manifest_entry.get('profile_id',0)}-{ptype}-{idx}"

    return f"""
  <div class="portrait-card" id="{card_id}"
       style="background:{bg_color};border-color:{bd_color}">
    <div class="portrait-header">
      <span class="portrait-type-tag">{type_label}</span>
      <span class="card-title">{name or f"{ptype}-{idx}"}</span>
      {portrait_badge(file_path)}
    </div>
    <div class="portrait-body">
      <div class="portrait-img-box">{img_tag}</div>
      <div class="info-col">{detail_html}</div>
    </div>
  </div>"""


def build_portraits_section(pid: int, manifest_entries: list, profile: dict) -> str:
    """为单个人物构建整个定妆照区块（按类型分组）。"""
    by_type: dict[str, list] = {"Relationship": [], "Pets": [], "Items": []}
    for entry in manifest_entries:
        t = entry.get("type", "")
        if t in by_type:
            by_type[t].append(entry)

    sections_html = ""
    for ptype, entries in by_type.items():
        if not entries:
            continue
        label = PORTRAIT_TYPE_LABELS[ptype]
        cards = "\n".join(build_portrait_card(e, profile) for e in entries)
        sections_html += f"""
    <div class="portrait-group">
      <div class="portrait-group-title">{label}
        <span class="portrait-group-count">{len(entries)} 张</span>
      </div>
      <div class="portrait-grid">{cards}</div>
    </div>"""

    if not sections_html:
        return '<p class="empty">暂无定妆照</p>'
    return sections_html


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None):
    global EVENTS_PATH, DIALOGUE_PATH, QA_FORMATTED_PATH, IMAGES_DIR
    global VOICE_MIXED_DIR, VOICE_MESSAGE_DIR, MANIFEST_PATH, PROFILES_PATH
    global OUTPUT_HTML, _HTML_DIR
    parser = argparse.ArgumentParser(description="按人物生成事件、图片、对话和音频检查页")
    parser.add_argument("--events", default=str(EVENTS_PATH))
    parser.add_argument("--dialogue", default=str(DIALOGUE_PATH))
    parser.add_argument("--qa", default=str(QA_FORMATTED_PATH))
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    parser.add_argument("--voice-mixed-dir", default=str(VOICE_MIXED_DIR))
    parser.add_argument("--voice-message-dir", default=str(VOICE_MESSAGE_DIR))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--profiles", default=str(PROFILES_PATH))
    parser.add_argument("--output", default=str(OUTPUT_HTML))
    args = parser.parse_args(argv)
    EVENTS_PATH = resolve_path(args.events)
    DIALOGUE_PATH = resolve_path(args.dialogue)
    QA_FORMATTED_PATH = resolve_path(args.qa)
    IMAGES_DIR = resolve_path(args.images_dir)
    VOICE_MIXED_DIR = resolve_path(args.voice_mixed_dir)
    VOICE_MESSAGE_DIR = resolve_path(args.voice_message_dir)
    MANIFEST_PATH = resolve_path(args.manifest)
    PROFILES_PATH = resolve_path(args.profiles)
    OUTPUT_HTML = resolve_path(args.output)
    _HTML_DIR = OUTPUT_HTML.parent.resolve()
    # ── 事件数据 ──
    if not EVENTS_PATH.exists():
        raise FileNotFoundError(f"事件文件不存在：{EVENTS_PATH}")
    events = load_json_or_jsonl(EVENTS_PATH)
    events_source_name = EVENTS_PATH.name

    if PROFILES_PATH.exists():
        profiles_data = load_json_or_jsonl(PROFILES_PATH)
    else:
        profiles_data = []

    # events_with_anchors.jsonl 已经直接保存 user_shared_image_description。
    img_desc_by_tid: dict[tuple, str] = {}

    # ── 对话数据（按 task_id 索引）──
    dialogue_by_tid: dict = {}
    dialogue_source_name = DIALOGUE_PATH.name
    if DIALOGUE_PATH.exists():
        dialogues = load_json_or_jsonl(DIALOGUE_PATH)
        for d in dialogues:
            tid = d.get("task_id")
            if tid:
                dialogue_by_tid[tid] = d
    qa_dialogues = load_qa_formatted_dialogues(QA_FORMATTED_PATH)
    if qa_dialogues:
        enriched = 0
        for tid, qa_entry in qa_dialogues.items():
            if tid in dialogue_by_tid and enrich_dialogue_audio_paths(dialogue_by_tid[tid], qa_entry):
                enriched += 1
        dialogue_source_name = f"{DIALOGUE_PATH.name} (+ audio paths from {QA_FORMATTED_PATH.name}: {enriched})"

    groups: dict[int, list] = defaultdict(list)
    names:  dict[int, str]  = {}
    for entry in events:
        p_id = entry.get("p_id", 0)
        groups[p_id].append(entry)
        if p_id not in names:
            # 优先从 profile.Basic.name 获取人物名；fallback 到旧 events 结构的 profile_str
            if p_id < len(profiles_data) and isinstance(profiles_data[p_id], dict):
                nm = ((profiles_data[p_id].get("Basic") or {}).get("name") or "").strip()
                names[p_id] = nm if nm else f"Person {p_id}"
            else:
                m = re.search(r"name:\s*(.+)", entry.get("profile_str", ""))
                names[p_id] = entry.get("profile_name") or (m.group(1).strip() if m else f"Person {p_id}")

    # ── 定妆照数据 ──
    # 直接从 profile 对象的 img_path 字段读取图片，避免用文件名 index 与 Items
    # 数组重新配对造成图片和文字错位。
    portrait_by_pid = collect_portraits_from_profiles(profiles_data)

    person_ids = sorted(set(list(groups.keys()) + list(portrait_by_pid.keys())))

    # ── 统计 ──
    total_events    = len(events)
    total_with_img  = sum(
        1 for e in events
        if (e.get("event") or {}).get("user_shared_image_description", "none") != "none"
    )
    total_generated = sum(
        1 for e in events
        if find_event_image(e["p_id"], e.get("task_id", "")).exists()
    )
    total_portraits  = sum(len(v) for v in portrait_by_pid.values())
    total_dialogues  = len(dialogue_by_tid)
    total_audio_turns = sum(
        1 for d in dialogue_by_tid.values()
        for t in (d.get("event") or {}).get("dialog", [])
        if t.get("role") == "user" and (t.get("background_audio") or t.get("audio_path"))
    )

    # ── Tab 按钮 ──
    tab_buttons = "\n".join(
        f'<button class="tab-btn" onclick="showTab({pid})" id="btn-{pid}">'
        f'P{pid}&nbsp;{escape(names.get(pid, f"Person {pid}"))} '
        f'({len(groups.get(pid, []))}件 / {len(portrait_by_pid.get(pid, []))}照)</button>'
        for pid in person_ids
    )

    # ── Tab 面板 ──
    tab_panels_html = ""
    for pid in person_ids:
        profile = profiles_data[pid] if pid < len(profiles_data) else {}

        event_cards = "\n".join(
            build_event_card(
                e,
                dialogue_by_tid.get(e.get("task_id")),
                img_desc_override=img_desc_by_tid.get((pid, str(e.get("task_id", "")))),
            )
            for e in groups.get(pid, [])
        )
        portraits_html = build_portraits_section(
            pid, portrait_by_pid.get(pid, []), profile
        )

        tab_panels_html += f"""
<div class="tab-panel" id="panel-{pid}">
  <!-- ──── Section 1: 事件图片 ──── -->
  <div class="section-header" onclick="toggleSection('events-{pid}','arrow-ev-{pid}')">
    <span>🖼 事件图片</span>
    <span class="section-count">{len(groups.get(pid, []))} 条</span>
    <span class="arrow" id="arrow-ev-{pid}">▼</span>
  </div>
  <div class="section-body" id="events-{pid}">
    {event_cards or '<p class="empty">暂无事件</p>'}
  </div>

  <!-- ──── Section 2: 定妆照 ──── -->
  <div class="section-header" onclick="toggleSection('portraits-{pid}','arrow-pt-{pid}')">
    <span>🎨 定妆照</span>
    <span class="section-count">{len(portrait_by_pid.get(pid, []))} 张</span>
    <span class="arrow" id="arrow-pt-{pid}">▼</span>
  </div>
  <div class="section-body" id="portraits-{pid}">
    {portraits_html}
  </div>
</div>"""

    # ── HTML ──
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>图片检查 — 按人物</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', 'PingFang SC', sans-serif; background: #f0f2f5; color: #1a1a2e; }}

  /* ── 顶栏 ── */
  header {{ background: #1a1a2e; color: #eee; padding: 14px 24px;
            display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap; }}
  header h1 {{ font-size: 18px; font-weight: 700; }}
  header .summary {{ font-size: 12px; color: #aaa; }}

  /* ── 人物 Tab ── */
  .tabs {{ display: flex; gap: 8px; padding: 12px 24px; background: #fff;
           border-bottom: 2px solid #e5e7eb; flex-wrap: wrap;
           position: sticky; top: 0; z-index: 100; }}
  .tab-btn {{ padding: 6px 18px; border: 1.5px solid #c7c9cf; border-radius: 20px;
              background: #f5f6f8; cursor: pointer; font-size: 13px; transition: .15s; }}
  .tab-btn:hover  {{ background: #e8eaf6; border-color: #9fa8da; }}
  .tab-btn.active {{ background: #1a1a2e; color: #fff; border-color: #1a1a2e; font-weight: 600; }}
  .tab-panel {{ display: none; padding: 16px 24px; }}
  .tab-panel.active {{ display: block; }}

  /* ── 区块折叠 ── */
  .section-header {{ display: flex; align-items: center; gap: 10px; cursor: pointer;
                     background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
                     padding: 12px 18px; margin-bottom: 10px; user-select: none;
                     font-size: 15px; font-weight: 700; color: #1a1a2e; }}
  .section-header:hover {{ background: #f5f6f8; }}
  .section-count {{ font-size: 12px; color: #6b7280; font-weight: 400; background: #f3f4f6;
                    padding: 2px 8px; border-radius: 10px; }}
  .arrow {{ margin-left: auto; font-size: 12px; color: #9ca3af; transition: transform .2s; }}
  .arrow.closed {{ transform: rotate(-90deg); }}
  .section-body {{ margin-bottom: 24px; }}

  /* ── 事件卡片 ── */
  .card {{ background: #fff; border-radius: 12px; margin-bottom: 14px;
           box-shadow: 0 2px 8px rgba(0,0,0,.07); overflow: hidden; }}
  .card-header {{ display: flex; align-items: center; gap: 10px; padding: 9px 16px;
                  background: #f8f9fb; border-bottom: 1px solid #eee; }}
  .card-title {{ font-size: 13px; font-weight: 600; font-family: monospace; color: #444; }}
  .card-body {{ display: flex; gap: 0; }}
  .img-col {{ flex: 0 0 290px; padding: 12px; background: #fafafa;
              border-right: 1px solid #eee; display: flex; flex-direction: column;
              align-items: center; }}
  .img-box img {{ width: 266px; height: 266px; object-fit: contain;
                  border-radius: 8px; border: 1px solid #ddd; }}
  .no-img {{ width: 266px; height: 180px; display: flex; align-items: center;
             justify-content: center; background: #f3f4f6; border-radius: 8px;
             color: #9ca3af; font-size: 13px; border: 1px dashed #d1d5db; }}
  .no-visual {{ background: #fafafa; color: #cbd5e1; }}
  .info-col {{ flex: 1; padding: 12px 16px; overflow: auto; min-width: 0; }}

  /* ── 状态徽章 ── */
  .badge {{ font-size: 11px; padding: 2px 9px; border-radius: 10px; font-weight: 600; }}
  .badge.ok      {{ background: #d1fae5; color: #065f46; }}
  .badge.missing {{ background: #fef3c7; color: #92400e; }}
  .badge.noimg   {{ background: #f3f4f6; color: #6b7280; }}

  /* ── 元信息表格 ── */
  table.meta {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px; }}
  table.meta th {{ width: 64px; padding: 5px 8px; text-align: right; color: #9ca3af;
                   font-weight: 500; white-space: nowrap; vertical-align: top; }}
  table.meta td {{ padding: 5px 8px; line-height: 1.65; word-break: break-all; }}
  table.meta tr {{ border-bottom: 1px solid #f3f4f6; }}
  tr.scene td  {{ font-weight: 600; color: #1e3a5f; }}
  tr.imgdesc td {{ color: #374151; }}
  tr.audio td  {{ color: #7c3aed; }}
  tr.anchor td {{ color: #b45309; font-weight: 500; }}
  code {{ font-size: 11px; background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }}

  /* ── 偏好区块 ── */
  .pref-section {{ display: flex; gap: 10px; }}
  .pref-col {{ flex: 1; min-width: 0; }}
  .pref-title {{ font-size: 11px; font-weight: 700; color: #6b7280; text-transform: uppercase;
                 letter-spacing: .05em; margin-bottom: 5px;
                 display: flex; align-items: baseline; flex-wrap: wrap; gap: 5px; }}
  .reflected-label {{ font-weight: 400; text-transform: none; letter-spacing: 0; color: #9ca3af; }}
  .pill {{ display: inline-block; font-size: 10px; padding: 1px 6px;
           background: #e0e7ff; color: #3730a3; border-radius: 8px;
           font-family: monospace; white-space: nowrap; }}
  .pref-item {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
                padding: 7px 9px; margin-bottom: 5px; font-size: 12px; line-height: 1.6; }}
  .pref-header {{ display: flex; flex-wrap: wrap; gap: 5px; align-items: baseline; margin-bottom: 3px; }}
  .pref-cat {{ font-family: monospace; font-size: 11px; background: #e0e7ff;
               color: #3730a3; padding: 1px 6px; border-radius: 4px; }}
  .pref-sub {{ font-weight: 600; color: #374151; }}
  .pref-src {{ font-size: 11px; color: #9ca3af; }}
  .pref-content {{ color: #374151; margin-bottom: 3px; }}
  ul.rationale {{ margin-left: 14px; color: #6b7280; font-size: 11px; line-height: 1.5; }}
  ul.rationale li {{ margin-bottom: 2px; }}
  .empty-prefs {{ font-size: 12px; color: #d1d5db; }}

  /* ── 定妆照区块 ── */
  .portrait-group {{ margin-bottom: 20px; }}
  .portrait-group-title {{ font-size: 13px; font-weight: 700; color: #374151;
                           margin-bottom: 10px; padding: 6px 0;
                           border-bottom: 2px solid #e5e7eb;
                           display: flex; align-items: center; gap: 8px; }}
  .portrait-group-count {{ font-size: 11px; color: #9ca3af; font-weight: 400;
                           background: #f3f4f6; padding: 1px 7px; border-radius: 8px; }}
  .portrait-grid {{ display: flex; flex-wrap: wrap; gap: 14px; }}
  .portrait-card {{ border: 1.5px solid #e5e7eb; border-radius: 12px; overflow: hidden;
                    width: 260px; flex-shrink: 0;
                    box-shadow: 0 1px 6px rgba(0,0,0,.06); }}
  .portrait-header {{ display: flex; align-items: center; gap: 8px; padding: 8px 12px;
                      border-bottom: 1px solid rgba(0,0,0,.06); }}
  .portrait-type-tag {{ font-size: 10px; font-weight: 700; padding: 2px 7px;
                        border-radius: 8px; background: #e0e7ff; color: #3730a3; }}
  .portrait-body {{ display: flex; flex-direction: column; padding: 10px; gap: 10px; }}
  .portrait-img-box img {{ width: 236px; height: 236px; object-fit: contain;
                           border-radius: 8px; border: 1px solid #e5e7eb; display: block; }}
  .portrait-img-box .no-img {{ width: 236px; height: 180px; }}

  .empty {{ color: #9ca3af; font-size: 13px; padding: 24px 0; text-align: center; }}

  /* ── 对话区块 ── */
  .dlg-section {{ border-top: 1px solid #f0f0f0; }}
  .dlg-header  {{ display: flex; align-items: center; gap: 10px; cursor: pointer;
                  padding: 8px 16px; background: #f8f9fb; user-select: none;
                  font-size: 13px; font-weight: 600; color: #374151; }}
  .dlg-header:hover {{ background: #f0f2f5; }}
  .dlg-body    {{ padding: 10px 16px; max-height: 480px; overflow-y: auto;
                  display: flex; flex-direction: column; gap: 6px; }}
  .turn-user   {{ background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px;
                  padding: 8px 12px; }}
  .turn-ai     {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px;
                  padding: 8px 12px; }}
  .turn-meta   {{ display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
                  margin-bottom: 4px; }}
  .turn-role-badge {{ font-size: 10px; font-weight: 700; padding: 1px 7px;
                      border-radius: 8px; }}
  .turn-user .turn-role-badge {{ background: #dbeafe; color: #1d4ed8; }}
  .turn-ai   .turn-role-badge {{ background: #f3f4f6; color: #6b7280; }}
  .turn-idx  {{ font-size: 10px; color: #9ca3af; font-family: monospace; }}
  .turn-img-mark  {{ font-size: 13px; }}
  .turn-audio-badge {{ font-size: 10px; background: #ede9fe; color: #6d28d9;
                       padding: 1px 7px; border-radius: 8px; white-space: nowrap; }}
  .turn-content {{ font-size: 13px; line-height: 1.65; color: #1f2937;
                   word-break: break-all; margin-bottom: 4px; }}
  .turn-audio   {{ width: 100%; height: 32px; margin-top: 4px; }}
  .dlg-empty    {{ color: #9ca3af; font-size: 12px; padding: 8px; }}

  @media (max-width: 800px) {{
    .card-body {{ flex-direction: column; }}
    .img-col {{ border-right: none; border-bottom: 1px solid #eee; flex: none; }}
    .pref-section {{ flex-direction: column; }}
    .portrait-grid {{ flex-direction: column; }}
    .portrait-card {{ width: 100%; }}
  }}
</style>
</head>
<body>
<header>
  <h1>图片检查 — 按人物</h1>
  <div class="summary">
    事件图片：{total_events} 条 / 含图像 {total_with_img} / 已生成 {total_generated} 张 &nbsp;|&nbsp;
    对话：{total_dialogues} 条 / 语音轮次 {total_audio_turns} 个 &nbsp;|&nbsp;
    定妆照：{total_portraits} 张 &nbsp;|&nbsp;
    数据：{events_source_name} / 对话音频：{dialogue_source_name}
  </div>
</header>

<div class="tabs">
  {tab_buttons}
</div>

{tab_panels_html}

<script>
function showTab(pid) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('panel-' + pid).classList.add('active');
  document.getElementById('btn-' + pid).classList.add('active');
}}

function toggleSection(bodyId, arrowId) {{
  var body  = document.getElementById(bodyId);
  var arrow = document.getElementById(arrowId);
  if (body.style.display === 'none') {{
    body.style.display = '';
    arrow.classList.remove('closed');
  }} else {{
    body.style.display = 'none';
    arrow.classList.add('closed');
  }}
}}

showTab({person_ids[0]});
</script>
</body>
</html>"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size // 1024
    print(f"生成完成：{OUTPUT_HTML}  ({size_kb} KB)")
    print(f"事件 {total_events} 条，含图像 {total_with_img} 条，已生成图片 {total_generated} 张")
    print(f"对话 {total_dialogues} 条，语音轮次 {total_audio_turns} 个")
    print(f"定妆照 {total_portraits} 张")


if __name__ == "__main__":
    main()
