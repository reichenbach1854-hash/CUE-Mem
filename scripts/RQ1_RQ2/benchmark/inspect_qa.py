"""
Human-Annotated QA 可视化检查器。

读取 data/dialog/ 下所有 history_with_qa_p*.json 文件，生成交互式 HTML 页面，
支持按 人物 / category / qa_type 筛选 QA，展开每条 QA 的 memory clue 等。

用法（从 benchmark/ 目录执行）：
    python inspect_qa.py
    # 然后用浏览器打开 inspect_qa.html
"""

from __future__ import annotations
import json
import os
import re
from pathlib import Path
from collections import defaultdict

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmark.paths import DATA_ROOT, QA_ROOT, RESULT_ROOT, resolve_runtime_path

# ---------------------------------------------------------------------------
# 路径配置
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent          # benchmark/
DATA_DIR     = DATA_ROOT
DIALOG_DIR   = DATA_ROOT / "dialog"
IMAGE_DIR    = DATA_ROOT / "image"
VOICE_DIR    = DATA_ROOT / "voice"
OUTPUT_HTML  = resolve_runtime_path(
    os.environ.get("CUE_MEM_INSPECT_OUTPUT", "inspect_qa.html"),
    root=RESULT_ROOT,
)

QA_FORMATTED_DATA = QA_ROOT / "qa_formatted_data_000_002.json"

_HTML_DIR = OUTPUT_HTML.parent.resolve()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def escape(s) -> str:
    return str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rel_src(path: Path) -> str:
    """返回相对于 HTML 所在目录的路径（正斜杠，浏览器友好）。"""
    if not path or not path.exists():
        return ""
    try:
        return path.resolve().relative_to(_HTML_DIR).as_posix()
    except ValueError:
        return Path(os.path.relpath(path.resolve(), _HTML_DIR)).as_posix()


def parse_clue(clue: str) -> tuple[str, str]:
    """
    返回 (clue_type, clue_id)
    clue_type: 'round' | 'image' | 'audio' | 'unknown'
    """
    if re.match(r'^D\d+:\d+$', clue):
        return ('round', clue)
    if clue.endswith('.png'):
        return ('image', clue[:-4])   # e.g. "D03-001"
    if clue.endswith('.wav'):
        return ('audio', clue[:-4])
    return ('unknown', clue)


def find_image(img_id: str) -> Path:
    """根据 image_id（如 D03-001）找到图片文件。"""
    m = re.match(r'^(D\d+)-', img_id)
    if m:
        session = m.group(1)
        p = IMAGE_DIR / session / f"{img_id}.png"
        if p.exists():
            return p
    # fallback: 全局搜索
    for p in IMAGE_DIR.rglob(f"{img_id}.png"):
        return p
    return IMAGE_DIR / f"?/{img_id}.png"   # 占位（不存在）


def find_audio(audio_id: str) -> Path:
    """根据 voice_id（如 D03-001）找到音频文件。"""
    m = re.match(r'^(D\d+)-', audio_id)
    if m:
        session = m.group(1)
        p = VOICE_DIR / session / f"{audio_id}.wav"
        if p.exists():
            return p
    for p in VOICE_DIR.rglob(f"{audio_id}.wav"):
        return p
    return VOICE_DIR / f"?/{audio_id}.wav"


# ---------------------------------------------------------------------------
# 建立索引
# ---------------------------------------------------------------------------

def build_indexes_per_pid(sessions: list) -> dict[int, tuple[dict, dict, dict, dict]]:
    """
    为每个 p_id 分别建立索引，避免不同人物共享 session_id 造成碰撞。

    返回:
      { p_id: (turns_by_round, images_by_id, audio_by_id, sessions_by_id) }
    """
    by_pid: dict[int, tuple[dict, dict, dict, dict]] = {}
    pid_sessions: dict[int, list] = defaultdict(list)
    for s in sessions:
        pid_sessions[s.get("_p_id", 0)].append(s)

    for p_id, p_sessions in pid_sessions.items():
        turns_by_round: dict[str, dict] = {}
        images_by_id: dict[str, Path] = {}
        audio_by_id: dict[str, Path] = {}
        sessions_by_id: dict[str, dict] = {}

        for session in p_sessions:
            sid = session.get("session_id", "")
            if sid:
                sessions_by_id[sid] = session
            for turn in session.get("dialogues", []):
                rnd = turn.get("round", "")
                if rnd:
                    turns_by_round[rnd] = turn

                for img_id, img_path_str in zip(
                    turn.get("image_id", []),
                    turn.get("input_image", [])
                ):
                    if img_id and img_id not in images_by_id:
                        p = (DIALOG_DIR / img_path_str).resolve()
                        if not p.exists():
                            p = find_image(img_id)
                        images_by_id[img_id] = p

                for v_id, v_path_str in zip(
                    turn.get("voice_id", []),
                    turn.get("input_voice_message", [])
                ):
                    if v_id and v_id not in audio_by_id:
                        p = (DIALOG_DIR / v_path_str).resolve()
                        if not p.exists():
                            p = find_audio(v_id)
                        audio_by_id[v_id] = p

        by_pid[p_id] = (turns_by_round, images_by_id, audio_by_id, sessions_by_id)

    return by_pid


def build_indexes_from_formatted_data(formatted_data_path: Path) -> dict[int, tuple[dict, dict, dict, dict]]:
    """
    从 qa_formatted_data JSON 构建 clue 索引。

    每个 event 的 dialog_list 中，.wav / .png 键保存了音频 / 图片的相对路径；
    将它们解析为 (turns_by_round, images_by_id, audio_by_id, sessions_by_id)，
    与 build_indexes_per_pid 返回格式一致。
    """
    with open(formatted_data_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    root = formatted_data_path.resolve().parent.parent   # 项目根目录

    by_pid: dict[int, tuple[dict, dict, dict, dict]] = {}

    for person in data:
        p_id = person.get("p_id", 0)
        turns_by_round: dict[str, dict] = {}
        images_by_id: dict[str, Path] = {}
        audio_by_id: dict[str, Path] = {}
        sessions_by_id: dict[str, dict] = {}

        for event in person.get("events", []):
            sid = event.get("session_id", "")
            scene = event.get("scene_description", "")
            date = scene.split(";")[0].strip() if ";" in scene else ""
            img_desc = event.get("user_shared_image_description", "") or ""

            dialogues: list[dict] = []
            for dl in event.get("dialog_list", []):
                rnd = dl.get("round", "")
                img_ids: list[str] = []
                img_caps: list[str] = []
                v_ids: list[str] = []
                v_caps: list[str] = []
                img_path_map: dict[str, Path] = {}
                audio_path_map: dict[str, Path] = {}

                for key, val in dl.items():
                    if key.endswith(".png"):
                        iid = key[:-4]
                        img_ids.append(iid)
                        img_caps.append(img_desc)
                        p = (root / val).resolve()
                        img_path_map[iid] = p
                        images_by_id[iid] = p
                    elif key.endswith(".wav"):
                        vid = key[:-4]
                        v_ids.append(vid)
                        v_caps.append(dl.get("user", ""))
                        p = (root / val).resolve()
                        audio_path_map[vid] = p
                        audio_by_id[vid] = p

                turn: dict = {
                    "round": rnd,
                    "user": dl.get("user", ""),
                    "user_voice_message_caption": dl.get("user", ""),
                    "assistant": dl.get("assistant", ""),
                    "image_id": img_ids,
                    "image_caption": img_caps,
                    "voice_id": v_ids,
                    "voice_caption": v_caps,
                    "_img_paths": img_path_map,
                    "_audio_paths": audio_path_map,
                }
                dialogues.append(turn)
                if rnd:
                    turns_by_round[rnd] = turn

            if sid:
                sessions_by_id[sid] = {
                    "session_id": sid,
                    "date": date,
                    "dialogues": dialogues,
                }

        by_pid[p_id] = (turns_by_round, images_by_id, audio_by_id, sessions_by_id)

    return by_pid


# ---------------------------------------------------------------------------
# HTML 渲染
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    "preference":     ("#eef2ff", "#6366f1"),
    "entity":         ("#f0fdf4", "#16a34a"),
    "refusal":        ("#fff7ed", "#ea580c"),
    "recommendation": ("#fdf4ff", "#9333ea"),
    "overthinking":   ("#fef2f2", "#dc2626"),
}

QATYPE_COLORS = {
    "explicit": ("#dcfce7", "#166534"),
    "implicit": ("#fef9c3", "#854d0e"),
}


def cat_badge(category: str) -> str:
    bg, fg = CATEGORY_COLORS.get(category, ("#f3f4f6", "#374151"))
    return f'<span class="badge" style="background:{bg};color:{fg}">{escape(category)}</span>'


def qatype_badge(qa_type: str) -> str:
    bg, fg = QATYPE_COLORS.get(qa_type, ("#f3f4f6", "#6b7280"))
    return f'<span class="badge" style="background:{bg};color:{fg}">{escape(qa_type or "—")}</span>'


def render_turn(turn: dict, highlight: bool = False) -> str:
    rnd         = escape(turn.get("round", ""))
    user_cap    = escape(turn.get("user_voice_message_caption", ""))
    assistant   = escape(turn.get("assistant", ""))
    img_ids     = turn.get("image_id", [])
    v_ids       = turn.get("voice_id", [])
    img_caps    = turn.get("image_caption", [])
    v_caps      = turn.get("voice_caption", [])

    highlight_cls = " turn-highlight" if highlight else ""

    # 图片缩略图
    imgs_html = ""
    img_path_map = turn.get("_img_paths", {})
    for i, img_id in enumerate(img_ids):
        p   = img_path_map.get(img_id) or find_image(img_id)
        src = rel_src(p)
        cap = escape(img_caps[i]) if i < len(img_caps) else ""
        if src:
            imgs_html += f'<div class="turn-media"><img src="{src}" alt="{escape(img_id)}" title="{cap}"><div class="media-cap">{cap}</div></div>'
        else:
            imgs_html += f'<div class="turn-media no-file">📷 {escape(img_id)}<br><small>{cap}</small></div>'

    # 音频
    audios_html = ""
    audio_path_map = turn.get("_audio_paths", {})
    for i, v_id in enumerate(v_ids):
        p   = audio_path_map.get(v_id) or find_audio(v_id)
        src = rel_src(p)
        cap = escape(v_caps[i]) if i < len(v_caps) else escape(turn.get("user_voice_message_caption", ""))
        if src:
            audios_html += (
                f'<div class="turn-media">'
                f'<audio controls preload="none"><source src="{src}" type="audio/wav"></audio>'
                f'<div class="media-cap">🎙 {cap}</div>'
                f'</div>'
            )
        else:
            audios_html += f'<div class="turn-media no-file">🔊 {escape(v_id)}<br><small>{cap}</small></div>'

    media_block = f'<div class="turn-media-row">{imgs_html}{audios_html}</div>' if (imgs_html or audios_html) else ""

    return f"""<div class="turn{highlight_cls}">
  <div class="turn-meta"><span class="round-badge">{rnd}</span></div>
  {f'<div class="turn-user-text">👤 {user_cap}</div>' if user_cap else ''}
  {media_block}
  {f'<div class="turn-assistant-text">🤖 {assistant}</div>' if assistant else ''}
</div>"""


def render_session(session: dict, highlight_rounds: set[str]) -> str:
    sid   = escape(session.get("session_id", ""))
    date  = escape(session.get("date", ""))
    turns = session.get("dialogues", [])

    turns_html = "\n".join(
        render_turn(t, highlight=t.get("round", "") in highlight_rounds)
        for t in turns
    )
    n_img   = sum(len(t.get("image_id", [])) for t in turns)
    n_audio = sum(len(t.get("voice_id", [])) for t in turns)
    stats   = f"{len(turns)} 轮"
    if n_img:   stats += f" · {n_img} 图"
    if n_audio: stats += f" · {n_audio} 音"

    body_id  = f"sess-body-{sid}"
    arrow_id = f"sess-arr-{sid}"
    return f"""<div class="session-block">
  <div class="session-header" onclick="toggleSection('{body_id}','{arrow_id}')">
    <span class="session-id">{sid}</span>
    <span class="session-date">{date}</span>
    <span class="section-count">{stats}</span>
    <span class="arrow closed" id="{arrow_id}">▼</span>
  </div>
  <div class="session-body" id="{body_id}" style="display:none">
    {turns_html or '<p class="empty">（无对话）</p>'}
  </div>
</div>"""


def render_qa_card(qa: dict, sessions_by_id: dict, turns_by_round: dict,
                   images_by_id: dict, audio_by_id: dict, card_idx: int,
                   profiles_map: dict | None = None) -> str:
    qa_id    = escape(qa.get("qa_id", ""))
    question = escape(qa.get("question", ""))
    answer   = escape(qa.get("answer", ""))
    category = (qa.get("point") or qa.get("category") or "").lower()
    qa_type  = qa.get("qa_type", "")
    session_id = qa.get("session_id", "")
    clues    = qa.get("clue", [])

    # ── clue 解析 ─────────────────────────────────────────────
    clue_rounds: list[str] = []
    clue_images: list[str] = []
    clue_audios: list[str] = []

    for c in clues:
        ctype, cid = parse_clue(c)
        if ctype == 'round':   clue_rounds.append(cid)
        elif ctype == 'image': clue_images.append(cid)
        elif ctype == 'audio': clue_audios.append(cid)

    # ── clue 渲染 ──────────────────────────────────────────────
    clue_sections = ""

    # 1. 图片 clue
    if clue_images:
        imgs_html = ""
        for img_id in clue_images:
            p   = images_by_id.get(img_id) or find_image(img_id)
            src = rel_src(p)
            if src:
                imgs_html += f'<div class="clue-media"><img src="{src}" alt="{escape(img_id)}"><div class="media-cap">{escape(img_id)}</div></div>'
            else:
                imgs_html += f'<div class="clue-media no-file">📷 {escape(img_id)}</div>'
        clue_sections += f'<div class="clue-group"><div class="clue-group-title">🖼 图片线索 ({len(clue_images)})</div><div class="clue-media-row">{imgs_html}</div></div>'

    # 2. 音频 clue
    if clue_audios:
        auds_html = ""
        for v_id in clue_audios:
            p   = audio_by_id.get(v_id) or find_audio(v_id)
            src = rel_src(p)
            # 查对应 caption（从 turns 里找）
            cap = ""
            for rnd, turn in turns_by_round.items():
                if v_id in turn.get("voice_id", []):
                    idx = turn["voice_id"].index(v_id)
                    caps = turn.get("voice_caption", [])
                    cap = caps[idx] if idx < len(caps) else turn.get("user_voice_message_caption", "")
                    break
            if src:
                auds_html += (
                    f'<div class="clue-media">'
                    f'<audio controls preload="none"><source src="{src}" type="audio/wav"></audio>'
                    f'<div class="media-cap">{escape(v_id)}<br><small>{escape(cap)}</small></div>'
                    f'</div>'
                )
            else:
                auds_html += f'<div class="clue-media no-file">🔊 {escape(v_id)}<br><small>{escape(cap)}</small></div>'
        clue_sections += f'<div class="clue-group"><div class="clue-group-title">🎙 音频线索 ({len(clue_audios)})</div><div class="clue-media-row">{auds_html}</div></div>'

    # 3. 对话轮 clue
    if clue_rounds:
        # 按 session 分组，把对应 session 的完整对话展示出来，高亮涉及轮次
        rounds_set = set(clue_rounds)
        involved_sessions: dict[str, list[str]] = defaultdict(list)
        for r in clue_rounds:
            m = re.match(r'^(D\d+):', r)
            if m:
                involved_sessions[m.group(1)].append(r)

        sessions_html = ""
        for sid_key in sorted(involved_sessions.keys()):
            if sid_key in sessions_by_id:
                highlighted = set(involved_sessions[sid_key])
                sessions_html += render_session(sessions_by_id[sid_key], highlighted)
            else:
                # session 不存在，仅列出 round id
                rounds_list = ", ".join(f'<code>{escape(r)}</code>' for r in involved_sessions[sid_key])
                sessions_html += f'<p class="empty">Session {escape(sid_key)}: {rounds_list}</p>'

        clue_sections += f"""<div class="clue-group">
  <div class="clue-group-title">💬 对话轮线索 ({len(clue_rounds)} 轮，涉及 {len(involved_sessions)} 个 Session)</div>
  {sessions_html}
</div>"""

    if not clue_sections:
        clue_sections = '<p class="empty">（无 memory clue）</p>'

    # ── 关联 session（非 clue 的原始 session）──
    session_block = ""
    if session_id and session_id in sessions_by_id:
        # 如果该 session 未在 clue rounds 里，则单独展示
        clue_session_ids = set(re.match(r'^(D\d+):', r).group(1)
                               for r in clue_rounds if re.match(r'^(D\d+):', r))
        if session_id not in clue_session_ids:
            session_block = f"""<div class="clue-group">
  <div class="clue-group-title">🗂 关联 Session（{escape(session_id)}）</div>
  {render_session(sessions_by_id[session_id], set())}
</div>"""

    body_id  = f"qa-clue-{card_idx}"
    arrow_id = f"qa-arr-{card_idx}"

    # extra metadata for entity/refusal
    extra_meta = ""
    if qa.get("subcategory"):
        extra_meta += f'<span class="meta-tag">子类: {escape(qa["subcategory"])}</span>'
    if qa.get("entity_name"):
        extra_meta += f'<span class="meta-tag">实体: {escape(qa["entity_name"])}</span>'
    if qa.get("dimension"):
        extra_meta += f'<span class="meta-tag">维度: {escape(qa["dimension"])}</span>'
    if qa.get("signal_source"):
        extra_meta += f'<span class="meta-tag">信号: {escape(qa["signal_source"])}</span>'
    if qa.get("rationale"):
        extra_meta += f'<span class="meta-tag">原因: {escape(qa["rationale"])}</span>'

    clue_count = len(clues)
    clue_label = f"{clue_count} 条线索" if clue_count else "无线索"

    p_id = qa.get("p_id", 0)
    char_name = (profiles_map or {}).get(p_id, f"p{p_id}")
    char_badge = f'<span class="badge" style="background:#dbeafe;color:#1e40af">{escape(char_name)}</span>'
    return f"""<div class="qa-card" data-category="{escape(category)}" data-qatype="{escape(qa_type)}" data-pid="{p_id}">
  <div class="qa-header">
    <span class="qa-id">{qa_id}</span>
    {char_badge}
    {cat_badge(category)}
    {qatype_badge(qa_type)}
    {f'<span class="meta-tag sess-tag">📅 {escape(session_id)}</span>' if session_id else ''}
    {extra_meta}
  </div>
  <div class="qa-body">
    <div class="qa-question">{question.replace(chr(10), '<br>')}</div>
    <div class="qa-answer">✅ 正确答案：<strong>{answer}</strong></div>
  </div>
  <div class="qa-clue-header" onclick="toggleSection('{body_id}','{arrow_id}')">
    <span>🔍 Memory Clue</span>
    <span class="section-count">{clue_label}</span>
    <span class="arrow closed" id="{arrow_id}">▼</span>
  </div>
  <div class="qa-clue-body" id="{body_id}" style="display:none">
    {clue_sections}
    {session_block}
  </div>
</div>"""


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def load_all_qa_files(dialog_dir: Path) -> tuple[list, list, dict]:
    """Load all history_with_qa_p*.json files.

    Returns (all_sessions, all_qa, profiles_map).
    profiles_map: {p_id: character_name}
    """
    all_sessions = []
    all_qa = []
    profiles_map: dict[int, str] = {}

    qa_files = sorted(dialog_dir.glob("history_with_qa_p*.json"))
    if not qa_files:
        qa_files = sorted(dialog_dir.glob("history_with_qa*.json"))

    for f in qa_files:
        print(f"  加载 {f.name} ...")
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        profile = data.get("character_profile", {})
        sessions = data.get("multi_session_dialogues", [])
        qa_list = data.get("human-annotated QAs", [])

        p_id = None
        if qa_list:
            p_id = qa_list[0].get("p_id", None)

        if p_id is None:
            m = re.search(r'_p(\d+)\.json$', f.name)
            p_id = int(m.group(1)) if m else len(profiles_map)

        char_name = profile.get("name", f"user_{p_id}")
        profiles_map[p_id] = char_name

        for s in sessions:
            s["_p_id"] = p_id
        all_sessions.extend(sessions)

        for qa in qa_list:
            qa.setdefault("p_id", p_id)
        all_qa.extend(qa_list)

        print(f"    p_id={p_id} ({char_name}): {len(sessions)} sessions, {len(qa_list)} QAs")

    return all_sessions, all_qa, profiles_map


def main():
    print(f"扫描 {DIALOG_DIR} ...")
    all_sessions, qa_list, profiles_map = load_all_qa_files(DIALOG_DIR)

    if not qa_list:
        print("ERROR: 未找到任何 QA 数据。")
        return

    title_names = " / ".join(profiles_map[k] for k in sorted(profiles_map))
    print(f"\n  人物: {title_names}")
    print(f"  Sessions: {len(all_sessions)}, QA items: {len(qa_list)}")
    sessions = all_sessions

    # 从 qa_formatted_data 构建 clue 索引（含图片 / 音频路径）
    print(f"加载 clue 索引: {QA_FORMATTED_DATA} ...")
    indexes_per_pid = build_indexes_from_formatted_data(QA_FORMATTED_DATA)

    total_turns = sum(len(idx[0]) for idx in indexes_per_pid.values())
    total_imgs = sum(len(idx[1]) for idx in indexes_per_pid.values())
    total_auds = sum(len(idx[2]) for idx in indexes_per_pid.values())
    print(f"  对话轮: {total_turns}, 图片: {total_imgs}, 音频: {total_auds}")

    # 统计各类别
    from collections import Counter
    cat_counts = Counter((qa.get("point") or qa.get("category") or "").lower() for qa in qa_list)
    type_counts = Counter(qa.get("qa_type", "") for qa in qa_list)

    # 渲染 QA 卡片 — 每条 QA 使用其所属人物的索引
    print("渲染 QA 卡片 ...")
    _empty_idx = ({}, {}, {}, {})

    def _render_card(i: int, qa: dict) -> str:
        pid = qa.get("p_id", 0)
        turns, imgs, auds, sess = indexes_per_pid.get(pid, _empty_idx)
        return render_qa_card(qa, sess, turns, imgs, auds, i, profiles_map)

    cards_html = "\n".join(_render_card(i, qa) for i, qa in enumerate(qa_list))

    # 过滤按钮
    categories = sorted(cat_counts.keys())
    qa_types   = [t for t in ["explicit", "implicit"] if type_counts.get(t, 0) > 0]

    cat_buttons = "".join(
        f'<button class="filter-btn active" data-filter="category" data-value="{escape(c)}" '
        f'onclick="toggleFilter(this)">'
        f'{escape(c)} <span class="btn-count">{cat_counts[c]}</span></button>'
        for c in categories
    )
    type_buttons = "".join(
        f'<button class="filter-btn active" data-filter="qatype" data-value="{escape(t)}" '
        f'onclick="toggleFilter(this)">'
        f'{escape(t)} <span class="btn-count">{type_counts[t]}</span></button>'
        for t in qa_types
    )
    # unknown types
    for t, cnt in type_counts.items():
        if t and t not in ("explicit", "implicit"):
            type_buttons += (
                f'<button class="filter-btn active" data-filter="qatype" data-value="{escape(t)}" '
                f'onclick="toggleFilter(this)">'
                f'{escape(t)} <span class="btn-count">{cnt}</span></button>'
            )

    # 人物筛选按钮
    pid_counts = Counter(qa.get("p_id", 0) for qa in qa_list)
    sorted_pids = sorted(profiles_map.keys())
    pid_buttons = "".join(
        f'<button class="filter-btn active" data-filter="pid" data-value="{pid}" '
        f'onclick="toggleFilter(this)">'
        f'{escape(profiles_map[pid])} <span class="btn-count">{pid_counts.get(pid, 0)}</span></button>'
        for pid in sorted_pids
    )

    total_img_files = sum(
        1 for idx in indexes_per_pid.values() for p in idx[1].values() if p.exists()
    )
    total_audio_files = sum(
        1 for idx in indexes_per_pid.values() for p in idx[2].values() if p.exists()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>QA 检查 — {escape(title_names)}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI','PingFang SC',sans-serif; background: #f0f2f5; color: #1a1a2e; }}

/* ── 顶栏 ── */
header {{ background: #1a1a2e; color: #eee; padding: 12px 24px;
          display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap; }}
header h1 {{ font-size: 17px; font-weight: 700; }}
header .summary {{ font-size: 12px; color: #aaa; }}

/* ── 工具栏 ── */
.toolbar {{ position: sticky; top: 0; z-index: 200; background: #fff;
            border-bottom: 1px solid #e5e7eb; padding: 10px 24px;
            display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }}
.toolbar-group {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
.toolbar-label {{ font-size: 11px; font-weight: 700; color: #9ca3af;
                  text-transform: uppercase; letter-spacing: .06em; white-space: nowrap; }}
.filter-btn {{ padding: 4px 12px; border: 1.5px solid #d1d5db; border-radius: 16px;
               background: #fff; cursor: pointer; font-size: 12px;
               transition: .15s; display: flex; align-items: center; gap: 4px; }}
.filter-btn:hover  {{ border-color: #6366f1; color: #4338ca; }}
.filter-btn.active {{ background: #1a1a2e; color: #fff; border-color: #1a1a2e; }}
.filter-btn.active .btn-count {{ background: rgba(255,255,255,.2); }}
.btn-count {{ font-size: 10px; background: #f3f4f6; color: #6b7280;
              padding: 0 5px; border-radius: 8px; }}
.visible-count {{ font-size: 12px; color: #6b7280; margin-left: auto; white-space: nowrap; }}

.search-box {{ padding: 5px 12px; border: 1.5px solid #d1d5db; border-radius: 16px;
               font-size: 12px; width: 220px; outline: none; }}
.search-box:focus {{ border-color: #6366f1; }}

/* ── 主内容区 ── */
.main {{ max-width: 1100px; margin: 0 auto; padding: 18px 24px; }}

/* ── QA 卡片 ── */
.qa-card {{ background: #fff; border-radius: 12px; margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,.07); overflow: hidden; }}
.qa-header {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
              padding: 8px 14px; background: #f8f9fb; border-bottom: 1px solid #eee; }}
.qa-id {{ font-family: monospace; font-size: 11px; color: #9ca3af; flex-shrink: 0; }}
.badge {{ font-size: 11px; padding: 2px 9px; border-radius: 10px; font-weight: 600; }}
.meta-tag {{ font-size: 11px; color: #6b7280; background: #f3f4f6;
             padding: 2px 8px; border-radius: 8px; }}
.sess-tag {{ color: #7c3aed; background: #ede9fe; }}

.qa-body {{ padding: 12px 14px; }}
.qa-question {{ font-size: 13px; line-height: 1.75; color: #1f2937;
                white-space: pre-wrap; word-break: break-word; margin-bottom: 8px; }}
.qa-answer {{ font-size: 12px; color: #059669; }}

/* ── clue 折叠区 ── */
.qa-clue-header {{ display: flex; align-items: center; gap: 8px; cursor: pointer;
                   padding: 7px 14px; background: #f0f4ff; border-top: 1px solid #e5e7eb;
                   font-size: 12px; font-weight: 600; color: #3730a3;
                   user-select: none; }}
.qa-clue-header:hover {{ background: #e0e7ff; }}
.qa-clue-body {{ padding: 10px 14px; border-top: 1px solid #e0e7ff;
                 display: flex; flex-direction: column; gap: 12px; }}

/* ── clue 分组 ── */
.clue-group {{ border: 1px solid #e5e7eb; border-radius: 10px; overflow: hidden; }}
.clue-group-title {{ font-size: 12px; font-weight: 700; color: #374151;
                     padding: 6px 12px; background: #f9fafb; border-bottom: 1px solid #e5e7eb; }}
.clue-media-row {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 10px; }}
.clue-media {{ display: flex; flex-direction: column; align-items: center; gap: 4px; max-width: 180px; }}
.clue-media img {{ width: 170px; height: 170px; object-fit: contain;
                   border-radius: 8px; border: 1px solid #ddd; }}
.clue-media audio {{ width: 200px; height: 32px; }}
.media-cap {{ font-size: 10px; color: #6b7280; text-align: center;
              max-width: 200px; word-break: break-word; }}
.no-file {{ color: #ef4444; font-size: 11px; background: #fef2f2;
            border: 1px dashed #fca5a5; padding: 8px; border-radius: 6px;
            text-align: center; }}

/* ── Session 块 ── */
.session-block {{ border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-top: 6px; }}
.session-header {{ display: flex; align-items: center; gap: 8px; cursor: pointer;
                   padding: 7px 12px; background: #f9fafb; user-select: none;
                   font-size: 12px; border-bottom: 1px solid #e5e7eb; }}
.session-header:hover {{ background: #f0f2f5; }}
.session-id {{ font-family: monospace; font-weight: 700; color: #1d4ed8; }}
.session-date {{ font-size: 11px; color: #9ca3af; }}
.session-body {{ padding: 8px 10px; max-height: 600px; overflow-y: auto;
                 display: flex; flex-direction: column; gap: 6px; }}

/* ── 对话轮 ── */
.turn {{ border-radius: 8px; padding: 8px 10px; font-size: 12px; line-height: 1.65;
         border: 1px solid #e5e7eb; background: #fafafa; }}
.turn-highlight {{ border-color: #f59e0b; background: #fffbeb; box-shadow: 0 0 0 2px #fcd34d; }}
.turn-meta {{ margin-bottom: 4px; }}
.round-badge {{ font-family: monospace; font-size: 10px; background: #e0e7ff;
                color: #4338ca; padding: 1px 6px; border-radius: 6px; }}
.turn-user-text {{ color: #1d4ed8; margin-bottom: 4px; }}
.turn-assistant-text {{ color: #374151; font-style: italic; }}
.turn-media-row {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0; }}

/* ── 通用 ── */
.section-count {{ font-size: 11px; color: #6b7280; background: #f3f4f6;
                  padding: 1px 7px; border-radius: 8px; font-weight: 400; }}
.arrow {{ margin-left: auto; font-size: 11px; color: #9ca3af; transition: transform .2s; }}
.arrow.closed {{ transform: rotate(-90deg); }}
.empty {{ color: #9ca3af; font-size: 12px; padding: 8px; text-align: center; }}
code {{ font-size: 11px; background: #f3f4f6; padding: 1px 5px; border-radius: 3px; }}

@media (max-width: 700px) {{
  .toolbar {{ padding: 8px 12px; }}
  .main {{ padding: 10px 12px; }}
  .clue-media img {{ width: 130px; height: 130px; }}
}}
</style>
</head>
<body>
<header>
  <h1>QA 检查 — {escape(title_names)}</h1>
  <div class="summary">
    人物: {len(profiles_map)} &nbsp;|&nbsp;
    QA 总数: {len(qa_list)} &nbsp;|&nbsp;
    Sessions: {len(sessions)} &nbsp;|&nbsp;
    图片文件: {total_img_files}/{total_imgs} &nbsp;|&nbsp;
    音频文件: {total_audio_files}/{total_auds} &nbsp;|&nbsp;
    {"  ".join(f"{c}: {n}" for c, n in sorted(cat_counts.items()))}
  </div>
</header>

<div class="toolbar">
  <div class="toolbar-group">
    <span class="toolbar-label">人物</span>
    <button class="filter-btn active" data-filter="__all_pid__" onclick="toggleAllFilter('pid',this)">全部</button>
    {pid_buttons}
  </div>
  <div class="toolbar-group">
    <span class="toolbar-label">Category</span>
    <button class="filter-btn active" data-filter="__all_category__" onclick="toggleAllFilter('category',this)">全部</button>
    {cat_buttons}
  </div>
  <div class="toolbar-group">
    <span class="toolbar-label">Type</span>
    <button class="filter-btn active" data-filter="__all_qatype__" onclick="toggleAllFilter('qatype',this)">全部</button>
    {type_buttons}
  </div>
  <input class="search-box" type="text" placeholder="🔍 搜索题目…" oninput="applyFilters()">
  <span class="visible-count" id="visible-count">共 {len(qa_list)} 条</span>
</div>

<div class="main" id="qa-list">
{cards_html}
</div>

<script>
var activePids       = new Set({json.dumps([str(p) for p in sorted_pids])});
var activeCategories = new Set({json.dumps(categories)});
var activeQaTypes    = new Set({json.dumps([t for t in type_counts.keys() if t])});

function toggleFilter(btn) {{
  var filter = btn.dataset.filter;
  var value  = btn.dataset.value;
  if (filter === 'pid') {{
    if (activePids.has(value)) activePids.delete(value);
    else activePids.add(value);
  }} else if (filter === 'category') {{
    if (activeCategories.has(value)) activeCategories.delete(value);
    else activeCategories.add(value);
  }} else {{
    if (activeQaTypes.has(value)) activeQaTypes.delete(value);
    else activeQaTypes.add(value);
  }}
  btn.classList.toggle('active');
  applyFilters();
}}

function toggleAllFilter(filterType, btn) {{
  var btns = document.querySelectorAll('[data-filter="' + filterType + '"]');
  var allActive = btn.classList.contains('active');
  var targetSet = filterType === 'pid' ? activePids : filterType === 'category' ? activeCategories : activeQaTypes;
  if (allActive) targetSet.clear();
  else btns.forEach(b => {{ if (b.dataset.value) targetSet.add(b.dataset.value); }});
  btns.forEach(b => {{ if (b !== btn) b.classList.toggle('active', !allActive); }});
  btn.classList.toggle('active');
  applyFilters();
}}

function applyFilters() {{
  var query = document.querySelector('.search-box').value.toLowerCase();
  var cards = document.querySelectorAll('.qa-card');
  var visible = 0;
  cards.forEach(function(card) {{
    var pid    = card.dataset.pid || '0';
    var cat    = card.dataset.category || '';
    var qtype  = card.dataset.qatype  || '';
    var pidOk  = activePids.size === 0 || activePids.has(pid);
    var catOk  = activeCategories.size === 0 || activeCategories.has(cat);
    var typeOk = activeQaTypes.size === 0 || qtype === '' || activeQaTypes.has(qtype);
    var textOk = !query || card.textContent.toLowerCase().includes(query);
    var show = pidOk && catOk && typeOk && textOk;
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('visible-count').textContent = '显示 ' + visible + ' / {len(qa_list)} 条';
}}

function toggleSection(bodyId, arrowId) {{
  var body  = document.getElementById(bodyId);
  var arrow = document.getElementById(arrowId);
  if (!body) return;
  if (body.style.display === 'none') {{
    body.style.display = '';
    if (arrow) arrow.classList.remove('closed');
  }} else {{
    body.style.display = 'none';
    if (arrow) arrow.classList.add('closed');
  }}
}}
</script>
</body>
</html>"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size // 1024
    print(f"\n生成完成：{OUTPUT_HTML}  ({size_kb} KB)")
    print(f"QA 总计 {len(qa_list)} 条，用浏览器打开上面的 HTML 文件即可。")


if __name__ == "__main__":
    main()
