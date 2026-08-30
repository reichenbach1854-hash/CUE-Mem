"""生成 entity_qa_viewer.html，用于可视化检查实体文本选择题。

展示内容：
  - 题目信息与文本选项
  - 正确答案高亮
  - Memory clue（轮次文本、图片、音频）

用法:
    python qa/entity_qa_reviewer.py
    python qa/entity_qa_reviewer.py --input qa/qa_entity_mcq_000_002.json
    python qa/entity_qa_reviewer.py --output qa/my_entity_viewer.html
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
from scripts.qa.config import BENCHMARK_DATA_ROOT, qa_path

INPUT_PATH = qa_path("qa_entity_mcq_000_002.json")
OUTPUT_PATH = qa_path("entity_qa_viewer.html")
BENCHMARK_DATA_DIR = BENCHMARK_DATA_ROOT / "dialog" / "base"

ENTITY_TYPE_LABELS = {
    "Relationship": "人物",
    "Pets": "宠物",
    "Items": "物品",
}


def esc(text: str) -> str:
    return html.escape(str(text))


def file_to_data_uri(filepath: str, mime: str) -> str:
    try:
        candidate = Path(filepath)
        if not candidate.is_absolute() and not candidate.exists():
            benchmark_candidate = BENCHMARK_DATA_DIR / candidate
            candidate = benchmark_candidate if benchmark_candidate.exists() else resolve_path(candidate)
        with open(candidate, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return ""


def build_clue_maps(benchmark_data_dir: str):
    voice_map = {}
    image_map = {}
    round_map = {}

    for pid in range(3):
        path = os.path.join(str(benchmark_data_dir), f"history_with_qa_p{pid}.json")
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sess in data.get("multi_session_dialogues", []):
            sid = sess.get("session_id", "")
            sess_date = sess.get("date", "")
            for d in sess.get("dialogues", []):
                rnd = d.get("round", "")
                round_info = {"session_id": sid, "date": sess_date, "round": rnd, "p_id": pid}
                if d.get("user"):
                    round_info["user"] = d["user"]
                if d.get("user_voice_message_caption"):
                    round_info["voice_caption"] = d["user_voice_message_caption"]
                if d.get("assistant"):
                    round_info["assistant"] = d["assistant"]
                round_map[(pid, rnd)] = round_info

                for i, vid in enumerate(d.get("voice_id", []) or []):
                    paths = d.get("input_voice_message", []) or []
                    capts = d.get("voice_caption", []) or []
                    voice_map[(pid, vid)] = {
                        "path": paths[i] if i < len(paths) else "",
                        "caption": capts[i] if i < len(capts) else "",
                    }

                for i, iid in enumerate(d.get("image_id", []) or []):
                    paths = d.get("input_image", []) or []
                    capts = d.get("image_caption", []) or []
                    image_map[(pid, iid)] = {
                        "path": paths[i] if i < len(paths) else "",
                        "caption": capts[i] if i < len(capts) else "",
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


def build_clue_html(record: dict, voice_map, image_map, round_map):
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
                audio_html = ""
                if audio_path:
                    full_path = os.path.normpath(audio_path)
                    if os.path.exists(full_path):
                        data_uri = file_to_data_uri(full_path, "audio/wav")
                        if data_uri:
                            audio_html = f'<audio controls preload="none" src="{data_uri}"></audio>'
                    if not audio_html:
                        audio_html = f'<div class="clue-missing">音频文件: {esc(audio_path)}</div>'

                sess_parts.append(
                    f'''
                <div class="clue-item clue-audio">
                    <span class="clue-tag tag-voice">🔊 {esc(clue_str)}</span>
                    {audio_html}
                    {f'<div class="clue-caption">{esc(caption)}</div>' if caption else ''}
                </div>'''
                )
            elif ctype == "image":
                info = image_map.get((p_id, cid), {})
                img_path = info.get("path", "")
                caption = info.get("caption", "")
                img_html = ""
                if img_path:
                    full_path = os.path.normpath(img_path)
                    if os.path.exists(full_path):
                        data_uri = file_to_data_uri(full_path, "image/png")
                        if data_uri:
                            img_html = f'<img class="clue-img" src="{data_uri}" loading="lazy">'
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


def build_html(records: list, voice_map, image_map, round_map) -> str:
    by_etype = Counter(r.get("entity_type", "?") for r in records)
    by_dim = Counter(r.get("dimension", "?") for r in records)
    total = len(records)

    etype_options = "".join(
        f'<option value="{et}">{ENTITY_TYPE_LABELS.get(et, et)}</option>'
        for et in sorted(by_etype.keys())
    )
    dim_options = "".join(
        f'<option value="{d}">{d}</option>'
        for d in sorted(by_dim.keys())
    )
    stats_etype = "".join(
        f"<li>{ENTITY_TYPE_LABELS.get(et, et)}: {c}</li>"
        for et, c in sorted(by_etype.items(), key=lambda x: x[0])
    )

    cards = []
    for idx, record in enumerate(records):
        qa_id = record.get("qa_id", "")
        entity_type = record.get("entity_type", "")
        entity_label = ENTITY_TYPE_LABELS.get(entity_type, entity_type)
        entity_name = record.get("entity_name", "")
        entity_relation = record.get("entity_relation", "")
        dimension = record.get("dimension", "")
        question = record.get("Q", "")
        answer = record.get("A", "")
        options = record.get("options", {})

        option_html = []
        for letter in ["A", "B", "C", "D"]:
            text = options.get(letter, "")
            is_correct = letter == answer
            cls = "option-correct" if is_correct else "option-wrong"
            option_html.append(
                f'''
            <div class="option-line {cls}">
                <span class="option-letter">{letter}</span>
                <span class="option-text">{esc(text)}</span>
            </div>'''
            )

        relation_str = f" ({esc(entity_relation)})" if entity_relation else ""
        clue_html = build_clue_html(record, voice_map, image_map, round_map)
        cards.append(
            f'''
    <div class="qa-card" data-etype="{entity_type}" data-dim="{esc(dimension)}" data-qid="{esc(qa_id)}" id="q-{idx}">
        <div class="card-header">
            <span class="qa-index">#{idx + 1}</span>
            <span class="qa-id">{esc(qa_id)}</span>
            <span class="badge badge-{entity_type.lower()}">{esc(entity_label)}</span>
        </div>
        <div class="entity-row"><strong>{esc(entity_label)}：</strong>{esc(entity_name)}{relation_str}</div>
        <div class="dimension-row"><strong>考察维度：</strong>{esc(dimension)}</div>
        <div class="question-row"><strong>Q：</strong>{esc(question)}</div>
        <div class="options-box">
            {"".join(option_html)}
        </div>
        {clue_html}
    </div>'''
        )

    cards_html = "\n".join(cards)

    return f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>实体文本选择题检查</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    background: #f0f2f5; color: #333; padding: 20px;
}}
.header, .filter-bar, .qa-card {{
    max-width: 1200px; margin-left: auto; margin-right: auto;
    background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.header {{ padding: 24px; margin-bottom: 24px; }}
.filter-bar {{ padding: 16px 24px; margin-bottom: 20px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
.qa-card {{ padding: 24px; margin-bottom: 28px; }}
.header h1 {{ font-size: 22px; margin-bottom: 12px; }}
.stats {{ display: flex; gap: 32px; flex-wrap: wrap; font-size: 14px; color: #666; }}
.stats ul {{ list-style: none; }}
.stats li {{ margin: 2px 0; }}

.filter-bar label {{ font-size: 14px; font-weight: 500; }}
.filter-bar select, .filter-bar input {{
    padding: 6px 12px; border: 1px solid #d9d9d9; border-radius: 6px;
    font-size: 14px; outline: none;
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

.entity-row, .dimension-row {{
    font-size: 14px; color: #555; margin-bottom: 8px;
    padding: 8px 12px; background: #fafafa; border-radius: 6px; border-left: 3px solid #1677ff;
}}
.question-row {{
    font-size: 15px; margin: 12px 0 16px; line-height: 1.6;
}}
.options-box {{
    border: 1px solid #e8e8e8; border-radius: 10px; overflow: hidden; margin-bottom: 16px;
}}
.option-line {{
    display: grid; grid-template-columns: 44px 1fr; gap: 12px;
    padding: 12px 14px; border-bottom: 1px solid #f0f0f0; align-items: start;
}}
.option-line:last-child {{ border-bottom: none; }}
.option-letter {{
    width: 28px; height: 28px; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 700; color: #fff; background: #8c8c8c;
}}
.option-correct {{
    background: #f6ffed;
}}
.option-correct .option-letter {{
    background: #52c41a;
}}
.option-wrong {{
    background: #fff;
}}
.option-text {{
    line-height: 1.6; font-size: 14px;
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
    padding: 10px 12px; border-bottom: 1px solid #f5f5f5; font-size: 13px; line-height: 1.6;
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
    margin: 6px 0; border: 1px solid #e8e8e8;
}}
audio {{
    display: block; margin: 6px 0; height: 36px; width: 100%; max-width: 360px;
}}
</style>
</head>
<body>
<div class="header">
    <h1>实体文本选择题 — 可视化检查</h1>
    <div class="stats">
        <div><strong>总题数：</strong>{total}</div>
        <div><strong>按实体类型：</strong><ul>{stats_etype}</ul></div>
    </div>
</div>

<div class="filter-bar">
    <label>实体类型：</label>
    <select id="filterEtype">
        <option value="">全部</option>
        {etype_options}
    </select>
    <label>维度：</label>
    <select id="filterDim">
        <option value="">全部</option>
        {dim_options}
    </select>
    <label>搜索：</label>
    <input id="filterSearch" type="text" placeholder="qa_id / 实体名 / 题干" style="width:260px;">
    <button id="expandAll" style="padding:6px 14px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;font-size:13px;">展开全部 Clue</button>
    <button id="collapseAll" style="padding:6px 14px;border:1px solid #d9d9d9;border-radius:6px;cursor:pointer;font-size:13px;">折叠全部 Clue</button>
    <span id="filterCount" style="font-size:13px;color:#999;margin-left:auto;"></span>
</div>

{cards_html}

<script>
const cards = document.querySelectorAll('.qa-card');
const filterEtype = document.getElementById('filterEtype');
const filterDim = document.getElementById('filterDim');
const filterSearch = document.getElementById('filterSearch');
const filterCount = document.getElementById('filterCount');

function applyFilters() {{
    const etype = filterEtype.value;
    const dim = filterDim.value;
    const q = filterSearch.value.toLowerCase();
    let shown = 0;
    cards.forEach(c => {{
        const matchEtype = !etype || c.dataset.etype === etype;
        const matchDim = !dim || c.dataset.dim === dim;
        const matchSearch = !q || c.dataset.qid.toLowerCase().includes(q) || c.textContent.toLowerCase().includes(q);
        const vis = matchEtype && matchDim && matchSearch;
        c.style.display = vis ? '' : 'none';
        if (vis) shown++;
    }});
    filterCount.textContent = shown + ' / ' + cards.length;
}}
filterEtype.addEventListener('change', applyFilters);
filterDim.addEventListener('change', applyFilters);
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
    parser = argparse.ArgumentParser(description="生成实体文本选择题 HTML 检查页面")
    parser.add_argument("--input", default=INPUT_PATH, help="输入 JSON 路径")
    parser.add_argument("--output", default=OUTPUT_PATH, help="输出 HTML 路径")
    parser.add_argument("--benchmark-data-dir", default=BENCHMARK_DATA_DIR, help="benchmark history JSON 所在目录")
    args = parser.parse_args()
    args.input = resolve_path(args.input)
    args.output = resolve_path(args.output)
    benchmark_data_dir = resolve_path(args.benchmark_data_dir)

    records = load_json_or_jsonl(args.input)

    print(f"Loaded {len(records)} records from {args.input}")
    print("Building clue maps from benchmark data...")
    voice_map, image_map, round_map = build_clue_maps(str(benchmark_data_dir))
    print(f"  voice: {len(voice_map)}, image: {len(image_map)}, rounds: {len(round_map)}")

    html_text = build_html(records, voice_map, image_map, round_map)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
