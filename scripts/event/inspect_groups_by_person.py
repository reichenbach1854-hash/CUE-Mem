"""Generate an HTML inspection page for preference groups.

Default usage:
    python event/inspect_groups_by_person.py

The page shows the first N profile records from manual group JSON, with:
  - per-profile preference frequency tables
  - every group and its explicit / implicit preferences
  - recommended_main_scene
  - implicit_integration_guidance
"""

from __future__ import annotations

import argparse
import html
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from scripts.common.io import load_json_or_jsonl
from scripts.common.paths import project_path, resolve_path

DEFAULT_INPUT = project_path("event", "manual_profiles_with_anchors_groups.json")
DEFAULT_OUTPUT = project_path("event", "manual_groups_first5_inspection.html")
DEFAULT_MAX_PROFILES = 20


PrefKey = Tuple[str, str]


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def read_json(path: Path) -> Any:
    return load_json_or_jsonl(path)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def pref_sources(pref: Dict[str, Any]) -> List[str]:
    return [str(x) for x in as_list(pref.get("sources") or pref.get("evidence_sources"))]


def pref_anchors(pref: Dict[str, Any]) -> List[str]:
    anchors = pref.get("entity_anchors")
    if anchors is None:
        anchors = pref.get("entity_anchor")
    if anchors is None:
        return []
    if isinstance(anchors, list):
        return [str(x) for x in anchors if str(x).strip()]
    text = str(anchors).strip()
    return [text] if text else []


def pref_content(pref: Dict[str, Any]) -> str:
    return str(pref.get("content") or pref.get("preference") or "")


def pref_category(pref: Dict[str, Any]) -> str:
    return str(pref.get("category") or "")


def compute_profile_frequency(record: Dict[str, Any]) -> Tuple[Counter[PrefKey], Dict[PrefKey, Dict[str, Any]]]:
    counts: Counter[PrefKey] = Counter()
    meta: Dict[PrefKey, Dict[str, Any]] = {}

    for group in record.get("groups", []) or []:
        for pref_type, field in (("explicit", "explicit_preferences"), ("implicit", "implicit_preferences")):
            for pref in group.get(field, []) or []:
                if not isinstance(pref, dict):
                    continue
                category = pref_category(pref)
                if not category:
                    continue
                key = (pref_type, category)
                counts[key] += 1
                meta.setdefault(
                    key,
                    {
                        "pref_type": pref_type,
                        "category": category,
                        "subcategory": pref.get("subcategory", ""),
                        "content": pref_content(pref),
                        "sources": pref_sources(pref),
                        "anchors": pref_anchors(pref),
                    },
                )

    return counts, meta


def chips(items: Iterable[Any], css_class: str = "chip") -> str:
    values = [str(x) for x in items if str(x).strip()]
    if not values:
        return '<span class="muted">none</span>'
    return "".join(f'<span class="{css_class}">{esc(x)}</span>' for x in values)


def render_pref(pref: Dict[str, Any], pref_type: str) -> str:
    category = pref_category(pref)
    subcategory = pref.get("subcategory", "")
    content = pref_content(pref)
    sources = pref_sources(pref)
    anchors = pref_anchors(pref)
    type_label = "显式偏好" if pref_type == "explicit" else "隐式偏好"
    return f"""
      <div class="pref pref-{esc(pref_type)}">
        <div class="pref-head">
          <span class="badge {esc(pref_type)}">{type_label}</span>
          <strong>{esc(category)}</strong>
          <span class="subcat">{esc(subcategory)}</span>
        </div>
        <div class="content">{esc(content)}</div>
        <div class="kv"><span>sources</span><div>{chips(sources)}</div></div>
        <div class="kv"><span>anchors</span><div>{chips(anchors, "anchor")}</div></div>
      </div>
    """


def render_frequency_table(record: Dict[str, Any]) -> str:
    counts, meta = compute_profile_frequency(record)
    rows = []
    for key, frequency in sorted(
        counts.items(),
        key=lambda item: (item[0][0], -item[1], item[0][1]),
    ):
        item = meta[key]
        rows.append(
            f"""
            <tr>
              <td><span class="badge {esc(item['pref_type'])}">{esc(item['pref_type'])}</span></td>
              <td><code>{esc(item['category'])}</code></td>
              <td>{esc(item['subcategory'])}</td>
              <td class="content-cell">{esc(item['content'])}</td>
              <td>{chips(item['sources'])}</td>
              <td class="num">{frequency}</td>
            </tr>
            """
        )
    if not rows:
        return '<p class="muted">No preference frequencies found.</p>'

    return f"""
      <table class="freq-table">
        <thead>
          <tr>
            <th>type</th>
            <th>category_id</th>
            <th>subcategory</th>
            <th>content</th>
            <th>sources</th>
            <th>frequency</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    """


def render_group(group: Dict[str, Any]) -> str:
    gid = group.get("group_id", "")
    explicit_prefs = [p for p in group.get("explicit_preferences", []) or [] if isinstance(p, dict)]
    implicit_prefs = [p for p in group.get("implicit_preferences", []) or [] if isinstance(p, dict)]
    recommended_scene = group.get("recommended_main_scene", "")
    guidance = group.get("implicit_integration_guidance", "")
    planned_date = group.get("planned_date", "")

    explicit_html = "".join(render_pref(pref, "explicit") for pref in explicit_prefs)
    implicit_html = "".join(render_pref(pref, "implicit") for pref in implicit_prefs)

    explicit_cats = group.get("explicit_categories", []) or []
    implicit_cats = group.get("implicit_categories", []) or []

    return f"""
      <article class="group-card" data-group-id="{esc(gid)}">
        <div class="group-title">
          <h3>Group {esc(gid)}</h3>
          <div class="cat-line">
            <span>explicit</span>{chips(explicit_cats)}
            <span>implicit</span>{chips(implicit_cats)}
            {f'<span>date</span><span class="chip date">{esc(planned_date)}</span>' if planned_date else ''}
          </div>
        </div>
        <div class="pref-grid">
          <section>
            <h4>Explicit Preference</h4>
            {explicit_html or '<p class="muted">none</p>'}
          </section>
          <section>
            <h4>Implicit Preference</h4>
            {implicit_html or '<p class="muted">none</p>'}
          </section>
        </div>
        <div class="scene-block">
          <div>
            <h4>Recommended Main Scene</h4>
            <p>{esc(recommended_scene) or '<span class="muted">empty</span>'}</p>
          </div>
          <div>
            <h4>Implicit Integration Guidance</h4>
            <p>{esc(guidance) or '<span class="muted">empty</span>'}</p>
          </div>
        </div>
      </article>
    """


def render_profile(record: Dict[str, Any], index: int) -> str:
    p_id = record.get("p_id", index)
    name = record.get("profile_name", "")
    groups = record.get("groups", []) or []
    group_html = "\n".join(render_group(group) for group in groups)
    return f"""
      <section class="profile-section" id="profile-{esc(p_id)}">
        <div class="profile-head">
          <div>
            <div class="eyebrow">p_id={esc(p_id)}</div>
            <h2>{esc(name)}</h2>
          </div>
          <div class="stats">
            <span><strong>{esc(record.get('num_explicit_prefs', ''))}</strong> explicit prefs</span>
            <span><strong>{esc(record.get('num_implicit_prefs', ''))}</strong> implicit prefs</span>
            <span><strong>{len(groups)}</strong> groups</span>
          </div>
        </div>

        <details open class="panel">
          <summary>Preference Frequency</summary>
          {render_frequency_table(record)}
        </details>

        <details open class="panel">
          <summary>Groups</summary>
          <div class="groups">{group_html}</div>
        </details>
      </section>
    """


def render_global_frequency(records: List[Dict[str, Any]]) -> str:
    global_counts: Counter[Tuple[int, str, str]] = Counter()
    global_meta: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
    for idx, record in enumerate(records):
        p_id = int(record.get("p_id", idx))
        name = record.get("profile_name", "")
        counts, meta = compute_profile_frequency(record)
        for (pref_type, category), frequency in counts.items():
            key = (p_id, pref_type, category)
            global_counts[key] += frequency
            item = meta[(pref_type, category)]
            global_meta[key] = {**item, "p_id": p_id, "profile_name": name}

    rows = []
    for key, frequency in sorted(global_counts.items(), key=lambda item: (item[0][0], item[0][1], -item[1], item[0][2])):
        item = global_meta[key]
        rows.append(
            f"""
            <tr>
              <td>{esc(item['p_id'])}</td>
              <td>{esc(item['profile_name'])}</td>
              <td><span class="badge {esc(item['pref_type'])}">{esc(item['pref_type'])}</span></td>
              <td><code>{esc(item['category'])}</code></td>
              <td>{esc(item['subcategory'])}</td>
              <td class="content-cell">{esc(item['content'])}</td>
              <td class="num">{frequency}</td>
            </tr>
            """
        )

    return f"""
      <details class="panel">
        <summary>Selected Profiles Frequency Overview</summary>
        <table class="freq-table">
          <thead>
            <tr>
              <th>p_id</th>
              <th>profile</th>
              <th>type</th>
              <th>category_id</th>
              <th>subcategory</th>
              <th>content</th>
              <th>frequency</th>
            </tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </details>
    """


def build_html(records: List[Dict[str, Any]], source_path: Path) -> str:
    profile_sections = "\n".join(render_profile(record, idx) for idx, record in enumerate(records))
    nav_links = "\n".join(
        f'<a href="#profile-{esc(record.get("p_id", idx))}">p{esc(record.get("p_id", idx))} {esc(record.get("profile_name", ""))}</a>'
        for idx, record in enumerate(records)
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Manual Groups Inspection</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: #ffffff;
      --line: #dfe5ef;
      --text: #1f2937;
      --muted: #6b7280;
      --blue: #2563eb;
      --green: #047857;
      --amber: #a16207;
      --soft-blue: #eef4ff;
      --soft-green: #ecfdf5;
      --soft-amber: #fffbeb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,.96);
      backdrop-filter: blur(8px);
      padding: 18px 24px;
    }}
    header h1 {{ margin: 0 0 6px; font-size: 22px; }}
    header p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 12px 24px;
      background: #fff;
      border-bottom: 1px solid var(--line);
    }}
    nav a {{
      text-decoration: none;
      color: var(--blue);
      background: var(--soft-blue);
      border: 1px solid #c7d8ff;
      padding: 6px 10px;
      border-radius: 7px;
      font-size: 13px;
      font-weight: 650;
    }}
    main {{ padding: 20px 24px 60px; }}
    .profile-section {{
      margin: 0 auto 24px;
      max-width: 1440px;
    }}
    .profile-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 12px;
    }}
    .profile-head h2 {{ margin: 0; font-size: 24px; }}
    .eyebrow {{ color: var(--muted); font-weight: 700; font-size: 12px; text-transform: uppercase; }}
    .stats {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .stats span {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 6px 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .stats strong {{ color: var(--text); }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 14px;
      overflow: hidden;
    }}
    summary {{
      cursor: pointer;
      font-weight: 750;
      padding: 12px 14px;
      background: #f8fafc;
      border-bottom: 1px solid var(--line);
    }}
    details:not([open]) summary {{ border-bottom: 0; }}
    .freq-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .freq-table th, .freq-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      vertical-align: top;
      text-align: left;
    }}
    .freq-table th {{
      color: #475569;
      background: #fbfdff;
      font-size: 12px;
    }}
    .content-cell {{ max-width: 560px; }}
    .num {{ text-align: right; font-weight: 800; }}
    .groups {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(560px, 1fr));
      gap: 12px;
      padding: 14px;
    }}
    .group-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }}
    .group-title {{
      padding: 12px;
      border-bottom: 1px solid var(--line);
      background: #fcfdff;
    }}
    .group-title h3 {{ margin: 0 0 8px; font-size: 17px; }}
    .cat-line {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
    }}
    .pref-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 12px;
    }}
    .pref-grid h4, .scene-block h4 {{
      margin: 0 0 7px;
      color: #334155;
      font-size: 13px;
    }}
    .pref {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 150px;
    }}
    .pref-explicit {{ background: var(--soft-blue); }}
    .pref-implicit {{ background: var(--soft-green); }}
    .pref-head {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .subcat {{ color: var(--muted); font-size: 12px; }}
    .content {{ font-weight: 650; margin-bottom: 8px; }}
    .kv {{
      display: grid;
      grid-template-columns: 60px 1fr;
      gap: 8px;
      margin-top: 6px;
      font-size: 12px;
    }}
    .kv > span:first-child {{ color: var(--muted); font-weight: 700; }}
    .chip, .anchor {{
      display: inline-block;
      margin: 0 5px 5px 0;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef2ff;
      color: #3730a3;
      border: 1px solid #dbe3ff;
      font-size: 12px;
      font-weight: 650;
    }}
    .anchor {{
      background: #fff7ed;
      color: #9a3412;
      border-color: #fed7aa;
    }}
    .date {{
      background: #fef3c7;
      color: var(--amber);
      border-color: #fde68a;
    }}
    .badge {{
      display: inline-block;
      border-radius: 6px;
      padding: 2px 6px;
      font-size: 12px;
      font-weight: 800;
      background: #e5e7eb;
      color: #374151;
    }}
    .badge.explicit {{ background: #dbeafe; color: #1d4ed8; }}
    .badge.implicit {{ background: #d1fae5; color: #047857; }}
    .scene-block {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 0 12px 12px;
    }}
    .scene-block > div {{
      border-top: 1px solid var(--line);
      padding-top: 10px;
    }}
    .scene-block p {{ margin: 0; }}
    .muted {{ color: var(--muted); }}
    code {{
      color: #334155;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 5px;
      padding: 1px 5px;
    }}
    @media (max-width: 900px) {{
      .profile-head, .pref-grid, .scene-block {{ grid-template-columns: 1fr; display: grid; }}
      .groups {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Manual Groups Inspection</h1>
    <p>Source: {esc(source_path)} · Profiles shown: {len(records)}</p>
  </header>
  <nav>{nav_links}</nav>
  <main>
    {render_global_frequency(records)}
    {profile_sections}
  </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an HTML page for inspecting manual preference groups.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to manual group JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output HTML.")
    parser.add_argument("--max-profiles", type=int, default=DEFAULT_MAX_PROFILES, help="Number of profiles to render; 0 means all.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if args.max_profiles < 0:
        raise ValueError("--max-profiles must be >= 0")

    records = read_json(input_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected top-level list in {input_path}")
    selected = records if args.max_profiles == 0 else records[: args.max_profiles]

    html_text = build_html(selected, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Profiles rendered: {len(selected)} / {len(records)}")


if __name__ == "__main__":
    main()
