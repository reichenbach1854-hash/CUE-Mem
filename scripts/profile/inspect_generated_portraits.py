#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an HTML gallery for all portrait/object reference images under
profile/generated_portraits, with the preference each image belongs to.

Default output:
  profile/generated_portraits_gallery.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from scripts.common.paths import project_path, resolve_path


DEFAULT_PROFILES = project_path("profile", "profiles_with_anchors_with_images_entity.json")
FALLBACK_PROFILES = [
    project_path("profile", "profiles_with_anchors_with_images_all.json"),
    project_path("profile", "profiles_with_anchors_with_items.json"),
    project_path("profile", "profiles_with_anchors.jsonl"),
]
DEFAULT_PORTRAITS_DIR = project_path("profile", "generated_portraits")
DEFAULT_MANIFEST = DEFAULT_PORTRAITS_DIR / "manifest.json"
DEFAULT_OUTPUT = project_path("profile", "generated_portraits_gallery.html")

PREFERENCE_CATEGORIES = [
    "FoodAndDrink",
    "HomeAndSpace",
    "BodyAndHealth",
    "HobbiesAndEntertainment",
    "WorkAndLearning",
    "MobilityAndTravel",
    "Relationship",
    "Pets",
]


def load_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def profile_name(profile: Dict[str, Any], p_id: int) -> str:
    basic = profile.get("Basic") or {}
    return clean_text(
        profile.get("profile_name")
        or profile.get("name")
        or basic.get("name")
        or f"profile_{p_id}"
    )


def pref_content(pref: Dict[str, Any]) -> str:
    return clean_text(pref.get("preference") or pref.get("content"))


def pref_sources(pref: Dict[str, Any]) -> List[str]:
    return [str(x) for x in (pref.get("evidence_sources") or pref.get("sources") or [])]


def pref_anchors(pref: Dict[str, Any]) -> List[str]:
    anchors = pref.get("entity_anchors")
    if anchors is None:
        anchors = pref.get("entity_anchor")
    return [clean_text(x) for x in as_list(anchors) if clean_text(x)]


def category_id(category: str, index: int) -> str:
    if category == "Relationship":
        return f"Relationship-{index}"
    if category == "Pets":
        return f"BasicPets-{index}"
    return f"{category}-{index}"


def basic_relation_record(profile: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    rels = ((profile.get("Basic") or {}).get("Relationship") or [])
    if not (0 <= index < len(rels)) or not isinstance(rels[index], dict):
        return None
    rel = rels[index]
    content_parts = [
        clean_text(rel.get("relation")),
        clean_text(rel.get("name")),
        clean_text(rel.get("info")),
        clean_text(rel.get("appearance")),
    ]
    return {
        "category_id": f"Relationship-{index}",
        "subcategory": clean_text(rel.get("relation")) or "Relationship",
        "expression_type": "explicit",
        "sources": ["basic"],
        "preference": "；".join(p for p in content_parts if p),
        "entity_anchors": [],
    }


def basic_pet_record(profile: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    pets = ((profile.get("Basic") or {}).get("Pets") or [])
    if not (0 <= index < len(pets)) or not isinstance(pets[index], dict):
        return None
    pet = pets[index]
    content_parts = [
        clean_text(pet.get("name")),
        clean_text(pet.get("info")),
        clean_text(pet.get("appearance")),
    ]
    return {
        "category_id": f"BasicPets-{index}",
        "subcategory": clean_text(pet.get("name")) or "Pets",
        "expression_type": "explicit",
        "sources": ["basic"],
        "preference": "；".join(p for p in content_parts if p),
        "entity_anchors": [],
    }


def iter_preference_records(profile: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for cat in PREFERENCE_CATEGORIES:
        values = profile.get(cat)
        if not isinstance(values, list):
            continue
        for idx, pref in enumerate(values):
            if not isinstance(pref, dict):
                continue
            yield {
                "category_id": category_id(cat, idx),
                "subcategory": clean_text(pref.get("subcategory")) or cat,
                "expression_type": clean_text(pref.get("expression_type")),
                "sources": pref_sources(pref),
                "preference": pref_content(pref),
                "entity_anchors": pref_anchors(pref),
            }


def build_item_preference_index(profile: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    for record in iter_preference_records(profile):
        for anchor in record["entity_anchors"]:
            index.setdefault(anchor, []).append(record)
    return index


def item_record(profile: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    items = profile.get("Items") or []
    if not (0 <= index < len(items)) or not isinstance(items[index], dict):
        return None
    item = items[index]
    desc = clean_text(item.get("description"))
    pref_index = build_item_preference_index(profile)
    prefs = list(pref_index.get(desc, []))

    # Some older Items can have source_subcategory only. Keep that context
    # even if exact anchor matching is not available.
    if not prefs:
        source_subcategory = clean_text(item.get("source_subcategory"))
        if source_subcategory:
            prefs.append(
                {
                    "category_id": "",
                    "subcategory": source_subcategory,
                    "expression_type": "",
                    "sources": [],
                    "preference": "",
                    "entity_anchors": [desc] if desc else [],
                }
            )
    return {
        "description": desc,
        "source_subcategory": clean_text(item.get("source_subcategory")),
        "preferences": prefs,
    }


def item_record_by_description(profile: Dict[str, Any], description: str) -> Optional[Dict[str, Any]]:
    desc = clean_text(description)
    if not desc:
        return None

    source_subcategory = ""
    for item in profile.get("Items") or []:
        if not isinstance(item, dict):
            continue
        if clean_text(item.get("description")) == desc:
            source_subcategory = clean_text(item.get("source_subcategory"))
            break

    pref_index = build_item_preference_index(profile)
    prefs = list(pref_index.get(desc, []))
    if not prefs and source_subcategory:
        prefs.append(
            {
                "category_id": "",
                "subcategory": source_subcategory,
                "expression_type": "",
                "sources": [],
                "preference": "",
                "entity_anchors": [desc],
            }
        )
    if not prefs and not source_subcategory:
        return None
    return {
        "description": desc,
        "source_subcategory": source_subcategory,
        "preferences": prefs,
    }


def description_from_source_name(source_name: str) -> str:
    # Manifest source names are usually "物品描述 (子类)".
    text = clean_text(source_name)
    match = re.match(r"^(.*?)\s*\([^()]*\)\s*$", text)
    return clean_text(match.group(1)) if match else text


def parse_portrait_filename(path: Path) -> Optional[Dict[str, Any]]:
    # profile_0_林悦_Items_xxx_4.png
    match = re.match(r"^profile_(\d+)_.+?_(Relationship|Pets|Items)_(.+)_(\d+)\.png$", path.name, re.I)
    if not match:
        return None
    return {
        "profile_id": int(match.group(1)),
        "type": match.group(2),
        "index": int(match.group(4)),
        "source_name": match.group(3),
        "file": str(path),
        "status": "file_only",
    }


def manifest_records(manifest_path: Path, portraits_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if manifest_path.exists():
        data = load_json_or_jsonl(manifest_path)
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    file_path = Path(clean_text(row.get("file")))
                    if not file_path.is_absolute():
                        file_path = portraits_dir / file_path
                    if file_path.exists() and file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                        rec = dict(row)
                        rec["file"] = str(file_path)
                        records.append(rec)

    seen = {str(Path(r["file"]).resolve()).lower() for r in records if r.get("file")}
    for img in sorted(portraits_dir.glob("*")):
        if img.name == "manifest.json" or img.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        key = str(img.resolve()).lower()
        if key in seen:
            continue
        parsed = parse_portrait_filename(img)
        if parsed:
            records.append(parsed)
        else:
            records.append(
                {
                    "profile_id": None,
                    "type": "Unknown",
                    "index": None,
                    "source_name": img.stem,
                    "file": str(img),
                    "status": "file_only",
                }
            )
    return records


def resolve_display_record(record: Dict[str, Any], profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    p_id = record.get("profile_id")
    try:
        p_id_int = int(p_id)
    except Exception:
        p_id_int = -1

    profile = profiles[p_id_int] if 0 <= p_id_int < len(profiles) else {}
    typ = clean_text(record.get("type"))
    try:
        idx = int(record.get("index"))
    except Exception:
        idx = -1

    preference_blocks: List[Dict[str, Any]] = []
    title = clean_text(record.get("source_name")) or Path(clean_text(record.get("file"))).stem
    subtitle = ""

    if profile:
        if typ == "Relationship":
            rel_record = basic_relation_record(profile, idx)
            if rel_record:
                preference_blocks = [rel_record]
                title = clean_text(((profile.get("Basic") or {}).get("Relationship") or [{}])[idx].get("name")) or title
                subtitle = rel_record["subcategory"]
        elif typ == "Pets":
            pet_record = basic_pet_record(profile, idx)
            if pet_record:
                preference_blocks = [pet_record]
                title = clean_text(((profile.get("Basic") or {}).get("Pets") or [{}])[idx].get("name")) or title
                subtitle = "Pets"
        elif typ == "Items":
            source_desc = description_from_source_name(clean_text(record.get("source_name")))
            item = None
            if source_desc:
                # For object portraits, the image filename / manifest source_name
                # is more stable than Items.index. Items order can change after
                # profile cleanup, while old image files keep their original name.
                item = item_record_by_description(profile, source_desc)
                if item is None:
                    item = {
                        "description": source_desc,
                        "source_subcategory": "",
                        "preferences": [],
                    }
            else:
                item = item_record(profile, idx)
            if item:
                title = item["description"] or title
                subtitle = item["source_subcategory"]
                preference_blocks = item["preferences"]

    return {
        "file": clean_text(record.get("file")),
        "profile_id": p_id_int if p_id_int >= 0 else "",
        "profile_name": profile_name(profile, p_id_int) if profile else "",
        "type": typ or "Unknown",
        "index": idx if idx >= 0 else "",
        "title": title,
        "subtitle": subtitle,
        "status": clean_text(record.get("status")),
        "image_model": clean_text(record.get("image_model")),
        "preference_blocks": preference_blocks,
    }


def rel_url(from_html: Path, target: Path) -> str:
    try:
        return Path(target).resolve().relative_to(from_html.resolve().parent).as_posix()
    except Exception:
        return Path(target).resolve().as_uri()


def render_pref_block(block: Dict[str, Any]) -> str:
    anchors = block.get("entity_anchors") or []
    sources = block.get("sources") or []
    anchor_html = "".join(f"<span>{esc(a)}</span>" for a in anchors) or "<em>无</em>"
    source_html = "".join(f"<span>{esc(s)}</span>" for s in sources) or "<em>无</em>"
    return f"""
      <div class="pref">
        <div class="pref-head">
          <strong>{esc(block.get("category_id") or "未匹配到具体 category")}</strong>
          <span>{esc(block.get("expression_type"))}</span>
          <span>{esc(block.get("subcategory"))}</span>
        </div>
        <p>{esc(block.get("preference") or "未在 profile 中找到完整偏好文本")}</p>
        <div class="chips"><b>sources</b>{source_html}</div>
        <div class="chips"><b>anchors</b>{anchor_html}</div>
      </div>
    """


def render_card(item: Dict[str, Any], output_path: Path) -> str:
    img_src = rel_url(output_path, Path(item["file"]))
    prefs = item.get("preference_blocks") or []
    prefs_html = "\n".join(render_pref_block(p) for p in prefs) or """
      <div class="pref missing">
        <p>未匹配到对应偏好。可检查 manifest 的 type/index 是否和 profile 文件一致。</p>
      </div>
    """
    search_blob = " ".join(
        [
            clean_text(item.get("profile_name")),
            clean_text(item.get("type")),
            clean_text(item.get("title")),
            clean_text(item.get("subtitle")),
            " ".join(clean_text(p.get("preference")) for p in prefs),
            " ".join(" ".join(p.get("entity_anchors") or []) for p in prefs),
        ]
    )
    return f"""
    <article class="card" data-profile="{esc(item.get("profile_id"))}" data-type="{esc(item.get("type"))}" data-search="{esc(search_blob.lower())}">
      <div class="image-wrap">
        <img src="{esc(img_src)}" loading="lazy" alt="{esc(item.get("title"))}">
      </div>
      <div class="body">
        <div class="meta">
          <span>p_id={esc(item.get("profile_id"))}</span>
          <span>{esc(item.get("profile_name"))}</span>
          <span>{esc(item.get("type"))}-{esc(item.get("index"))}</span>
        </div>
        <h2>{esc(item.get("title"))}</h2>
        <div class="sub">{esc(item.get("subtitle"))}</div>
        {prefs_html}
        <details>
          <summary>文件信息</summary>
          <code>{esc(item.get("file"))}</code>
          <div class="tiny">status={esc(item.get("status"))} | model={esc(item.get("image_model"))}</div>
        </details>
      </div>
    </article>
    """


def render_html(items: List[Dict[str, Any]], output_path: Path) -> str:
    cards = "\n".join(render_card(item, output_path) for item in items)
    profile_options = "\n".join(
        f'<option value="{esc(pid)}">p_id={esc(pid)}</option>'
        for pid in sorted({item["profile_id"] for item in items if item["profile_id"] != ""})
    )
    type_options = "\n".join(
        f'<option value="{esc(t)}">{esc(t)}</option>'
        for t in sorted({item["type"] for item in items})
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Generated Portraits Gallery</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #6b7280;
      --line: #d9dee7;
      --accent: #1d4ed8;
      --chip: #eef2ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(255,255,255,.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 22px;
    }}
    h1 {{ margin: 0 0 10px; font-size: 22px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 160px 160px;
      gap: 10px;
      max-width: 1180px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      padding: 9px 10px;
      color: var(--text);
    }}
    main {{
      padding: 22px;
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    .image-wrap {{
      background: #fafafa;
      border-bottom: 1px solid var(--line);
      aspect-ratio: 1 / 1;
      display: grid;
      place-items: center;
    }}
    img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: block;
    }}
    .body {{ padding: 13px 14px 14px; }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .meta span, .chips span {{
      background: var(--chip);
      border: 1px solid #dbe4ff;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      color: #243b7a;
    }}
    h2 {{
      margin: 0;
      font-size: 17px;
      line-height: 1.35;
    }}
    .sub {{ color: var(--muted); margin-top: 3px; min-height: 20px; }}
    .pref {{
      margin-top: 11px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
    }}
    .pref-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      align-items: center;
      margin-bottom: 6px;
    }}
    .pref-head strong {{
      color: var(--accent);
    }}
    .pref-head span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .pref p {{ margin: 0 0 8px; }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      align-items: center;
      margin-top: 5px;
    }}
    .chips b {{
      font-size: 12px;
      color: var(--muted);
      min-width: 58px;
    }}
    .missing {{ color: #9a3412; background: #fff7ed; }}
    details {{ margin-top: 10px; color: var(--muted); }}
    code {{
      display: block;
      margin-top: 6px;
      white-space: normal;
      word-break: break-all;
      font-size: 12px;
      background: #f3f4f6;
      padding: 6px;
      border-radius: 5px;
    }}
    .tiny {{ font-size: 12px; margin-top: 4px; }}
    .hidden {{ display: none; }}
    @media (max-width: 760px) {{
      .toolbar {{ grid-template-columns: 1fr; }}
      main {{ padding: 12px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Generated Portraits Gallery <span id="count"></span></h1>
    <div class="toolbar">
      <input id="q" placeholder="搜索人物、物品、偏好、anchor">
      <select id="profile"><option value="">全部 profile</option>{profile_options}</select>
      <select id="type"><option value="">全部类型</option>{type_options}</select>
    </div>
  </header>
  <main id="grid">{cards}</main>
  <script>
    const q = document.getElementById('q');
    const profile = document.getElementById('profile');
    const type = document.getElementById('type');
    const count = document.getElementById('count');
    const cards = Array.from(document.querySelectorAll('.card'));
    function apply() {{
      const query = q.value.trim().toLowerCase();
      const pid = profile.value;
      const typ = type.value;
      let visible = 0;
      for (const card of cards) {{
        const okQuery = !query || card.dataset.search.includes(query);
        const okPid = !pid || card.dataset.profile === pid;
        const okType = !typ || card.dataset.type === typ;
        const show = okQuery && okPid && okType;
        card.classList.toggle('hidden', !show);
        if (show) visible++;
      }}
      count.textContent = `(${{visible}}/${{cards.length}})`;
    }}
    q.addEventListener('input', apply);
    profile.addEventListener('change', apply);
    type.addEventListener('change', apply);
    apply();
  </script>
</body>
</html>
"""


def choose_profiles_path(path: Path) -> Path:
    if path.exists():
        return path
    for fallback in FALLBACK_PROFILES:
        if fallback.exists():
            return fallback
    raise FileNotFoundError(f"Profile file not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成定妆照 HTML 检查页面")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES), help="profile JSON/JSONL path")
    parser.add_argument("--portraits_dir", default=str(DEFAULT_PORTRAITS_DIR), help="generated_portraits directory")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="manifest.json path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output HTML path")
    args = parser.parse_args()

    profiles_path = choose_profiles_path(resolve_path(args.profiles))
    portraits_dir = resolve_path(args.portraits_dir)
    manifest_path = resolve_path(args.manifest)
    output_path = resolve_path(args.output)

    profiles = load_json_or_jsonl(profiles_path)
    if not isinstance(profiles, list):
        raise ValueError(f"profiles must be a list: {profiles_path}")

    records = manifest_records(manifest_path, portraits_dir)
    records.sort(key=lambda r: (r.get("profile_id") is None, r.get("profile_id") or -1, str(r.get("type")), r.get("index") or -1, str(r.get("source_name"))))
    resolved_items = [resolve_display_record(record, profiles) for record in records]
    items = [item for item in resolved_items if item.get("preference_blocks")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_html(items, output_path), encoding="utf-8")

    unmatched = len(resolved_items) - len(items)
    print(f"Profiles: {len(profiles)} from {profiles_path}")
    print(f"Images: {len(items)}")
    print(f"Skipped unmatched images: {unmatched}")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
