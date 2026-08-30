"""Build the derived profile artifact used by later profile/event scripts.

This is the single replacement for the former ``gen_profile_w_items.py`` and
``attach_events_to_profile.py`` pair.  It performs four deterministic steps:

1. collect and de-duplicate preference ``entity_anchors`` into ``Items``;
2. fill missing Basic evidence from event records;
3. clear stale event mounts;
4. mount current events onto preference entries and matching Items.

The input files are never modified.  The output defaults to a new file next
to the profile input.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from scripts.common.io import clean_text, load_record_list, write_json
from scripts.common.paths import project_path, resolve_path


TOP_LEVEL_CATEGORIES = (
    "FoodAndDrink",
    "HomeAndSpace",
    "BodyAndHealth",
    "HobbiesAndEntertainment",
    "WorkAndLearning",
    "MobilityAndTravel",
)
BASIC_CATEGORIES = ("Relationship", "Pets")
CATEGORY_ALIASES = {
    "BasicPets": "Pets",
    "BasicRelationship": "Relationship",
    "Pet": "Pets",
}


def parse_category_key(value: Any) -> tuple[str, int] | None:
    if not isinstance(value, str) or "-" not in value:
        return None
    name, index_text = value.rsplit("-", 1)
    try:
        return CATEGORY_ALIASES.get(name, name), int(index_text)
    except ValueError:
        return None


def category_entries(profile: dict[str, Any], category: str) -> list[dict[str, Any]]:
    if category in TOP_LEVEL_CATEGORIES:
        value = profile.get(category, [])
    elif category in BASIC_CATEGORIES:
        basic = profile.get("Basic") or {}
        value = basic.get(category, []) if isinstance(basic, dict) else []
        if not isinstance(value, list) or not value:
            value = profile.get(category, [])
    else:
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def locate_preference(profile: dict[str, Any], category_key: Any) -> dict[str, Any] | None:
    parsed = parse_category_key(category_key)
    if parsed is None:
        return None
    category, index = parsed
    entries = category_entries(profile, category)
    return entries[index] if 0 <= index < len(entries) else None


def iter_preference_entries(profile: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for category in (*TOP_LEVEL_CATEGORIES, *BASIC_CATEGORIES):
        yield from category_entries(profile, category)


def anchors_from(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return []


def collect_items(profile: dict[str, Any]) -> int:
    profile["Items"] = []
    seen: set[str] = set()
    for category in (*TOP_LEVEL_CATEGORIES, *BASIC_CATEGORIES):
        for preference in category_entries(profile, category):
            subcategory = clean_text(preference.get("subcategory"))
            raw_anchors = preference.get("entity_anchors", preference.get("entity_anchor", []))
            for anchor in anchors_from(raw_anchors):
                if anchor in seen:
                    continue
                seen.add(anchor)
                profile["Items"].append(
                    {
                        "description": anchor,
                        "source_category": category,
                        "source_subcategory": subcategory,
                        "event": "",
                        "source_task_id": None,
                    }
                )
    return len(profile["Items"])


def fill_basic_evidence(profile: dict[str, Any], record: dict[str, Any]) -> None:
    for field in ("explicit_preferences", "implicit_preferences"):
        preferences = record.get(field) or []
        if not isinstance(preferences, list):
            continue
        for preference in preferences:
            if not isinstance(preference, dict):
                continue
            parsed = parse_category_key(preference.get("category"))
            if parsed is None or parsed[0] not in BASIC_CATEGORIES:
                continue
            target = locate_preference(profile, preference.get("category"))
            if target is None:
                continue
            sources = preference.get("sources", preference.get("evidence_sources"))
            rationale = preference.get("rationale", preference.get("analysis"))
            if "evidence_sources" not in target and isinstance(sources, list):
                target["evidence_sources"] = list(sources)
            if "analysis" not in target and isinstance(rationale, list):
                target["analysis"] = list(rationale)


def clear_event_mounts(profile: dict[str, Any]) -> None:
    for preference in iter_preference_entries(profile):
        preference["events"] = []
    for item in profile.get("Items", []) or []:
        if isinstance(item, dict):
            item["event"] = ""
            item["source_task_id"] = None


def event_object(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("event")
    return value if isinstance(value, dict) else {}


def event_categories(record: dict[str, Any]) -> list[str]:
    event = event_object(record)
    values: list[Any] = []
    for key in ("explicit_preferences_reflected", "implicit_preferences_reflected"):
        values.extend(event.get(key) or [])
    if not values:
        for key in ("explicit_preferences", "implicit_preferences"):
            for preference in record.get(key) or []:
                if isinstance(preference, dict) and preference.get("category"):
                    values.append(preference["category"])
    return list(dict.fromkeys(clean_text(value) for value in values if clean_text(value)))


def compact_event(record: dict[str, Any]) -> dict[str, Any]:
    event = event_object(record)
    result: dict[str, Any] = {
        "task_id": record.get("task_id"),
        "group_id": record.get("group_id"),
        "p_id": record.get("p_id"),
        "recommended_main_scene": record.get("recommended_main_scene", ""),
    }
    for key in (
        "scene_description",
        "user_shared_image_description",
        "background_audio_info",
        "human_speech_content",
        "entity_anchor",
        "entity_anchors",
        "explicit_preferences_reflected",
        "implicit_preferences_reflected",
    ):
        if key in event:
            result[key] = event[key]
    return result


def event_text(record: dict[str, Any]) -> str:
    event = event_object(record)
    parts: list[str] = []
    scene = clean_text(event.get("scene_description"))
    image = clean_text(event.get("user_shared_image_description"))
    audio = clean_text(event.get("background_audio_info"))
    if scene and scene.casefold() != "none":
        parts.append(scene)
    if image and image.casefold() != "none":
        parts.append(f"图像描述：{image}")
    if audio and audio.casefold() != "none":
        parts.append(f"背景音：{audio}")
    return "\n".join(parts)


def selected_event_anchors(record: dict[str, Any]) -> list[str]:
    event = event_object(record)
    values = event.get("entity_anchors", event.get("entity_anchor", []))
    values = anchors_from(values)
    values.extend(anchors_from(record.get("entity_anchors", record.get("entity_anchor", []))))
    return list(dict.fromkeys(values))


def same_profile(left: Any, right: Any) -> bool:
    try:
        return int(left) == int(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def mount_events(profile: dict[str, Any], records: list[dict[str, Any]]) -> tuple[int, int]:
    p_id = profile.get("p_id")
    mounted_preferences = 0
    mounted_items = 0
    for record in records:
        if not same_profile(record.get("p_id"), p_id) or not record.get("task_id"):
            continue
        info = compact_event(record)
        for category_key in event_categories(record):
            target = locate_preference(profile, category_key)
            if target is None:
                continue
            events = target.setdefault("events", [])
            if any(isinstance(item, dict) and item.get("task_id") == info["task_id"] for item in events):
                continue
            events.append(info)
            mounted_preferences += 1

        description = event_text(record)
        if not description:
            continue
        anchors = selected_event_anchors(record)
        image_description = clean_text(event_object(record).get("user_shared_image_description"))
        for item in profile.get("Items", []) or []:
            if not isinstance(item, dict) or item.get("event"):
                continue
            item_description = clean_text(item.get("description"))
            if not item_description:
                continue
            anchor_match = any(
                item_description == anchor
                or item_description in anchor
                or anchor in item_description
                for anchor in anchors
            )
            text_match = bool(image_description and item_description in image_description)
            if anchor_match or text_match:
                item["event"] = description
                item["source_task_id"] = info["task_id"]
                mounted_items += 1
    return mounted_preferences, mounted_items


def derive_output_path(profile_path: Path) -> Path:
    return profile_path.with_name(f"{profile_path.stem}_with_items.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        default=str(project_path("profile", "profiles_with_anchors.jsonl")),
        help="profile JSON/JSONL input",
    )
    parser.add_argument(
        "--events",
        default=str(project_path("event", "events_with_anchors.jsonl")),
        help="event JSON/JSONL input",
    )
    parser.add_argument("--output", default=None, help="derived profile output path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    profiles_path = resolve_path(args.profiles)
    events_path = resolve_path(args.events)
    output_path = resolve_path(args.output) if args.output else derive_output_path(profiles_path)
    profiles = load_record_list(profiles_path)
    records = load_record_list(events_path)
    for index, profile in enumerate(profiles):
        profile.setdefault("p_id", index)

    total_items = 0
    for profile in profiles:
        total_items += collect_items(profile)
        clear_event_mounts(profile)
    for record in records:
        for profile in profiles:
            if same_profile(profile.get("p_id"), record.get("p_id")):
                fill_basic_evidence(profile, record)
                break

    mounted_preferences = mounted_items = 0
    for profile in profiles:
        preference_count, item_count = mount_events(profile, records)
        mounted_preferences += preference_count
        mounted_items += item_count
    write_json(output_path, profiles, indent=2)
    print(f"profiles={len(profiles)} events={len(records)}")
    print(f"items={total_items} mounted_preferences={mounted_preferences} mounted_items={mounted_items}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
