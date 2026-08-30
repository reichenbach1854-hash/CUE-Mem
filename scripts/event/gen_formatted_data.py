"""Convert task-level dialogue records into profile-grouped QA data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from typing import Any

from scripts.common.io import load_record_list, write_json
from scripts.common.paths import project_path, resolve_path


DEFAULT_INPUT = project_path("event", "dialogue_000_019_with_assets.jsonl")
DEFAULT_OUTPUT = project_path("qa", "qa_formatted_data_000_019.json")


def parse_scene_date(scene_description: str) -> datetime:
    return datetime.strptime(scene_description.split(";", 1)[0].strip(), "%m/%d/%Y")


def pick_audio_description(event: dict[str, Any]) -> str:
    human_speech = str(event.get("human_speech_content") or "").strip()
    background_audio = str(event.get("background_audio_info") or "").strip()
    if human_speech and human_speech.casefold() != "none":
        return human_speech
    if background_audio and background_audio.casefold() != "none":
        return background_audio
    return ""


def build_dialog_list(event: dict[str, Any], session_id: str) -> list[dict[str, Any]]:
    turns = event.get("dialog") or []
    dialog_list: list[dict[str, Any]] = []
    round_index = image_index = audio_index = 0
    i = 0
    while i < len(turns):
        user_turn = turns[i]
        if not isinstance(user_turn, dict) or user_turn.get("role") != "user":
            i += 1
            continue
        assistant_content = ""
        if i + 1 < len(turns) and isinstance(turns[i + 1], dict) and turns[i + 1].get("role") == "assistant":
            assistant_content = turns[i + 1].get("content", "") or ""
            i += 2
        else:
            i += 1
        item: dict[str, Any] = {
            "round": f"{session_id}:{round_index:02d}",
            "user": user_turn.get("content", "") or "",
            "assistant": assistant_content,
        }
        if user_turn.get("image_path"):
            image_index += 1
            item[f"{session_id}-{image_index:03d}.png"] = str(user_turn["image_path"]).strip()
        if user_turn.get("audio_path"):
            audio_index += 1
            item[f"{session_id}-{audio_index:03d}.wav"] = str(user_turn["audio_path"]).strip()
        dialog_list.append(item)
        round_index += 1
    return dialog_list


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    records = load_record_list(resolve_path(args.input))
    grouped: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if "p_id" in record:
            grouped[record["p_id"]].append(record)

    output: list[dict[str, Any]] = []
    for p_id in sorted(grouped):
        records_for_profile = sorted(
            grouped[p_id],
            key=lambda record: parse_scene_date(
                str((record.get("event") or {}).get("scene_description", ""))
            ),
        )
        events: list[dict[str, Any]] = []
        for index, record in enumerate(records_for_profile):
            event_data = dict(record.get("event") or {})
            event_data.update(
                {
                    "session_id": f"D{index:02d}",
                    "dialog_list": build_dialog_list(event_data, f"D{index:02d}"),
                    "task_id": record.get("task_id"),
                    "group_id": record.get("group_id"),
                    "recommended_main_scene": record.get("recommended_main_scene"),
                    "explicit_preferences": record.get("explicit_preferences", []),
                    "implicit_preferences": record.get("implicit_preferences", []),
                }
            )
            events.append(event_data)
        output.append({
            "p_id": p_id,
            "profile_str": records_for_profile[0].get("profile_str", ""),
            "events": events,
        })

    output_path = resolve_path(args.output)
    write_json(output_path, output, indent=4)
    print(f"records={len(records)} profiles={len(output)} events={sum(len(x['events']) for x in output)}")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
