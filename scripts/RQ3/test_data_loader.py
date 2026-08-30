"""Quick test for data_loader module."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import PROFILE_FILES
from .data_loader import (
    build_text_for_turn,
    clue_to_turn_ids,
    flatten_turns,
    load_profile,
    resolve_media_path,
)

p = load_profile(PROFILE_FILES[0])
print("Profile:", p["profile"]["name"])
print("p_id:", p["p_id"])
print("Sessions:", len(p["sessions"]))

turns = flatten_turns(p["sessions"])
print("Total turns:", len(turns))

voice_turns = [t for t in turns if t["voice_paths"]]
img_turns = [t for t in turns if t["image_paths"]]
text_only = len(turns) - len(voice_turns) - len(img_turns)
print(f"Voice: {len(voice_turns)}, Image: {len(img_turns)}, Text-only: {text_only}")
print("QAs:", len(p["qas"]))

# Test text building
t = voice_turns[0]
print(f"\n--- Turn {t['turn_id']} ---")
text = build_text_for_turn(t)
print(text[:300])

# Test media path resolution
if t["voice_paths"]:
    raw = t["voice_paths"][0]
    resolved = resolve_media_path(raw)
    print(f"\nVoice path: {raw}")
    print(f"Resolved: {resolved}")
    print(f"Exists: {resolved.exists() if resolved else False}")

if img_turns:
    it = img_turns[0]
    raw = it["image_paths"][0]
    resolved = resolve_media_path(raw)
    print(f"\nImage path: {raw}")
    print(f"Resolved: {resolved}")
    print(f"Exists: {resolved.exists() if resolved else False}")

# Test clue resolution
qa = p["qas"][0]
print(f"\n--- QA {qa['qa_id']} ---")
print(f"Point: {qa['point']}, qa_type: {qa['qa_type']}")
print(f"Clue (first 5): {qa['clue'][:5]}")
evidence = clue_to_turn_ids(qa["clue"], p["sessions"])
print(f"Evidence turns ({len(evidence)}): {evidence}")

# QA type distribution
from collections import Counter

point_counts = Counter(q["point"] for q in p["qas"])
print(f"\nQA point distribution: {dict(point_counts)}")
