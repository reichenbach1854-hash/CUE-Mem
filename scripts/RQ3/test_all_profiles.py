"""Verify all 3 profiles load correctly and media paths resolve."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "RQ3"

from .config import PROFILE_FILES
from .data_loader import flatten_turns, load_profile, resolve_media_path

for i, pf in enumerate(PROFILE_FILES):
    p = load_profile(pf)
    turns = flatten_turns(p["sessions"])
    voice_turns = [t for t in turns if t["voice_paths"]]
    img_turns = [t for t in turns if t["image_paths"]]

    # Check voice path resolution
    voice_ok = 0
    voice_fail = 0
    for t in voice_turns:
        rp = resolve_media_path(t["voice_paths"][0])
        if rp and rp.exists():
            voice_ok += 1
        else:
            voice_fail += 1
            if voice_fail <= 2:
                print(f"  VOICE FAIL: {t['voice_paths'][0]}")

    # Check image path resolution
    img_ok = 0
    img_fail = 0
    for t in img_turns:
        rp = resolve_media_path(t["image_paths"][0])
        if rp and rp.exists():
            img_ok += 1
        else:
            img_fail += 1
            if img_fail <= 2:
                print(f"  IMAGE FAIL: {t['image_paths'][0]}")

    print(f"p{i}: {len(p['sessions'])} sessions, {len(turns)} turns, "
          f"{len(p['qas'])} QAs | "
          f"voice {voice_ok}/{len(voice_turns)} ok, "
          f"image {img_ok}/{len(img_turns)} ok")
