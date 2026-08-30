#!/usr/bin/env python3
"""Generate character preference groups with an OpenAI-compatible chat API.

The script reads local JSON/JSONL profiles, asks an LLM to create one-explicit /
one-implicit preference groups, validates the response locally, and asks the LLM
to repair invalid drafts.  Preference payloads in the output are always rebuilt
from the input profile so model output cannot silently alter anchors or evidence.

Requires Python 3.10+ and ``openai>=1.0.0``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from scripts.common.llm import env_value, openai_client
from scripts.common.paths import project_path, resolve_path


# ---------------------------------------------------------------------------
# Default configuration. Every value can be overridden by the command line.
# ---------------------------------------------------------------------------
INPUT_PATH = str(project_path("profile", "profiles_with_anchors.jsonl"))
OUTPUT_GROUPS_PATH = str(project_path("event", "manual_profiles_with_anchors_groups.json"))
OUTPUT_FREQUENCY_CSV = str(project_path("event", "manual_profiles_with_anchors_frequency.csv"))
OUTPUT_SUMMARY_CSV = str(project_path("event", "manual_profiles_with_anchors_summary.csv"))
MODEL = os.getenv("CUE_MEM_LLM_MODEL", "deepseek-v4-pro")
TEMPERATURE = 0.2
REASONING_EFFORT = "high"
MAX_RETRIES = 3
MAX_REPAIR_RETRIES = 2
TIMEOUT = 180.0
MAX_WORKERS = 1
MAX_COMPLETION_TOKENS = 24_000
MIN_GROUPS = 30
MAX_GROUPS = 45  # 30-45 is the strict validation range requested by the user.


PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class PreferenceCandidate:
    """A canonical preference, with an output payload copied from its source."""

    category: str
    subcategory: str
    content: str
    sources: list[str]
    expression_type: str
    anchor_fields: dict[str, Any] = field(default_factory=dict)

    def to_output_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "category": self.category,
            "subcategory": self.subcategory,
            "content": self.content,
            "sources": list(self.sources),
        }
        # Keep the exact anchor field spelling used by the source profile.
        value.update(self.anchor_fields)
        return value

    def manifest_dict(self) -> dict[str, Any]:
        """The compact but lossless preference description provided to the LLM."""

        value = self.to_output_dict()
        value["expression_type"] = self.expression_type
        return value


@dataclass
class NormalizedProfile:
    p_id: int
    profile_name: str
    basic: dict[str, Any]
    explicit: list[PreferenceCandidate]
    implicit: list[PreferenceCandidate]

    @property
    def preference_map(self) -> dict[str, PreferenceCandidate]:
        return {pref.category: pref for pref in self.explicit + self.implicit}

    @property
    def explicit_map(self) -> dict[str, PreferenceCandidate]:
        return {pref.category: pref for pref in self.explicit}

    @property
    def implicit_map(self) -> dict[str, PreferenceCandidate]:
        return {pref.category: pref for pref in self.implicit}

    def prompt_manifest(self) -> dict[str, Any]:
        return {
            "p_id": self.p_id,
            "profile_name": self.profile_name,
            "basic_context": self.basic,
            "explicit_preferences": [pref.manifest_dict() for pref in self.explicit],
            "implicit_preferences": [pref.manifest_dict() for pref in self.implicit],
        }


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls


@dataclass
class LLMCallResult:
    content: str
    usage: TokenUsage


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    explicit_frequency: Counter[str]
    implicit_frequency: Counter[str]
    explicit_covered: list[str]
    implicit_covered: list[str]
    explicit_unused: list[str]
    implicit_unused: list[str]


@dataclass
class ProcessResult:
    p_id: int
    profile_name: str
    record: dict[str, Any]
    validation: ValidationResult
    usage: TokenUsage
    elapsed_seconds: float
    repair_rounds: int


def log(message: str) -> None:
    """Avoid interleaved progress lines when API calls run concurrently."""

    with PRINT_LOCK:
        print(message, flush=True)


def coerce_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).strip()


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [coerce_string(item) for item in value if coerce_string(item)]
    converted = coerce_string(value)
    return [converted] if converted else []


def remove_markdown_fences(text: str) -> str:
    text = text.strip().lstrip("\ufeff")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|JSON)?\s*", "", text, count=1)
        text = re.sub(r"\s*```\s*$", "", text, count=1)
    return text.strip()


def remove_trailing_commas(text: str) -> str:
    """Conservative cleanup for a common form of otherwise-valid dirty JSON."""

    return re.sub(r",\s*([}\]])", r"\1", text)


def extract_balanced_json(text: str) -> Optional[str]:
    """Return the first balanced JSON object/array while respecting strings."""

    start_positions = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    if not start_positions:
        return None
    start = min(start_positions)
    opening = text[start]
    closing_for = {"{": "}", "[": "]"}
    stack = [opening]
    in_string = False
    escaped = False

    for index in range(start + 1, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in closing_for:
            stack.append(char)
        elif char in ("}", "]"):
            if not stack or closing_for[stack[-1]] != char:
                return None
            stack.pop()
            if not stack:
                return text[start : index + 1]
    return None


def parse_loose_json(text: str) -> Any:
    """Parse fenced JSON, prose-wrapped JSON, and simple trailing-comma JSON."""

    cleaned = remove_markdown_fences(text)
    candidates = [cleaned, remove_trailing_commas(cleaned)]
    balanced = extract_balanced_json(cleaned)
    if balanced:
        candidates.extend([balanced, remove_trailing_commas(balanced)])

    errors: list[str] = []
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{exc.msg} at line {exc.lineno}, column {exc.colno}")
    raise ValueError("; ".join(dict.fromkeys(errors)) or "unable to parse JSON")


def unwrap_profile_container(value: Any) -> list[dict[str, Any]]:
    """Convert a JSON list, a single profile object, or known wrapper keys to profiles."""

    if isinstance(value, Mapping):
        for wrapper_key in ("profiles", "items", "data", "results"):
            nested = value.get(wrapper_key)
            if isinstance(nested, list):
                value = nested
                break
        else:
            return [dict(value)]

    if not isinstance(value, list):
        raise ValueError("input top level must be a JSON object, JSON list, or JSONL records")

    # Some APIs/files accidentally emit [[profile, ...]]. Unwrap only singleton
    # list layers so a valid list of profile dictionaries is never flattened.
    while len(value) == 1 and isinstance(value[0], list):
        value = value[0]

    profiles = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(profiles) != len(value):
        raise ValueError("every profile in the input list must be a JSON object")
    return profiles


def load_profiles(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON, JSONL, or a slightly dirty JSON document from local disk."""

    input_path = resolve_path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"input profile file does not exist: {input_path}")
    raw = input_path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        raise ValueError(f"input profile file is empty: {input_path}")

    try:
        return unwrap_profile_container(parse_loose_json(raw))
    except ValueError as whole_document_error:
        records: list[dict[str, Any]] = []
        line_errors: list[str] = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            candidate = line.strip().rstrip(",")
            if not candidate or candidate in ("[", "]"):
                continue
            try:
                parsed = parse_loose_json(candidate)
            except ValueError as exc:
                line_errors.append(f"line {line_number}: {exc}")
                continue
            if not isinstance(parsed, Mapping):
                line_errors.append(f"line {line_number}: JSONL item is not an object")
                continue
            records.append(dict(parsed))
        if records and not line_errors:
            return records
        details = " | ".join(line_errors[:5])
        raise ValueError(
            f"failed to parse {input_path} as JSON ({whole_document_error}) or JSONL ({details})"
        ) from whole_document_error


def basic_candidate(
    category: str,
    subcategory: str,
    item: Any,
) -> PreferenceCandidate:
    """Turn Basic.Relationship/Pets entries into explicit grouping candidates."""

    if isinstance(item, Mapping):
        parts: list[str] = []
        name = coerce_string(item.get("name"))
        relation = coerce_string(item.get("relation"))
        info = coerce_string(item.get("info"))
        appearance = coerce_string(item.get("appearance"))
        if name:
            parts.append(name)
        if relation:
            parts.append(f"关系：{relation}")
        if info:
            parts.append(info)
        if appearance:
            parts.append(f"外观：{appearance}")
        content = "；".join(parts) or json.dumps(item, ensure_ascii=False)
        anchor_fields = {
            key: item[key]
            for key in ("entity_anchor", "entity_anchors")
            if key in item
        }
    else:
        content = coerce_string(item)
        anchor_fields = {}

    # Basic records have no explicit modality evidence. Labeling their descriptive
    # appearance as a visual source would incorrectly permit visual implicit pairs.
    return PreferenceCandidate(
        category=category,
        subcategory=subcategory,
        content=content,
        sources=["basic"],
        expression_type="explicit",
        anchor_fields=anchor_fields,
    )


def preference_candidate(category: str, item: Mapping[str, Any]) -> PreferenceCandidate:
    anchors = {
        key: item[key]
        for key in ("entity_anchor", "entity_anchors")
        if key in item
    }
    sources = normalize_string_list(item.get("sources", item.get("evidence_sources", [])))
    return PreferenceCandidate(
        category=category,
        subcategory=coerce_string(item.get("subcategory", "")),
        content=coerce_string(item.get("content", item.get("preference", ""))),
        sources=sources,
        expression_type=coerce_string(item.get("expression_type", "")).lower(),
        anchor_fields=anchors,
    )


def iter_preference_category_lists(profile: Mapping[str, Any]) -> Iterable[tuple[str, list[Any]]]:
    """Yield top-level category lists in source order, with a nested fallback."""

    ignored = {"Basic", "basic", "id", "p_id", "profile_name", "name", "persona", "mbti"}
    found = False
    for key, value in profile.items():
        if key in ignored or key == "preferences":
            continue
        if isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            found = True
            yield str(key), value

    nested = profile.get("preferences")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
                yield str(key), value
    elif not found:
        # No silent fallback to arbitrary recursion: a deeply nested profile may
        # carry unrelated lists, and grouping them would change category indexing.
        return


def normalize_profile(profile: Mapping[str, Any], input_index: int) -> NormalizedProfile:
    """Create stable category IDs and preserve source order/anchor spellings."""

    basic_value = profile.get("Basic", profile.get("basic", {}))
    basic = dict(basic_value) if isinstance(basic_value, Mapping) else {}
    raw_id = profile.get("id", profile.get("p_id", input_index))
    try:
        p_id = int(raw_id)
    except (TypeError, ValueError):
        p_id = input_index
    profile_name = coerce_string(basic.get("name", profile.get("profile_name", profile.get("name", ""))))
    profile_name = profile_name or f"profile_{p_id}"

    explicit: list[PreferenceCandidate] = []
    implicit: list[PreferenceCandidate] = []

    relationships = basic.get("Relationship", [])
    relationship_items = relationships if isinstance(relationships, list) else [relationships]
    for index, item in enumerate(relationship_items):
        if item is None:
            continue
        explicit.append(basic_candidate(f"Relationship-{index}", "Relationship", item))

    pets = basic.get("Pets", [])
    pet_items = pets if isinstance(pets, list) else [pets]
    for index, item in enumerate(pet_items):
        if item is None:
            continue
        explicit.append(basic_candidate(f"BasicPets-{index}", "Basic.Pets", item))

    for category_name, items in iter_preference_category_lists(profile):
        for item_index, item in enumerate(items):
            if not isinstance(item, Mapping):
                continue
            expression_type = coerce_string(item.get("expression_type", "")).lower()
            if expression_type not in {"explicit", "implicit"}:
                continue
            candidate = preference_candidate(f"{category_name}-{item_index}", item)
            if not candidate.content:
                # Keep it in the manifest/validator rather than renumbering later
                # entries, so category indices always remain source-order indices.
                candidate = PreferenceCandidate(
                    category=candidate.category,
                    subcategory=candidate.subcategory,
                    content="",
                    sources=candidate.sources,
                    expression_type=candidate.expression_type,
                    anchor_fields=candidate.anchor_fields,
                )
            (explicit if expression_type == "explicit" else implicit).append(candidate)

    categories = [pref.category for pref in explicit + implicit]
    if len(categories) != len(set(categories)):
        raise ValueError(f"profile {p_id} has duplicate normalized category IDs")
    return NormalizedProfile(p_id, profile_name, basic, explicit, implicit)


def has_visual_source(preference: PreferenceCandidate | Mapping[str, Any]) -> bool:
    sources = preference.sources if isinstance(preference, PreferenceCandidate) else preference.get("sources", [])
    for source in normalize_string_list(sources):
        normalized = source.casefold()
        if normalized in {"visual", "image", "vision", "图片", "图像", "视觉"}:
            return True
    return False


def is_basic_visual_subject(preference: PreferenceCandidate | Mapping[str, Any]) -> bool:
    """Basic Relationship/Pets entries may serve as visual subjects without anchors."""

    category = preference.category if isinstance(preference, PreferenceCandidate) else preference.get("category", "")
    return coerce_string(category).startswith(("Relationship-", "BasicPets-"))


def is_explicit_visual_subject(preference: PreferenceCandidate | Mapping[str, Any]) -> bool:
    """Explicit preferences that must be paired with implicit visual preferences."""

    return has_visual_source(preference) or is_basic_visual_subject(preference)


def can_pair_with_visual_implicit(preference: PreferenceCandidate | Mapping[str, Any]) -> bool:
    if is_basic_visual_subject(preference):
        return True
    return has_visual_source(preference) and bool(anchor_texts(preference))


def anchor_texts(preference: PreferenceCandidate | Mapping[str, Any]) -> list[str]:
    if isinstance(preference, PreferenceCandidate):
        fields = preference.anchor_fields
    else:
        fields = {
            key: preference[key]
            for key in ("entity_anchor", "entity_anchors")
            if key in preference
        }

    values: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            if value.strip():
                values.append(value.strip())
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for nested in value:
                visit(nested)
        else:
            converted = coerce_string(value)
            if converted:
                values.append(converted)

    for value in fields.values():
        visit(value)
    return values


def feasibility_errors(profile: NormalizedProfile, strict_implicit_gt_explicit: bool) -> list[str]:
    """Catch mathematical/data impossibilities before spending API tokens."""

    errors: list[str] = []
    explicit_count = len(profile.explicit)
    implicit_count = len(profile.implicit)
    if explicit_count == 0:
        errors.append("profile has no explicit preferences")
    if implicit_count == 0:
        errors.append("profile has no implicit preferences")

    required_groups = max(MIN_GROUPS, explicit_count * 2, implicit_count * 3)
    if required_groups > MAX_GROUPS:
        errors.append(
            f"minimum required group count is {required_groups}, exceeding {MAX_GROUPS} "
            f"(explicit={explicit_count}, implicit={implicit_count})"
        )

    visual_explicit = [pref for pref in profile.explicit if can_pair_with_visual_implicit(pref)]
    visual_implicit = [pref for pref in profile.implicit if has_visual_source(pref)]
    visual_subject_explicit = [pref for pref in profile.explicit if is_explicit_visual_subject(pref)]
    non_visual_subject_explicit = [
        pref for pref in profile.explicit if not is_explicit_visual_subject(pref)
    ]
    non_visual_implicit = [pref for pref in profile.implicit if not has_visual_source(pref)]
    if visual_implicit and not visual_explicit:
        errors.append(
            "profile has implicit visual preferences but no explicit candidate allowed for visual pairing"
        )
    visual_partition_min_groups = max(
        2 * len(visual_subject_explicit),
        3 * len(visual_implicit),
    )
    non_visual_partition_min_groups = max(
        2 * len(non_visual_subject_explicit),
        3 * len(non_visual_implicit),
    )
    if visual_partition_min_groups + non_visual_partition_min_groups > MAX_GROUPS:
        errors.append(
            "minimum required group count exceeds the maximum after enforcing visual-subject pairing: "
            f"visual_partition={visual_partition_min_groups}, "
            f"non_visual_partition={non_visual_partition_min_groups}, "
            f"total={visual_partition_min_groups + non_visual_partition_min_groups}, max={MAX_GROUPS}"
        )
    for preference in visual_implicit:
        if not anchor_texts(preference):
            errors.append(f"implicit visual preference {preference.category} has no entity anchor")
    for preference in profile.explicit:
        if has_visual_source(preference) and not is_basic_visual_subject(preference) and not anchor_texts(preference):
            # This does not make all grouping impossible, but it cannot be used for
            # an implicit visual pair, which is important enough to tell the model.
            errors.append(f"explicit visual preference {preference.category} has no entity anchor")

    if strict_implicit_gt_explicit:
        errors.append(
            "strict implicit_total_frequency > explicit_total_frequency is impossible "
            "when every group has exactly one explicit and one implicit preference"
        )
    return errors


def build_generation_prompt(profile: NormalizedProfile) -> str:
    manifest = json.dumps(profile.prompt_manifest(), ensure_ascii=False, indent=2)
    return f"""You are creating preference groups for one character profile.
Return ONLY one valid JSON object. Do not use Markdown, code fences, explanations, or comments.

Your job is to pair the supplied preference candidates into {MIN_GROUPS} to {MAX_GROUPS} groups.
Each group MUST contain exactly one explicit preference and exactly one implicit preference.
Use only category IDs from the supplied manifest. Copy the selected preference dictionaries exactly;
do not invent, paraphrase, rename, omit, or modify entity_anchor/entity_anchors fields.

Hard constraints:
1. Every explicit candidate appears in at least 2 groups; every implicit candidate appears in at least 3 groups.
2. Every candidate must be covered. Spread partners: do not repeatedly bind an explicit candidate
   to only one implicit candidate when other natural pairings are available.
3. The explicit and implicit preference in each pair must naturally co-occur in one coherent
   real-world spatial event; prefer natural compatibility over mechanical coverage.
4. If an implicit candidate has source "visual", its explicit partner must be visually usable.
   Normally this means the explicit partner has source "visual" and an entity_anchor/entity_anchors.
   Exception: Basic Relationship-* and BasicPets-* may also be used as visual subjects even though
   their source is "basic" and they may have no entity anchor. When this exception is used, the
   person/pet described by the Basic item is the centered main subject. For non-Basic visual
   pairings, the explicit anchor is the centered main subject and the implicit anchor can only be
   at an edge/corner.
5. If an explicit candidate is a visual subject, its implicit partner MUST have source "visual".
   Visual subjects include:
   - any explicit preference whose sources include "visual";
   - Relationship-* preferences;
   - BasicPets-* preferences.
   Therefore explicit visual/Relationship/BasicPets + implicit non-visual is invalid. Implicit
   non-visual preferences must be paired only with explicit preferences that are not visual subjects.
6. Implicit audio does NOT require an explicit audio partner.
7. Basic Relationship-* and BasicPets-* are explicit candidates. They must be covered like other
   explicit preferences and are always treated as visual subjects for pairing constraints.
8. recommended_main_scene is mandatory, concise, and describes ONLY the explicit preference's
   main scene. It MUST NOT contain any implicit object, action, environment clue, audio clue,
   hint, edge/corner placement, or indirect reference. Do not use words such as "background sound",
   "edge", "corner", "implicit", "hint", "背景音", "边缘", "角落", "隐式", or "暗示".
9. If a perfect plan is difficult, preserve all hard constraints and choose the most natural
   spatial co-occurrence. Never fabricate an invalid category or preference.

Return this exact top-level shape:
{{
  "groups": [
    {{
      "group_id": 0,
      "explicit_categories": ["<one explicit category ID>"],
      "implicit_categories": ["<one implicit category ID>"],
      "explicit_preferences": [{{"category": "...", "subcategory": "...", "content": "...", "sources": []}}],
      "implicit_preferences": [{{"category": "...", "subcategory": "...", "content": "...", "sources": []}}],
      "recommended_main_scene": "only an explicit-preference main scene"
    }}
  ],
  "coverage": {{
    "explicit_covered": [], "implicit_covered": [],
    "explicit_unused": [], "implicit_unused": []
  }}
}}

The local program will recompute coverage and restore preference dictionaries from the source;
nevertheless include coverage in your response. group_id must be consecutive integers starting at 0.

SOURCE PROFILE MANIFEST:
{manifest}
"""


def build_repair_prompt(
    profile: NormalizedProfile,
    current_payload: Any,
    validation_errors: Sequence[str],
) -> str:
    manifest = json.dumps(profile.prompt_manifest(), ensure_ascii=False, indent=2)
    current = json.dumps(current_payload, ensure_ascii=False, indent=2)
    errors = json.dumps(list(validation_errors), ensure_ascii=False, indent=2)
    return f"""Repair an invalid character preference grouping.
Return ONLY one valid JSON object, with no Markdown or explanation.

Keep all already-valid groups unchanged whenever possible, but return the complete replacement
object with both "groups" and "coverage". Use only the supplied source category IDs. Each group
must contain exactly one explicit and exactly one implicit preference, group IDs 0..N-1, and there
must be {MIN_GROUPS}-{MAX_GROUPS} groups. Every explicit appears >=2 times and every implicit >=3.
Avoid fixed pairings; use natural shared spatial scenes. For implicit visual, the explicit partner
must be visually usable: either a normal explicit visual item with entity_anchor/entity_anchors, or
a Basic Relationship-* / BasicPets-* item used as the centered person/pet subject without requiring
an entity anchor. For normal visual pairs, both anchors must naturally fit in one photograph
(explicit centered, implicit only edge/corner).
Additionally, every explicit visual subject MUST pair with an implicit visual preference. Explicit
visual subjects are explicit preferences whose sources include visual, plus Relationship-* and
BasicPets-* preferences. explicit visual/Relationship/BasicPets + implicit non-visual is invalid;
implicit non-visual preferences may only pair with explicit preferences that are not visual subjects.
recommended_main_scene is required and can describe ONLY the explicit main scene: it may not mention
or hint at the implicit item/action/environment/audio, nor say background sound/edge/corner/implicit/hint
(including 背景音、边缘、角落、隐式、暗示).

LOCAL VALIDATOR ERRORS:
{errors}

CURRENT INVALID PAYLOAD:
{current}

SOURCE PROFILE MANIFEST:
{manifest}
"""


def usage_value(usage: Any, name: str) -> int:
    if usage is None:
        return 0
    value = getattr(usage, name, None)
    if value is None and isinstance(usage, Mapping):
        value = usage.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                chunks.append(coerce_string(item.get("text", item.get("content", ""))))
            else:
                chunks.append(coerce_string(item))
        return "".join(chunks)
    return coerce_string(content)


def call_llm(args: argparse.Namespace, prompt: str) -> LLMCallResult:
    """Call an OpenAI-compatible Chat Completions endpoint with backoff."""
    client = openai_client(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    last_error: Optional[Exception] = None
    for attempt in range(1, args.max_retries + 1):
        request: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": "You are a precise JSON-only data generation service."},
                {"role": "user", "content": prompt},
            ],
            "temperature": args.temperature,
        }
        if args.seed is not None:
            request["seed"] = args.seed
        if args.reasoning_effort != "none":
            request["reasoning_effort"] = args.reasoning_effort
        if args.max_completion_tokens > 0:
            request[args.completion_limit_param] = args.max_completion_tokens

        try:
            response = client.chat.completions.create(**request)
            if not getattr(response, "choices", None):
                raise RuntimeError("API response has no choices")
            text = message_content_to_text(response.choices[0].message.content)
            if not text.strip():
                raise RuntimeError("API response contains an empty message")
            usage = TokenUsage(
                prompt_tokens=usage_value(getattr(response, "usage", None), "prompt_tokens"),
                completion_tokens=usage_value(getattr(response, "usage", None), "completion_tokens"),
                calls=1,
            )
            return LLMCallResult(text, usage)
        except Exception as exc:  # network/provider errors are intentionally retried
            last_error = exc
            if attempt >= args.max_retries:
                break
            delay = min(30.0, 2.0 ** (attempt - 1)) + random.uniform(0.0, 0.5)
            log(f"  API attempt {attempt}/{args.max_retries} failed: {exc}; retrying in {delay:.1f}s")
            time.sleep(delay)

    raise RuntimeError(f"LLM call failed after {args.max_retries} attempt(s): {last_error}")


def parse_llm_json(text: str) -> Any:
    """Parse an LLM response and unwrap harmless singleton result wrappers."""

    value = parse_loose_json(text)
    while True:
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, Mapping)):
            value = value[0]
            continue
        if isinstance(value, Mapping):
            for key in ("result", "data", "output"):
                nested = value.get(key)
                if isinstance(nested, (list, Mapping)):
                    value = nested
                    break
            else:
                break
            continue
        break
    return value


def as_category_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [coerce_string(item) for item in value if coerce_string(item)]


def categories_from_raw_group(raw_group: Mapping[str, Any], kind: str) -> Any:
    category_key = f"{kind}_categories"
    preferences_key = f"{kind}_preferences"
    if category_key in raw_group:
        return raw_group.get(category_key)
    preferences = raw_group.get(preferences_key)
    if isinstance(preferences, list):
        return [coerce_string(item.get("category")) for item in preferences if isinstance(item, Mapping)]
    return []


def materialize_groups(profile: NormalizedProfile, payload: Any) -> tuple[list[dict[str, Any]], Any]:
    """Make source-authoritative output groups while retaining invalid structure for repair."""

    if isinstance(payload, Mapping):
        raw_groups = payload.get("groups", [])
    else:
        raw_groups = payload
    if isinstance(raw_groups, list) and len(raw_groups) == 1 and isinstance(raw_groups[0], list):
        raw_groups = raw_groups[0]
    if not isinstance(raw_groups, list):
        return [], {"groups": raw_groups}

    preference_map = profile.preference_map
    groups: list[dict[str, Any]] = []
    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            groups.append({"group_id": None, "raw_group": raw_group})
            continue
        explicit_categories_raw = categories_from_raw_group(raw_group, "explicit")
        implicit_categories_raw = categories_from_raw_group(raw_group, "implicit")
        explicit_categories = as_category_list(explicit_categories_raw)
        implicit_categories = as_category_list(implicit_categories_raw)

        def materialize_preferences(categories: list[str]) -> list[dict[str, Any]]:
            values: list[dict[str, Any]] = []
            for category in categories:
                preference = preference_map.get(category)
                if preference is None:
                    values.append({"category": category, "subcategory": "", "content": "", "sources": []})
                else:
                    values.append(preference.to_output_dict())
            return values

        groups.append(
            {
                "group_id": raw_group.get("group_id", index),
                "explicit_categories": explicit_categories_raw,
                "implicit_categories": implicit_categories_raw,
                "explicit_preferences": materialize_preferences(explicit_categories),
                "implicit_preferences": materialize_preferences(implicit_categories),
                "recommended_main_scene": coerce_string(raw_group.get("recommended_main_scene", "")),
            }
        )
    return groups, {"groups": raw_groups}


GENERIC_CLUES = {
    "喜欢", "偏好", "习惯", "日常", "用户", "使用", "需要", "认为", "能够", "表明", "说明",
    "一种", "这个", "那个", "进行", "以及", "自己", "画面", "用户在", "她在", "生活",
}
FORBIDDEN_SCENE_TERMS = (
    "背景音", "背景声音", "背景里", "背景中", "边缘", "角落", "隐式", "暗示",
    "background sound", "at the edge", "in the corner", "implicit", "hint",
)


def implicit_scene_clues(preference: PreferenceCandidate) -> set[str]:
    """Extract conservative Chinese/English phrase clues for scene leakage checks."""

    clues: set[str] = set(anchor_texts(preference))
    if len(preference.subcategory) >= 2:
        clues.add(preference.subcategory)

    content = preference.content.strip()
    if content:
        for phrase in re.split(r"[，。；、：:！!？?（）()\n]+", content):
            phrase = phrase.strip()
            if len(phrase) >= 3:
                clues.add(phrase)
            for run in re.findall(r"[\u4e00-\u9fff]{3,}", phrase):
                for width in range(3, min(6, len(run)) + 1):
                    clues.update(run[start : start + width] for start in range(0, len(run) - width + 1))
            clues.update(word for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", phrase))

    return {
        clue.strip()
        for clue in clues
        if len(clue.strip()) >= 3 and clue.strip() not in GENERIC_CLUES
    }


def validate_groups(
    profile: NormalizedProfile,
    groups: list[dict[str, Any]],
    strict_implicit_gt_explicit: bool = False,
) -> ValidationResult:
    """Strict structural, coverage, modality, pairing, and scene-leakage validation."""

    errors: list[str] = []
    warnings: list[str] = []
    explicit_map = profile.explicit_map
    implicit_map = profile.implicit_map
    explicit_frequency: Counter[str] = Counter()
    implicit_frequency: Counter[str] = Counter()
    explicit_partners: dict[str, set[str]] = defaultdict(set)
    implicit_partners: dict[str, set[str]] = defaultdict(set)

    if not isinstance(groups, list):
        errors.append("groups must be a list")
        groups = []
    if not (MIN_GROUPS <= len(groups) <= MAX_GROUPS):
        errors.append(f"num_groups must be in {MIN_GROUPS}-{MAX_GROUPS}; got {len(groups)}")

    for expected_id, group in enumerate(groups):
        if not isinstance(group, Mapping):
            errors.append(f"group {expected_id}: group must be an object")
            continue
        prefix = f"group {group.get('group_id', expected_id)!r}"
        if group.get("group_id") != expected_id:
            errors.append(f"{prefix}: group_id must be consecutive; expected {expected_id}")

        explicit_categories_raw = group.get("explicit_categories")
        implicit_categories_raw = group.get("implicit_categories")
        explicit_preferences = group.get("explicit_preferences")
        implicit_preferences = group.get("implicit_preferences")
        if not isinstance(explicit_categories_raw, list):
            errors.append(f"{prefix}: explicit_categories must be a list")
        if not isinstance(implicit_categories_raw, list):
            errors.append(f"{prefix}: implicit_categories must be a list")
        if not isinstance(explicit_preferences, list):
            errors.append(f"{prefix}: explicit_preferences must be a list")
        if not isinstance(implicit_preferences, list):
            errors.append(f"{prefix}: implicit_preferences must be a list")

        explicit_categories = as_category_list(explicit_categories_raw)
        implicit_categories = as_category_list(implicit_categories_raw)
        exp_prefs = explicit_preferences if isinstance(explicit_preferences, list) else []
        imp_prefs = implicit_preferences if isinstance(implicit_preferences, list) else []
        if len(explicit_categories) != 1 or len(exp_prefs) != 1:
            errors.append(f"{prefix}: must contain exactly 1 explicit category and 1 explicit preference")
        if len(implicit_categories) != 1 or len(imp_prefs) != 1:
            errors.append(f"{prefix}: must contain exactly 1 implicit category and 1 implicit preference")
        if len(set(explicit_categories)) != len(explicit_categories):
            errors.append(f"{prefix}: explicit_categories contains duplicates")
        if len(set(implicit_categories)) != len(implicit_categories):
            errors.append(f"{prefix}: implicit_categories contains duplicates")

        actual_exp_categories = [coerce_string(pref.get("category")) for pref in exp_prefs if isinstance(pref, Mapping)]
        actual_imp_categories = [coerce_string(pref.get("category")) for pref in imp_prefs if isinstance(pref, Mapping)]
        if explicit_categories != actual_exp_categories:
            errors.append(f"{prefix}: explicit_categories and explicit_preferences categories disagree")
        if implicit_categories != actual_imp_categories:
            errors.append(f"{prefix}: implicit_categories and implicit_preferences categories disagree")

        explicit = explicit_map.get(explicit_categories[0]) if len(explicit_categories) == 1 else None
        implicit = implicit_map.get(implicit_categories[0]) if len(implicit_categories) == 1 else None
        if len(explicit_categories) == 1 and explicit is None:
            errors.append(f"{prefix}: {explicit_categories[0]} is not a known explicit category")
        if len(implicit_categories) == 1 and implicit is None:
            errors.append(f"{prefix}: {implicit_categories[0]} is not a known implicit category")
        if explicit is not None:
            explicit_frequency[explicit.category] += 1
        if implicit is not None:
            implicit_frequency[implicit.category] += 1
        if explicit is not None and implicit is not None:
            explicit_partners[explicit.category].add(implicit.category)
            implicit_partners[implicit.category].add(explicit.category)

            if is_explicit_visual_subject(explicit) and not has_visual_source(implicit):
                errors.append(
                    f"{prefix}: explicit visual subject {explicit.category} must pair with an implicit visual preference; "
                    f"got non-visual implicit {implicit.category}"
                )
            if has_visual_source(implicit):
                if not can_pair_with_visual_implicit(explicit):
                    errors.append(
                        f"{prefix}: implicit visual {implicit.category} requires a visual-capable explicit partner"
                    )
                if not is_basic_visual_subject(explicit) and not anchor_texts(explicit):
                    errors.append(f"{prefix}: visual explicit {explicit.category} lacks an entity anchor")
                if not anchor_texts(implicit):
                    errors.append(f"{prefix}: visual implicit {implicit.category} lacks an entity anchor")

        scene = coerce_string(group.get("recommended_main_scene", ""))
        if not scene:
            errors.append(f"{prefix}: recommended_main_scene is empty")
        else:
            scene_folded = scene.casefold()
            for term in FORBIDDEN_SCENE_TERMS:
                if term.casefold() in scene_folded:
                    errors.append(f"{prefix}: recommended_main_scene contains forbidden term {term!r}")
            if implicit is not None:
                matched = next(
                    (clue for clue in implicit_scene_clues(implicit) if clue.casefold() in scene_folded),
                    None,
                )
                if matched:
                    errors.append(
                        f"{prefix}: recommended_main_scene leaks implicit preference clue {matched!r}"
                    )

    for candidate in profile.explicit:
        frequency = explicit_frequency[candidate.category]
        if frequency < 2:
            errors.append(f"explicit {candidate.category} appears {frequency} time(s); requires at least 2")
        if frequency >= 2 and len(profile.implicit) >= 2 and len(explicit_partners[candidate.category]) < 2:
            errors.append(f"explicit {candidate.category} is locked to one implicit partner")
    for candidate in profile.implicit:
        frequency = implicit_frequency[candidate.category]
        if frequency < 3:
            errors.append(f"implicit {candidate.category} appears {frequency} time(s); requires at least 3")
        if frequency >= 3 and len(profile.explicit) >= 2 and len(implicit_partners[candidate.category]) < 2:
            errors.append(f"implicit {candidate.category} is locked to one explicit partner")

    explicit_total = sum(explicit_frequency.values())
    implicit_total = sum(implicit_frequency.values())
    if strict_implicit_gt_explicit and not implicit_total > explicit_total:
        errors.append(
            f"implicit total frequency ({implicit_total}) must exceed explicit total frequency ({explicit_total})"
        )
    elif implicit_total == explicit_total and explicit_total:
        warnings.append(
            "implicit and explicit total frequencies are equal by design: each group has exactly one of each"
        )

    explicit_covered = [candidate.category for candidate in profile.explicit if explicit_frequency[candidate.category] > 0]
    implicit_covered = [candidate.category for candidate in profile.implicit if implicit_frequency[candidate.category] > 0]
    explicit_unused = [candidate.category for candidate in profile.explicit if explicit_frequency[candidate.category] == 0]
    implicit_unused = [candidate.category for candidate in profile.implicit if implicit_frequency[candidate.category] == 0]
    return ValidationResult(
        is_valid=not errors,
        errors=errors,
        warnings=warnings,
        explicit_frequency=explicit_frequency,
        implicit_frequency=implicit_frequency,
        explicit_covered=explicit_covered,
        implicit_covered=implicit_covered,
        explicit_unused=explicit_unused,
        implicit_unused=implicit_unused,
    )


def build_profile_record(
    profile: NormalizedProfile,
    groups: list[dict[str, Any]],
    validation: ValidationResult,
) -> dict[str, Any]:
    return {
        "p_id": profile.p_id,
        "profile_name": profile.profile_name,
        "num_explicit_prefs": len(profile.explicit),
        "num_implicit_prefs": len(profile.implicit),
        "num_groups": len(groups),
        "groups": groups,
        "coverage": {
            "explicit_covered": validation.explicit_covered,
            "implicit_covered": validation.implicit_covered,
            "explicit_unused": validation.explicit_unused,
            "implicit_unused": validation.implicit_unused,
        },
    }


def process_profile(profile: NormalizedProfile, args: argparse.Namespace) -> ProcessResult:
    started = time.monotonic()
    usage = TokenUsage()
    preflight = feasibility_errors(profile, args.strict_implicit_gt_explicit)
    if preflight:
        validation = validate_groups(profile, [], args.strict_implicit_gt_explicit)
        validation.errors = preflight + validation.errors
        validation.is_valid = False
        return ProcessResult(
            profile.p_id,
            profile.profile_name,
            build_profile_record(profile, [], validation),
            validation,
            usage,
            time.monotonic() - started,
            0,
        )

    current_payload: Any = {"groups": []}
    groups: list[dict[str, Any]] = []
    validation: Optional[ValidationResult] = None
    errors_for_repair: list[str] = []

    for generation_round in range(args.max_repair_retries + 1):
        if generation_round == 0:
            prompt = build_generation_prompt(profile)
            phase = "draft"
        else:
            prompt = build_repair_prompt(profile, current_payload, errors_for_repair)
            phase = f"repair {generation_round}/{args.max_repair_retries}"

        try:
            call = call_llm(args, prompt)
            usage.add(call.usage)
            try:
                parsed = parse_llm_json(call.content)
            except ValueError as exc:
                current_payload = {"unparsed_response": call.content[:8_000]}
                groups = []
                validation = validate_groups(profile, groups, args.strict_implicit_gt_explicit)
                validation.errors.insert(0, f"LLM JSON parse failed: {exc}")
                validation.is_valid = False
                errors_for_repair = validation.errors
                if args.verbose:
                    log(f"[{profile.p_id}] {phase}: JSON parse failed; response starts: {call.content[:500]!r}")
                continue
            groups, current_payload = materialize_groups(profile, parsed)
            validation = validate_groups(profile, groups, args.strict_implicit_gt_explicit)
            if validation.is_valid:
                break
            errors_for_repair = validation.errors
            if args.verbose:
                log(f"[{profile.p_id}] {phase}: {len(validation.errors)} validation issue(s)")
        except Exception as exc:
            validation = validate_groups(profile, groups, args.strict_implicit_gt_explicit)
            validation.errors.insert(0, f"{phase} API failure: {exc}")
            validation.is_valid = False
            errors_for_repair = validation.errors
            break

    assert validation is not None
    record = build_profile_record(profile, groups, validation)
    return ProcessResult(
        profile.p_id,
        profile.profile_name,
        record,
        validation,
        usage,
        time.monotonic() - started,
        min(args.max_repair_retries, max(0, usage.calls - 1)),
    )


def extract_record_groups(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = record.get("groups", [])
    return groups if isinstance(groups, list) else []


def compute_frequency(profile: NormalizedProfile, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return one CSV row per source preference, including zero-frequency entries."""

    explicit_counter: Counter[str] = Counter()
    implicit_counter: Counter[str] = Counter()
    for group in extract_record_groups(record):
        if not isinstance(group, Mapping):
            continue
        explicit_counter.update(as_category_list(group.get("explicit_categories")))
        implicit_counter.update(as_category_list(group.get("implicit_categories")))

    rows: list[dict[str, Any]] = []
    for pref_type, preferences, counter in (
        ("explicit", profile.explicit, explicit_counter),
        ("implicit", profile.implicit, implicit_counter),
    ):
        for preference in preferences:
            rows.append(
                {
                    "p_id": profile.p_id,
                    "profile_name": profile.profile_name,
                    "pref_type": pref_type,
                    "category_id": preference.category,
                    "subcategory": preference.subcategory,
                    "content": preference.content,
                    "frequency": counter[preference.category],
                }
            )
    return rows


def recompute_record_validation(profile: NormalizedProfile, record: Mapping[str, Any], args: argparse.Namespace) -> ValidationResult:
    return validate_groups(profile, extract_record_groups(record), args.strict_implicit_gt_explicit)


def atomic_write_text(path: Path, content: str) -> None:
    """Replace an output atomically so an interrupted checkpoint stays usable."""

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def save_outputs(
    records: list[dict[str, Any]],
    profiles_by_id: Mapping[int, NormalizedProfile],
    args: argparse.Namespace,
) -> list[ValidationResult]:
    """Write groups JSON plus frequency/summary CSV files from the final records."""

    for output_path in (args.output_groups, args.output_frequency, args.output_summary):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    frequency_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validations: list[ValidationResult] = []
    for record in records:
        p_id = record.get("p_id")
        profile = profiles_by_id.get(p_id)
        if profile is None:
            continue
        validation = recompute_record_validation(profile, record, args)
        validations.append(validation)
        # A resumed record may come from an older run. Keep its derived fields
        # authoritative and in sync with the source profile and current groups.
        record["profile_name"] = profile.profile_name
        record["num_explicit_prefs"] = len(profile.explicit)
        record["num_implicit_prefs"] = len(profile.implicit)
        record["num_groups"] = len(extract_record_groups(record))
        record["coverage"] = {
            "explicit_covered": validation.explicit_covered,
            "implicit_covered": validation.implicit_covered,
            "explicit_unused": validation.explicit_unused,
            "implicit_unused": validation.implicit_unused,
        }
        frequency_rows.extend(compute_frequency(profile, record))
        summary_rows.append(
            {
                "p_id": profile.p_id,
                "profile_name": profile.profile_name,
                "num_explicit_prefs": len(profile.explicit),
                "num_implicit_prefs": len(profile.implicit),
                "num_groups": len(extract_record_groups(record)),
                "explicit_total_frequency": sum(validation.explicit_frequency.values()),
                "implicit_total_frequency": sum(validation.implicit_frequency.values()),
                "explicit_covered_count": len(validation.explicit_covered),
                "implicit_covered_count": len(validation.implicit_covered),
                "explicit_unused_count": len(validation.explicit_unused),
                "implicit_unused_count": len(validation.implicit_unused),
                "is_valid": validation.is_valid,
            }
        )

    atomic_write_csv(
        Path(args.output_frequency),
        ["p_id", "profile_name", "pref_type", "category_id", "subcategory", "content", "frequency"],
        frequency_rows,
    )
    atomic_write_csv(
        Path(args.output_summary),
        [
            "p_id", "profile_name", "num_explicit_prefs", "num_implicit_prefs", "num_groups",
            "explicit_total_frequency", "implicit_total_frequency", "explicit_covered_count",
            "implicit_covered_count", "explicit_unused_count", "implicit_unused_count", "is_valid",
        ],
        summary_rows,
    )
    atomic_write_text(Path(args.output_groups), json.dumps(records, ensure_ascii=False, indent=2))
    return validations


def load_existing_records(path: str | Path) -> dict[int, dict[str, Any]]:
    output_path = Path(path)
    if not output_path.is_file():
        return {}
    parsed = parse_loose_json(output_path.read_text(encoding="utf-8-sig"))
    if not isinstance(parsed, list):
        raise ValueError(f"existing groups file must contain a JSON list: {output_path}")
    records: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, Mapping) or "p_id" not in item:
            continue
        try:
            records[int(item["p_id"])] = dict(item)
        except (TypeError, ValueError):
            continue
    return records


def parse_profile_ids(values: Optional[list[str]]) -> Optional[set[int]]:
    if not values:
        return None
    result: set[int] = set()
    for raw in values:
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                result.add(int(part))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(f"invalid profile id: {part!r}") from exc
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=INPUT_PATH, help="local JSON or JSONL profile file")
    parser.add_argument("--output_groups", default=OUTPUT_GROUPS_PATH)
    parser.add_argument("--output_frequency", default=OUTPUT_FREQUENCY_CSV)
    parser.add_argument("--output_summary", default=OUTPUT_SUMMARY_CSV)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api_key", default=None, help="runtime API key override; otherwise CUE_MEM_LLM_API_KEY")
    parser.add_argument("--base_url", default=None, help="runtime OpenAI-compatible endpoint override")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument(
        "--reasoning_effort",
        choices=("none", "low", "medium", "high"),
        default=REASONING_EFFORT,
        help="reasoning budget sent to supported APIs; default: high",
    )
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES, help="retries for one API request")
    parser.add_argument("--max_repair_retries", type=int, default=MAX_REPAIR_RETRIES)
    parser.add_argument("--timeout", type=float, default=TIMEOUT, help="per-request timeout in seconds")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="parallel profile requests; 1 disables concurrency")
    parser.add_argument("--only_profile_ids", nargs="*", help="IDs, e.g. --only_profile_ids 0 2,5")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="process only the first N selected profiles in source order; useful for a small API test",
    )
    parser.add_argument("--resume", action="store_true", help="skip p_id values already valid in output_groups")
    parser.add_argument(
        "--checkpoint_every", type=int, default=1,
        help="atomically save outputs after every N completed profiles; default: 1",
    )
    parser.add_argument("--dry_run", action="store_true", help="print generation prompts without calling the API")
    parser.add_argument("--seed", type=int, default=None, help="passed only when specified and supported by the API")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max_completion_tokens", type=int, default=MAX_COMPLETION_TOKENS)
    parser.add_argument(
        "--completion_limit_param",
        choices=("max_completion_tokens", "max_tokens"),
        default="max_completion_tokens",
        help="use max_tokens for older OpenAI-compatible servers",
    )
    parser.add_argument(
        "--strict_implicit_gt_explicit",
        action="store_true",
        help="enforce the mathematically incompatible implicit_total > explicit_total rule",
    )
    return parser


def print_result(result: ProcessResult) -> None:
    state = "VALID" if result.validation.is_valid else "INVALID"
    log(
        f"[{state}] p_id={result.p_id} name={result.profile_name!r} "
        f"groups={result.record['num_groups']} repairs={result.repair_rounds} "
        f"tokens={result.usage.prompt_tokens}+{result.usage.completion_tokens} "
        f"time={result.elapsed_seconds:.1f}s"
    )
    if not result.validation.is_valid:
        for error in result.validation.errors[:8]:
            log(f"  - {error}")
        if len(result.validation.errors) > 8:
            log(f"  - ... {len(result.validation.errors) - 8} more validation error(s)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_retries < 1 or args.max_repair_retries < 0 or args.workers < 1:
        parser.error("max_retries/workers must be >= 1 and max_repair_retries must be >= 0")
    if args.sample is not None and args.sample < 1:
        parser.error("--sample must be >= 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint_every must be >= 1")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("temperature must be between 0 and 2")

    started = time.monotonic()
    raw_profiles = load_profiles(args.input)
    normalized_profiles = [normalize_profile(profile, index) for index, profile in enumerate(raw_profiles)]
    profiles_by_id = {profile.p_id: profile for profile in normalized_profiles}
    if len(profiles_by_id) != len(normalized_profiles):
        raise ValueError("profile IDs must be unique; duplicate id/p_id values were found")

    only_ids = parse_profile_ids(args.only_profile_ids)
    selected = [profile for profile in normalized_profiles if only_ids is None or profile.p_id in only_ids]
    if only_ids is not None:
        missing = sorted(only_ids - set(profiles_by_id))
        if missing:
            raise ValueError(f"requested profile IDs are not present in input: {missing}")
    if not selected:
        raise ValueError("no profiles selected")
    if args.sample is not None:
        selected = selected[:args.sample]
        log(f"Sample mode: selected the first {len(selected)} profile(s) in source order.")

    if args.dry_run:
        for profile in selected:
            print(f"\n{'=' * 24} p_id={profile.p_id} {profile.profile_name} {'=' * 24}")
            print(build_generation_prompt(profile))
        log(f"Dry run complete: printed {len(selected)} prompt(s); no API call or output file was made.")
        return 0
    if not args.api_key and not env_value("CUE_MEM_LLM_API_KEY"):
        parser.error("--api_key is required (or set CUE_MEM_LLM_API_KEY)")

    records_by_id = load_existing_records(args.output_groups) if args.resume else {}
    completed_ids: set[int] = set()
    if args.resume:
        for profile in selected:
            existing = records_by_id.get(profile.p_id)
            if existing is not None and recompute_record_validation(profile, existing, args).is_valid:
                completed_ids.add(profile.p_id)
        skipped = len(completed_ids)
        if skipped:
            log(f"Resume: skipping {skipped} profile(s) already valid in {args.output_groups}")
        invalid_to_retry = sum(1 for profile in selected if profile.p_id in records_by_id and profile.p_id not in completed_ids)
        if invalid_to_retry:
            log(f"Resume: reprocessing {invalid_to_retry} invalid/incomplete profile(s).")
    todo = [profile for profile in selected if profile.p_id not in completed_ids]
    log(f"Loaded {len(normalized_profiles)} profiles; processing {len(todo)} profile(s) with workers={args.workers}.")

    results: list[ProcessResult] = []

    def checkpoint() -> list[ValidationResult]:
        output_records = [
            records_by_id[profile.p_id]
            for profile in normalized_profiles
            if profile.p_id in records_by_id
        ]
        validations = save_outputs(output_records, profiles_by_id, args)
        log(f"Checkpoint saved: {len(output_records)} profile record(s).")
        return validations

    if args.workers == 1:
        for profile in todo:
            result = process_profile(profile, args)
            results.append(result)
            records_by_id[profile.p_id] = result.record
            print_result(result)
            if len(results) % args.checkpoint_every == 0:
                checkpoint()
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_profile = {executor.submit(process_profile, profile, args): profile for profile in todo}
            for future in as_completed(future_to_profile):
                profile = future_to_profile[future]
                try:
                    result = future.result()
                except Exception as exc:  # Defensive: persist a visible invalid record.
                    validation = validate_groups(profile, [], args.strict_implicit_gt_explicit)
                    validation.errors.insert(0, f"unhandled worker failure: {exc}")
                    validation.is_valid = False
                    result = ProcessResult(
                        profile.p_id, profile.profile_name, build_profile_record(profile, [], validation),
                        validation, TokenUsage(), 0.0, 0,
                    )
                results.append(result)
                records_by_id[profile.p_id] = result.record
                print_result(result)
                if len(results) % args.checkpoint_every == 0:
                    checkpoint()

    validations = checkpoint()
    total_usage = TokenUsage()
    for result in results:
        total_usage.add(result.usage)
    valid_count = sum(validation.is_valid for validation in validations)
    elapsed = time.monotonic() - started
    log(
        f"Finished: valid={valid_count}/{len(validations)}, API calls={total_usage.calls}, "
        f"tokens={total_usage.prompt_tokens}+{total_usage.completion_tokens}, elapsed={elapsed:.1f}s"
    )
    log(f"Groups JSON: {Path(args.output_groups).resolve()}")
    log(f"Frequency CSV: {Path(args.output_frequency).resolve()}")
    log(f"Summary CSV: {Path(args.output_summary).resolve()}")
    return 0 if valid_count == len(validations) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted by user")
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
