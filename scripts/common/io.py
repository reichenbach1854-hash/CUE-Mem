"""JSON/JSONL and small CLI parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json_or_jsonl(path: str | Path) -> Any:
    """Load a JSON document, a single object, or a JSONL file."""

    source = Path(path)
    text = source.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError as document_error:
        rows: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip().rstrip(",")
            if not stripped or stripped in {"[", "]"}:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as line_error:
                raise ValueError(
                    f"{source} is neither valid JSON nor JSONL; line {line_number}: {line_error}"
                ) from document_error
        return rows


def load_record_list(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON/JSONL file and normalize its top level to dictionaries."""

    value = load_json_or_jsonl(path)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON object, list, or JSONL records")
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object record: {type(item).__name__}")
        records.append(item)
    return records


def write_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    """Create parent directories and write UTF-8 JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent),
        encoding="utf-8",
    )


def parse_id_set(values: Iterable[str] | None, *, cast=int) -> set[Any] | None:
    """Parse repeated CLI values containing comma- or space-separated IDs."""

    if not values:
        return None
    result: set[Any] = set()
    for raw in values:
        for token in str(raw).replace(",", " ").split():
            if token:
                result.add(cast(token))
    return result


def clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()
