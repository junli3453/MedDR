from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable

from meddr.schemas import ResearchSample


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Data file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number}."
                )
            yield value


def load_samples(path: str | Path) -> list[ResearchSample]:
    return [ResearchSample.from_dict(row) for row in read_jsonl(path)]


def load_pickle(path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Data file not found: {source}")
    with source.open("rb") as handle:
        return pickle.load(handle)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
        )
