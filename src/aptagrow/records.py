"""Shared data records and newline-delimited JSON helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Candidate:
    sequence: str
    structure: str
    free_energy: float
    binding_energy: float | None = None
    pdbqt_path: str | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "Candidate":
        return cls(
            sequence=str(value["sequence"]),
            structure=str(value.get("structure", "")),
            free_energy=float(value.get("free_energy", 0.0)),
            binding_energy=(
                None if value.get("binding_energy") is None else float(value["binding_energy"])
            ),
            pdbqt_path=value.get("pdbqt_path"),
        )


def read_jsonl(path: str | Path) -> Iterator[Candidate]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield Candidate.from_dict(json.loads(line))


def write_jsonl(path: str | Path, records: Iterable[Candidate]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count

