"""Memory-bounded enumerative construction of the de novo DNA library."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice, product
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

from .config import Config
from .records import Candidate, read_jsonl, write_jsonl
from .secondary import ViennaRNAFolder, has_base_pair


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.count) if self.count else math.nan


def enumerate_sequences(alphabet: Sequence[str], length: int) -> Iterator[str]:
    for bases in product(alphabet, repeat=length):
        yield "".join(bases)


def _fold_length(
    folder: ViennaRNAFolder,
    alphabet: Sequence[str],
    length: int,
    stable_output: Path,
    max_sequences: int | None,
) -> dict:
    stable_output.parent.mkdir(parents=True, exist_ok=True)
    source = enumerate_sequences(alphabet, length)
    if max_sequences is not None:
        source = islice(source, max_sequences)

    total = 0
    stable = 0
    stats = RunningStats()
    with stable_output.open("w", encoding="utf-8", newline="\n") as handle:
        for sequence in source:
            result = folder.fold(sequence)
            total += 1
            stats.update(result.free_energy)
            if has_base_pair(result.structure):
                stable += 1
                handle.write(
                    json.dumps(
                        asdict(Candidate(sequence, result.structure, result.free_energy)),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return {
        "length": length,
        "enumerated": total,
        "stable": stable,
        "mean_free_energy": stats.mean,
        "std_free_energy": stats.std,
        "minimum_free_energy": stats.minimum,
        "maximum_free_energy": stats.maximum,
        "stable_file": str(stable_output),
    }


def threshold_jsonl(input_path: Path, output_path: Path, sigma: float) -> dict:
    stats = RunningStats()
    for candidate in read_jsonl(input_path):
        stats.update(candidate.free_energy)
    if not stats.count:
        raise ValueError(f"No stable candidates found in {input_path}")
    threshold = stats.mean - sigma * stats.std
    retained = write_jsonl(
        output_path,
        (candidate for candidate in read_jsonl(input_path) if candidate.free_energy <= threshold),
    )
    return {
        "input_count": stats.count,
        "retained_count": retained,
        "mean_free_energy": stats.mean,
        "std_free_energy": stats.std,
        "threshold": threshold,
        "output_file": str(output_path),
    }


def build_library(config: Config, max_sequences: int | None = None) -> dict:
    """Fold every enumerated sequence through the reported length-12 stage.

    ``max_sequences`` is only a smoke-test aid. Omit it for manuscript-scale runs.
    Enumeration is streamed, so the full combinatorial pool is never held in RAM.
    """
    alphabet = tuple(config.get("library", "alphabet", default=list("ACGT")))
    initial_length = int(config.get("library", "initial_length", default=10))
    additional = int(config.get("library", "additional_rounds", default=2))
    sigma = float(config.get("library", "threshold_sigma", default=1.5))
    parameter_file = config.get("tools", "vienna_dna_parameter_file", default="dna_mathews2004.par")
    folder = ViennaRNAFolder(parameter_file)

    output_dir = config.path("paths", "output_dir") / "01_library"
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    # Rounds 1-9 generate lengths 2-10. Two subsequent rounds generate 11-12 nt.
    for length in range(2, initial_length + additional + 1):
        stable_path = output_dir / f"stable_length_{length}.jsonl"
        summaries.append(
            _fold_length(folder, alphabet, length, stable_path, max_sequences=max_sequences)
        )

    final_stable = Path(summaries[-1]["stable_file"])
    filtered_path = output_dir / "low_free_energy_candidates.jsonl"
    screening = threshold_jsonl(final_stable, filtered_path, sigma)
    report = {
        "smoke_test_limit": max_sequences,
        "rounds": summaries,
        "final_screening": screening,
    }
    with (output_dir / "library_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report

