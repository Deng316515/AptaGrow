"""ViennaRNA DNA folding and distribution-adaptive screening."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Iterable, Sequence


DNA_ALPHABET = frozenset("ACGT")


def normalize_dna(sequence: str) -> str:
    normalized = sequence.strip().upper()
    invalid = set(normalized) - DNA_ALPHABET
    if not normalized or invalid:
        raise ValueError(f"Expected a non-empty DNA sequence; invalid symbols: {sorted(invalid)}")
    return normalized


def has_base_pair(structure: str) -> bool:
    return "(" in structure and ")" in structure


def dynamic_threshold(values: Sequence[float], sigma: float = 1.5) -> float:
    if not values:
        raise ValueError("Cannot calculate a threshold for an empty distribution")
    if sigma < 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be a finite non-negative number")
    return fmean(values) - sigma * pstdev(values)


@dataclass(frozen=True)
class FoldResult:
    sequence: str
    structure: str
    free_energy: float


class ViennaRNAFolder:
    """Thin, cached adapter around ViennaRNA 2.4.18 ``RNA.fold``."""

    def __init__(self, parameter_file: str | Path = "dna_mathews2004.par") -> None:
        try:
            import RNA  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ViennaRNA is required. Create the Conda environment from environment.yml."
            ) from exc
        self._rna = RNA
        parameter = str(parameter_file)
        try:
            RNA.read_parameter_file(parameter)
        except Exception as exc:
            raise RuntimeError(f"Unable to load ViennaRNA DNA parameters: {parameter}") from exc

    @lru_cache(maxsize=100_000)
    def fold(self, sequence: str) -> FoldResult:
        dna = normalize_dna(sequence)
        structure, energy = self._rna.fold(dna)
        return FoldResult(dna, str(structure), float(energy))


def filter_by_energy(records: Iterable[FoldResult], sigma: float = 1.5) -> list[FoldResult]:
    materialized = list(records)
    threshold = dynamic_threshold([item.free_energy for item in materialized], sigma)
    return [item for item in materialized if item.free_energy <= threshold]
