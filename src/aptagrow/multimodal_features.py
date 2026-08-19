"""Operational definitions of the manuscript's fixed-width 1D/2D inputs."""

from __future__ import annotations

import numpy as np

from .feature_utils import longest_run


def sequence_features(sequence: str, maximum_length: int) -> np.ndarray:
    """Return the original 20-slot primary-sequence model input.

    Five physically defined descriptors occupy slots 1-5. Slots 6-20 are
    reserved zero channels retained for compatibility with the trained model's
    fixed input interface; they carry no sample-specific information.
    """
    length = max(len(sequence), 1)
    dinucleotides = {sequence[index : index + 2] for index in range(len(sequence) - 1)}
    values = np.zeros(20, dtype=np.float32)
    values[:5] = [
        (sequence.count("G") + sequence.count("C")) / length,
        sequence.count("A") / length,
        sequence.count("T") / length,
        length / max(maximum_length, 1),
        len(dinucleotides) / max(length - 1, 1),
    ]
    return values


def structure_features(structure: str, free_energy: float) -> np.ndarray:
    """Return the original 15-slot secondary-structure model input.

    Five physically defined descriptors occupy slots 1-5. Slots 6-15 are
    reserved zero channels retained for compatibility with the trained model's
    fixed input interface; they carry no sample-specific information.
    """
    length = max(len(structure), 1)
    values = np.zeros(15, dtype=np.float32)
    values[:5] = [
        (structure.count("(") + structure.count(")")) / length,
        structure.count(".") / length,
        longest_run(structure, {"(", ")"}) / length,
        longest_run(structure, {"."}) / length,
        free_energy / length,
    ]
    return values
