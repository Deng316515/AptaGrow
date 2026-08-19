"""Terminal rigid-motif placement using ViennaRNA free-energy selection."""

from __future__ import annotations

from pathlib import Path

from .config import Config
from .records import Candidate, read_jsonl, write_jsonl
from .secondary import ViennaRNAFolder


def engineer_terminal_motifs(config: Config, input_path: str | Path | None = None) -> dict:
    source = Path(input_path) if input_path else (
        config.path("paths", "output_dir") / "04_multimodal_clustering" / "representatives.jsonl"
    )
    records = list(read_jsonl(source))
    motif = str(config.get("motif_engineering", "primary_motif", default="CGCGCTTCGGCGCG"))
    folder = ViennaRNAFolder(config.get("tools", "vienna_dna_parameter_file", default="dna_mathews2004.par"))
    engineered: list[Candidate] = []
    placements: list[dict] = []
    for item in records:
        variants = {"5_prime": motif + item.sequence, "3_prime": item.sequence + motif}
        folds = {position: folder.fold(sequence) for position, sequence in variants.items()}
        position = min(folds, key=lambda key: folds[key].free_energy)
        selected = folds[position]
        engineered.append(Candidate(selected.sequence, selected.structure, selected.free_energy))
        placements.append(
            {
                "original_sequence": item.sequence,
                "engineered_sequence": selected.sequence,
                "placement": position,
                "original_free_energy": item.free_energy,
                "engineered_free_energy": selected.free_energy,
            }
        )
    output = config.path("paths", "output_dir") / "05_motif_engineering"
    write_jsonl(output / "engineered_candidates.jsonl", engineered)
    import json

    (output / "placements.json").write_text(json.dumps(placements, indent=2), encoding="utf-8")
    return {"engineered_candidates": len(engineered), "output": str(output)}

