"""Two-phase docking-guided 3'-extension and dynamic selection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import fmean, pstdev

from .config import Config
from .docking import VinaGPU
from .records import Candidate, read_jsonl, write_jsonl
from .secondary import ViennaRNAFolder, dynamic_threshold
from .structure3d import SimRNAPredictor


LOGGER = logging.getLogger(__name__)


def should_screen(round_number: int, phase1_last: int = 17, module_size: int = 3) -> bool:
    if round_number <= phase1_last:
        return (round_number - 11) % module_size == 0
    return True


def select_by_binding_energy(candidates: list[Candidate], sigma: float = 1.5) -> tuple[list[Candidate], float]:
    values = [item.binding_energy for item in candidates if item.binding_energy is not None]
    if len(values) != len(candidates) or not values:
        raise ValueError("Every candidate must have a binding energy before screening")
    threshold = dynamic_threshold([float(value) for value in values], sigma)
    return [item for item in candidates if float(item.binding_energy) <= threshold], threshold


class EvolutionRunner:
    def __init__(self, config: Config) -> None:
        self.config = config
        parameter = config.get("tools", "vienna_dna_parameter_file", default="dna_mathews2004.par")
        self.folder = ViennaRNAFolder(parameter)
        self.predictor = SimRNAPredictor(config, self.folder)
        self.docker = VinaGPU(config)
        self.alphabet = tuple(config.get("library", "alphabet", default=list("ACGT")))

    def _expand(self, parents: list[Candidate], round_number: int, round_dir: Path) -> list[Candidate]:
        candidates: list[Candidate] = []
        structure_dir = round_dir / "structures"
        pose_dir = round_dir / "poses"
        for parent in parents:
            for nucleotide in self.alphabet:
                sequence = parent.sequence + nucleotide
                folded = self.folder.fold(sequence)
                receptor = self.predictor.predict(sequence, structure_dir)
                docking = self.docker.dock(receptor, pose_dir)
                candidates.append(
                    Candidate(
                        sequence=sequence,
                        structure=folded.structure,
                        free_energy=folded.free_energy,
                        binding_energy=docking.binding_energy,
                        pdbqt_path=str(receptor),
                    )
                )
        LOGGER.info("Round %d generated %d docked candidates", round_number, len(candidates))
        return candidates

    def run(self, seeds_path: str | Path | None = None) -> dict:
        source = Path(seeds_path) if seeds_path else (
            self.config.path("paths", "output_dir") / "02_initial_clustering" / "representative_seeds.jsonl"
        )
        parents = list(read_jsonl(source))
        if not parents:
            raise ValueError(f"No seed sequences in {source}")
        parameters = self.config.get("evolution", default={})
        first_round = int(parameters.get("first_round", 12))
        last_round = int(parameters.get("last_round", 24))
        phase1_last = int(parameters.get("phase1_last_round", 17))
        module_size = int(parameters.get("phase1_rounds_per_module", 3))
        sigma = float(parameters.get("threshold_sigma", 1.5))
        output_dir = self.config.path("paths", "output_dir") / "03_evolution"
        output_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []

        for round_number in range(first_round, last_round + 1):
            round_dir = output_dir / f"round_{round_number:02d}"
            raw = self._expand(parents, round_number, round_dir)
            write_jsonl(round_dir / "all_candidates.jsonl", raw)
            energies = [float(item.binding_energy) for item in raw]
            screened = should_screen(round_number, phase1_last, module_size)
            threshold = None
            survivors = raw
            if screened:
                survivors, threshold = select_by_binding_energy(raw, sigma)
                if not survivors:
                    raise RuntimeError(f"Dynamic screening removed every candidate in round {round_number}")
                write_jsonl(round_dir / "selected_candidates.jsonl", survivors)
            parents = survivors
            entry = {
                "round": round_number,
                "generated": len(raw),
                "selected": len(survivors),
                "screened": screened,
                "mean_binding_energy": fmean(energies),
                "std_binding_energy": pstdev(energies),
                "minimum_binding_energy": min(energies),
                "threshold": threshold,
            }
            history.append(entry)
            (round_dir / "statistics.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")

        write_jsonl(output_dir / "final_candidates.jsonl", parents)
        report = {"rounds": history, "final_candidates": len(parents)}
        (output_dir / "evolution_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

