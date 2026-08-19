"""SimRNA tertiary-structure prediction and PDBQT preparation."""

from __future__ import annotations

from dataclasses import dataclass
from glob import glob
import hashlib
import logging
from pathlib import Path
import shutil
import subprocess
import threading

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .config import Config
from .records import Candidate, read_jsonl, write_jsonl
from .secondary import ViennaRNAFolder, has_base_pair, normalize_dna


LOGGER = logging.getLogger(__name__)
_PYMOL_LOCK = threading.Lock()


@dataclass(frozen=True)
class TrajectoryFrame:
    header: str
    coordinates: str
    energy: float
    vector: np.ndarray


def read_trafl(path: str | Path) -> list[TrajectoryFrame]:
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[TrajectoryFrame] = []
    for index in range(0, len(lines) - 1, 2):
        fields = lines[index].split()
        if len(fields) < 5:
            continue
        try:
            energy = float(fields[2])
            vector = np.asarray([float(value) for value in lines[index + 1].split()], dtype=float)
        except ValueError:
            continue
        if np.isfinite(energy) and np.all(np.isfinite(vector)):
            frames.append(TrajectoryFrame(lines[index], lines[index + 1], energy, vector))
    if not frames:
        raise ValueError(f"No valid trajectory frames in {path}")
    return frames


def select_representative_frame(
    input_path: str | Path,
    output_path: str | Path,
    clusters: int = 3,
    max_frames: int = 200,
) -> TrajectoryFrame:
    """Select the medoid of the lowest-mean-energy hierarchical cluster.

    SimRNA trajectories can contain hundreds of thousands of frames. The lowest
    energy ``max_frames`` are used, matching the tractable selection strategy in
    the supplied implementation while making the cluster/medoid rule explicit.
    """
    frames = sorted(read_trafl(input_path), key=lambda frame: frame.energy)[:max_frames]
    widths = {frame.vector.size for frame in frames}
    if len(widths) != 1:
        raise ValueError("Inconsistent coordinate-vector lengths in trajectory")
    if len(frames) < 3:
        selected = frames[0]
    else:
        matrix = np.vstack([frame.vector for frame in frames])
        distances = pdist(matrix, metric="euclidean") / np.sqrt(matrix.shape[1])
        labels = fcluster(linkage(distances, method="average"), t=min(clusters, len(frames)), criterion="maxclust")
        cluster_id = min(set(labels), key=lambda label: np.mean([f.energy for f, x in zip(frames, labels) if x == label]))
        indices = np.flatnonzero(labels == cluster_id)
        within = squareform(pdist(matrix[indices], metric="euclidean"))
        medoid = int(indices[int(np.argmin(within.mean(axis=1)))])
        selected = frames[medoid]
    Path(output_path).write_text(
        f"{selected.header}\n{selected.coordinates}\n", encoding="utf-8"
    )
    return selected


def _template_pdb(sequence_rna: str, standard_dir: Path, output_path: Path) -> None:
    atom_sets = {
        "A": {"P", "C4'", "N9", "C2", "C6"},
        "C": {"P", "C4'", "N1", "C2", "C4"},
        "G": {"P", "C4'", "N9", "C2", "C6"},
        "U": {"P", "C4'", "N1", "C2", "C4"},
    }
    atom_index = 1
    with output_path.open("w", encoding="ascii", newline="\n") as output:
        for residue_index, base in enumerate(sequence_rna, start=1):
            source = standard_dir / f"{base}.pdb"
            if not source.is_file():
                raise FileNotFoundError(f"Missing standard nucleotide template: {source}")
            required = set(atom_sets[base])
            if residue_index == 1:
                required.remove("P")
                required.add("O5'")
            for line in source.read_text(encoding="ascii", errors="ignore").splitlines():
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                atom_name = line[12:16].strip()
                if atom_name not in required:
                    continue
                mapped = "P" if residue_index == 1 and atom_name == "O5'" else atom_name
                element = "P" if mapped == "P" else (line[76:78].strip() or mapped[0])
                x = float(line[30:38]) + (residue_index - 1) * 10.0
                y = float(line[38:46])
                z = float(line[46:54])
                output.write(
                    f"ATOM  {atom_index:5d} {mapped:>4} {base:>3} A{residue_index:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
                )
                atom_index += 1
        output.write("TER\nEND\n")


def _rna_to_dna_pymol(input_pdb: Path, output_pdb: Path) -> None:
    try:
        from pymol import cmd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMOL 2.5.4 is required for RNA-to-DNA conversion") from exc
    with _PYMOL_LOCK:
        cmd.reinitialize()
        cmd.feedback("disable", "all", "everything")
        cmd.load(str(input_pdb), "aptamer")
        uracils: list[tuple[str, str]] = []
        cmd.iterate("aptamer and resn U and name N1", "items.append((chain,resi))", space={"items": uracils})
        if uracils:
            cmd.wizard("nucmutagenesis")
            wizard = cmd.get_wizard()
            for chain, residue in uracils:
                cmd.select("target_u", f"aptamer and chain {chain} and resi {residue} and resn U")
                wizard.do_select("target_u")
                wizard.set_mode("Thymine")
                wizard.apply()
            cmd.set_wizard()
        cmd.remove("aptamer and name O2'")
        for source, target in (("A", "DA"), ("C", "DC"), ("G", "DG"), ("T", "DT"), ("U", "DT")):
            cmd.alter(f"aptamer and resn {source}", f"resn='{target}'")
        cmd.save(str(output_pdb), "aptamer")
        cmd.delete("all")


class SimRNAPredictor:
    def __init__(self, config: Config, folder: ViennaRNAFolder) -> None:
        self.config = config
        self.folder = folder
        self.simrna_dir = config.path("tools", "simrna_dir", required=True)
        self.simrna = self.simrna_dir / "SimRNA"
        self.trafl2pdbs = self.simrna_dir / "SimRNA_trafl2pdbs"
        self.standard_dir = config.path("paths", "standard_nucleotides_dir", required=True)
        self.mgltools_dir = config.path("tools", "mgltools_dir", required=True)
        self.pythonsh = self.mgltools_dir / "bin" / "pythonsh"
        self.prepare_receptor = self.mgltools_dir / "MGLToolsPckgs" / "AutoDockTools" / "Utilities24" / "prepare_receptor4.py"
        for path in (self.simrna, self.trafl2pdbs, self.pythonsh, self.prepare_receptor):
            if not path.is_file():
                raise FileNotFoundError(path)

    def predict(self, sequence: str, output_dir: str | Path) -> Path:
        dna = normalize_dna(sequence)
        folded = self.folder.fold(dna)
        if not has_base_pair(folded.structure):
            raise ValueError(f"Sequence has no predicted base pair: {dna}")
        uid = hashlib.sha256(dna.encode()).hexdigest()[:16]
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pdbqt = output / f"aptamer_{uid}.pdbqt"
        if pdbqt.is_file() and pdbqt.stat().st_size:
            return pdbqt

        work = self.config.path("paths", "cache_dir") / "simrna" / uid
        work.mkdir(parents=True, exist_ok=True)
        sequence_file = work / "sequence.txt"
        restraints_file = work / "restraints.txt"
        config_file = work / "simrna.conf"
        template = work / "template.pdb"
        sequence_file.write_text(dna.replace("T", "U"), encoding="ascii")
        restraints_file.write_text(folded.structure, encoding="ascii")
        sim = self.config.get("simrna", default={})
        config_file.write_text(
            "\n".join(
                [
                    f"NUMBER_OF_ITERATIONS {int(sim.get('iterations_per_replica', 50000))}",
                    f"TRA_WRITE_IN_EVERY_N_ITERATIONS {int(sim.get('trajectory_stride', 5000))}",
                    f"INIT_TEMP {float(sim.get('initial_temperature', 1.35))}",
                    f"FINAL_TEMP {float(sim.get('final_temperature', 0.90))}",
                    f"NUMBER_OF_REPLICAS {int(sim.get('replicas', 5))}",
                    f"SECOND_STRC_RESTRAINTS_WEIGHT {float(sim.get('secondary_structure_weight', 5.0))}",
                    "",
                ]
            ),
            encoding="ascii",
        )
        _template_pdb(dna.replace("T", "U"), self.standard_dir, template)
        prefix = work / "simulation"
        subprocess.run(
            [
                str(self.simrna), "-c", str(config_file), "-s", str(sequence_file),
                "-S", str(restraints_file), "-d", str(self.simrna_dir / "data"),
                "-o", str(prefix), "-R", str(self.config.get("project", "random_seed", default=42)),
            ],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        trajectories = sorted(work.glob("simulation*.trafl"))
        if not trajectories:
            raise RuntimeError(f"SimRNA did not create a trajectory for {dna}")
        representative = work / "representative.trafl"
        select_representative_frame(
            trajectories[0], representative, clusters=int(sim.get("representative_clusters", 3))
        )
        subprocess.run(
            [str(self.trafl2pdbs), str(template), str(representative), "1", "AA"],
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        pdbs = [Path(item) for item in glob(str(work / "representative-*_AA.pdb"))]
        if not pdbs:
            raise RuntimeError(f"trafl2pdbs did not create a PDB for {dna}")
        dna_pdb = output / f"aptamer_{uid}.pdb"
        _rna_to_dna_pymol(pdbs[0], dna_pdb)
        subprocess.run(
            [
                str(self.pythonsh), str(self.prepare_receptor), "-r", str(dna_pdb),
                "-o", str(pdbqt), "-U", "nphs", "-A", "hydrogens",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not pdbqt.is_file() or not pdbqt.stat().st_size:
            raise RuntimeError(f"MGLTools did not create {pdbqt}")
        shutil.rmtree(work)
        LOGGER.info("Predicted 3D structure for %s", dna)
        return pdbqt


def prepare_candidate_structures(
    config: Config,
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate PDBQT files for a candidate set and write an updated JSONL manifest."""

    source = Path(input_path)
    if not source.is_absolute():
        source = (config.root / source).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    parameters = config.get("multimodal", default={})
    candidate_round = int(parameters.get("candidate_round", 22))
    candidate_file = str(parameters.get("candidate_file", "selected_candidates.jsonl"))
    if output_path is None:
        target = (
            config.path("paths", "output_dir")
            / "03_evolution"
            / f"round_{candidate_round:02d}"
            / candidate_file
        )
    else:
        target = Path(output_path)
        if not target.is_absolute():
            target = (config.root / target).resolve()

    records = list(read_jsonl(source))
    if not records:
        raise ValueError(f"No candidates in {source}")
    expected_length = int(parameters.get("expected_sequence_length", 23))
    invalid = [item.sequence for item in records if len(item.sequence) != expected_length]
    if invalid:
        raise ValueError(
            f"Structure preparation requires {expected_length}-nt candidates; "
            f"invalid sequences include {', '.join(invalid[:5])}"
        )

    folder = ViennaRNAFolder(config)
    predictor = SimRNAPredictor(config, folder)
    structure_dir = target.parent / "pdbqt"
    prepared: list[Candidate] = []
    for item in records:
        pdbqt_path = predictor.predict(item.sequence, structure_dir)
        prepared.append(
            Candidate(
                sequence=item.sequence,
                structure=item.structure,
                free_energy=item.free_energy,
                binding_energy=item.binding_energy,
                pdbqt_path=str(pdbqt_path),
            )
        )
    count = write_jsonl(target, prepared)
    return {
        "input": str(source),
        "output": str(target),
        "pdbqt_directory": str(structure_dir),
        "prepared_count": count,
    }
