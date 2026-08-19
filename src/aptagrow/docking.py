"""Vina-GPU docking with geometry derived from the receptor coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess

import numpy as np

from .config import Config


def read_pdbqt_coordinates(path: str | Path) -> np.ndarray:
    coordinates: list[list[float]] = []
    in_first_model = False
    has_models = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MODEL"):
            if in_first_model:
                break
            has_models = True
            in_first_model = True
            continue
        if line.startswith("ENDMDL") and in_first_model:
            break
        if not line.startswith(("ATOM", "HETATM")) or (has_models and not in_first_model):
            continue
        try:
            point = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        if np.all(np.isfinite(point)):
            coordinates.append(point)
    if not coordinates:
        raise ValueError(f"No valid atom coordinates in {path}")
    return np.asarray(coordinates, dtype=float)


def docking_box(receptor_pdbqt: str | Path, extension_angstrom: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    coordinates = read_pdbqt_coordinates(receptor_pdbqt)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    return (minimum + maximum) / 2.0, (maximum - minimum) + extension_angstrom


@dataclass(frozen=True)
class DockingResult:
    binding_energy: float
    pose_path: Path


class VinaGPU:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.work_dir = config.path("tools", "vina_gpu_dir", required=True)
        self.executable = self.work_dir / str(
            config.get("tools", "vina_gpu_executable", default="AutoDock-Vina-GPU-2-1")
        )
        self.ligand = config.path("paths", "pfoa_ligand_pdbqt", required=True)
        if not self.executable.is_file():
            raise FileNotFoundError(self.executable)
        if not self.ligand.is_file():
            raise FileNotFoundError(self.ligand)

    def dock(self, receptor: str | Path, output_dir: str | Path) -> DockingResult:
        receptor_path = Path(receptor).resolve()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        uid = hashlib.sha256(receptor_path.read_bytes() + self.ligand.read_bytes()).hexdigest()[:16]
        pose = output / f"dock_{uid}.pdbqt"
        parameters = self.config.get("docking", default={})
        center, size = docking_box(
            receptor_path, float(parameters.get("box_extension_angstrom", 10.0))
        )
        command = [
            str(self.executable),
            "--receptor", str(receptor_path),
            "--ligand", str(self.ligand),
            "--center_x", f"{center[0]:.3f}",
            "--center_y", f"{center[1]:.3f}",
            "--center_z", f"{center[2]:.3f}",
            "--size_x", f"{size[0]:.3f}",
            "--size_y", f"{size[1]:.3f}",
            "--size_z", f"{size[2]:.3f}",
            "--out", str(pose.resolve()),
            "--num_modes", str(int(parameters.get("num_modes", 1))),
            "--thread", str(int(parameters.get("gpu_threads", 1024))),
            "--search_depth", str(int(parameters.get("search_depth", 50))),
            "--opencl_binary_path", str(self.work_dir),
        ]
        subprocess.run(command, cwd=self.work_dir, check=True, capture_output=True, text=True)
        if not pose.is_file():
            raise RuntimeError("Vina-GPU completed without writing the requested pose")
        match = re.search(r"^REMARK VINA RESULT:\s+(-?\d+(?:\.\d+)?)", pose.read_text(errors="replace"), re.MULTILINE)
        if not match:
            raise ValueError(f"No Vina score in {pose}")
        return DockingResult(float(match.group(1)), pose)

