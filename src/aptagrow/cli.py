"""Command-line entry points; computational work occurs only after a subcommand."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

from .config import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aptagrow",
        description="AptaGrow de novo aptamer generation and screening pipeline",
    )
    parser.add_argument("--config", default="config/default.yaml", help="YAML configuration file")
    subcommands = parser.add_subparsers(dest="command", required=True)

    build = subcommands.add_parser("build-library", help="Enumerate, fold, and FE-screen the DNA library")
    build.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Smoke-test limit per length; omit for manuscript-scale enumeration",
    )
    initial = subcommands.add_parser("cluster-initial", help="Select initial seeds by TF-IDF/UMAP/HDBSCAN")
    initial.add_argument("--input", type=Path, default=None)
    evolution = subcommands.add_parser("evolve", help="Run Rounds 12-24 docking-guided evolution")
    evolution.add_argument("--input", type=Path, default=None)
    structures = subcommands.add_parser(
        "prepare-structures",
        help=(
            "Generate PDBQT structures for a candidate JSONL file with "
            "SimRNA, PyMOL, and MGLTools"
        ),
    )
    structures.add_argument(
        "--input",
        type=Path,
        default=Path("data/round22/selected_candidates.jsonl"),
        help="Candidate JSONL file (default: deposited Round-22 candidates)",
    )
    structures.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prepared JSONL path; defaults to the configured Round-22 evolution location",
    )
    final = subcommands.add_parser(
        "cluster-final",
        help="Run the Round-22 1D/2D ablation and multimodal contrastive clustering",
    )
    final.add_argument("--input", type=Path, default=None)
    motifs = subcommands.add_parser("engineer-motifs", help="Select 5' or 3' rigid-motif placement")
    motifs.add_argument("--input", type=Path, default=None)
    subcommands.add_parser("run-all", help="Run all five computational stages in manuscript order")
    subcommands.add_parser("doctor", help="Report configured paths and external-tool availability")
    return parser


def _doctor(config) -> dict:
    checks: dict[str, object] = {
        "config_root": str(config.root),
        "declared_versions": config.get("software_versions", default={}),
        "paths": {},
    }
    for section, key in (
        ("tools", "simrna_dir"),
        ("tools", "mgltools_dir"),
        ("tools", "vina_gpu_dir"),
        ("paths", "standard_nucleotides_dir"),
        ("paths", "pfoa_ligand_pdbqt"),
    ):
        try:
            path = config.path(section, key, required=True)
            checks["paths"][f"{section}.{key}"] = {"path": str(path), "exists": path.exists()}
        except ValueError as error:
            checks["paths"][f"{section}.{key}"] = {"configured": False, "message": str(error)}
    dependencies: dict[str, bool] = {}
    for module in ("RNA", "numpy", "sklearn", "umap", "hdbscan", "torch", "torch_geometric"):
        try:
            __import__(module)
            dependencies[module] = True
        except ImportError:
            dependencies[module] = False
    checks["python_dependencies"] = dependencies
    pymol_command = str(config.get("tools", "pymol_executable", default="pymol"))
    checks["pymol_executable"] = shutil.which(pymol_command)
    return checks


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    logging.basicConfig(
        level=getattr(logging, str(config.get("project", "log_level", default="INFO")).upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if arguments.command == "doctor":
        report = _doctor(config)
    elif arguments.command == "build-library":
        from .library import build_library

        report = build_library(config, arguments.max_sequences)
    elif arguments.command == "cluster-initial":
        from .initial_clustering import cluster_initial_library

        report = cluster_initial_library(config, arguments.input)
    elif arguments.command == "evolve":
        from .evolution import EvolutionRunner

        report = EvolutionRunner(config).run(arguments.input)
    elif arguments.command == "prepare-structures":
        from .structure3d import prepare_candidate_structures

        report = prepare_candidate_structures(config, arguments.input, arguments.output)
    elif arguments.command == "cluster-final":
        from .multimodal import run_multimodal_clustering

        report = run_multimodal_clustering(config, arguments.input)
    elif arguments.command == "engineer-motifs":
        from .motifs import engineer_terminal_motifs

        report = engineer_terminal_motifs(config, arguments.input)
    elif arguments.command == "run-all":
        from .evolution import EvolutionRunner
        from .initial_clustering import cluster_initial_library
        from .library import build_library
        from .motifs import engineer_terminal_motifs
        from .multimodal import run_multimodal_clustering

        report = {
            "library": build_library(config),
            "initial_clustering": cluster_initial_library(config),
            "evolution": EvolutionRunner(config).run(),
            "multimodal": run_multimodal_clustering(config),
            "motif_engineering": engineer_terminal_motifs(config),
        }
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(arguments.command)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
