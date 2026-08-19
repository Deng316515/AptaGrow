"""One-/two-dimensional descriptor ablation and manuscript-result checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .clustering import ClusterResult, project_cluster_and_score
from .config import Config
from .initial_clustering import build_manuscript_feature_matrix
from .records import Candidate


def resolve_final_candidate_source(config: Config, input_path: str | Path | None = None) -> Path:
    """Resolve the manuscript-selected Round-22 candidate pool."""

    if input_path is not None:
        return Path(input_path)
    parameters = config.get("multimodal", default={})
    candidate_round = int(parameters.get("candidate_round", 22))
    candidate_file = str(parameters.get("candidate_file", "selected_candidates.jsonl"))
    return (
        config.path("paths", "output_dir")
        / "03_evolution"
        / f"round_{candidate_round:02d}"
        / candidate_file
    )


def validate_final_candidate_pool(records: list[Candidate], expected_length: int) -> None:
    """Reject a pool that cannot represent the manuscript's final 23-nt analysis."""

    invalid_lengths = sorted(
        {len(item.sequence) for item in records if len(item.sequence) != expected_length}
    )
    if invalid_lengths:
        raise ValueError(
            f"Final clustering requires {expected_length}-nt Round-22 candidates; "
            f"encountered lengths {invalid_lengths}"
        )
    missing_binding_energies = [item.sequence for item in records if item.binding_energy is None]
    if missing_binding_energies:
        raise ValueError(
            "Every final-clustering candidate requires a binding energy; missing values for "
            + ", ".join(missing_binding_energies[:5])
        )


def run_descriptor_ablation(
    records: list[Candidate],
    *,
    feature_parameters: dict,
    clustering_parameters: dict,
    seed: int,
    output_dir: Path,
) -> ClusterResult:
    """Cluster the same candidate pool using only the manuscript's 556D descriptors."""

    features = build_manuscript_feature_matrix(records, feature_parameters)
    result = project_cluster_and_score(features, parameters=clustering_parameters, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_1d2d_assignments.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        coordinate_names = [
            f"umap_{index}" for index in range(1, result.coordinates.shape[1] + 1)
        ]
        writer.writerow(["sequence", "binding_energy", "cluster", *coordinate_names])
        for candidate, label, point in zip(records, result.labels, result.coordinates):
            writer.writerow(
                [candidate.sequence, candidate.binding_energy, int(label), *map(float, point)]
            )
    return result


def write_ablation_report(
    output_path: Path,
    *,
    baseline_metrics: dict[str, Any],
    multimodal_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Write calculated ablation metrics and the observed improvements."""

    baseline_silhouette = baseline_metrics.get("silhouette_score")
    multimodal_silhouette = multimodal_metrics.get("silhouette_score")
    baseline_db = baseline_metrics.get("davies_bouldin_index")
    multimodal_db = multimodal_metrics.get("davies_bouldin_index")
    improvements = {
        "silhouette_score_increase": (
            None
            if baseline_silhouette is None or multimodal_silhouette is None
            else float(multimodal_silhouette) - float(baseline_silhouette)
        ),
        "davies_bouldin_index_decrease": (
            None
            if baseline_db is None or multimodal_db is None
            else float(baseline_db) - float(multimodal_db)
        ),
    }
    report = {
        "protocol": {
            "candidate_pool": "Round 22 selected 23-nt candidates",
            "baseline": "556D 1D sequence and 2D secondary-structure descriptors",
            "comparison": "hierarchical 1D/2D + six-view + 3D spatial-graph embedding",
            "metric_space": "UMAP coordinates excluding HDBSCAN noise",
        },
        "one_d_two_d_only": baseline_metrics,
        "multimodal": multimodal_metrics,
        "improvement": improvements,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
