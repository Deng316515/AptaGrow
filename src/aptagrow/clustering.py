"""Shared UMAP, HDBSCAN, and internal-validation calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClusterResult:
    """Projected coordinates, labels, cluster identifiers, and summary metrics."""

    coordinates: np.ndarray
    labels: np.ndarray
    cluster_ids: list[int]
    metrics: dict[str, Any]


def resolve_min_cluster_size(samples: int, parameters: dict) -> int:
    """Resolve either the initial-library or final small-pool HDBSCAN policy."""

    if "hdbscan_min_cluster_size" not in parameters:
        requested = max(20, samples // 50)
    else:
        requested = int(parameters["hdbscan_min_cluster_size"])
        small_pool_threshold = parameters.get("hdbscan_small_pool_threshold")
        if small_pool_threshold is not None and samples < int(small_pool_threshold):
            requested = int(parameters.get("hdbscan_small_pool_min_cluster_size", requested))
    return min(requested, max(2, samples - 1))


def project_cluster_and_score(
    features,
    *,
    parameters: dict,
    seed: int,
) -> ClusterResult:
    """Apply the manuscript UMAP/HDBSCAN protocol and score non-noise points."""

    try:
        import hdbscan  # type: ignore
        import umap  # type: ignore
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import normalize
    except ImportError as exc:
        raise RuntimeError("Install environment.yml before running clustering") from exc

    samples = int(features.shape[0])
    if samples < 3:
        raise ValueError("At least three candidates are required for clustering")
    input_dimensions = int(features.shape[1])
    l2_normalized = bool(parameters.get("l2_normalize", False))
    projected_features = normalize(features, norm="l2") if l2_normalized else features
    effective_min_cluster_size = resolve_min_cluster_size(samples, parameters)
    coordinates = umap.UMAP(
        n_components=int(parameters.get("umap_components", 3)),
        metric=str(parameters.get("umap_metric", "cosine")),
        n_neighbors=min(int(parameters.get("umap_neighbors", 30)), samples - 1),
        random_state=seed,
        min_dist=float(parameters.get("umap_min_dist", 0.0)),
    ).fit_transform(projected_features)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=int(parameters.get("hdbscan_min_samples", 5)),
        cluster_selection_method=str(parameters.get("hdbscan_selection_method", "eom")),
        metric="euclidean",
    ).fit_predict(coordinates)
    cluster_ids = sorted(int(value) for value in set(labels) if int(value) != -1)
    valid = labels != -1
    metrics: dict[str, Any] = {
        "samples": samples,
        "input_dimensions": input_dimensions,
        "projection_dimensions": int(coordinates.shape[1]),
        "clusters": len(cluster_ids),
        "noise_points": int((labels == -1).sum()),
        "min_cluster_size": effective_min_cluster_size,
        "l2_normalized_before_umap": l2_normalized,
        "silhouette_score": None,
        "davies_bouldin_index": None,
    }
    if len(cluster_ids) > 1 and int(valid.sum()) > len(cluster_ids):
        metrics["silhouette_score"] = float(silhouette_score(coordinates[valid], labels[valid]))
        metrics["davies_bouldin_index"] = float(
            davies_bouldin_score(coordinates[valid], labels[valid])
        )
    return ClusterResult(coordinates, labels, cluster_ids, metrics)
