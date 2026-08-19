"""TF-IDF, biological-feature, UMAP, and HDBSCAN seed selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .clustering import project_cluster_and_score
from .config import Config
from .feature_utils import longest_run
from .records import Candidate, read_jsonl, write_jsonl


SEQUENCE_TFIDF_DIMENSIONS = 500
STRUCTURE_TFIDF_DIMENSIONS = 50
BIOLOGICAL_DIMENSIONS = 6
MANUSCRIPT_FEATURE_DIMENSIONS = 556

def biological_features(sequence: str, structure: str) -> np.ndarray:
    """Return the six biological descriptors specified in the Methods."""
    length = max(len(sequence), 1)
    paired = structure.count("(") + structure.count(")")
    unpaired = structure.count(".")
    max_stem = longest_run(structure, {"(", ")"})
    max_loop = longest_run(structure, {"."})
    return np.asarray(
        [
            (sequence.count("G") + sequence.count("C")) / length,
            (sequence.count("A") + sequence.count("T")) / length,
            paired / max(unpaired, 1),
            max_stem,
            max_loop,
            max_stem / max(max_loop, 1),
        ],
        dtype=np.float32,
    )


def _top_variance_columns(matrix: sparse.spmatrix, count: int) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    if matrix.shape[1] <= count:
        return matrix
    means = np.asarray(matrix.mean(axis=0)).ravel()
    squared_means = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    variances = squared_means - means**2
    indices = np.argpartition(variances, -count)[-count:]
    return matrix[:, np.sort(indices)]


def _tfidf(corpus: Iterable[str], ngram_range: tuple[int, int], count: int) -> sparse.csr_matrix:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=ngram_range,
        lowercase=False,
        dtype=np.float32,
        norm="l2",
    )
    return _top_variance_columns(vectorizer.fit_transform(corpus), count)


def build_manuscript_feature_matrix(records: list[Candidate], section: dict) -> sparse.csr_matrix:
    """Build the manuscript-defined 500 + 50 + 6 representation."""

    seq_count = int(section.get("sequence_features", SEQUENCE_TFIDF_DIMENSIONS))
    struct_count = int(section.get("structure_features", STRUCTURE_TFIDF_DIMENSIONS))
    if (seq_count, struct_count) != (SEQUENCE_TFIDF_DIMENSIONS, STRUCTURE_TFIDF_DIMENSIONS):
        raise ValueError(
            "The manuscript-defined representation requires exactly "
            "500 sequence TF-IDF and 50 structure TF-IDF dimensions"
        )
    seq_matrix = _tfidf(
        (item.sequence for item in records),
        tuple(section.get("sequence_ngram_range", [3, 5])),
        seq_count,
    )
    struct_matrix = _tfidf(
        (item.structure for item in records),
        tuple(section.get("structure_ngram_range", [2, 4])),
        struct_count,
    )
    biology = np.vstack([biological_features(item.sequence, item.structure) for item in records])
    biology = StandardScaler().fit_transform(biology).astype(np.float32)
    features = sparse.hstack([seq_matrix, struct_matrix, sparse.csr_matrix(biology)], format="csr")
    if features.shape[1] != MANUSCRIPT_FEATURE_DIMENSIONS:
        raise ValueError(
            f"Expected {MANUSCRIPT_FEATURE_DIMENSIONS} features but obtained {features.shape[1]}; "
            "the corpus did not contain enough distinct n-grams"
        )
    return features


def cluster_initial_library(config: Config, input_path: str | Path | None = None) -> dict:
    source = Path(input_path) if input_path else (
        config.path("paths", "output_dir") / "01_library" / "low_free_energy_candidates.jsonl"
    )
    records = list(read_jsonl(source))
    if len(records) < 3:
        raise ValueError("At least three candidates are required for clustering")

    section = config.get("initial_clustering", default={})
    features = build_manuscript_feature_matrix(records, section)
    result = project_cluster_and_score(
        features,
        parameters=section,
        seed=int(config.get("project", "random_seed", default=42)),
    )
    embedding = result.coordinates
    labels = result.labels
    cluster_ids = result.cluster_ids
    representatives: list[Candidate] = []
    for cluster_id in cluster_ids:
        indices = np.flatnonzero(labels == cluster_id)
        best_index = min(indices, key=lambda index: records[int(index)].free_energy)
        representatives.append(records[int(best_index)])

    output_dir = config.path("paths", "output_dir") / "02_initial_clustering"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "representative_seeds.jsonl", representatives)
    with (output_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sequence", "structure", "free_energy", "cluster", "umap_1", "umap_2", "umap_3"])
        for item, label, point in zip(records, labels, embedding):
            writer.writerow([item.sequence, item.structure, item.free_energy, int(label), *map(float, point)])

    metrics = dict(result.metrics)
    metrics["features"] = metrics.pop("input_dimensions")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics
