"""Hierarchical multimodal contrastive representation and final clustering."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
import random
import threading
from typing import Any

import numpy as np

from .ablation import (
    resolve_final_candidate_source,
    run_descriptor_ablation,
    validate_final_candidate_pool,
    write_ablation_report,
)
from .clustering import project_cluster_and_score
from .config import Config
from .multimodal_features import sequence_features, structure_features
from .records import Candidate, read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)
_PYMOL_RENDER_LOCK = threading.Lock()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_spatial_graph(pdbqt_path: str | Path, distance_threshold: float = 8.0):
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError("Install the 'multimodal' dependencies") from exc

    atom_names: list[str] = []
    coordinates: list[list[float]] = []
    for line in Path(pdbqt_path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        if name not in {"P", "C4'", "C1'"}:
            continue
        try:
            point = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        if np.all(np.isfinite(point)):
            atom_names.append(name)
            coordinates.append(point)
    if not coordinates:
        raise ValueError(f"No P/C4'/C1' backbone atoms in {pdbqt_path}")

    xyz = np.asarray(coordinates, dtype=np.float32)
    origin = xyz.mean(axis=0, keepdims=True)
    centered = xyz - origin
    one_hot = np.zeros((len(atom_names), 3), dtype=np.float32)
    atom_index = {"P": 0, "C4'": 1, "C1'": 2}
    for index, name in enumerate(atom_names):
        one_hot[index, atom_index[name]] = 1.0
    node_features = np.hstack([centered, one_hot])
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
    rows, columns = np.where((distances > 0) & (distances <= distance_threshold))
    edge_index = np.vstack([rows, columns]).astype(np.int64)
    edge_weight = (1.0 / distances[rows, columns]).astype(np.float32)
    return Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        edge_weight=torch.from_numpy(edge_weight),
    )


def render_six_views(pdbqt_path: str | Path, output_dir: str | Path, image_size: int = 224) -> np.ndarray:
    try:
        from PIL import Image
        from pymol import cmd  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMOL and Pillow are required for structural rendering") from exc

    source = Path(pdbqt_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    uid = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = output / f"views_{uid}.npy"
    if cache.is_file():
        return np.load(cache)
    operations = [
        ("front", None),
        ("right", ("y", 90)),
        ("back", ("y", 180)),
        ("bottom", ("x", -90)),
        ("top", ("x", 90)),
        ("left", ("y", -90)),
    ]
    images: list[np.ndarray] = []
    with _PYMOL_RENDER_LOCK:
        for label, turn in operations:
            cmd.reinitialize()
            cmd.feedback("disable", "all", "everything")
            cmd.load(str(source), "aptamer")
            cmd.hide("everything", "all")
            cmd.show("sticks", "aptamer")
            cmd.color("cyan", "aptamer")
            cmd.bg_color("white")
            cmd.orient("aptamer")
            if turn:
                cmd.turn(turn[0], turn[1])
            target = output / f"{uid}_{label}.png"
            cmd.png(str(target), width=image_size, height=image_size, dpi=300, ray=1, quiet=1)
            with Image.open(target) as image:
                images.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
            cmd.delete("all")
    stack = np.stack(images)
    np.save(cache, stack)
    return stack


def _torch_components():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from torch_geometric.data import Batch
        from torch_geometric.nn import GCNConv, global_mean_pool
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError("Install with: pip install -e '.[multimodal]'") from exc
    return torch, nn, F, DataLoader, Dataset, Batch, GCNConv, global_mean_pool, resnet18


torch, nn, F, DataLoader, Dataset, Batch, GCNConv, global_mean_pool, resnet18 = _torch_components()


class MultiViewVisualEncoder(nn.Module):
    """Shared ResNet18-style encoder plus direction-specific view-pair fusion."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.direction_groups = nn.ModuleList(
            [nn.Sequential(nn.Linear(1024, 256), nn.ReLU(inplace=True)) for _ in range(3)]
        )
        self.output = nn.Sequential(nn.Linear(768, 256), nn.ReLU(inplace=True))

    def forward(self, images):
        batch, views, height, width, channels = images.shape
        pixels = images.permute(0, 1, 4, 2, 3).reshape(batch * views, channels, height, width)
        per_view = self.backbone(pixels).reshape(batch, views, 512)
        groups = []
        for module, indices in zip(self.direction_groups, ((0, 1), (2, 3), (4, 5))):
            groups.append(module(torch.cat([per_view[:, indices[0]], per_view[:, indices[1]]], dim=1)))
        return self.output(torch.cat(groups, dim=1))


class SpatialGCNEncoder(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList([GCNConv(6, 64), GCNConv(64, 64), GCNConv(64, 128)])
        self.dropout = dropout

    def forward(self, graph):
        features = graph.x
        for index, convolution in enumerate(self.convolutions):
            features = convolution(features, graph.edge_index, graph.edge_weight)
            if index < 2:
                features = F.dropout(F.relu(features), p=self.dropout, training=self.training)
        return global_mean_pool(features, graph.batch)


class HierarchicalFusion(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.sequence_projection = nn.Linear(20, 128)
        self.structure_projection = nn.Linear(15, 128)
        self.semantic_projection = nn.Linear(128, 256)
        self.visual_projection = nn.Linear(256, 256)
        self.graph_projection = nn.Linear(128, 256)
        self.attention = nn.Linear(768, 3)
        self.fusion = nn.Sequential(nn.Linear(768, embedding_dim), nn.LayerNorm(embedding_dim), nn.ReLU())
        self.projector = nn.Sequential(nn.Linear(embedding_dim, 256), nn.ReLU(), nn.Linear(256, 128))

    def forward(self, sequence, structure, visual, graph, project: bool = True):
        semantic = F.relu(self.sequence_projection(sequence)) + F.relu(self.structure_projection(structure))
        modalities = [
            self.semantic_projection(semantic),
            self.visual_projection(visual),
            self.graph_projection(graph),
        ]
        concatenated = torch.cat(modalities, dim=1)
        weights = F.softmax(self.attention(concatenated), dim=1)
        weighted = torch.cat([value * weights[:, index : index + 1] for index, value in enumerate(modalities)], dim=1)
        embedding = self.fusion(weighted)
        projected = F.normalize(self.projector(embedding), dim=1) if project else embedding
        return projected, weights


class AptaGrowModel(nn.Module):
    def __init__(self, embedding_dim: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.visual = MultiViewVisualEncoder()
        self.graph = SpatialGCNEncoder(dropout)
        self.fusion = HierarchicalFusion(embedding_dim)

    def forward(self, images, graphs, sequence, structure, project: bool = True):
        return self.fusion(sequence, structure, self.visual(images), self.graph(graphs), project=project)


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, first, second):
        batch = first.size(0)
        joined = F.normalize(torch.cat([first, second], dim=0), dim=1)
        logits = joined @ joined.T / self.temperature
        logits.fill_diagonal_(-float("inf"))
        positives = torch.cat([torch.diag(logits, batch), torch.diag(logits, -batch)])
        return (-positives + torch.logsumexp(logits, dim=1)).mean()


def augment_visuals(images):
    augmented = images.clone()
    for index in range(images.size(0)):
        rotation = random.choice((1, 2, 3))
        augmented[index] = torch.rot90(augmented[index], rotation, dims=(1, 2))
        if random.random() < 0.5:
            augmented[index] = torch.flip(augmented[index], dims=(2,))
        if random.random() < 0.5:
            augmented[index] = torch.flip(augmented[index], dims=(1,))
        augmented[index] = torch.clamp(augmented[index] + 0.05 * torch.randn_like(augmented[index]), 0, 1)
    return augmented


class AptamerDataset(Dataset):
    def __init__(self, records: list[Candidate], image_dir: Path, image_size: int, distance: float) -> None:
        self.records = records
        self.image_dir = image_dir
        self.image_size = image_size
        self.distance = distance
        self.maximum_length = max(len(item.sequence) for item in records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        if not item.pdbqt_path or not Path(item.pdbqt_path).is_file():
            raise FileNotFoundError(f"Missing PDBQT for {item.sequence}: {item.pdbqt_path}")
        images = render_six_views(item.pdbqt_path, self.image_dir, self.image_size)
        return {
            "images": torch.from_numpy(images.astype(np.float32) / 255.0),
            "graph": build_spatial_graph(item.pdbqt_path, self.distance),
            "sequence_features": torch.from_numpy(sequence_features(item.sequence, self.maximum_length)),
            "structure_features": torch.from_numpy(structure_features(item.structure, item.free_energy)),
            "candidate": item,
        }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in samples]),
        "graphs": Batch.from_data_list([item["graph"] for item in samples]),
        "sequence_features": torch.stack([item["sequence_features"] for item in samples]),
        "structure_features": torch.stack([item["structure_features"] for item in samples]),
        "candidates": [item["candidate"] for item in samples],
    }


def _move(batch: dict[str, Any], device):
    return (
        batch["images"].to(device),
        batch["graphs"].to(device),
        batch["sequence_features"].to(device),
        batch["structure_features"].to(device),
    )


def train_contrastive(
    model,
    loader,
    parameters: dict,
    device,
    model_path: Path,
    experiment_id: str,
) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 1e-4)),
        weight_decay=float(parameters.get("weight_decay", 1e-6)),
    )
    criterion = NTXentLoss(float(parameters.get("temperature", 0.5)))
    model.train()
    losses: list[float] = []
    for epoch in range(int(parameters.get("epochs", 200))):
        total = 0.0
        batches = 0
        for batch in loader:
            images, graphs, sequence, structure = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            first, _ = model(images, graphs, sequence, structure)
            second, _ = model(augment_visuals(images), graphs, sequence, structure)
            loss = criterion(first, second)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        losses.append(total / max(batches, 1))
        LOGGER.info("Contrastive epoch %d/%d loss %.6f", epoch + 1, int(parameters.get("epochs", 200)), losses[-1])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "loss": losses,
            "experiment_fingerprint": experiment_id,
        },
        model_path,
    )


def extract_embeddings(model, loader, device) -> tuple[np.ndarray, list[Candidate], np.ndarray]:
    model.eval()
    vectors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    candidates: list[Candidate] = []
    with torch.no_grad():
        for batch in loader:
            images, graphs, sequence, structure = _move(batch, device)
            embedding, attention = model(images, graphs, sequence, structure, project=False)
            vectors.append(embedding.cpu().numpy())
            weights.append(attention.cpu().numpy())
            candidates.extend(batch["candidates"])
    return np.vstack(vectors), candidates, np.vstack(weights)


def experiment_fingerprint(records: list[Candidate], parameters: dict) -> str:
    """Prevent checkpoint reuse with another pool or another parameter set."""

    payload = json.dumps(
        {
            "parameters": parameters,
            "candidates": [
                {
                    "sequence": item.sequence,
                    "structure": item.structure,
                    "free_energy": item.free_energy,
                    "binding_energy": item.binding_energy,
                }
                for item in records
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_multimodal_clustering(config: Config, input_path: str | Path | None = None) -> dict:
    source = resolve_final_candidate_source(config, input_path)
    records = list(read_jsonl(source))
    if len(records) < 3:
        raise ValueError("At least three final candidates are required")
    parameters = config.get("multimodal", default={})
    expected_length = int(parameters.get("expected_sequence_length", 23))
    validate_final_candidate_pool(records, expected_length)
    seed = int(config.get("project", "random_seed", default=42))
    set_random_seed(seed)
    output = config.path("paths", "output_dir") / "04_multimodal_clustering"
    output.mkdir(parents=True, exist_ok=True)
    baseline = run_descriptor_ablation(
        records,
        feature_parameters=config.get("initial_clustering", default={}),
        clustering_parameters=parameters,
        seed=seed,
        output_dir=output,
    )
    cache = config.path("paths", "cache_dir") / "multimodal_images"
    dataset = AptamerDataset(
        records,
        cache,
        int(parameters.get("image_size", 224)),
        float(parameters.get("graph_distance_angstrom", 8.0)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(parameters.get("batch_size", 8)),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_samples,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AptaGrowModel(
        embedding_dim=int(parameters.get("embedding_dim", 128)),
        dropout=float(parameters.get("dropout", 0.3)),
    ).to(device)
    model_path = output / "model.pt"
    fingerprint = experiment_fingerprint(records, parameters)
    if model_path.is_file():
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get("experiment_fingerprint") != fingerprint:
            raise ValueError(
                f"Checkpoint {model_path} belongs to a different candidate pool or configuration; "
                "move it aside and rerun cluster-final"
            )
        model.load_state_dict(checkpoint["model"])
    else:
        train_contrastive(model, loader, parameters, device, model_path, fingerprint)
    evaluation_loader = DataLoader(
        dataset,
        batch_size=int(parameters.get("batch_size", 8)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    embeddings, ordered, attention = extract_embeddings(model, evaluation_loader, device)
    multimodal = project_cluster_and_score(embeddings, parameters=parameters, seed=seed)
    representatives_with_clusters: list[tuple[int, Candidate]] = []
    for cluster_id in multimodal.cluster_ids:
        indices = np.flatnonzero(multimodal.labels == cluster_id)
        best = min(indices, key=lambda index: float(ordered[int(index)].binding_energy))
        representatives_with_clusters.append((cluster_id, ordered[int(best)]))
    representatives = [candidate for _, candidate in representatives_with_clusters]
    write_jsonl(output / "representatives.jsonl", representatives)
    with (output / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        coordinate_names = [
            f"umap_{index}" for index in range(1, multimodal.coordinates.shape[1] + 1)
        ]
        writer.writerow(
            [
                "sequence",
                "binding_energy",
                "cluster",
                *coordinate_names,
                "attention_semantic",
                "attention_visual",
                "attention_graph",
            ]
        )
        for candidate, label, point, weights in zip(
            ordered, multimodal.labels, multimodal.coordinates, attention
        ):
            writer.writerow(
                [
                    candidate.sequence,
                    candidate.binding_energy,
                    int(label),
                    *map(float, point),
                    *map(float, weights),
                ]
            )
    with (output / "representative_aptamers.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["aptamer_id", "cluster", "sequence", "length", "binding_energy"])
        for index, (cluster_id, candidate) in enumerate(representatives_with_clusters, start=1):
            writer.writerow(
                [
                    f"Apt{index}",
                    cluster_id,
                    candidate.sequence,
                    len(candidate.sequence),
                    candidate.binding_energy,
                ]
            )
    metrics = dict(multimodal.metrics)
    metrics.update(
        {
            "source": str(source),
            "candidate_round": int(parameters.get("candidate_round", 22)),
            "expected_sequence_length": expected_length,
        }
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return write_ablation_report(
        output / "ablation_metrics.json",
        baseline_metrics=baseline.metrics,
        multimodal_metrics=metrics,
    )
