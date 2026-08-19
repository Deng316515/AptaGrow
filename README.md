# AptaGrow

**A data-independent generative framework for the de novo design of aptamers**

AptaGrow is a structure-guided workflow for generating DNA aptamers without
SELEX-derived training sequences. Starting from the four nucleotides A, C, G,
and T, the software enumerates candidate sequences, predicts secondary structure
and free energy, compresses the stable library through sequence–structure
clustering, evolves representative seeds by PFOA docking, and resolves the final
candidate space with hierarchical multimodal contrastive learning.

This repository accompanies the manuscript **“AptaGrow: A data-independent
generative framework for the de novo design of aptamers.”** It provides the
complete computational workflow, a configuration-driven software environment,
the 27-member Round-22 candidate dataset and clustering results, and the standard
nucleotide templates used for public PDBQT structure generation.

## Highlights

- **Data-independent generation** — no prior SELEX reads or target-specific
  training set are required.
- **Memory-bounded enumeration** — the combinatorial library is streamed to disk
  rather than accumulated in RAM.
- **Distribution-adaptive selection** — both folding free energy (FE) and binding
  energy (BE) use the reported threshold `mean − 1.5 × population SD`.
- **Two-stage evolution** — Rounds 12–17 use three-round modules; Rounds 18–24
  apply selection after every round.
- **Multiscale structural representation** — 1D sequence, 2D secondary structure,
  six orthographic molecular images, and an 8 Å spatial backbone graph.
- **Reproducible execution** — deterministic seed, pinned Python environment,
  configuration-driven paths, stable JSONL/CSV outputs, and unit tests.
- **Deposited analysis dataset** — all 27 Round-22 candidates, six final cluster
  representatives, and A/C/G/U structure templates are included.

## Repository structure

```text
AptaGrow/
├── README.md                     # Installation, usage, inputs, and outputs
├── environment.yml              # Recommended locked Conda environment
├── requirements.txt             # Pip dependencies (ViennaRNA excluded)
├── pyproject.toml                # Installable Python package and CLI
├── config/
│   ├── default.yaml              # Portable paths and manuscript parameters
│   └── reference_linux.yaml      # Reported /root Linux environment
├── docs/
│   └── environment_setup.md      # Reproducible external-tool installation
├── data/
│   ├── README.md                 # Deposited data and input preparation
│   ├── round22/
│   │   ├── selected_candidates.jsonl
│   │   ├── round22_all_sequences_clustered_apt.csv
│   │   └── representative_aptamers.csv
│   ├── reference/std_nucleotides/  # A/C/G/U PDB templates
│   └── input/pfoa_ligand.pdbqt     # Deposited PFOA docking ligand
├── src/aptagrow/
│   ├── cli.py                    # Command-line orchestration only
│   ├── config.py                 # YAML loading and path resolution
│   ├── records.py                # Candidate schema and JSONL I/O
│   ├── secondary.py              # ViennaRNA DNA folding and thresholds
│   ├── library.py                # Enumerative initial library construction
│   ├── initial_clustering.py     # TF-IDF/UMAP/HDBSCAN seed selection
│   ├── feature_utils.py          # Shared descriptor helper functions
│   ├── structure3d.py            # Public SimRNA-to-PDBQT preparation workflow
│   ├── docking.py                # Docking-box calculation and Vina-GPU
│   ├── evolution.py              # Rounds 12–24 evolutionary screening
│   ├── multimodal_features.py    # Explicit 20D and 15D feature definitions
│   ├── multimodal.py             # Visual/graph encoders and SimCLR clustering
│   └── motifs.py                 # 5′/3′ rigid-motif FE selection
└── tests/
    └── test_core.py              # Deterministic unit tests for core logic
```

## Workflow

```text
A/C/G/T
   │
   ├─ enumerate and ViennaRNA-fold through 10 nt
   ├─ extend two rounds to 12 nt
   └─ retain stable sequences below the adaptive FE threshold
          │
          ├─ sequence TF-IDF (3–5-mers)
          ├─ structure TF-IDF (2–4-mers)
          ├─ six biological descriptors
          └─ UMAP (3D) + HDBSCAN → one lowest-FE seed per cluster
                 │
                 └─ SimRNA → DNA PDBQT → Vina-GPU/PFOA
                        │
                        ├─ Phase 1: Rounds 12–17, screen after R14 and R17
                        └─ Phase 2: Rounds 18–24, screen every round
                               │
                               ├─ six-view ResNet18-style visual encoder
                               ├─ 8 Å P/C4′/C1′ spatial GCN
                               ├─ 1D + 2D semantic encoder
                               └─ attention fusion + SimCLR + UMAP/HDBSCAN
                                      │
                                      └─ terminal rigid-motif engineering
```

## System requirements

The reported workflow was designed for Linux with an NVIDIA GPU. Pure Python
stages can run on CPU, but SimRNA enumeration and Vina-GPU docking at manuscript
scale require substantial compute and storage.

### Python environment

- Python 3.9
- ViennaRNA 2.4.18 with `dna_mathews2004.par`
- NumPy, SciPy, pandas, scikit-learn, UMAP-learn, HDBSCAN
- PyTorch, torchvision, and PyTorch Geometric
- PyMOL 2.5.4 for RNA-to-DNA conversion and six-view rendering

Create the recommended environment:

```bash
conda env create -f environment.yml
conda activate aptagrow
pip install -e .
```

`requirements.txt` is provided for journal systems that accept pip manifests,
but ViennaRNA 2.4.18 and PyMOL should be installed through Conda or their official
distributions. `environment.yml` is therefore the authoritative environment
definition.

### External software

Install these separately because their binaries cannot be redistributed in this
archive:

| Software | Reference version | Purpose |
|---|---:|---|
| SimRNA | 3.20 | REMC tertiary-structure prediction |
| MGLTools | 1.5.7 | `prepare_receptor4.py` and PDBQT preparation |
| PyMOL | 2.5.4 | RNA→DNA mutation and orthographic rendering |
| Vina-GPU | 2.1 | GPU-accelerated PFOA docking |
| Boost | 1.84.0 | Vina-GPU build dependency |
| CUDA / OpenCL | 12.4 / 3.0 | NVIDIA GPU runtime and API target |

The external tools must be obtained from their official sources and used under
their respective licenses.

## Input preparation

See [`data/README.md`](data/README.md) for the deposited input layout. The
repository includes `data/input/pfoa_ligand.pdbqt` with its AutoDock partial
charges and torsion topology, together with the required `A.pdb`, `C.pdb`,
`G.pdb`, and `U.pdb` templates under `data/reference/std_nucleotides/`.
Local installations of SimRNA, MGLTools, PyMOL, and Vina-GPU are required for
the external-tool stages.

No absolute path is embedded in the source. Set the tool locations as environment
variables:

```bash
export SIMRNA_DIR=/opt/SimRNA_64bitIntel_Linux
export MGLTOOLS_DIR=/opt/mgltools_x86_64Linux2_1.5.7
export VINA_GPU_DIR=/opt/AutoDock-Vina-GPU-2.1
```

Alternatively, replace the `${...}` values in `config/default.yaml` with local
paths. Relative data and result paths are resolved from the repository root, not
from the current shell directory. The reported Linux environment is captured in
`config/reference_linux.yaml`; complete installation commands
are provided in [`docs/environment_setup.md`](docs/environment_setup.md).

Check the environment before launching expensive work:

```bash
aptagrow --config config/default.yaml doctor

# Reported /root Linux layout
aptagrow --config config/reference_linux.yaml doctor
```

## Quick start

Run each stage explicitly so that outputs can be inspected and checkpointed:

```bash
# 1. Enumerate and FE-screen the de novo library
aptagrow --config config/default.yaml build-library

# 2. Select low-FE, structurally diverse seeds
aptagrow --config config/default.yaml cluster-initial

# 3. Run the two-phase PFOA docking evolution (Rounds 12–24)
aptagrow --config config/default.yaml evolve

# 4. Train the multimodal contrastive model and cluster final candidates
aptagrow --config config/default.yaml cluster-final

# 5. Choose the lower-FE 5′ or 3′ rigid-motif placement
aptagrow --config config/default.yaml engineer-motifs
```

To launch the entire workflow without intermediate intervention:

```bash
aptagrow --config config/default.yaml run-all
```

To run the final analysis directly from the deposited 27-member Round-22 pool,
first generate its PDBQT structures through the public structure-preparation
workflow and then launch final clustering:

```bash
aptagrow --config config/default.yaml prepare-structures \
  --input data/round22/selected_candidates.jsonl
aptagrow --config config/default.yaml cluster-final
```

The first command uses the deposited A/C/G/U templates with SimRNA, PyMOL, and
MGLTools and writes the generated PDBQT files plus an updated candidate manifest
to `results/03_evolution/round_22/`.

The full enumerative stage is intentionally large. Confirm the installation first
with a bounded smoke test:

```bash
aptagrow --config config/default.yaml build-library --max-sequences 256
```

The smoke-test limit changes the scientific sample and must never be used for
reported results.

## Stage details and outputs

### 1. De novo library construction

`library.py` enumerates every A/C/G/T sequence for each length in lexicographic
order and folds it immediately with ViennaRNA’s DNA Mathews 2004 parameters. A
sequence is considered structurally stable when its dot-bracket string contains
at least one complete base pair. The code records stable candidates as JSONL and
uses a two-pass streaming calculation for the final threshold, so millions of
records do not need to reside in memory.

Key outputs in `results/01_library/`:

- `stable_length_<N>.jsonl` — stable sequences at each enumerated length.
- `low_free_energy_candidates.jsonl` — final stable candidates satisfying the
  adaptive FE threshold.
- `library_summary.json` — counts and FE distribution statistics per length.

The manuscript reports 1,048,576 length-10 sequences, 48,129 stable length-10
sequences, 2,112,896 stable candidates after two further extensions, and 200,204
sequences after FE screening. These are validation targets, not hard-coded
assertions.

### 2. Initial library compression

`initial_clustering.py` builds:

- 500 sequence TF-IDF dimensions from overlapping 3–5-mers;
- 50 structure TF-IDF dimensions from overlapping 2–4-mer dot-bracket grammar;
- 6 standardized biological features: GC content, AT content, paired/unpaired
  ratio, maximum stem run, maximum loop run, and their ratio.

The resulting 556-dimensional sparse matrix is reduced to three dimensions with
UMAP (`metric=cosine`, `n_neighbors=30`) and clustered with HDBSCAN
(`min_cluster_size=max(20,N//50)`, `min_samples=5`, `method=eom`). The lowest-FE
member of each non-noise cluster becomes a seed.

Outputs in `results/02_initial_clustering/`:

- `representative_seeds.jsonl`
- `assignments.csv`
- `metrics.json`

The manuscript reports 21 clusters, silhouette score 0.7707, and
Davies–Bouldin index 0.3743.

### 3. Tertiary structure prediction and docking evolution

For each sequence, the public structure-preparation workflow in `structure3d.py`
creates ViennaRNA constraints and runs five SimRNA replicas from 1.35 to 0.90 for
50,000 iterations per replica. The selected trajectory frame is converted to
PDB, mutated from RNA to DNA in PyMOL, and prepared as a receptor PDBQT with
MGLTools. Temporary work is isolated by a SHA-256 sequence identifier to prevent
filename collisions.

`docking.py` computes the receptor heavy-atom bounding box, extends each dimension
by 10 Å, and runs Vina-GPU with search depth 50, one mode, and 1,024 GPU threads.
The returned Vina score is the BE used for screening.

`evolution.py` implements the reported schedule exactly:

- Rounds 12–14: four 3′ variants per parent per round; screen after Round 14.
- Rounds 15–17: repeat; screen after Round 17.
- Rounds 18–24: generate four variants and screen after every round.

Every screened distribution uses `mean − 1.5 × population SD`. There is no hidden
“keep at least N” fallback because none is described in Methods.

Outputs in `results/03_evolution/` include each round’s candidates, selected
candidates, PDBQT structures, docking poses, statistics, and a final Round-24
candidate file. Round 24 remains part of the reported trajectory; comparison of
the per-round mean and minimum BE identifies Round 22 as the optimum used for the
final clustering analysis. Starting from the 12-nt seeds, the Round-22 selected
pool contains 23-nt candidates.

The complete Round-22 pool is deposited in
[`data/round22/round22_all_sequences_clustered_apt.csv`](data/round22/round22_all_sequences_clustered_apt.csv).
It contains 27 unique 23-nt candidates assigned to six clusters. The corresponding
machine-readable input is
[`data/round22/selected_candidates.jsonl`](data/round22/selected_candidates.jsonl),
and the six lowest-BE cluster representatives are listed in
[`data/round22/representative_aptamers.csv`](data/round22/representative_aptamers.csv).

### 4. Hierarchical multimodal contrastive clustering

`multimodal.py` constructs three complementary streams:

- **Semantic:** fixed-width 20-slot sequence and 15-slot secondary-structure
  inputs, independently projected and combined by residual addition into 128
  dimensions. Five physically defined descriptors are populated in each input;
  the remaining slots are reserved zero channels retained for compatibility with
  the original trained-model interface. They carry no sample-specific information.
- **Visual:** six 224 × 224 orthographic stick views processed by a shared
  ResNet18-style backbone and three direction-specific view-pair projections,
  producing a 256-dimensional embedding.
- **Graph:** P, C4′, and C1′ atoms as 6-feature nodes (centered xyz + atom one-hot),
  edges within 8 Å weighted by `1/d`, and three GCN layers
  `6 → 64 → 64 → 128` with dropout 0.3.

Each stream is projected to 256 dimensions. A softmax attention gate weights all
three streams before a 128-dimensional fused embedding is produced. A
`128 → 256 → 128` normalized projection head is trained with NT-Xent loss for 200
epochs (`lr=1e-4`, `weight_decay=1e-6`, `temperature=0.5`). Only the image stream
is augmented; the graph and 1D/2D features remain unchanged.

Before multimodal clustering, the code performs the reported tertiary-encoding
ablation on exactly the same Round-22 candidate pool. The baseline reconstructs
the manuscript-defined 556-dimensional representation (500 sequence TF-IDF,
50 secondary-structure TF-IDF, and 6 sequence/structure-derived biological
descriptors), then applies the same seeded UMAP, HDBSCAN, and non-noise scoring
protocol used for the multimodal embedding. This isolates the contribution of
the six-view visual and 3D spatial-graph streams rather than changing the
candidate population or clustering settings.

For both arms, features are L2-normalized and reduced to 10 dimensions with UMAP
(`metric=cosine`, `n_neighbors=15`, `min_dist=0`, seed 42). HDBSCAN uses
`min_samples=2`, `method=eom`, and `min_cluster_size=5` for pools of at least 50
candidates or 3 for smaller pools. Silhouette and Davies–Bouldin statistics are
calculated in this reduced space after excluding HDBSCAN noise points.

Both metric pairs are calculated directly from the supplied candidate pool.
No reported result values are embedded in the configuration or used for runtime
comparison. The lowest-BE member of each discovered multimodal cluster is named
sequentially as Apt1, Apt2, and so forth in ascending cluster-label order.

Outputs in `results/04_multimodal_clustering/`:

- `model.pt` — state dictionary, loss history, and an input/configuration fingerprint
  that prevents reuse with a different experiment;
- `assignments.csv` — cluster, UMAP coordinates, and attention weights;
- `ablation_1d2d_assignments.csv` — 1D/2D-only cluster assignments and UMAP coordinates;
- `representatives.jsonl` — lowest-BE representative per cluster;
- `representative_aptamers.csv` — sequential Apt identifiers, clusters, lengths, and BE values;
- `metrics.json` — sample, cluster, noise, silhouette, and DB statistics.
- `ablation_metrics.json` — both calculated metric pairs and their observed improvement.

### 5. Terminal rigid-motif engineering

`motifs.py` appends `CGCGCTTCGGCGCG` to both termini and retains whichever complete
sequence has the lower ViennaRNA FE. The output records the selected terminus and
pre/post-engineering FE. Tertiary prediction and PFOA docking can then be repeated
using the same stage-3 classes; those compute-intensive post-engineering reruns
are deliberately not triggered implicitly.

## Output format

Large candidate collections use newline-delimited JSON. Each line is an
independent record:

```json
{"sequence":"AACG...","structure":"(((...)))","free_energy":-4.2,"binding_energy":-8.3,"pdbqt_path":"results/.../aptamer_abcd.pdbqt"}
```

JSONL permits streaming, recovery after interruption, and inspection with common
command-line tools. Cluster assignments use CSV; compact run summaries use JSON.

## Reproducibility and validation

- All Python and CUDA random seeds are set from `project.random_seed` (default 42).
- UMAP receives the same deterministic seed.
- No computation occurs when modules are imported.
- External commands use argument arrays rather than shell strings.
- Failed external tools raise an error instead of silently returning fabricated
  structures, zero features, or default docking boxes.
- PDBQT coordinates are validated as finite before box calculation.
- The final model checkpoint stores only a state dictionary and loss history.

Run the unit tests from the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests cover DNA validation, base-pair detection, the population-SD threshold,
the invariant `500 + 50 + 6 = 556` representation, the exact 20-slot/15-slot
interface, Round-22/23-nt final-pool selection,
the deposited 27-candidate dataset and Apt1-Apt6 mapping, the two-phase screening
schedule, and docking-box geometry.

## Scope of this source release

This package contains the AptaGrow generation, structural screening, docking
evolution, multimodal clustering, and motif-engineering workflow represented by
the supplied source. GROMACS simulations, ITC measurements, and PFAS contact
analyses are separate validation activities and are intentionally outside the
scope of this Source Code submission.
