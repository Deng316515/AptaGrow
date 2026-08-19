# Data and reference resources

This directory contains the deposited Round-22 candidate set, its final cluster
assignments, the six representative aptamers, and the standard nucleotide
templates used by the public structure-preparation workflow.

```text
data/
├── round22/
│   ├── README.md
│   ├── selected_candidates.jsonl
│   ├── round22_all_sequences_clustered_apt.csv
│   └── representative_aptamers.csv
├── reference/
│   └── std_nucleotides/
│       ├── README.md
│       ├── SHA256SUMS.txt
│       ├── A.pdb
│       ├── C.pdb
│       ├── G.pdb
│       └── U.pdb
└── input/
    ├── README.md
    ├── SHA256SUMS.txt
    └── pfoa_ligand.pdbqt
```

The Round-22 dataset contains 27 unique 23-nt DNA aptamer candidates. The CSV
table is the complete cluster-assignment table. `selected_candidates.jsonl`
contains the same sequence, secondary-structure, FE, and BE fields in the format
accepted by the command-line workflow. `representative_aptamers.csv` lists the
lowest-BE member of each cluster as Apt1-Apt6.

PDBQT structures are generated from the deposited sequences and A/C/G/U templates
through the documented SimRNA, PyMOL, and MGLTools workflow:

```bash
aptagrow --config config/default.yaml prepare-structures \
  --input data/round22/selected_candidates.jsonl
```

This writes generated PDBQT files and a prepared candidate manifest under
`results/03_evolution/round_22/`, which is the default input location for
`aptagrow cluster-final`.

The deposited `data/input/pfoa_ligand.pdbqt` file is the configured Vina-GPU
docking input. It contains AutoDock atom types, encoded partial charges, and the
torsion topology used by the docking workflow. Its integrity metadata is provided
in `data/input/SHA256SUMS.txt`.

The four standard nucleotide PDB files are supplied in
`data/reference/std_nucleotides/` and are used to construct the SimRNA conversion
template for each candidate sequence.
