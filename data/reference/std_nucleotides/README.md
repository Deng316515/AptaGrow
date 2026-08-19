# Standard nucleotide templates

`A.pdb`, `C.pdb`, `G.pdb`, and `U.pdb` are the standard single-nucleotide
templates used by `aptagrow.structure3d`. For each candidate, the workflow
constructs a sequence-specific SimRNA conversion template from these four files,
selects a representative SimRNA trajectory frame, converts the RNA structure to
DNA with PyMOL, and prepares the receptor PDBQT with MGLTools.

The files are workflow inputs and should remain unchanged. Their SHA-256 digests
are recorded in `SHA256SUMS.txt` for integrity checking.
