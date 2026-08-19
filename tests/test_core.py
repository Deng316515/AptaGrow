import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from aptagrow.ablation import (
    resolve_final_candidate_source,
    validate_final_candidate_pool,
)
from aptagrow.config import Config
from aptagrow.clustering import resolve_min_cluster_size
from aptagrow.docking import docking_box, read_pdbqt_coordinates
from aptagrow.evolution import should_screen
from aptagrow.initial_clustering import (
    BIOLOGICAL_DIMENSIONS,
    MANUSCRIPT_FEATURE_DIMENSIONS,
    SEQUENCE_TFIDF_DIMENSIONS,
    STRUCTURE_TFIDF_DIMENSIONS,
    biological_features,
    build_manuscript_feature_matrix,
)
from aptagrow.multimodal_features import sequence_features, structure_features
from aptagrow.records import Candidate, read_jsonl, write_jsonl
from aptagrow.secondary import dynamic_threshold, has_base_pair, normalize_dna
from aptagrow.structure3d import prepare_candidate_structures


class CoreLogicTests(unittest.TestCase):
    def test_deposited_round22_dataset_and_representatives(self):
        repository = Path(__file__).resolve().parents[1]
        candidates = list(read_jsonl(repository / "data" / "round22" / "selected_candidates.jsonl"))
        self.assertEqual(len(candidates), 27)
        self.assertEqual(len({item.sequence for item in candidates}), 27)
        self.assertTrue(all(len(item.sequence) == 23 for item in candidates))
        self.assertTrue(all(item.binding_energy is not None for item in candidates))

        with (repository / "data" / "round22" / "representative_aptamers.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            representatives = list(csv.DictReader(handle))
        self.assertEqual(
            [row["aptamer_id"] for row in representatives],
            [f"Apt{i}" for i in range(1, 7)],
        )
        self.assertEqual([int(row["cluster_id"]) for row in representatives], list(range(6)))

    def test_prepare_structures_writes_generated_pdbqt_manifest(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / "candidates.jsonl"
            write_jsonl(source, [Candidate("A" * 23, "." * 23, -1.0, -8.0)])
            config = Config(
                data={
                    "paths": {"output_dir": "results"},
                    "multimodal": {
                        "candidate_round": 22,
                        "candidate_file": "selected_candidates.jsonl",
                        "expected_sequence_length": 23,
                    },
                },
                root=root,
            )
            generated = root / "results" / "03_evolution" / "round_22" / "pdbqt" / "aptamer.pdbqt"
            with patch("aptagrow.structure3d.ViennaRNAFolder"), patch(
                "aptagrow.structure3d.SimRNAPredictor"
            ) as predictor:
                predictor.return_value.predict.return_value = generated
                report = prepare_candidate_structures(config, source)

            manifest = Path(str(report["output"]))
            prepared = list(read_jsonl(manifest))
            self.assertEqual(report["prepared_count"], 1)
            self.assertEqual(prepared[0].pdbqt_path, str(generated))

    def test_dynamic_threshold_population_standard_deviation(self):
        values = [-1.0, -2.0, -3.0, -4.0]
        expected = np.mean(values) - 1.5 * np.std(values)
        self.assertAlmostEqual(dynamic_threshold(values), expected)

    def test_sequence_validation_and_structure_detection(self):
        self.assertEqual(normalize_dna(" acgt "), "ACGT")
        self.assertTrue(has_base_pair("((..))"))
        self.assertFalse(has_base_pair("......"))
        with self.assertRaises(ValueError):
            normalize_dna("ACGU")

    def test_biological_feature_count(self):
        features = biological_features("AACCGGTT", "((....))")
        self.assertEqual(features.shape, (BIOLOGICAL_DIMENSIONS,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_manuscript_feature_dimensions(self):
        self.assertEqual(
            SEQUENCE_TFIDF_DIMENSIONS + STRUCTURE_TFIDF_DIMENSIONS + BIOLOGICAL_DIMENSIONS,
            MANUSCRIPT_FEATURE_DIMENSIONS,
        )

    def test_manuscript_feature_matrix_is_actually_556_dimensional(self):
        generator = np.random.default_rng(42)
        records = []
        for index in range(300):
            sequence = "".join(generator.choice(list("ACGT"), size=23))
            structure = "".join(generator.choice(list(".()"), size=23))
            records.append(Candidate(sequence, structure, -float(index % 10), -8.0))
        matrix = build_manuscript_feature_matrix(
            records,
            {
                "sequence_features": 500,
                "structure_features": 50,
                "sequence_ngram_range": [3, 5],
                "structure_ngram_range": [2, 4],
            },
        )
        self.assertEqual(matrix.shape, (300, MANUSCRIPT_FEATURE_DIMENSIONS))

    def test_two_phase_screening_schedule(self):
        screened = [round_number for round_number in range(12, 18) if should_screen(round_number)]
        self.assertEqual(screened, [14, 17])
        self.assertTrue(all(should_screen(round_number) for round_number in range(18, 25)))

    def test_final_clustering_uses_round_22_selected_candidates(self):
        config = Config(
            data={
                "paths": {"output_dir": "results"},
                "multimodal": {
                    "candidate_round": 22,
                    "candidate_file": "selected_candidates.jsonl",
                },
            },
            root=Path.cwd(),
        )
        expected = Path.cwd() / "results" / "03_evolution" / "round_22" / "selected_candidates.jsonl"
        self.assertEqual(resolve_final_candidate_source(config), expected)

    def test_final_candidate_pool_requires_23_nt_and_binding_energy(self):
        valid = [Candidate("A" * 23, "." * 23, -1.0, -8.0)]
        validate_final_candidate_pool(valid, 23)
        with self.assertRaises(ValueError):
            validate_final_candidate_pool([Candidate("A" * 25, "." * 25, -1.0, -8.0)], 23)
        with self.assertRaises(ValueError):
            validate_final_candidate_pool([Candidate("A" * 23, "." * 23, -1.0)], 23)

    def test_final_and_initial_minimum_cluster_size_policies(self):
        final_parameters = {
            "hdbscan_min_cluster_size": 5,
            "hdbscan_small_pool_threshold": 50,
            "hdbscan_small_pool_min_cluster_size": 3,
        }
        self.assertEqual(resolve_min_cluster_size(71, final_parameters), 5)
        self.assertEqual(resolve_min_cluster_size(40, final_parameters), 3)
        self.assertEqual(resolve_min_cluster_size(1000, {}), 20)

    def test_multimodal_fixed_width_and_reserved_channels(self):
        sequence = sequence_features("AACCGGTT", maximum_length=24)
        structure = structure_features("((....))", free_energy=-2.4)
        self.assertEqual(sequence.shape, (20,))
        self.assertEqual(structure.shape, (15,))
        self.assertTrue(np.all(np.isfinite(sequence)))
        self.assertTrue(np.all(np.isfinite(structure)))
        self.assertEqual(np.count_nonzero(sequence), 5)
        self.assertEqual(np.count_nonzero(structure), 5)
        self.assertTrue(np.all(sequence[5:] == 0))
        self.assertTrue(np.all(structure[5:] == 0))

    def test_docking_box_uses_receptor_span_plus_extension(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            receptor = Path(temporary) / "receptor.pdbqt"
            receptor.write_text(
                "ATOM      1  P   DA  A   1       0.000   1.000   2.000  1.00  0.00      P\n"
                "ATOM      2  P   DA  A   2      10.000  11.000  12.000  1.00  0.00      P\n",
                encoding="ascii",
            )
            center, size = docking_box(receptor, 10.0)
        self.assertTrue(np.allclose(center, [5.0, 6.0, 7.0]))
        self.assertTrue(np.allclose(size, [20.0, 20.0, 20.0]))

    def test_deposited_pfoa_ligand_is_valid_pdbqt(self):
        repository = Path(__file__).resolve().parents[1]
        ligand = repository / "data" / "input" / "pfoa_ligand.pdbqt"
        coordinates = read_pdbqt_coordinates(ligand)
        lines = ligand.read_text(encoding="utf-8").splitlines()
        self.assertEqual(coordinates.shape, (26, 3))
        self.assertTrue(np.all(np.isfinite(coordinates)))
        self.assertIn("TORSDOF 8", lines)
        self.assertEqual(sum(line.startswith("BRANCH") for line in lines), 8)
        self.assertEqual(sum(line.startswith("ENDBRANCH") for line in lines), 8)


if __name__ == "__main__":
    unittest.main()
