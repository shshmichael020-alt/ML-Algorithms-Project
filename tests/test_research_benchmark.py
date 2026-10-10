"""Regression tests for the research benchmark; no real CICIDS data is required."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "run_benchmark.py"

# Import the actual benchmark module so model and pipeline contracts are tested.
sys.path.insert(0, str(ROOT / "research"))
import run_benchmark as benchmark  # noqa: E402


class BenchmarkPipelineTests(unittest.TestCase):
    def test_pipeline_uses_expected_estimator_for_each_name(self):
        expected = {
            "Logistic Regression (L2)": LogisticRegression,
            "Logistic Regression (L1)": LogisticRegression,
            "Logistic Regression (Elastic Net)": LogisticRegression,
            "Ridge Classifier": RidgeClassifier,
            "SGD Classifier (log-loss, L2)": SGDClassifier,
        }
        for name, estimator_type in expected.items():
            with self.subTest(model=name):
                pipeline = benchmark.make_pipeline(name, seed=7)
                self.assertIsInstance(pipeline.named_steps["imputer"], benchmark.SimpleImputer)
                self.assertIsInstance(pipeline.named_steps["scaler"], benchmark.StandardScaler)
                self.assertIsInstance(pipeline.named_steps["model"], estimator_type)

    def test_preprocessor_is_fitted_from_training_rows_only(self):
        X_train = pd.DataFrame({
            "signal": [0.0, 2.0, np.nan, 6.0, 8.0, 10.0, np.nan, 14.0, 16.0],
            "other": [1.0, 3.0, 2.0, 5.0, 7.0, 9.0, 8.0, 12.0, 13.0],
        })
        y_train = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        X_test_with_extreme_value = pd.DataFrame({"signal": [1e12], "other": [-1e12]})

        pipeline = benchmark.make_pipeline("Logistic Regression (L2)", seed=7)
        pipeline.fit(X_train, y_train)
        train_medians = X_train.median().to_numpy()

        np.testing.assert_allclose(
            pipeline.named_steps["imputer"].statistics_,
            train_medians,
        )
        # Transforming unusual test data must not alter fitted imputer statistics.
        before = pipeline.named_steps["imputer"].statistics_.copy()
        pipeline.predict(X_test_with_extreme_value)
        np.testing.assert_array_equal(pipeline.named_steps["imputer"].statistics_, before)

    def test_end_to_end_outputs_metrics_and_plots_without_real_dataset(self):
        rng = np.random.default_rng(123)
        labels = np.array(["BENIGN", "DDoS", "PortScan"] * 30)
        centers = {"BENIGN": 0.0, "DDoS": 5.0, "PortScan": 10.0}
        all_data = pd.DataFrame({
            "Flow Feature 1": [centers[label] + rng.normal(0, 0.4) for label in labels],
            "Flow Feature 2": [centers[label] * -0.5 + rng.normal(0, 0.4) for label in labels],
            "Feature With Missing Values": rng.normal(size=len(labels)),
            "Entirely Empty Feature": np.nan,
            "Label": labels,
        })
        all_data.loc[::17, "Flow Feature 1"] = np.nan
        all_data.loc[::29, "Flow Feature 2"] = np.inf

        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            data_dir = temp_path / "data"
            output_dir = temp_path / "results"
            data_dir.mkdir()
            filenames = [
                "Tuesday-WorkingHours.pcap_ISCX.csv",
                "Wednesday-workingHours.pcap_ISCX.csv",
                "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            ]
            # Each file contains all three categories, to exercise stratified split.
            for index, filename in enumerate(filenames):
                all_data.iloc[index * 30:(index + 1) * 30].to_csv(data_dir / filename, index=False)

            command = [
                sys.executable, str(SCRIPT),
                "--data-dir", str(data_dir),
                "--target-column", "Label",
                "--test-size", "0.2",
                "--seed", "42",
                "--output-dir", str(output_dir),
            ]
            completed = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, timeout=180
            )
            self.assertEqual(
                completed.returncode, 0,
                msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            )

            summary_path = output_dir / "benchmark_test_0.2.csv"
            self.assertTrue(summary_path.is_file())
            summary = pd.read_csv(summary_path)
            self.assertEqual(len(summary), 5)
            for metric in ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro"]:
                self.assertTrue(summary[metric].between(0, 1).all(), metric)

            metadata = json.loads((output_dir / "metadata_test_0.2.json").read_text())
            self.assertEqual(metadata["input_rows_after_target_filter"], 90)
            self.assertEqual(metadata["train_rows"], 72)
            self.assertEqual(metadata["test_rows"], 18)
            self.assertIn("Entirely Empty Feature", metadata["features_dropped_all_missing_in_training"])
            self.assertEqual(len(metadata["input_files"]), 3)
            self.assertEqual(len(metadata["input_files"][0]["sha256"]), 64)
            self.assertIn("scikit_learn_version", metadata)
            self.assertIn("feature_names_used", metadata)

            pngs = list(output_dir.glob("*.png"))
            self.assertEqual(len(pngs), 12, f"Expected 5 raw + 5 normalized confusion matrices + 2 summaries, found {len(pngs)}")
            self.assertTrue(all(path.stat().st_size > 100 for path in pngs))
            json_reports = list(output_dir.glob("*_test_0.2.json"))
            self.assertEqual(len(json_reports), 6, "Expected five model reports plus run metadata")


if __name__ == "__main__":
    unittest.main()
