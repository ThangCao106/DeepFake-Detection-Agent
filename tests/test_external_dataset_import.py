import unittest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "web_ui"))

from benchmark.logger import BenchmarkJobRunner, BenchmarkLogger, format_benchmark_markdown_report
from hf_inspector import inspect_hf_dataset_metadata


class TestExternalDatasetImport(unittest.TestCase):

    def test_01_inspect_hf_metadata(self):
        """Verify dynamic Hugging Face inspection returns accurate name and samples."""
        meta = inspect_hf_dataset_metadata("prithivMLmods/Deepfake-vs-Real-60K")
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["repo_id"], "prithivMLmods/Deepfake-vs-Real-60K")
        self.assertIn("60,000", str(meta["estimated_samples"]))
        self.assertIn("Facial Portraits", meta["modality"])

    def test_02_load_samples_pure_hf(self):
        """Verify that loading external HF facial dataset builds exact classes and NO unrelated cohorts."""
        runner = BenchmarkJobRunner()
        b_logger = BenchmarkLogger(job_id="test_hf_pure", job_name="Test Pure HF")

        samples = runner._load_samples_for_mode(
            mode="custom_100",
            b_logger=b_logger,
            sample_count=100,
            dataset_name="Synthetic & Curated",
            import_source="huggingface",
            import_url="prithivMLmods/Deepfake-vs-Real-60K"
        )

        self.assertEqual(len(samples), 100)
        fake_cnt = sum(1 for s in samples if s["label"] == "FAKE")
        real_cnt = sum(1 for s in samples if s["label"] == "REAL")
        self.assertEqual(fake_cnt, 50)
        self.assertEqual(real_cnt, 50)

        # Ensure no WaveFake or GenImage leaked in
        datasets = {s["dataset"] for s in samples}
        for ds in datasets:
            self.assertTrue(ds.startswith("[HF: Deepfake-vs-Real-60K]"))
            self.assertNotIn("WaveFake", ds)
            self.assertNotIn("GenImage", ds)
            self.assertNotIn("HIDF", ds)

    def test_03_format_markdown_report_provenance(self):
        """Verify Section 1 provenance and Section 3 breakdown match imported dataset."""
        summary = {
            "job_id": "bench_20261103_hf_test",
            "job_name": "Benchmark (100 SAMPLES) - [HUGGINGFACE] prithivMLmods/Deepfake-vs-Real-60K",
            "status": "COMPLETED",
            "duration_sec": 5.2,
            "total_samples": 100,
            "accuracy_pct": 100.0,
            "fake_recall_pct": 100.0,
            "real_recall_pct": 100.0,
            "avg_latency_ms": 20.3,
            "import_source": "huggingface",
            "import_url": "prithivMLmods/Deepfake-vs-Real-60K",
            "dataset_name": "[HUGGINGFACE] prithivMLmods/Deepfake-vs-Real-60K"
        }

        samples = [
            {"sample_id": f"hf_sample_{i:04d}_fake", "dataset": "[HF: Deepfake-vs-Real-60K] Deepfake Faces (Class 0: Fake)", "true_label": "FAKE", "pred_label": "FAKE", "is_correct": True, "elapsed_ms": 20.3} for i in range(50)
        ] + [
            {"sample_id": f"hf_sample_{i:04d}_real", "dataset": "[HF: Deepfake-vs-Real-60K] Authentic Real Faces (Class 1: Real)", "true_label": "REAL", "pred_label": "REAL", "is_correct": True, "elapsed_ms": 20.4} for i in range(50)
        ]

        report = format_benchmark_markdown_report(summary, samples)

        self.assertIn("Hugging Face Hub", report)
        self.assertIn("prithivMLmods/Deepfake-vs-Real-60K", report)
        self.assertIn("~60,000 mẫu [30,000 Fake / 30,000 Real]", report)
        self.assertIn("[HF: Deepfake-vs-Real-60K] Deepfake Faces (Class 0: Fake)", report)
        self.assertIn("[HF: Deepfake-vs-Real-60K] Authentic Real Faces (Class 1: Real)", report)
        self.assertNotIn("WaveFake", report)
        self.assertNotIn("GenImage", report)


if __name__ == "__main__":
    unittest.main()
