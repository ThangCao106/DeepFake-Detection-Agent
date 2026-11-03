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
        """Verify dynamic Hugging Face inspection returns accurate name, modalities and samples."""
        # 1. Image test
        meta_img = inspect_hf_dataset_metadata("prithivMLmods/Deepfake-vs-Real-60K")
        self.assertEqual(meta_img["status"], "ok")
        self.assertEqual(meta_img["repo_id"], "prithivMLmods/Deepfake-vs-Real-60K")
        self.assertIn("60,000", str(meta_img["estimated_samples"]))
        self.assertIn("Facial Portraits", meta_img["modality"])

        # 2. Audio Model test
        meta_audio = inspect_hf_dataset_metadata("MelodyMachine/Deepfake-audio-detection-V2")
        self.assertEqual(meta_audio["status"], "ok")
        self.assertTrue(meta_audio["is_model"])
        self.assertIn("Audio Deepfake", meta_audio["modality"])

        # 3. Multimodal Video test
        meta_mm = inspect_hf_dataset_metadata("ControlNet/AV-Deepfake1M")
        self.assertEqual(meta_mm["status"], "ok")
        self.assertIn("Audio-Visual", meta_mm["modality"])

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

        self.assertIn("Hugging Face", report)
        self.assertIn("prithivMLmods/Deepfake-vs-Real-60K", report)
        self.assertIn("60,000", report)
        self.assertIn("[HF: Deepfake-vs-Real-60K] Deepfake Faces (Class 0: Fake)", report)
        self.assertIn("[HF: Deepfake-vs-Real-60K] Authentic Real Faces (Class 1: Real)", report)
        self.assertNotIn("WaveFake", report)
        self.assertNotIn("GenImage", report)

    def test_04_audio_deepfake_load(self):
        """Verify loading MelodyMachine Audio Deepfake builds accurate audio classes."""
        runner = BenchmarkJobRunner()
        b_logger = BenchmarkLogger(job_id="test_hf_audio", job_name="Test Audio HF")

        samples = runner._load_samples_for_mode(
            mode="custom_40",
            b_logger=b_logger,
            sample_count=40,
            dataset_name="MelodyMachine Audio",
            import_source="huggingface",
            import_url="MelodyMachine/Deepfake-audio-detection-V2"
        )
        self.assertEqual(len(samples), 40)
        self.assertTrue(any("Synthetic Voice Clone" in s["dataset"] for s in samples))
        self.assertTrue(any("Authentic Human Speech" in s["dataset"] for s in samples))

    def test_05_multimodal_deepfake_load(self):
        """Verify loading ControlNet/AV-Deepfake1M builds accurate multimodal classes."""
        runner = BenchmarkJobRunner()
        b_logger = BenchmarkLogger(job_id="test_hf_av", job_name="Test AV HF")

        samples = runner._load_samples_for_mode(
            mode="custom_40",
            b_logger=b_logger,
            sample_count=40,
            dataset_name="AV-Deepfake1M",
            import_source="huggingface",
            import_url="ControlNet/AV-Deepfake1M"
        )
        self.assertEqual(len(samples), 40)
        self.assertTrue(any("Audio-Visual Deepfake" in s["dataset"] for s in samples))
        self.assertTrue(any("Authentic Audio-Visual Media" in s["dataset"] for s in samples))

    def test_06_gdrive_sample_names_match_drive(self):
        """Verify that GDrive samples use real filenames (real_img_ / fake_faceswap_) matching Google Drive."""
        runner = BenchmarkJobRunner()
        b_logger = BenchmarkLogger(job_id="test_gdrive_sync", job_name="Test GDrive Sync")

        samples = runner._load_samples_for_mode(
            mode="custom_60",
            b_logger=b_logger,
            sample_count=60,
            dataset_name="Google Drive Pentest Vault",
            import_source="gdrive",
            import_url="https://drive.google.com/drive/folders/1WBBHlhNAKwUR8WNgm-dnSNaVyzHj-Ys"
        )
        self.assertEqual(len(samples), 60)
        real_samples = [s for s in samples if s["label"] == "REAL"]
        fake_samples = [s for s in samples if s["label"] == "FAKE"]
        self.assertEqual(len(real_samples), 30)
        self.assertEqual(len(fake_samples), 30)

        # Check that NO generic gdrive_real_ or gdrive_fake_ placeholders exist
        for s in samples:
            self.assertFalse(s["sample_id"].startswith("gdrive_real_"), f"Generic ID found: {s['sample_id']}")
            self.assertFalse(s["sample_id"].startswith("gdrive_fake_"), f"Generic ID found: {s['sample_id']}")

        # Check that REAL samples start with real_img_ and FAKE samples start with fake_faceswap_
        for s in real_samples:
            self.assertTrue(s["sample_id"].startswith("real_img_") or s["sample_id"].startswith("threads_"), f"Invalid real name: {s['sample_id']}")
            self.assertTrue(s["filename"].startswith("real_img_") or s["filename"].startswith("threads_"), f"Invalid real filename: {s['filename']}")

        for s in fake_samples:
            self.assertTrue(s["sample_id"].startswith("fake_faceswap_"), f"Invalid fake name: {s['sample_id']}")
            self.assertTrue(s["filename"].startswith("fake_faceswap_"), f"Invalid fake filename: {s['filename']}")


if __name__ == "__main__":
    unittest.main()
