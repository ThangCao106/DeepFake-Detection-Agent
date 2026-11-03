"""
Test Suite for 3-Pillar Multi-Agent Calibration Deliberation Council.
Validates:
1. Empirical telemetry gathering from Benchmark & Pentest memory.
2. Formulation of calibration proposals (alpha, Platt T*, Youden tau*, Focal gamma).
3. 3-Pillar Deliberation (Gemini, Meta Muse, Claude) with consensus verdict & Ed25519-SHA256 seal.
4. Gated weight application to calibration_params.json and vietnamese_demographic_params.json.
5. Markdown deliberation audit report generation in report_colab/council_deliberations/.
6. Frontend UI wiring in web_ui/templates/index.html and web_ui/static/js/app.js.
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.calibration_deliberation_council import CalibrationDeliberationCouncil


class TestCalibrationDeliberationCouncil(unittest.TestCase):

    def test_01_empirical_evidence_gathering(self):
        """Test that evidence aggregator pulls metrics from benchmark & pentest storage."""
        evidence = CalibrationDeliberationCouncil.gather_empirical_evidence()
        self.assertIn("benchmark_summary", evidence)
        self.assertIn("pentest_summary", evidence)
        self.assertIn("active_parameters", evidence)

        bench = evidence["benchmark_summary"]
        self.assertGreater(bench.get("total_samples", 0), 0)
        self.assertGreater(bench.get("accuracy_pct", 0), 90.0)

        pent = evidence["pentest_summary"]
        self.assertGreater(pent.get("total_media", 0), 0)
        self.assertIn("fpr_percent", pent)
        self.assertIn("ece", pent)

    def test_02_proposal_formulation(self):
        """Test parameter proposal generation with boundary safety bounds."""
        evidence = CalibrationDeliberationCouncil.gather_empirical_evidence()
        
        # Test default proposal
        prop = CalibrationDeliberationCouncil.formulate_calibration_proposal(evidence)
        self.assertIn("current_parameters", prop)
        self.assertIn("proposed_parameters", prop)
        self.assertIn("delta", prop)
        self.assertIn("vietnamese_cohorts_boost", prop)

        proposed_params = prop["proposed_parameters"]
        self.assertGreaterEqual(proposed_params["alpha_human"], 0.40)
        self.assertLessEqual(proposed_params["alpha_human"], 0.80)
        self.assertAlmostEqual(proposed_params["alpha_human"] + proposed_params["alpha_general"], 1.0, places=4)

        # Test requested alpha within bounds
        prop_req = CalibrationDeliberationCouncil.formulate_calibration_proposal(evidence, requested_alpha=0.65)
        self.assertEqual(prop_req["proposed_parameters"]["alpha_human"], 0.65)
        self.assertEqual(prop_req["proposed_parameters"]["alpha_general"], 0.35)

    def test_03_deliberation_consensus_and_verdict(self):
        """Test 3-Pillar deliberation transcript, consensus verdict, and cryptographic seal."""
        delib = CalibrationDeliberationCouncil.conduct_deliberation(requested_alpha=0.62)
        
        self.assertIn("deliberation_id", delib)
        self.assertIn("decision_hash", delib)
        self.assertIn("consensus_verdict", delib)
        self.assertIn("transcript", delib)
        self.assertIn("is_approved_for_training", delib)

        # Verify 3 pillars participated
        transcript = delib["transcript"]
        self.assertEqual(len(transcript), 3)
        speakers = [t["speaker"] for t in transcript]
        self.assertTrue(any("Gemini" in s for s in speakers))
        self.assertTrue(any("Meta Muse" in s for s in speakers))
        self.assertTrue(any("Claude" in s for s in speakers))

        # Decision hash verification
        self.assertEqual(len(delib["decision_hash"]), 64)
        self.assertTrue(delib["deliberation_id"].startswith("COUNCIL-CERT-"))

    def test_04_gated_weights_application(self):
        """Test applying approved calibration weights updates files safely."""
        delib = CalibrationDeliberationCouncil.conduct_deliberation(requested_alpha=0.60)
        res = CalibrationDeliberationCouncil.apply_approved_parameters(delib, user="test_suite")
        
        self.assertEqual(res.get("status"), "success")
        self.assertIn("parameters", res)
        self.assertIn("audit_trail", res)

        # Check calibration_params.json
        calib_file = PROJECT_ROOT / "custom_vit_model" / "calibration_params.json"
        self.assertTrue(calib_file.exists())
        with open(calib_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved.get("optimal_alpha"), 0.60)
        self.assertIn("council_deliberation_id", saved)
        self.assertEqual(saved.get("authorized_by_council"), True)
        self.assertEqual(saved.get("council_deliberation_id"), delib["deliberation_id"])

    def test_05_archive_report_generation(self):
        """Test that Markdown audit reports are archived in report_colab/council_deliberations/."""
        reports_dir = PROJECT_ROOT / "report_colab" / "council_deliberations"
        self.assertTrue(reports_dir.exists())
        md_files = list(reports_dir.glob("*.md"))
        self.assertGreater(len(md_files), 0, "At least one council deliberation markdown report should exist")
        
        # Verify content of the latest report
        latest_report = max(md_files, key=lambda f: f.stat().st_mtime)
        report_text = latest_report.read_text(encoding="utf-8")
        self.assertIn("HỘI ĐỒNG LLM TRANH LUẬN", report_text)
        self.assertIn("Google Gemini", report_text)
        self.assertIn("Meta Muse", report_text)
        self.assertIn("Claude", report_text)

    def test_06_ui_template_and_js_wiring(self):
        """Test that index.html and app.js correctly wire the Council Deliberation features."""
        index_html_path = PROJECT_ROOT / "web_ui" / "templates" / "index.html"
        app_js_path = PROJECT_ROOT / "web_ui" / "static" / "js" / "app.js"
        
        index_text = index_html_path.read_text(encoding="utf-8")
        app_js_text = app_js_path.read_text(encoding="utf-8")

        # HTML checks
        self.assertIn("openCouncilDebateModal()", index_text)
        self.assertIn("councilDebateModalBackdrop", index_text)
        self.assertIn("councilDebateModalBody", index_text)
        self.assertIn("btnApplyCouncilWeights", index_text)

        # JS checks
        self.assertIn("function openCouncilDebateModal", app_js_text)
        self.assertIn("function renderCouncilDeliberation", app_js_text)
        self.assertIn("function closeCouncilDebateModal", app_js_text)
        self.assertIn("function applyCouncilApprovedWeights", app_js_text)
        self.assertIn("/api/benchmark/debate_calibration", app_js_text)
        self.assertIn("/api/benchmark/apply_calibrated_weights_council", app_js_text)


if __name__ == "__main__":
    unittest.main()
