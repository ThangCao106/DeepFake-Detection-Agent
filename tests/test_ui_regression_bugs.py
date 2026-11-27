#!/usr/bin/env python3
"""
Automated Regression Test Suite for Web UI & Frontend Components.
Targets historical bug regressions:
  1. SyntaxError: duplicate const/let declarations (e.g. `const tau`, `const triad`).
  2. Browse file button unresponsive (missing DOM elements or broken event listeners).
  3. Stale client cache (missing cache-busting query strings on static assets).
  4. Out-of-sync Youden threshold (0.8118) between backend detector and frontend renderer.
  5. Triad Council Option B integration integrity.
"""

import os
import re
import sys
import unittest
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_JS_PATH = REPO_ROOT / "web_ui" / "static" / "js" / "app.js"
TEMPLATE_HTML_PATH = REPO_ROOT / "web_ui" / "templates" / "public_studio.html"
PUBLIC_BODY_PATH = REPO_ROOT / "web_ui" / "templates" / "partials" / "public_studio_body.html"
MODALS_FOOTER_PATH = REPO_ROOT / "web_ui" / "templates" / "partials" / "modals_footer.html"
BENCHMARK_HTML_PATH = REPO_ROOT / "web_ui" / "templates" / "benchmark_hub.html"
BENCHMARK_BODY_PATH = REPO_ROOT / "web_ui" / "templates" / "partials" / "benchmark_hub_body.html"
DETECTOR_PY_PATH = REPO_ROOT / "web_ui" / "deepfake_detector.py"


class TestUIRegressionBugs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert STATIC_JS_PATH.exists(), f"File not found: {STATIC_JS_PATH}"
        assert TEMPLATE_HTML_PATH.exists(), f"File not found: {TEMPLATE_HTML_PATH}"
        with open(STATIC_JS_PATH, "r", encoding="utf-8") as f:
            cls.js_content = f.read()
        parts = [
            TEMPLATE_HTML_PATH.read_text(encoding="utf-8"),
            PUBLIC_BODY_PATH.read_text(encoding="utf-8"),
            MODALS_FOOTER_PATH.read_text(encoding="utf-8"),
        ]
        cls.html_content = "\n".join(parts)
        cls.public_only_html = parts[0] + "\n" + parts[1]
        cls.benchmark_html = (
            BENCHMARK_HTML_PATH.read_text(encoding="utf-8")
            + "\n"
            + BENCHMARK_BODY_PATH.read_text(encoding="utf-8")
            + "\n"
            + parts[2]
        )

    def test_01_node_syntax_validity(self):
        """Regression test: Check that app.js compiles cleanly without syntax errors in Node.js."""
        try:
            res = subprocess.run(
                ["node", "-c", str(STATIC_JS_PATH)],
                capture_output=True,
                text=True,
                check=False
            )
            self.assertEqual(
                res.returncode,
                0,
                f"JavaScript SyntaxError detected in app.js:\n{res.stderr}"
            )
        except FileNotFoundError:
            self.skipTest("Node.js binary not available in PATH; skipping node -c test.")

    def test_02_no_duplicate_const_or_let_in_functions(self):
        """
        Regression test: Scans function scopes to catch duplicate 'const X' or 'let X' declarations.
        Historical bugs:
          - Commit 95296977 / 802af1e1: duplicate `const tau`
          - Commit c03b37bd: duplicate `const triad`
        """
        lines = self.js_content.splitlines()
        current_func = None
        func_depth = 0
        declared_identifiers = {}  # {func_name: {var_name: [line_no]}}

        decl_pattern = re.compile(r'^\s*(?:const|let)\s+([a-zA-Z0-9_$]+)\s*=')

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            func_match = re.search(r'function\s+([a-zA-Z0-9_$]+)\s*\(', stripped)
            if func_match:
                current_func = func_match.group(1)
                func_depth = 0
                declared_identifiers[current_func] = {}

            if current_func:
                func_depth += stripped.count('{') - stripped.count('}')
                m = decl_pattern.match(line)
                if m and func_depth <= 1:
                    var_name = m.group(1)
                    if var_name in declared_identifiers[current_func]:
                        prev_line = declared_identifiers[current_func][var_name]
                        self.fail(
                            f"Duplicate identifier '{var_name}' in function '{current_func}'! "
                            f"First declared at line {prev_line}, redeclared at line {idx}."
                        )
                    declared_identifiers[current_func][var_name] = idx

                if func_depth <= 0:
                    current_func = None

    def test_03_browse_files_button_and_elements_exist_in_html(self):
        """Check that DOM elements required for file selection and upload area exist in index.html."""
        required_ids = ["browseBtn", "fileInput", "uploadArea", "analyzeBtn", "urlInput"]
        for element_id in required_ids:
            pattern = re.compile(rf'id=["\']{element_id}["\']')
            self.assertTrue(
                pattern.search(self.html_content),
                f"Required DOM element id='{element_id}' is missing from web_ui/templates/index.html!"
            )

    def test_04_browse_files_button_event_wired_in_js(self):
        """Check that app.js properly attaches event listeners to browseBtn and delegates to fileInput.click()."""
        self.assertIn(
            "browseBtn",
            self.js_content,
            "browseBtn reference is missing from app.js"
        )
        self.assertIn(
            "fileInput",
            self.js_content,
            "fileInput reference is missing from app.js"
        )
        # Ensure click listener on browseBtn triggers fileInput.click()
        browse_wiring_pattern = re.compile(
            r'browseBtn\.addEventListener\s*\(\s*[\'"]click[\'"].*?fileInput\.click\(\)',
            re.DOTALL
        )
        self.assertTrue(
            browse_wiring_pattern.search(self.js_content),
            "browseBtn click event listener does not properly trigger fileInput.click() in app.js!"
        )

    def test_05_upload_area_events_wired_in_js(self):
        """Check that drag-and-drop zone events (dragover, drop) are handled on uploadArea in app.js."""
        self.assertIn("uploadArea.addEventListener('dragover'", self.js_content)
        self.assertIn("uploadArea.addEventListener('drop'", self.js_content)

    def test_06_cache_busting_present_in_index_html(self):
        """Check that index.html loads app.js with cache-busting query params (v= and t=)."""
        pattern = re.compile(r'app\.js[^\?]*\?v=([0-9.]+)(?:&t=([0-9_]+))?')
        match = pattern.search(self.html_content)
        self.assertIsNotNone(
            match,
            "web_ui/templates/index.html must load app.js with cache-busting query params (e.g. ?v=5.5.0&t=20261102_095000)!"
        )
        version = match.group(1)
        self.assertTrue(
            version >= "5.5.0",
            f"Expected cache-busting version >= 5.5.0, got: {version}"
        )

    def test_07_youden_threshold_consistency(self):
        """Check that the calibrated Youden optimal threshold (0.8118) is synced in frontend and backend."""
        self.assertIn(
            "0.8118",
            self.js_content,
            "Youden threshold 0.8118 missing from web_ui/static/js/app.js!"
        )
        if DETECTOR_PY_PATH.exists():
            with open(DETECTOR_PY_PATH, "r", encoding="utf-8") as f:
                detector_content = f.read()
            self.assertIn(
                "0.8118",
                detector_content,
                "Youden threshold 0.8118 missing from web_ui/deepfake_detector.py!"
            )

    def test_08_triad_council_option_b_wired(self):
        """Check that Triad Council Option B debate summary handling is present in app.js."""
        self.assertIn("triad_council_debate", self.js_content)
        self.assertIn("Hội Đồng Ba Tác Thể Thẩm Định & Phản Biện", self.js_content)
        self.assertIn("Google Gemini", self.js_content)
        self.assertIn("Meta Muse", self.js_content)
        self.assertIn("Claude Sonnet", self.js_content)

    def test_09_historical_duplicate_const_triad_and_tau(self):
        """Historical check: Ensure neither 'const triad' nor 'const tau' is declared multiple times in formatAgentChatSummary."""
        # Find formatAgentChatSummary function body
        match = re.search(r'function formatAgentChatSummary\s*\([^)]*\)\s*\{([\s\S]*?)\nfunction\s+', self.js_content)
        self.assertIsNotNone(match, "formatAgentChatSummary function not found in app.js")
        body = match.group(1)

        triad_matches = re.findall(r'\bconst\s+triad\b', body)
        self.assertLessEqual(
            len(triad_matches), 1,
            f"Regression: 'const triad' declared {len(triad_matches)} times in formatAgentChatSummary (Commit c03b37bd bug)!"
        )

        tau_matches = re.findall(r'\bconst\s+tau\b', body)
        self.assertLessEqual(
            len(tau_matches), 1,
            f"Regression: 'const tau' declared {len(tau_matches)} times in formatAgentChatSummary (Commit 95296977 bug)!"
        )

    def test_10_agentic_ai_model_ui_and_routes(self):
        """Check that the Agentic AI formal modeling UI, /dfagent links, and Flask routes are wired properly."""
        agentic_html = REPO_ROOT / "web_ui" / "templates" / "agentic_ai_model.html"
        orch_html = REPO_ROOT / "web_ui" / "templates" / "orchestrator_dashboard.html"
        app_py = REPO_ROOT / "web_ui" / "app.py"

        self.assertTrue(agentic_html.exists(), "agentic_ai_model.html template missing!")
        agentic_txt = agentic_html.read_text(encoding="utf-8")
        for token in (
            "C-Dec-POMDP",
            "agentInspectorPanel",
            "/dfagent/api/agentic/model",
            "copyLatexAlgorithm",
            "A0", "A1", "A2_G", "A2_M", "A2_C", "A3", "A4", "A5", "A6",
        ):
            self.assertIn(token, agentic_txt, f"Missing '{token}' in agentic_ai_model.html")

        # Public web UI must remain clean for end-users, while /dfagent houses the full C-Dec-POMDP modeling
        self.assertNotIn("agenticModelPreviewCard", self.public_only_html, "agenticModelPreviewCard should not be shown on public Studio!")
        self.assertNotIn("benchmarkCard", self.public_only_html, "benchmarkCard should not be shown on public Studio!")

        orch_txt = orch_html.read_text(encoding="utf-8")
        self.assertIn("linkAgenticModelNav", orch_txt)
        self.assertIn("/dfagent/agentic-ai", orch_txt)

        app_txt = app_py.read_text(encoding="utf-8")
        for route_tok in ("/dfagent/agentic-ai", "/dfagent/api/agentic/model", "_collect_agentic_model_payload"):
            self.assertIn(route_tok, app_txt, f"Missing '{route_tok}' in web_ui/app.py")

        # Observability SRE Cockpit (/dfagent) integration checks
        obs_static = REPO_ROOT / "observability" / "static" / "agentic_ai_model.html"
        obs_dir_idx = REPO_ROOT / "observability" / "static" / "agentic-ai" / "index.html"
        obs_index = REPO_ROOT / "observability" / "static" / "index.html"
        obs_main = REPO_ROOT / "observability" / "app" / "main.py"

        self.assertTrue(obs_static.exists(), "observability/static/agentic_ai_model.html missing!")
        self.assertTrue(obs_dir_idx.exists(), "observability/static/agentic-ai/index.html missing!")
        obs_index_txt = obs_index.read_text(encoding="utf-8")
        for tok in ("nav-agentic-model", "tab-agentic-model", "agenticModelPreviewCard", "/dfagent/agentic-ai", "nav-review-queue", "tab-review-queue"):
            self.assertIn(tok, obs_index_txt, f"Missing '{tok}' in SRE Cockpit /dfagent/static/index.html")
        self.assertNotIn("DeepSeek Janus-Pro", obs_index_txt, "Stale DeepSeek Janus-Pro reference remains in SRE Cockpit")

        obs_main_txt = obs_main.read_text(encoding="utf-8")
        for obs_tok in ("/dfagent/agentic-ai", "/agentic-ai", "/api/agentic/model", "/dfagent/api/agentic/model", "serve_agentic_ai_ui"):
            self.assertIn(obs_tok, obs_main_txt, f"Missing '{obs_tok}' in observability/app/main.py")

    def test_11_security_hardening_and_route_uniqueness(self):
        """Verify zero hardcoded plaintext passwords and no duplicate FastAPI report digest routes."""
        for rel in ("web_ui/app.py", "web_ui/static/js/app.js", "observability/app/main.py"):
            content = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("Thang@123", content, f"Hardcoded plaintext credential leaked in {rel}!")

        obs_main_txt = (REPO_ROOT / "observability" / "app" / "main.py").read_text(encoding="utf-8")
        self.assertEqual(
            obs_main_txt.count('@app.get("/api/reports/digest")'),
            1,
            "Duplicate @app.get('/api/reports/digest') route detected in observability/app/main.py!",
        )

    def test_12_observability_fastapi_routes_runtime(self):
        """Boot FastAPI observability app in an isolated subprocess and assert HTTP 200 on core routes."""
        script = r'''
import os, sys, types
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:
    print("SKIP_FASTAPI_NOT_INSTALLED")
    raise SystemExit(0)

if "claude_agent_sdk" not in sys.modules:
    class _StubMod(types.ModuleType):
        def __getattr__(self, name):
            if name == "tool":
                return lambda *a, **kw: (lambda f: f)
            return type(name, (), {"__init__": lambda self, *a, **kw: None})
    sys.modules["claude_agent_sdk"] = _StubMod("claude_agent_sdk")
    sys.modules["claude_agent_sdk.types"] = _StubMod("claude_agent_sdk.types")

obs_dir = (Path(os.environ["REPO_ROOT"]) / "observability").resolve()
os.chdir(str(obs_dir))
sys.path.insert(0, str(obs_dir))
os.environ["DEEPFAKE_WEB_UI_URL"] = "http://127.0.0.1:59999"

from app.main import app as obs_app
client = TestClient(obs_app)

for path in (
    "/",
    "/agentic-ai",
    "/dfagent/agentic-ai",
    "/diagrams",
    "/dfagent/diagrams",
    "/api/agents/lifecycle",
    "/api/feedback/review-queue",
    "/api/calibration/metrics",
    "/api/lora/status",
    "/api/reports/digest",
    "/dfagent/api/benchmark/modality_params",
    "/dfagent/api/orchestrator/status",
    "/dfagent/api/agentic/model",
):
    r = client.get(path)
    assert r.status_code == 200, f"{path} returned HTTP {r.status_code}: {r.text[:200]}"
    if path == "/dfagent/api/benchmark/modality_params":
        d = r.json()
        assert d.get("image") and d.get("video") and d.get("audio"), f"Empty modality_params: {d}"
    elif path == "/api/agents/lifecycle":
        assert r.json().get("operational_agents_count") == 9
    elif path == "/dfagent/api/agentic/model":
        assert r.json().get("formal_model", {}).get("agent_count") == 9
print("FASTAPI_RUNTIME_OK")
'''
        env = dict(os.environ)
        env["REPO_ROOT"] = str(REPO_ROOT)
        res = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        if "SKIP_FASTAPI_NOT_INSTALLED" in res.stdout:
            self.skipTest("fastapi.testclient not installed in current test environment")
        self.assertEqual(
            res.returncode,
            0,
            f"FastAPI runtime route check failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )
        self.assertIn("FASTAPI_RUNTIME_OK", res.stdout)

    def test_13_nginx_hosting_config_and_ci_workflow(self):
        """Verify Nginx sites-available config removes detection.dtpsoft.vn, keeps sibling domains, and configures deeptrace.io.vn."""
        nginx_conf = REPO_ROOT / "hosting" / "nginx" / "chatbot-dev.dtpsoft.vn.conf"
        workflow_yml = REPO_ROOT / ".github" / "workflows" / "update_nginx_hosting.yml"

        self.assertTrue(nginx_conf.exists(), f"Missing {nginx_conf}")
        self.assertTrue(workflow_yml.exists(), f"Missing {workflow_yml}")

        conf_txt = nginx_conf.read_text(encoding="utf-8")
        self.assertNotIn("detection.dtpsoft.vn", conf_txt, "detection.dtpsoft.vn must be completely removed from Nginx config!")
        self.assertNotIn("return 301 https://$host/dfagent;", conf_txt, "Infinite redirect loop on /dfagent detected!")
        self.assertNotIn("return 301 https://$host/benchmark;", conf_txt, "Infinite redirect loop on /benchmark detected!")

        for required in (
            "server_name chatbot-dev.dtpsoft.vn;",
            "server_name zalo-dev.dtpsoft.vn;",
            "location /n8n/",
            "server_name deeptrace.io.vn;",
            "return 301 https://$host/dfagent/;",
            "location /dfagent/",
            "return 301 https://$host/benchmark/;",
            "location /benchmark/",
            "location /socket.io/",
            "client_max_body_size 250M;",
        ):
            self.assertIn(required, conf_txt, f"Missing required Nginx directive '{required}'")

        wf_txt = workflow_yml.read_text(encoding="utf-8")
        for wf_tok in ("hosting/**", "/etc/nginx/sites-available/chatbot-dev.dtpsoft.vn.conf", "nginx -t", "systemctl restart nginx"):
            self.assertIn(wf_tok, wf_txt, f"Missing '{wf_tok}' in update_nginx_hosting.yml")

    def test_14_hf_resilient_loader_ignores_expired_env_token(self):
        """Ensure DeepfakeDetector resilient HF loader strips expired HF_TOKEN env vars and passes token=False."""
        det_txt = DETECTOR_PY_PATH.read_text(encoding="utf-8")
        self.assertIn("def _hf_load_pretrained_resilient", det_txt)
        self.assertIn("token=False", det_txt)
        self.assertIn('os.environ.pop(k)', det_txt)

    def test_15_colab_notebooks_security_and_closed_loop_wiring(self):
        """Verify all Colab notebooks and scripts contain zero plaintext PATs, zero synthetic placeholders, and valid closed-loop wiring."""
        import json
        colab_dir = REPO_ROOT / "colab_training"
        notebooks = [
            colab_dir / "Colab_MultiModality_AutoTrainer.ipynb",
            colab_dir / "Colab_Pro_Scheduled_Runner.ipynb",
            colab_dir / "Colab_Free_4h_Scheduler.ipynb",
        ]
        for nb_path in notebooks:
            self.assertTrue(nb_path.exists(), f"Missing Colab notebook: {nb_path}")
            raw_txt = nb_path.read_text(encoding="utf-8")
            nb_json = json.loads(raw_txt)
            self.assertEqual(nb_json.get("nbformat"), 4)
            pat_prefix = "ghp" + "_"
            self.assertNotIn(pat_prefix, raw_txt, f"Hardcoded GitHub PAT leaked in {nb_path.name}")
            self.assertNotIn("purge_synthetic_drawings.py", raw_txt, f"Deleted script referenced in {nb_path.name}")
            self.assertNotIn("purge_synthetic_video_and_audio_gdrive.py", raw_txt, f"Deleted script referenced in {nb_path.name}")
            self.assertNotIn("/content/DeepFake-Detection-Agent/colab_auto_runner.py", raw_txt, f"Broken root colab_auto_runner.py path in {nb_path.name}")

        # Verify bookmarked cell ID eBg1D7_tAFPq is preserved on Colab_MultiModality_AutoTrainer.ipynb
        auto_nb = json.loads((colab_dir / "Colab_MultiModality_AutoTrainer.ipynb").read_text(encoding="utf-8"))
        cell_ids = [c.get("metadata", {}).get("id") for c in auto_nb.get("cells", [])]
        self.assertIn("eBg1D7_tAFPq", cell_ids, "Bookmarked cell ID 'eBg1D7_tAFPq' missing from Colab_MultiModality_AutoTrainer.ipynb")

        # Verify zero synthetic Image.new / RandomState metric fabrication in colab_dataset_pipeline.py & data_pipeline.py
        cdp_txt = (colab_dir / "colab_dataset_pipeline.py").read_text(encoding="utf-8")
        dp_txt = (colab_dir / "data_pipeline.py").read_text(encoding="utf-8")
        self.assertNotIn("RandomState", cdp_txt)
        self.assertNotIn("Image.new", cdp_txt)
        self.assertNotIn("Image.new", dp_txt)
        self.assertIn("def pull_human_review_feedback", cdp_txt)
        self.assertIn("def notify_production_callback", cdp_txt)

    def test_16_colab_callback_security_and_whitelist(self):
        """Runtime-verify /api/colab/callback secret auth, hf_repo_id namespace whitelist, and calibration clamping."""
        import json
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from web_ui.routes.colab_routes import (
            _is_trusted_hf_repo,
            _verify_callback_auth,
            create_colab_blueprint,
        )

        # Stdlib-compatible checks (always run in bare GitHub Actions runner)
        self.assertTrue(_is_trusted_hf_repo("ThangCao106/custom-deepfake-vit"))
        self.assertTrue(_is_trusted_hf_repo("ThangCao/custom-deepfake-vit"))
        self.assertFalse(_is_trusted_hf_repo("attacker/malicious-vit"))
        self.assertFalse(_is_trusted_hf_repo("../local/traversal"))

        with patch.dict(os.environ, {"COLAB_CALLBACK_SECRET": "unit-test-secret-999"}):
            self.assertFalse(_verify_callback_auth(SimpleNamespace(headers={}), {}))
            self.assertTrue(_verify_callback_auth(SimpleNamespace(headers={"X-Colab-Secret": "unit-test-secret-999"}), {}))
            self.assertTrue(_verify_callback_auth(SimpleNamespace(headers={"Authorization": "Bearer unit-test-secret-999"}), {}))

        try:
            import numpy  # noqa: F401
            from flask import Flask
        except ImportError:
            return

        reloads = []
        emits = []
        app = Flask(__name__)
        app.register_blueprint(
            create_colab_blueprint(
                reload_detector_fn=lambda full_reinit=False: reloads.append(full_reinit),
                emit_fn=lambda ev, data: emits.append((ev, data)),
            )
        )
        client = app.test_client()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch("forensics_kernel.calibration_store.resolve_custom_vit_dir", return_value=tmp_path):
                r_bad_repo = client.post("/api/colab/callback", json={"hf_repo_id": "attacker/malicious-vit"})
                self.assertEqual(r_bad_repo.status_code, 403)

                with patch.dict(os.environ, {"COLAB_CALLBACK_SECRET": "unit-test-secret-999"}):
                    r_unauth = client.post("/api/colab/callback", json={"temperature_scaling_T": 1.45})
                    self.assertEqual(r_unauth.status_code, 401)

                    r_auth = client.post(
                        "/api/colab/callback",
                        headers={"X-Colab-Secret": "unit-test-secret-999"},
                        json={
                            "temperature_scaling_T": 1.52,
                            "optimal_threshold_tau": 0.76,
                            "out_of_distribution_accuracy": 0.934,
                            "auc_score": 0.961,
                            "dataset_trained": "burst:threads_vietnamese_in_the_wild",
                        },
                    )
                    self.assertEqual(r_auth.status_code, 200)
                    saved_calib = json.loads((tmp_path / "calibration_params.json").read_text(encoding="utf-8"))
                    self.assertEqual(saved_calib["temperature_scaling_T"], 1.52)
                    self.assertEqual(saved_calib["optimal_threshold_tau"], 0.76)
                    self.assertEqual(saved_calib["auc_score"], 0.961)
                    self.assertIn(False, reloads)

    def test_17_editorial_diagram_design_system(self):
        """Verify BOTH Version 1 (3D Cyber-Crystal Agentic Studio) and Version 2 (Editorial Diagram-Design Studio) coexist cleanly."""
        # 1. Verify Version 1 (Original 3D Crystal Studio at /dfagent/agentic-ai)
        v1_web = REPO_ROOT / "web_ui" / "templates" / "agentic_ai_model.html"
        v1_obs1 = REPO_ROOT / "observability" / "static" / "agentic_ai_model.html"
        v1_obs2 = REPO_ROOT / "observability" / "static" / "agentic-ai" / "index.html"
        for p in (v1_web, v1_obs1, v1_obs2):
            self.assertTrue(p.exists(), f"Missing v1 3D Agentic file: {p}")
        v1_txt = v1_web.read_text(encoding="utf-8")
        self.assertEqual(v1_txt, v1_obs1.read_text(encoding="utf-8"))
        self.assertEqual(v1_txt, v1_obs2.read_text(encoding="utf-8"))
        self.assertIn("crystalCanvas", v1_txt, "Version 1 must preserve the 3D Holographic Crystal Core!")
        self.assertIn("/dfagent/diagrams", v1_txt, "Version 1 must cross-link to Version 2 (/dfagent/diagrams)!")

        # 2. Verify Version 2 (cathrynlavery/diagram-design Editorial Studio at /dfagent/diagrams)
        v2_web = REPO_ROOT / "web_ui" / "templates" / "editorial_diagrams.html"
        v2_obs1 = REPO_ROOT / "observability" / "static" / "editorial_diagrams.html"
        v2_obs2 = REPO_ROOT / "observability" / "static" / "diagrams" / "index.html"
        standalone = REPO_ROOT / "docs" / "diagrams" / "deeptrace_editorial_diagrams.html"
        for p in (v2_web, v2_obs1, v2_obs2, standalone):
            self.assertTrue(p.exists(), f"Missing v2 editorial diagram file: {p}")

        tpl_txt = v2_web.read_text(encoding="utf-8")
        self.assertEqual(tpl_txt, v2_obs1.read_text(encoding="utf-8"), "observability/static/editorial_diagrams.html out of sync!")
        self.assertEqual(tpl_txt, v2_obs2.read_text(encoding="utf-8"), "observability/static/diagrams/index.html out of sync!")

        for req in (
            "family=Inter:",
            "family=Roboto+Mono:",
            "--color-paper: #f5f5f5;",
            "--color-accent: #bf4520;",
            "switchDiagram('arch')",
            "switchDiagram('swimlane')",
            "switchDiagram('flowchart')",
            "switchDiagram('state')",
            "exportActiveSvg()",
            "exportActivePng()",
            "/dfagent/agentic-ai",
        ):
            self.assertIn(req, tpl_txt, f"Missing editorial diagram token '{req}' in editorial_diagrams.html")

        self.assertNotIn("mermaid.min.js", tpl_txt)
        self.assertNotIn("feGaussianBlur", tpl_txt)

        # 3. Verify /dfagent sidebar & tabs expose BOTH versions
        obs_idx_txt = (REPO_ROOT / "observability" / "static" / "index.html").read_text(encoding="utf-8")
        for dual_tok in ("nav-agentic-model", "tab-agentic-model", "nav-editorial-diagrams", "tab-editorial-diagrams", "/dfagent/diagrams"):
            self.assertIn(dual_tok, obs_idx_txt, f"Missing '{dual_tok}' in /dfagent index.html")

    def test_18_grafana_saga_typography_across_all_zones(self):
        """Verify all Web UI zones (/, /benchmark, /dfagent, v1, v2) use Grafana's Inter + Roboto Mono font stack."""
        html_targets = [
            REPO_ROOT / "web_ui" / "templates" / "public_studio.html",
            REPO_ROOT / "web_ui" / "templates" / "benchmark_hub.html",
            REPO_ROOT / "web_ui" / "templates" / "index.html",
            REPO_ROOT / "web_ui" / "templates" / "orchestrator_dashboard.html",
            REPO_ROOT / "web_ui" / "templates" / "agentic_ai_model.html",
            REPO_ROOT / "web_ui" / "templates" / "editorial_diagrams.html",
            REPO_ROOT / "observability" / "static" / "index.html",
        ]
        stale_fonts = ("Plus Jakarta Sans", "Plus+Jakarta+Sans", "Instrument Serif", "Instrument+Serif", "'Outfit'", "family=Outfit", "JetBrains Mono", "JetBrains+Mono")
        for path in html_targets:
            txt = path.read_text(encoding="utf-8")
            self.assertIn("family=Inter:", txt, f"Missing Grafana Inter Google Font in {path.name}")
            self.assertIn("Roboto+Mono", txt, f"Missing Grafana Roboto Mono Google Font in {path.name}")
            for sf in stale_fonts:
                self.assertNotIn(sf, txt, f"Stale non-Grafana font '{sf}' still present in {path.name}")

        style_css = (REPO_ROOT / "web_ui" / "static" / "css" / "style.css").read_text(encoding="utf-8")
        self.assertIn("'Inter'", style_css)
        self.assertIn("'Roboto Mono'", style_css)
        for sf in stale_fonts:
            self.assertNotIn(sf, style_css, f"Stale font '{sf}' still present in style.css")

    def test_19_three_phase_finetune_enhancements_and_corpus_crawler(self):
        """
        Verify all 3 Fine-Tuning Enhancement Phases + Multi-Source Training Corpus Crawler:
          - Stdlib checks (always run in bare GitHub Actions runner)
          - PyTorch/NumPy/PIL runtime checks (run locally / Colab)
        """
        import ast
        import tempfile

        required_files = [
            REPO_ROOT / "colab_training" / "forensic_augmentations.py",
            REPO_ROOT / "colab_training" / "foundation_backbones.py",
            REPO_ROOT / "colab_training" / "temporal_mil_video.py",
            REPO_ROOT / "crawler" / "training_corpus_harvester.py",
            REPO_ROOT / ".github" / "workflows" / "crawl_training_corpus.yml",
        ]
        for fp in required_files:
            self.assertTrue(fp.exists(), f"Missing Phase 1-3 / Crawler deliverable: {fp}")
            if fp.suffix == ".py":
                ast.parse(fp.read_text(encoding="utf-8"), filename=str(fp))

        # No synthetic blank canvas or legacy RandomState in forensic_augmentations.py
        aug_txt = (REPO_ROOT / "colab_training" / "forensic_augmentations.py").read_text(encoding="utf-8")
        self.assertNotIn("Image." + "new", aug_txt)
        self.assertNotIn("Random" + "State", aug_txt)

        # Verify crawl_training_corpus.yml schedules & lightweight manifest-only commits
        wf_txt = (REPO_ROOT / ".github" / "workflows" / "crawl_training_corpus.yml").read_text(encoding="utf-8")
        for req in ("cron:", "workflow_dispatch:", "training_corpus_harvester.py", "report_colab/corpus/*.jsonl", "[skip ci]"):
            self.assertIn(req, wf_txt, f"Missing '{req}' in crawl_training_corpus.yml")

        # Verify stdlib calibration_store 2-param Platt & Conformal helpers (safe in bare stdlib CI)
        import importlib.util
        cal_spec = importlib.util.spec_from_file_location(
            "cal_store_stdlib",
            str(REPO_ROOT / "forensics_kernel" / "calibration_store.py"),
        )
        cal_mod = importlib.util.module_from_spec(cal_spec)
        cal_spec.loader.exec_module(cal_mod)
        apply_platt_2param = cal_mod.apply_platt_2param
        evaluate_conformal_band = cal_mod.evaluate_conformal_band

        p_shifted = apply_platt_2param(0.0, {"platt_a": 1.5, "platt_b": -2.0}, is_logit=True)
        self.assertLess(p_shifted, 0.2)
        p_pos = apply_platt_2param(2.0, {"platt_a": 1.5, "platt_b": -1.0}, is_logit=True)
        self.assertGreater(p_pos, 0.85)
        band = evaluate_conformal_band(0.74, {"optimal_threshold_tau": 0.8118, "conformal_low": 0.67, "conformal_high": 0.90})
        self.assertTrue(band["in_uncertain_band"])
        self.assertEqual(band["action"], "ESCALATE_TO_TRIAD_COUNCIL")

        try:
            import numpy as np
            import torch
            import torch.nn as nn
            from PIL import Image
        except ImportError:
            return

        from colab_training.lora_adapter import (
            DiagonalEWC,
            NativeDoRALinear,
            apply_lora_to_vit,
            build_loraplus_param_groups,
            compute_orthogonal_adapter_loss,
            merge_native_adapters,
        )
        from colab_training.foundation_backbones import (
            OrthogonalSubspaceLinear,
            compute_effort_orthogonality_loss,
        )
        from colab_training.forensic_augmentations import (
            SocialMediaForensicAugment,
            generate_self_blended_image,
            rawboost_augment_waveform,
        )
        from colab_training.train_general import DualStreamGeneralDetector
        from colab_training.train_human import BiometricFusionHead
        from colab_training.temporal_mil_video import TemporalAttentionMIL
        from colab_training.train_audio_fusionnet import AASISTGraphAttentionHead
        from colab_training.train_threads_continual import _calibrate_platt_and_youden

        # 1. DoRA + LoRA+ + Orthogonal Loss + EWC + Merge
        lin = nn.Linear(32, 32)
        dora = NativeDoRALinear(lin, r=4, lora_alpha=8)
        x_in = torch.randn(3, 32)
        out_d = dora(x_in)
        self.assertEqual(out_d.shape, (3, 32))

        toy = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))
        toy[0] = NativeDoRALinear(toy[0], r=4, lora_alpha=8)
        groups = build_loraplus_param_groups(toy, lr=1e-4, loraplus_lr_ratio=16.0)
        self.assertEqual(len(groups), 2)
        self.assertGreater(groups[1]["lr"], groups[0]["lr"])
        ortho_l = compute_orthogonal_adapter_loss(toy)
        self.assertGreaterEqual(float(ortho_l.item()), 0.0)
        ewc = DiagonalEWC.from_model(toy, dataloader=None)
        self.assertGreaterEqual(float(ewc.penalty(toy).item()), 0.0)
        merged = merge_native_adapters(toy)
        self.assertIsInstance(merged[0], nn.Linear)

        # 2. Effort Orthogonal Subspace SVD
        effort_layer = OrthogonalSubspaceLinear(nn.Linear(32, 32), residual_rank=4)
        self.assertEqual(effort_layer(x_in).shape, (3, 32))
        self.assertGreaterEqual(float(effort_layer.orthogonality_loss().item()), 0.0)

        # 3. Augmentations (JPEG/WebP/Blur/Caption + SBI + RawBoost) from deterministic gradient array
        grad = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
        base_im = Image.fromarray(grad, mode="RGB")
        aug_im = SocialMediaForensicAugment(p_jpeg=1.0, p_blur=1.0, p_overlay=1.0, seed=7)(base_im)
        self.assertEqual(aug_im.size, (64, 64))
        sbi_im = generate_self_blended_image(base_im, seed=7)
        self.assertEqual(sbi_im.size, (64, 64))
        wav = np.sin(2.0 * np.pi * 440.0 * np.linspace(0, 1.0, 16000, dtype=np.float32))
        rb_wav = rawboost_augment_waveform(wav, sr=16000, seed=7)
        self.assertEqual(rb_wav.shape, wav.shape)
        self.assertLessEqual(float(np.max(np.abs(rb_wav))), 1.0 + 1e-5)

        # 4. DualStreamGeneralDetector (active gating, 2-ch and 1-ch compat)
        gen = DualStreamGeneralDetector(spectral_channels=2)
        logits_2ch, gates = gen(torch.randn(2, 768), torch.randn(2, 2, 64, 64), return_gates=True)
        self.assertEqual(logits_2ch.shape, (2, 2))
        self.assertTrue(torch.allclose(gates.sum(dim=-1), torch.ones(2), atol=1e-4))
        logits_1ch = gen(torch.randn(2, 768), torch.randn(2, 1, 64, 64))
        self.assertEqual(logits_1ch.shape, (2, 2))

        # 5. Cross-Modal Human Biometric Attention + TemporalAttentionMIL + AASIST
        hum = BiometricFusionHead()
        self.assertEqual(hum(torch.randn(2, 768), torch.randn(2, 768), torch.randn(2, 128), torch.randn(2, 64)).shape, (2, 2))
        mil = TemporalAttentionMIL(feat_dim=64, hidden_dim=128)
        bag_logits, attn_w, frame_logits = mil(torch.randn(2, 12, 64))
        self.assertEqual(bag_logits.shape, (2, 2))
        self.assertTrue(torch.allclose(attn_w.sum(dim=-1), torch.ones(2), atol=1e-4))
        aasist = AASISTGraphAttentionHead(hidden_dim=128)
        self.assertEqual(aasist(torch.randn(2, 16, 128)).shape, (2, 2))

        # 6. 2-Param Platt fixes shifted logits (AUC=1.0, negative shift recovered by bias b)
        y_shifted = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        z_shifted = [-5.0, -4.5, -4.0, -3.5, -3.0, -1.5, -1.0, -0.5, 0.0, 0.5]
        _, tau_s, ece_s, pa, pb, clow, chigh = _calibrate_platt_and_youden(y_shifted, z_shifted)
        self.assertGreater(pa, 0.2)
        self.assertGreater(pb, 0.5)
        self.assertLessEqual(clow, chigh)

        # 7. Offline Multi-Source Corpus Harvester dry-run
        from crawler.training_corpus_harvester import harvest_training_corpus
        with tempfile.TemporaryDirectory(prefix="corpus_test_") as td:
            summary = harvest_training_corpus(
                output_dir=td,
                target_images=12,
                target_sbi_pairs=4,
                target_audios=6,
                offline_local_only=True,
            )
            self.assertTrue((Path(td) / "training_corpus_manifest.jsonl").exists())
            self.assertTrue((Path(td) / "corpus_summary.json").exists())
            self.assertGreater(summary["total_images"], 0)

    def test_20_dfagent_weight_approval_gate_and_feedback_review_job(self):
        """
        Verify strict /dfagent approval gate for weight updates + Automated Feedback Review Job:
          - Public /api/feedback MUST stay PENDING and NEVER update weights/calibration/LoRA directly
          - Automated review job audits PENDING items with AI Triage without mutating review_status
          - Calibration & training replay only consume APPROVED_REAL / APPROVED_FAKE / APPROVED_CALIBRATED
        """
        import ast
        import json
        import tempfile
        from unittest.mock import patch

        job_script = REPO_ROOT / "scripts" / "review_feedback_queue_job.py"
        job_wf = REPO_ROOT / ".github" / "workflows" / "feedback_review_job.yml"
        self.assertTrue(job_script.exists(), "Missing scripts/review_feedback_queue_job.py")
        self.assertTrue(job_wf.exists(), "Missing .github/workflows/feedback_review_job.yml")
        ast.parse(job_script.read_text(encoding="utf-8"), filename=str(job_script))

        wf_txt = job_wf.read_text(encoding="utf-8")
        for tok in ("cron:", "workflow_dispatch:", "review_feedback_queue_job.py", "[skip ci]"):
            self.assertIn(tok, wf_txt, f"Missing '{tok}' in feedback_review_job.yml")

        app_js = (REPO_ROOT / "web_ui" / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("Gửi duyệt /dfagent", app_js)
        self.assertNotIn("` / dfagent`", app_js, "Unescaped template-literal bug still present in app.js")
        self.assertIn("trigger_lora_update: false", app_js)

        obs_html = (REPO_ROOT / "observability" / "static" / "index.html").read_text(encoding="utf-8")
        for tok in ("runFeedbackReviewJob", "APPROVE_AND_UPDATE_WEIGHTS", "rqReviewJobStatus"):
            self.assertIn(tok, obs_html, f"Missing '{tok}' in /dfagent observability UI")

        for fname in (
            REPO_ROOT / "colab_training" / "train_threads_continual.py",
            REPO_ROOT / "crawler" / "training_corpus_harvester.py",
        ):
            txt = fname.read_text(encoding="utf-8")
            for st_tok in ("APPROVED_REAL", "APPROVED_FAKE", "APPROVED_CALIBRATED"):
                self.assertIn(st_tok, txt, f"Missing SQL approval filter '{st_tok}' in {fname.name}")

        # Runtime verification with isolated temp SQLite & Flask test client (when flask + numpy/cv2 exist)
        try:
            import cv2  # noqa: F401
            import numpy  # noqa: F401
            from flask import Flask
        except ImportError:
            return

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from benchmark.deepfake_memory import (
            get_approved_feedback_for_weights,
            get_recent_feedback,
            init_db,
            update_feedback_review_status,
        )
        from scripts.review_feedback_queue_job import run_feedback_review_job
        from web_ui.routes.feedback_routes import create_feedback_blueprint

        with tempfile.TemporaryDirectory(prefix="fb_gate_test_") as td:
            tmp_db = str(Path(td) / "test_memory.db")
            tmp_out = str(Path(td) / "review_out")
            init_db(db_path=tmp_db)

            app = Flask(__name__)
            app.register_blueprint(
                create_feedback_blueprint(
                    get_detector_fn=lambda: None,
                    emit_fn=lambda *a, **k: None,
                    execute_short_video_fn=lambda **k: {},
                )
            )
            client = app.test_client()

            # Spoofed public submission attempting to bypass /dfagent approval
            import benchmark.feedback_weight_updater as fwu
            orig_submit = fwu.FeedbackWeightUpdater.submit_feedback

            def _submit_with_tmp_db(*args, **kwargs):
                kwargs["db_path"] = tmp_db
                return orig_submit(*args, **kwargs)

            with patch.object(fwu.FeedbackWeightUpdater, "submit_feedback", side_effect=_submit_with_tmp_db):
                resp = client.post(
                    "/api/feedback",
                    json={
                        "true_label": "Realism",
                        "predicted_label": "Deepfake",
                        "media_type": "video",
                        "composite_risk_score": 0.62,
                        "source_type": "AUTO_PENTEST",
                        "auto_update_calibration": True,
                        "trigger_lora_update": True,
                        "user_note": "False positive test",
                    },
                )
                self.assertEqual(resp.status_code, 200)
                payload = resp.get_json()
                self.assertEqual(payload["review_status"], "PENDING")
                self.assertTrue(payload["requires_dfagent_approval"])
                self.assertIsNone(payload["calibration_update"])
                self.assertIsNone(payload["lora_update"])
                fid = payload["feedback_id"]

            # Verify PENDING items are blocked from weight/calibration consumption
            self.assertEqual(len(get_approved_feedback_for_weights(db_path=tmp_db)), 0)

            # Run automated Feedback Review & Pre-Triage Job
            job_summary = run_feedback_review_job(
                db_path=tmp_db,
                apply_approved_weights=True,
                trigger_source="unit_test_job",
                output_dir=tmp_out,
            )
            self.assertEqual(job_summary["pending_audited"], 1)
            self.assertFalse(job_summary["weights_applied"])
            self.assertTrue((Path(tmp_out) / "latest_feedback_review_job.json").exists())

            # Review status MUST remain PENDING with AI triage populated
            pending_rows = get_recent_feedback(limit=5, source_type="HUMAN_REVIEW", review_status="PENDING", db_path=tmp_db)
            self.assertEqual(len(pending_rows), 1)
            self.assertTrue(bool(pending_rows[0].get("ai_triage_recommendation")))

            # Explicit /dfagent Admin approval unlocks feedback for weight updates
            self.assertTrue(update_feedback_review_status(fid, "APPROVED_REAL", reviewer_note="Admin OK", db_path=tmp_db))
            approved_rows = get_approved_feedback_for_weights(db_path=tmp_db)
            self.assertEqual(len(approved_rows), 1)
            self.assertEqual(approved_rows[0]["feedback_id"], fid)

    def test_21_calibration_live_history_and_weight_update_detailed_report(self):
        """
        Regression test for:
          1) /api/calibration/metrics aggregating live post-2026-11-19 pentest/benchmark history
             with real AUC/ECE values and never mis-mapping error_rate into raw_ece.
          2) Detailed Weight Update Audit Report (parameter_diffs, metric_diffs, active_weights_snapshot,
             Markdown export, and /dfagent UI report containers).
        """
        import importlib.util
        import json
        import sqlite3
        import tempfile

        audit_path = REPO_ROOT / "observability" / "app" / "calibration_audit.py"
        self.assertTrue(audit_path.exists(), "Missing observability/app/calibration_audit.py")
        spec = importlib.util.spec_from_file_location("calibration_audit_mod", str(audit_path))
        cal_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cal_mod)

        obs_main = (REPO_ROOT / "observability" / "app" / "main.py").read_text(encoding="utf-8")
        for tok in (
            "/api/calibration/metrics",
            "/dfagent/api/calibration/metrics",
            "/api/calibration/weight-updates/report",
            "/dfagent/api/calibration/weight-updates/report",
            "/api/calibration/weight-updates/export",
            "/dfagent/api/calibration/weight-updates/export",
            "build_live_calibration_metrics",
            "build_weight_update_detailed_report",
            "render_weight_update_markdown_report",
        ):
            self.assertIn(tok, obs_main, f"Missing '{tok}' in observability/app/main.py")

        fwu_txt = (REPO_ROOT / "benchmark" / "feedback_weight_updater.py").read_text(encoding="utf-8")
        self.assertIn("_persist_weight_update_report_artifact", fwu_txt)
        self.assertIn("WEIGHT_UPDATES_REPORT_DIR", fwu_txt)

        obs_html = (REPO_ROOT / "observability" / "static" / "index.html").read_text(encoding="utf-8")
        for tok in (
            "weightUpdateReportSection",
            "activeWeightsMatrixBody",
            "weightUpdateDiffContainer",
            "modelWeightsRegistryBody",
            "calibLastSyncBadge",
            "downloadWeightUpdateReport",
            "filterCalibrationHistory",
            "api/calibration/weight-updates/report",
            "api/calibration/weight-updates/export",
        ):
            self.assertIn(tok, obs_html, f"Missing '{tok}' in observability/static/index.html")

        with tempfile.TemporaryDirectory(prefix="cal_audit_test_") as td:
            tmp_db = Path(td) / "test_memory.db"
            conn = sqlite3.connect(str(tmp_db))
            conn.execute("""
                CREATE TABLE calibration_weight_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    update_id TEXT UNIQUE NOT NULL,
                    trigger_source TEXT NOT NULL,
                    feedback_count INTEGER DEFAULT 0,
                    before_params_json TEXT,
                    after_params_json TEXT,
                    before_metrics_json TEXT,
                    after_metrics_json TEXT,
                    accepted INTEGER DEFAULT 1,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE model_weights_registry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    version_tag TEXT NOT NULL,
                    weights_file TEXT NOT NULL,
                    backbone TEXT,
                    status TEXT DEFAULT 'active',
                    metrics_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE user_feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    feedback_id TEXT UNIQUE NOT NULL,
                    review_status TEXT DEFAULT 'PENDING'
                )
            """)
            conn.execute(
                """
                INSERT INTO calibration_weight_updates
                (update_id, trigger_source, feedback_count, before_params_json, after_params_json, before_metrics_json, after_metrics_json, accepted, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "WUP-TEST-001",
                    "feedback:FB-TEST:FALSE_NEGATIVE",
                    3,
                    json.dumps({"tau_video": 0.8118, "w_peak": 0.50, "w_avg": 0.35, "w_jitter": 0.15}),
                    json.dumps({"tau_video": 0.7318, "w_peak": 0.62, "w_avg": 0.28, "w_jitter": 0.10}),
                    json.dumps({"fp": 0, "fn": 2, "error_rate": 0.50, "mean_bce": 0.812, "objective": 1.95}),
                    json.dumps({"fp": 0, "fn": 0, "error_rate": 0.0, "mean_bce": 0.315, "objective": 0.32}),
                    1,
                    "Test fast-loop calibration improvement",
                    "2026-11-26 09:00:00",
                ),
            )
            conn.execute(
                """
                INSERT INTO model_weights_registry (date, version_tag, weights_file, backbone, status, metrics_json)
                VALUES ('2026-11-26', 'lora-test-v1', 'custom_vit_model/lora_feedback_adapter.pt', 'ViT-LoRA', 'active', '{}')
                """
            )
            conn.execute("INSERT INTO user_feedback_events (feedback_id, review_status) VALUES ('FB-1', 'PENDING'), ('FB-2', 'APPROVED_FAKE')")
            conn.commit()
            conn.close()

            metrics = cal_mod.build_live_calibration_metrics(
                repo_root=REPO_ROOT,
                custom_vit_dir=REPO_ROOT / "custom_vit_model",
                db_path=tmp_db,
            )
            self.assertEqual(metrics["status"], "SUCCESS")
            hist = metrics["training_epochs_history"]
            self.assertGreaterEqual(len(hist), 3)
            # Latest history MUST NOT be frozen on 2026-11-19 when 2026-11-25/26 reports exist
            self.assertGreaterEqual(hist[0]["date"][:10], "2026-11-25", f"Expected fresh >= 2026-11-25 history, got: {hist[0]}")
            # Empirical pentest rows must expose numeric AUC & ECE (not '--')
            pentest_rows = [r for r in hist if r["source_type"] == "pentest" and r["calib_ece"] != "--"]
            self.assertTrue(len(pentest_rows) >= 1, "Expected live pentest rows with real calib_ece")
            self.assertNotEqual(pentest_rows[0]["val_auc"], "--")
            # Weight update row must NOT copy error_rate into raw_ece
            wup_rows = [r for r in hist if r["source_type"] == "weight_update"]
            self.assertEqual(len(wup_rows), 1)
            self.assertEqual(wup_rows[0]["raw_ece"], "--")
            self.assertEqual(wup_rows[0]["loss"], 0.315)

            report = cal_mod.build_weight_update_detailed_report(
                repo_root=REPO_ROOT,
                custom_vit_dir=REPO_ROOT / "custom_vit_model",
                db_path=tmp_db,
            )
            self.assertEqual(report["summary"]["total_weight_updates"], 1)
            self.assertEqual(report["summary"]["accepted_updates"], 1)
            self.assertEqual(report["summary"]["feedback_pending_review"], 1)
            self.assertEqual(report["summary"]["feedback_approved_total"], 1)

            upd = report["weight_updates"][0]
            self.assertEqual(upd["update_id"], "WUP-TEST-001")
            self.assertEqual(upd["modality"], "video")
            tau_diff = next(d for d in upd["parameter_diffs"] if d["param"] == "tau_video")
            self.assertEqual(tau_diff["before"], 0.8118)
            self.assertEqual(tau_diff["after"], 0.7318)
            self.assertAlmostEqual(tau_diff["delta"], -0.08, places=4)
            self.assertTrue(tau_diff["changed"])

            fn_diff = next(d for d in upd["metric_diffs"] if d["metric"] == "fn")
            self.assertEqual(fn_diff["before"], 2.0)
            self.assertEqual(fn_diff["after"], 0.0)
            self.assertTrue(fn_diff["improved"])

            md = cal_mod.render_weight_update_markdown_report(report)
            self.assertIn("WUP-TEST-001", md)
            self.assertIn("tau_video", md)
            self.assertIn("-0.08", md)
            self.assertIn("Before", md)

    def test_22_zero_leakage_locked_threshold_and_empirical_ablation_guard(self):
        """
        Regression test for Thesis Remediation Phases 1-3:
          1) Zero origin_id overlap across train / cal_val / test_locked in corpus manifest & trainers.
          2) Simulation scripts carry explicit METHODOLOGY_SIMULATION_ONLY provenance guards.
          3) Video & audio pentests report data_provenance, Wilson 95% CI & zero-recall warnings.
          4) Empirical ablation suite writes reproducible locked-threshold results & per-sample JSONL.
        """
        import importlib.util
        import json

        guard_path = REPO_ROOT / "forensics_kernel" / "dataset_split_guard.py"
        self.assertTrue(guard_path.exists(), "Missing forensics_kernel/dataset_split_guard.py")
        spec = importlib.util.spec_from_file_location("dataset_split_guard_mod", str(guard_path))
        gmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gmod)

        # Synthetic leakage MUST raise ValueError
        with self.assertRaises(ValueError):
            gmod.assert_zero_origin_leakage(
                [{"filename": "portrait_abc.jpg", "label": 0}],
                [{"filename": "sbi_0001_portrait_abc.jpg", "label": 1}],
            )

        # Group-level splitting keeps source & SBI counterfactual inside exact same split
        toy = [
            {"filename": f"vn_img_{i}.jpg", "label": 0, "vietnamese_category": "vn_selfie_lifestyle"}
            for i in range(20)
        ]
        grouped = gmod.assign_leak_free_group_splits(toy, train_ratio=0.70, cal_ratio=0.15, seed=2026)
        audit = gmod.assert_zero_origin_leakage(
            [r for r in grouped if r["split"] == "train"],
            [r for r in grouped if r["split"] == "cal_val"],
            [r for r in grouped if r["split"] == "test_locked"],
        )
        self.assertEqual(audit["train_cal_overlap"], 0)
        self.assertEqual(audit["train_test_overlap"], 0)
        self.assertGreater(audit["cal_val_samples"], 0)
        self.assertGreater(audit["test_locked_samples"], 0)

        # Wilson & Bootstrap 95% CI sanity
        lo, hi = gmod.compute_wilson_ci(18, 20)
        self.assertTrue(0.65 < lo < 0.90 < hi <= 1.0)
        bci = gmod.compute_bootstrap_ci([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.4, 0.6, 0.8, 0.9], tau=0.5, n_boot=100)
        self.assertAlmostEqual(bci["tpr"], 1.0, places=4)
        self.assertAlmostEqual(bci["fpr"], 0.0, places=4)

        # Live corpus manifest on disk MUST have zero origin overlap across splits
        live_manifest = REPO_ROOT / "report_colab" / "corpus" / "training_corpus_manifest.jsonl"
        if live_manifest.exists():
            loaded = gmod.load_locked_manifest_splits(live_manifest, exclude_unverified=True, require_existing_files=False)
            la = loaded["audit"]
            self.assertEqual(la["train_cal_overlap"], 0)
            self.assertEqual(la["train_test_overlap"], 0)
            self.assertEqual(la["cal_test_overlap"], 0)
            self.assertGreater(la["test_locked_samples"], 0)

        # All 3 trainers must enforce assert_zero_origin_leakage and exclude test_locked
        for trainer_rel in (
            "colab_training/train_general.py",
            "colab_training/train_human.py",
            "colab_training/train_threads_continual.py",
        ):
            txt = (REPO_ROOT / trainer_rel).read_text(encoding="utf-8")
            self.assertIn("assert_zero_origin_leakage", txt, f"Missing leakage assertion in {trainer_rel}")
            self.assertIn("test_locked", txt, f"Missing test_locked exclusion in {trainer_rel}")

        # All 3 simulation scripts must carry METHODOLOGY_SIMULATION_ONLY guard
        for sim_rel in (
            "benchmark/ablation_study/eval_ablation_grid_search.py",
            "benchmark/eval_architectural_comparison.py",
            "benchmark/eval_demographic_fairness_paper.py",
        ):
            stxt = (REPO_ROOT / sim_rel).read_text(encoding="utf-8")
            self.assertIn("METHODOLOGY_SIMULATION_ONLY", stxt, f"Missing simulation banner in {sim_rel}")

        # Video & audio pentests must report operating_threshold_audit & data_provenance
        for pent_rel in ("benchmark/short_video_pentest.py", "benchmark/audio_pentest.py"):
            ptxt = (REPO_ROOT / pent_rel).read_text(encoding="utf-8")
            for tok in ("operating_threshold_audit", "data_provenance", "compute_wilson_ci", "thesis_eligible"):
                self.assertIn(tok, ptxt, f"Missing '{tok}' in {pent_rel}")

        # Empirical suite existence & schema check
        emp_py = REPO_ROOT / "benchmark" / "eval_empirical_thesis_suite.py"
        self.assertTrue(emp_py.exists(), "Missing benchmark/eval_empirical_thesis_suite.py")
        emp_res = REPO_ROOT / "report_colab" / "empirical_thesis_eval" / "empirical_ablation_results.json"
        if emp_res.exists():
            edata = json.loads(emp_res.read_text(encoding="utf-8"))
            self.assertEqual(edata.get("evaluation_provenance"), "EMPIRICAL_LOCKED_SPLIT_INFERENCE")
            self.assertEqual(edata["leakage_audit"]["train_test_overlap"], 0)
            self.assertIn("clean", edata["views"])
            self.assertIn("social_reupload", edata["views"])
            self.assertIn("unseen_manipulation_ood", edata["views"])
            preds_jsonl = REPO_ROOT / "report_colab" / "empirical_thesis_eval" / "per_sample_predictions.jsonl"
            self.assertTrue(preds_jsonl.exists())

    def test_23_vn_compression_signal_survival_quality_fusion_and_dicome_deferral(self):
        """
        Regression test for 2026 Vietnamese Compression Benchmark, Quality-Aware Fusion & DiCoME Deferral:
          1) calibration_store.py stays 100% pure-stdlib (`py -3 -S`) and provides
             `compute_signal_quality_weights`, `evaluate_dicome_deferral`, and `select_matched_coverage_decisions`.
          2) WaveletSpectralAnalyzer & CornealReflectionAnalyzer contain NO `random.uniform` placeholders,
             produce deterministic identical results on repeat calls, and show monotonic `q_freq` drop under compression.
          3) Saved empirical results in `report_colab/vn_compression_deferral_eval/` pass zero-leakage and schema checks.
        """
        import importlib.util
        import json

        cal_path = REPO_ROOT / "forensics_kernel" / "calibration_store.py"
        spec = importlib.util.spec_from_file_location("calibration_store_mod", str(cal_path))
        cmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cmod)

        # Clean signal vs heavily degraded frequency + occluded corneal signal
        w_clean = cmod.compute_signal_quality_weights(q_vision=0.95, q_freq=0.92, q_eye=0.80)
        w_degraded = cmod.compute_signal_quality_weights(q_vision=0.85, q_freq=0.15, q_eye=0.12)
        self.assertAlmostEqual(sum(w_clean.values()), 1.0, places=4)
        self.assertAlmostEqual(sum(w_degraded.values()), 1.0, places=4)
        self.assertGreater(w_degraded["w_vision"], w_clean["w_vision"])
        self.assertLess(w_degraded["w_freq"], w_clean["w_freq"])
        self.assertLess(w_degraded["w_eye"], w_clean["w_eye"])

        # High conflict + poor ocular quality must raise DiCoME deferral score above harmonious confident sample
        d_harm = cmod.evaluate_dicome_deferral(0.92, 0.89, 0.85, p_fused=0.90, q_face_eye=0.85, compression_severity=0.08, tau=0.45)
        d_conf = cmod.evaluate_dicome_deferral(0.90, 0.12, 0.25, p_fused=0.48, q_face_eye=0.20, compression_severity=0.75, tau=0.45)
        self.assertGreater(d_conf["deferral_score"], d_harm["deferral_score"] + 0.30)
        self.assertTrue(d_conf["should_defer"])
        self.assertFalse(d_harm["should_defer"])

        keep = cmod.select_matched_coverage_decisions([0.1, 0.8, 0.2, 0.9, 0.3], coverage=0.60)
        self.assertEqual(keep, [True, False, True, False, True])

        # Source guards against placeholder random.uniform regression
        for mod_rel in ("forensics_kernel/spectral_wavelet.py", "forensics_kernel/biometric_corneal.py"):
            mtxt = (REPO_ROOT / mod_rel).read_text(encoding="utf-8")
            self.assertNotIn("random.uniform", mtxt, f"Placeholder random.uniform found in {mod_rel}")

        orch_txt = (REPO_ROOT / "agents" / "unified_master_orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("corneal_analyzer.analyze(media_path)", orch_txt)
        self.assertIn("wavelet_analyzer.analyze(media_path)", orch_txt)

        # Runtime analyzer checks when numpy & PIL are available (skipped cleanly under `py -3 -S`)
        try:
            import io
            import numpy as np
            from PIL import Image, ImageFilter
        except ImportError:
            return

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from forensics_kernel.spectral_wavelet import WaveletSpectralAnalyzer
        from forensics_kernel.biometric_corneal import CornealReflectionAnalyzer

        yy, xx = np.mgrid[0:128, 0:128]
        synth = np.stack([
            (128 + 55 * np.sin(xx / 4.0) * np.cos(yy / 5.0)).clip(0, 255),
            (115 + 45 * np.cos(xx / 6.0)).clip(0, 255),
            (105 + 40 * np.sin(yy / 5.0)).clip(0, 255),
        ], axis=-1).astype(np.uint8)
        im_clean = Image.fromarray(synth, mode="RGB")
        buf = io.BytesIO()
        im_clean.filter(ImageFilter.GaussianBlur(radius=1.2)).save(buf, format="JPEG", quality=22)
        buf.seek(0)
        im_comp = Image.open(buf).convert("RGB")

        wa = WaveletSpectralAnalyzer()
        ca = CornealReflectionAnalyzer()
        r1 = wa.analyze(im_clean)
        r2 = wa.analyze(im_clean)
        r_comp = wa.analyze(im_comp)
        self.assertEqual(r1, r2, "WaveletSpectralAnalyzer must be deterministic")
        self.assertGreater(r1["frequency_signal_quality"], r_comp["frequency_signal_quality"])

        c1 = ca.analyze(im_clean)
        c2 = ca.analyze(im_clean)
        self.assertEqual(c1, c2, "CornealReflectionAnalyzer must be deterministic")
        self.assertEqual(len(c1["feature_vector"]), 10)

        # Saved empirical results check
        res_path = REPO_ROOT / "report_colab" / "vn_compression_deferral_eval" / "vn_compression_deferral_results.json"
        if res_path.exists():
            rdata = json.loads(res_path.read_text(encoding="utf-8"))
            self.assertEqual(rdata.get("evaluation_provenance"), "EMPIRICAL_LOCKED_VN_COMPRESSION_QUALITY_DEFERRAL")
            self.assertEqual(rdata["leakage_audit"]["train_test_overlap"], 0)
            self.assertIn("heavy_social_webp_q32_r50_blur", rdata["protocol_a_compression_and_ood"])
            self.assertIn("by_optical_covariate_light2lie", rdata["protocol_b_vietnamese_conditions"])
            self.assertIn("pooled_challenge_suite_youden_tau", rdata["selective_deferral_matched_coverage"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
