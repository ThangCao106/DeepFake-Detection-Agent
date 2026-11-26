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


if __name__ == "__main__":
    unittest.main(verbosity=2)
