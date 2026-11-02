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
TEMPLATE_HTML_PATH = REPO_ROOT / "web_ui" / "templates" / "index.html"
DETECTOR_PY_PATH = REPO_ROOT / "web_ui" / "deepfake_detector.py"


class TestUIRegressionBugs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert STATIC_JS_PATH.exists(), f"File not found: {STATIC_JS_PATH}"
        assert TEMPLATE_HTML_PATH.exists(), f"File not found: {TEMPLATE_HTML_PATH}"
        with open(STATIC_JS_PATH, "r", encoding="utf-8") as f:
            cls.js_content = f.read()
        with open(TEMPLATE_HTML_PATH, "r", encoding="utf-8") as f:
            cls.html_content = f.read()

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
