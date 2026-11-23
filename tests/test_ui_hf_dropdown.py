import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestUIHfDropdown(unittest.TestCase):
    def setUp(self):
        self.html_path = PROJECT_ROOT / "web_ui" / "templates" / "benchmark_hub.html"
        self.partial_path = PROJECT_ROOT / "web_ui" / "templates" / "partials" / "benchmark_hub_body.html"
        self.js_path = PROJECT_ROOT / "web_ui" / "static" / "js" / "app.js"
        with open(self.html_path, "r", encoding="utf-8") as f:
            hub_html = f.read()
        with open(self.partial_path, "r", encoding="utf-8") as f:
            partial_html = f.read()
        self.html = hub_html + "\n" + partial_html
        with open(self.js_path, "r", encoding="utf-8") as f:
            self.js = f.read()

    def test_custom_hf_select_dropdown_exists(self):
        self.assertIn('id="customHfPresetSelect"', self.html)
        self.assertIn('onchange="handleHfPresetChange()"', self.html)

    def test_all_13_presets_present_in_select(self):
        expected_repos = [
            "MelodyMachine/Deepfake-audio-detection-V2",
            "garystafford/deepfake-audio-detection",
            "UniDataPro/real-vs-fake-human-voice-deepfake-audio",
            "unfake/fake_voices",
            "thenewsupercell/celeb-df-audio-dataset",
            "prithivMLmods/Deepfake-vs-Real-60K",
            "OpenRL/DeepFakeFace",
            "jzousz/GenImage",
            "dragonintelligence/CIFAKE-image-dataset",
            "ThangCao106/Threads-DFBench-Vietnamese",
            "ControlNet/AV-Deepfake1M",
            "bitmind/FaceForensicsC23",
            "nuriachandra/Deepfake-Eval-2024"
        ]
        for repo in expected_repos:
            self.assertIn(repo, self.html, f"Missing preset repo in index.html: {repo}")

    def test_main_benchmark_dataset_select_has_hf_optgroup(self):
        self.assertIn('optgroup label="🤗 Hugging Face Deepfake Hub (13 Datasets)"', self.html)
        self.assertIn('onchange="handleDatasetSelectChange()"', self.html)

    def test_js_functions_defined(self):
        self.assertIn('function handleHfPresetChange()', self.js)
        self.assertIn('function handleDatasetSelectChange()', self.js)
        self.assertIn('function handleImportSourceChange()', self.js)

if __name__ == '__main__':
    unittest.main()

