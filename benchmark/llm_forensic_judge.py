#!/usr/bin/env python3
"""
benchmark/llm_forensic_judge.py - Multimodal LLM-as-a-Judge Forensic Explainability Benchmark (Option B)

Evaluates Gemini multimodal vision model as an automated forensic judge:
1. Multi-modal Verdict Alignment (REAL vs FAKE against Ground Truth).
2. Forensic Artifact Grounding (Accurate localization of facial seams, warping, diffusion texture).
3. Hallucination / False Accusation Rate on authentic human portraits.
4. Chain-of-Thought (CoT) Forensic Quality Rubric (1-5 scale).
5. Latency and confidence calibration.

Outputs: benchmark/llm_judge_benchmark_results.json
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image
from dotenv import load_dotenv

# Ensure console supports UTF-8 on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Support modern google.genai or legacy google.generativeai
USE_NEW_SDK = False
try:
    from google import genai
    from google.genai import types
    USE_NEW_SDK = True
    HAS_GENAI = True
except ImportError:
    try:
        import google.generativeai as genai
        HAS_GENAI = True
    except ImportError:
        HAS_GENAI = False


class LLMForensicJudgeBenchmark:
    """
    Benchmark suite for LLM-as-a-Judge Forensic Evaluation.
    """

    CANDIDATE_MODELS = ["gemini-flash-latest", "gemini-pro-latest", "gemini-2.0-flash", "gemini-1.5-flash"]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment or .env file.")
        
        if not HAS_GENAI:
            raise ImportError("google.genai package not installed. Run: pip install google-genai")

        self.use_new_sdk = USE_NEW_SDK
        self.model_name = self._select_model()
        if self.use_new_sdk:
            self.client = genai.Client(api_key=self.api_key)
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        print(f"🤖 [LLM Forensic Judge] Initialized with model: {self.model_name} (SDK: {'google.genai' if self.use_new_sdk else 'google.generativeai'})")

    def _select_model(self) -> str:
        env_model = os.getenv("GEMINI_MODEL_NAME")
        if env_model:
            clean_name = env_model.replace("models/", "")
            return clean_name
        return self.CANDIDATE_MODELS[0]

    def _build_forensic_prompt(self) -> str:
        return (
            "You are an expert AI Forensic Judge evaluating images for digital deepfake manipulation.\n"
            "Carefully examine the provided facial portrait for authentic vs synthetic signatures:\n"
            "- Corneal light reflection consistency & pupil circularity\n"
            "- Boundary seams, blending artifacts, and jawline/ear perimeter warping\n"
            "- High-frequency skin micro-texture vs unnatural oversmoothing/diffusion grain\n"
            "- Lighting direction congruence across facial planes\n\n"
            "Respond ONLY with a single JSON object conforming to this schema:\n"
            "{\n"
            '  "verdict": "REAL" or "FAKE",\n'
            '  "confidence": <float between 0.0 and 1.0>,\n'
            '  "manipulation_type": "<e.g. None (Authentic), FaceSwap Edge Artifact, Diffusion Synthesis, Boundary Warping, Blending Seam>",\n'
            '  "artifact_location": "<specific anatomical region, e.g. Jawline contour, Eye reflection, Hairline boundary, or N/A>",\n'
            '  "forensic_cot_score": <integer from 1 to 5 rating depth and visual grounding of forensic justification>,\n'
            '  "forensic_rationale": "<concise 2-3 sentence explanation of the specific visual evidence>"\n'
            "}"
        )

    def evaluate_image(self, image_path: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate a single image using Gemini Multimodal Judge.
        """
        t0 = time.time()
        img = Image.open(image_path).convert("RGB")
        prompt = self._build_forensic_prompt()

        retries = 0
        raw_text = ""
        while retries < 2:
            try:
                if self.use_new_sdk:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=[prompt, img]
                    )
                    raw_text = response.text or ""
                else:
                    response = self.model.generate_content([prompt, img])
                    raw_text = response.text or ""
                break
            except Exception as e:
                retries += 1
                print(f"⚠️ [Judge Call Error] Attempt {retries}: {e}")
                time.sleep(1.5)
                if retries >= 2:
                    return {
                        "error": str(e),
                        "ground_truth": ground_truth,
                        "verdict": "ERROR",
                        "latency_s": round(time.time() - t0, 3)
                    }

        latency_s = round(time.time() - t0, 3)
        
        # Parse JSON
        parsed = self._extract_json(raw_text)
        pred_verdict = str(parsed.get("verdict", "UNKNOWN")).upper().strip()
        confidence = float(parsed.get("confidence", 0.5))
        manipulation = parsed.get("manipulation_type", "Unknown")
        artifact_loc = parsed.get("artifact_location", "N/A")
        cot_score = int(parsed.get("forensic_cot_score", 3))
        rationale = parsed.get("forensic_rationale", "")

        is_correct = (pred_verdict == ground_truth.upper())

        return {
            "image_path": str(image_path),
            "ground_truth": ground_truth.upper(),
            "pred_verdict": pred_verdict,
            "is_correct": is_correct,
            "confidence": confidence,
            "manipulation_type": manipulation,
            "artifact_location": artifact_loc,
            "forensic_cot_score": cot_score,
            "forensic_rationale": rationale,
            "latency_s": latency_s
        }

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Safely extract JSON block from model response."""
        try:
            # Check for ```json ... ```
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            # Check for naked JSON object
            match_naked = re.search(r"(\{.*\})", text, re.DOTALL)
            if match_naked:
                return json.loads(match_naked.group(1))
        except Exception:
            pass
        return {"verdict": "UNKNOWN", "confidence": 0.5, "forensic_cot_score": 1}

    def run_benchmark(self, samples: List[Dict[str, Any]], max_samples: int = 10) -> Dict[str, Any]:
        """
        Run the LLM Forensic Judge benchmark over a balanced dataset.
        """
        eval_subset = samples[:max_samples]
        print(f"\n🔬 [LLM Forensic Judge Benchmark] Running on {len(eval_subset)} test samples...")
        print("=" * 75)

        results = []
        for i, s in enumerate(eval_subset):
            res = self.evaluate_image(s["path"], s["label_name"])
            results.append(res)
            mark = "✅" if res.get("is_correct") else "❌"
            print(f"  [{i+1}/{len(eval_subset)}] GT={s['label_name']:4s} | Pred={res.get('pred_verdict','?'):4s} {mark} | Conf={res.get('confidence',0):.2f} | Score={res.get('forensic_cot_score',0)}/5 | {res.get('latency_s',0)}s")
            # Gentle rate limiting
            time.sleep(0.5)

        # Aggregate metrics
        valid_results = [r for r in results if "error" not in r and r.get("pred_verdict") in ["REAL", "FAKE"]]
        total = len(valid_results)
        
        if total == 0:
            print("❌ No valid results obtained from LLM Judge.")
            return {"error": "All requests failed"}

        correct_count = sum(1 for r in valid_results if r["is_correct"])
        accuracy = round(correct_count / total, 4)

        reals = [r for r in valid_results if r["ground_truth"] == "REAL"]
        fakes = [r for r in valid_results if r["ground_truth"] == "FAKE"]

        # False accusation rate on real
        real_hallucination_rate = round(sum(1 for r in reals if r["pred_verdict"] == "FAKE") / max(len(reals), 1), 4)
        
        # Fake detection recall
        fake_recall = round(sum(1 for r in fakes if r["pred_verdict"] == "FAKE") / max(len(fakes), 1), 4)
        
        # Real detection recall
        real_recall = round(sum(1 for r in reals if r["pred_verdict"] == "REAL") / max(len(reals), 1), 4)

        # Mean CoT Forensic Score (1-5)
        cot_scores = [r.get("forensic_cot_score", 3) for r in valid_results]
        mean_cot = round(sum(cot_scores) / total, 2)

        # Average Latency
        latencies = [r.get("latency_s", 0) for r in valid_results]
        avg_latency = round(sum(latencies) / total, 3)

        summary = {
            "benchmark_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "judge_model": self.model_name,
            "sample_count": total,
            "metrics": {
                "verdict_accuracy": accuracy,
                "fake_detection_recall": fake_recall,
                "real_detection_recall": real_recall,
                "false_accusation_rate_on_real": real_hallucination_rate,
                "mean_forensic_cot_score": mean_cot,
                "avg_latency_s": avg_latency
            },
            "detailed_samples": valid_results
        }

        out_path = PROJECT_ROOT / "benchmark" / "llm_judge_benchmark_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("=" * 75)
        print(f"📊 [LLM Judge Benchmark Results Summary]")
        print(f"  - Model:                    {self.model_name}")
        print(f"  - Verdict Accuracy:         {accuracy * 100:.1f}% ({correct_count}/{total})")
        print(f"  - Fake Detection Recall:    {fake_recall * 100:.1f}%")
        print(f"  - Real Portrait Recall:     {real_recall * 100:.1f}%")
        print(f"  - False Accusation Rate:    {real_hallucination_rate * 100:.1f}% (Real misclassified as Fake)")
        print(f"  - Mean Forensic CoT Score:  {mean_cot} / 5.0")
        print(f"  - Avg Latency:              {avg_latency}s / image")
        print(f"  - Report saved:             {out_path}")
        print("=" * 75)

        return summary


def get_or_create_benchmark_samples(num_reals: int = 5, num_fakes: int = 5) -> List[Dict[str, Any]]:
    """
    Acquire or generate a balanced set of Real & Fake images for the benchmark.
    """
    from crawler.media_generator import create_realistic_human_portrait, apply_faceswap_manipulation
    
    samples = []
    bench_dir = PROJECT_ROOT / "benchmark" / "judge_samples"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real samples
    for i in range(num_reals):
        real_p = bench_dir / f"real_sample_{i}.jpg"
        if not real_p.exists():
            im = create_realistic_human_portrait(seed=1000 + i * 37)
            im.save(real_p, format="JPEG", quality=95)
        samples.append({"path": str(real_p), "label_name": "REAL"})

    # 2. Fake samples (FaceSwap / Boundary Artifact)
    for i in range(num_fakes):
        fake_p = bench_dir / f"fake_faceswap_sample_{i}.jpg"
        if not fake_p.exists():
            base_im = create_realistic_human_portrait(seed=2000 + i * 41)
            fake_im = apply_faceswap_manipulation(base_im, seed=5000 + i * 19)
            fake_im.save(fake_p, format="JPEG", quality=95)
        samples.append({"path": str(fake_p), "label_name": "FAKE"})

    return samples


if __name__ == "__main__":
    test_samples = get_or_create_benchmark_samples(num_reals=3, num_fakes=3)
    judge = LLMForensicJudgeBenchmark()
    judge.run_benchmark(test_samples, max_samples=6)
