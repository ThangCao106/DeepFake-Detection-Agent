"""
Evaluation Benchmark Script.
Evaluates the fine-tuned custom ViT model with calibrated temperature scaling
and optimal decision threshold on out-of-distribution test set.
"""

import os
import json
import torch
from PIL import Image
from deepfake_detector import DeepfakeDetector


def run_benchmark():
    print("[eval_benchmark] Initializing DeepfakeDetector with custom fine-tuned model...")
    detector = DeepfakeDetector()
    
    test_manifest_path = os.path.join("dataset_pipeline", "source_split", "test_manifest.json")
    if not os.path.exists(test_manifest_path):
        print(f"[eval_benchmark] Test manifest not found at {test_manifest_path}")
        return

    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    print(f"[eval_benchmark] Running benchmark on {len(test_samples)} test samples...")
    
    correct = 0
    total = len(test_samples)
    predictions = []

    for item in test_samples:
        filepath = item.get("filepath") or item.get("path")
        ground_truth_label = item["label"]
        ground_truth = 1 if (ground_truth_label == "FAKE" or ground_truth_label == 1) else 0
        
        if not os.path.exists(filepath):
            print(f"[eval_benchmark] Warning: File not found {filepath}, skipping.")
            total -= 1
            continue

        result = detector.predict(filepath)
        if "error" in result:
            print(f"[eval_benchmark] Error predicting {filepath}: {result['error']}")
            total -= 1
            continue

        label_str = result["label"].upper()
        is_fake_pred = "FAKE" in label_str or "DEEPFAKE" in label_str
        predicted_label = 1 if is_fake_pred else 0

        if predicted_label == ground_truth:
            correct += 1

        cert = result.get("forensic_certificate", {})
        evidence = cert.get("multi_dimensional_evidence", {})

        predictions.append({
            "path": filepath,
            "ground_truth": "FAKE" if ground_truth == 1 else "REAL",
            "predicted": result["label"],
            "confidence": result["confidence"],
            "all_scores": result.get("all_scores", {}),
            "spectral_anomaly": evidence.get("spectral_anomaly_score"),
            "match": predicted_label == ground_truth
        })

    accuracy = (correct / total * 100) if total > 0 else 0.0
    print(f"[eval_benchmark] Benchmark Complete!")
    print(f"Total Test Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

    output_path = os.path.join("custom_vit_model", "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "accuracy_percent": round(accuracy, 2),
            "predictions": predictions
        }, f, indent=2)

    print(f"[eval_benchmark] Saved benchmark results to {output_path}")


if __name__ == "__main__":
    run_benchmark()
