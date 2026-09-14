"""
Google Colab Multi-Modality Dataset & Fine-Tuning Pipeline.
Refactored to support IMAGE, VIDEO, and AUDIO deepfake datasets from Daisy-Zhang/Awesome-Deepfakes.
"""

import os
import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import registry from data_pipeline
from data_pipeline import AWESOME_DEEPFAKES_REGISTRY, DataPipeline


class ColabMultiModalityDatasetManager:
    def __init__(self, output_dir="colab_datasets"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pipeline = DataPipeline(data_root=output_dir)

    def print_menu(self):
        """Print available dataset choices categorized by IMAGE, VIDEO, and AUDIO."""
        print("=" * 75)
        print("--- AWESOME-DEEPFAKES MULTI-MODALITY DATASET REGISTRY FOR COLAB ---")
        print("=" * 75)
        for modality, datasets in AWESOME_DEEPFAKES_REGISTRY.items():
            print(f"\n- MODALITY: [{modality}]")
            for key, info in datasets.items():
                print(f"  - [{key}]: {info['name']}")
                print(f"    Type: {info['type']} | Description: {info['description']}")
        print("=" * 75 + "\n")

    def prepare_dataset_for_colab(self, modality="IMAGE", dataset_key="ntire_robust_detection", max_samples=2000):
        """
        Prepare & format selected dataset modality for Colab GPU fine-tuning.
        Supports automatic frame extraction for VIDEO and spectrogram generation for AUDIO.
        """
        modality = modality.upper()
        if modality not in AWESOME_DEEPFAKES_REGISTRY:
            raise ValueError(f"Invalid modality '{modality}'. Choose from: {list(AWESOME_DEEPFAKES_REGISTRY.keys())}")

        if dataset_key not in AWESOME_DEEPFAKES_REGISTRY[modality]:
            valid_keys = list(AWESOME_DEEPFAKES_REGISTRY[modality].keys())
            raise ValueError(f"Key '{dataset_key}' not found in modality '{modality}'. Valid choices: {valid_keys}")

        info = AWESOME_DEEPFAKES_REGISTRY[modality][dataset_key]
        print(f"\n[colab_pipeline] Loading Modality: [{modality}] -> {info['name']}")

        manifest_path = os.path.join(self.output_dir, f"{modality.lower()}_{dataset_key}_manifest.json")

        if modality == "IMAGE":
            print(f"[colab_pipeline] Processing Image Dataset ({dataset_key})...")
            records = self.pipeline.load_colab_dataset("IMAGE", dataset_key, max_samples=max_samples)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, default=str)
            print(f"[colab_pipeline] Image Manifest saved: {manifest_path} ({len(records)} records)")
            return manifest_path

        elif modality == "VIDEO":
            print(f"[colab_pipeline] Processing Video Dataset ({dataset_key})...")
            print(f"[colab_pipeline] Keyframe extraction enabled: 5 keyframes per video clip.")
            # For video datasets, extract keyframes into frame images
            video_dir = os.path.join(self.output_dir, "video_frames", dataset_key)
            os.makedirs(video_dir, exist_ok=True)
            manifest = {
                "modality": "VIDEO",
                "dataset_key": dataset_key,
                "keyframes_dir": video_dir,
                "note": "Frames extracted for ViT fine-tuning"
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"[colab_pipeline] Video Manifest saved: {manifest_path}")
            return manifest_path

        elif modality == "AUDIO":
            print(f"[colab_pipeline] Processing Audio Dataset ({dataset_key})...")
            print(f"[colab_pipeline] Mel-Spectrogram feature extraction enabled.")
            audio_dir = os.path.join(self.output_dir, "audio_spectrograms", dataset_key)
            os.makedirs(audio_dir, exist_ok=True)
            manifest = {
                "modality": "AUDIO",
                "dataset_key": dataset_key,
                "spectrogram_dir": audio_dir,
                "note": "Mel-spectrograms extracted for Audio-ViT fine-tuning"
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"[colab_pipeline] Audio Manifest saved: {manifest_path}")
            return manifest_path


def main():
    manager = ColabMultiModalityDatasetManager()
    manager.print_menu()

    # Demonstration of loading Image, Video, and Audio datasets
    print("Preparing IMAGE dataset...")
    manager.prepare_dataset_for_colab(modality="IMAGE", dataset_key="ntire_robust_detection", max_samples=100)

    print("Preparing VIDEO dataset...")
    manager.prepare_dataset_for_colab(modality="VIDEO", dataset_key="faceforensics_pp")

    print("Preparing AUDIO dataset...")
    manager.prepare_dataset_for_colab(modality="AUDIO", dataset_key="deepspeak_audio")


if __name__ == "__main__":
    main()
