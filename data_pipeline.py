"""
Data Pipeline Module for Custom Deepfake Detection Fine-Tuning.
Handles:
1. Multi-dataset Ingestion (CIFAKE, GenImage, Custom Deepfake)
2. Data Cleaning (corrupted images, low resolution, invalid formats)
3. Perceptual Hash Duplicate Removal (MD5 + Difference Hash)
4. Source-Based Out-of-Distribution Split (Train Generators vs Unseen Test Generators)
"""

import os
import sys
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


class DataPipeline:
    def __init__(self, data_root="dataset_pipeline"):
        self.data_root = data_root
        self.clean_root = os.path.join(data_root, "clean_data")
        self.split_root = os.path.join(data_root, "source_split")

    @staticmethod
    def compute_dhash(pil_image, hash_size=8):
        """Compute difference hash (dhash) for near-duplicate image detection."""
        try:
            resized = pil_image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
            pixels = np.array(resized)
            # Compare adjacent pixels
            diff = pixels[:, 1:] > pixels[:, :-1]
            return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        except Exception:
            return None

    @staticmethod
    def compute_md5(filepath):
        """Compute MD5 hash for exact file duplicate detection."""
        hasher = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def clean_image(self, filepath, min_dim=32):
        """Verify image readability, dimensions, and color mode."""
        try:
            with Image.open(filepath) as img:
                img.verify()
            with Image.open(filepath) as img:
                w, h = img.size
                if w < min_dim or h < min_dim:
                    return None, f"Image too small ({w}x{h})"
                # Convert to RGB
                rgb_img = img.convert("RGB")
                return rgb_img, "Valid"
        except Exception as e:
            return None, f"Corrupted file: {e}"

    def build_synthetic_multi_source_dataset(self, samples_per_source=20):
        """
        Synthesize multi-source datasets representing CIFAKE, GenImage, and Custom Deepfake
        for pipeline execution verification.
        """
        sources = {
            "CIFAKE": {"type": "gan_cifar", "noise": 10.0, "grid_freq": 12.0},
            "GenImage": {"type": "latent_diffusion", "noise": 25.0, "grid_freq": 18.0},
            "Deepfake_v1": {"type": "face_swap", "noise": 35.0, "grid_freq": 24.0},
            "New_Generators_Test": {"type": "dit_flow_matching", "noise": 45.0, "grid_freq": 32.0}
        }

        print(f"[data_pipeline] Generating multi-source datasets in {self.data_root}...")
        np.random.seed(42)

        raw_dir = os.path.join(self.data_root, "raw_sources")
        os.makedirs(raw_dir, exist_ok=True)

        dataset_manifest = {}

        for src_name, params in sources.items():
            src_dir_real = os.path.join(raw_dir, src_name, "real")
            src_dir_fake = os.path.join(raw_dir, src_name, "fake")
            os.makedirs(src_dir_real, exist_ok=True)
            os.makedirs(src_dir_fake, exist_ok=True)

            file_list = []

            for i in range(samples_per_source):
                np.random.seed(hash(f"{src_name}_{i}") % (2**32))
                w, h = 224, 224
                x = np.linspace(0, 1, w)
                y = np.linspace(0, 1, h)
                xx, yy = np.meshgrid(x, y)

                # Real image generation with unique phase offset
                phase_x = np.random.uniform(0.1, 5.0)
                phase_y = np.random.uniform(0.1, 5.0)
                r = (np.sin(xx * phase_x) * 0.5 + 0.5) * 200 + 40 + np.random.uniform(-10, 10, (h, w))
                g = (np.cos(yy * phase_y) * 0.5 + 0.5) * 180 + 50 + np.random.uniform(-10, 10, (h, w))
                b = (np.sin(xx + yy) * 0.5 + 0.5) * 210 + 30 + np.random.uniform(-10, 10, (h, w))
                real_np = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
                real_img = Image.fromarray(real_np).filter(ImageFilter.GaussianBlur(radius=0.5))
                
                real_path = os.path.join(src_dir_real, f"{src_name}_real_{i+1:03d}.jpg")
                real_img.save(real_path, quality=95)
                file_list.append({"path": real_path, "label": "REAL", "source": src_name})

                # Fake image generation (with periodic grid & noise artifacts)
                fake_np = np.stack([r, g, b], axis=-1).astype(np.float32)
                freq = params["grid_freq"] + np.random.uniform(-1.0, 1.0)
                grid = (np.sin(xx * freq * np.pi) * np.cos(yy * freq * np.pi)) * params["noise"]
                fake_np += np.stack([grid, grid, grid], axis=-1)

                cy_s, cy_e = int(h*0.25), int(h*0.75)
                cx_s, cx_e = int(w*0.25), int(w*0.75)
                high_noise = np.random.normal(0, params["noise"], (cy_e - cy_s, cx_e - cx_s, 3))
                fake_np[cy_s:cy_e, cx_s:cx_e] += high_noise

                fake_np = np.clip(fake_np, 0, 255).astype(np.uint8)
                fake_img = Image.fromarray(fake_np)

                fake_path = os.path.join(src_dir_fake, f"{src_name}_fake_{i+1:03d}.jpg")
                fake_img.save(fake_path, quality=95)
                file_list.append({"path": fake_path, "label": "FAKE", "source": src_name})

            dataset_manifest[src_name] = file_list

        print(f"[data_pipeline] Built raw multi-source dataset across {len(sources)} sources.")
        return raw_dir

    def process_data_cleaning_and_deduplication(self, raw_dir):
        """Run cleaning, exact MD5 deduplication, and dhash near-duplicate removal."""
        print("[data_pipeline] Starting Data Cleaning & Duplicate Removal...")
        
        seen_md5 = set()
        seen_dhash = set()

        cleaned_records = []
        stats = {"total_processed": 0, "corrupted": 0, "exact_duplicates": 0, "near_duplicates": 0, "cleaned": 0}

        for root, _, files in os.walk(raw_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    stats["total_processed"] += 1
                    filepath = os.path.join(root, file)

                    # 1. Clean & Verify Image
                    img, msg = self.clean_image(filepath)
                    if img is None:
                        stats["corrupted"] += 1
                        continue

                    # 2. Check MD5 Exact Duplicate
                    file_md5 = self.compute_md5(filepath)
                    if file_md5 in seen_md5:
                        stats["exact_duplicates"] += 1
                        continue
                    seen_md5.add(file_md5)

                    # 3. Check Perceptual Near Duplicate
                    img_dhash = self.compute_dhash(img)
                    if img_dhash in seen_dhash:
                        stats["near_duplicates"] += 1
                        continue
                    seen_dhash.add(img_dhash)

                    # Save cleaned image
                    rel_dir = os.path.relpath(root, raw_dir)
                    save_dir = os.path.join(self.clean_root, rel_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, file)
                    img.save(save_path, quality=95)

                    label = "FAKE" if "fake" in save_path.lower() else "REAL"
                    source_name = rel_dir.split(os.sep)[0] if os.sep in rel_dir else rel_dir

                    cleaned_records.append({
                        "filepath": save_path,
                        "label": label,
                        "source": source_name,
                        "md5": file_md5,
                        "dhash": str(img_dhash)
                    })
                    stats["cleaned"] += 1

        print(f"[data_pipeline] Cleaning Complete: {stats['cleaned']}/{stats['total_processed']} valid unique images retained.")
        print(f"                Duplicates Pruned: {stats['exact_duplicates']} exact MD5, {stats['near_duplicates']} near dhash.")
        return cleaned_records

    def execute_source_based_split(self, cleaned_records, train_sources=["CIFAKE", "GenImage", "Deepfake_v1"], test_sources=["New_Generators_Test"]):
        """
        Perform Source-Based Out-of-Distribution Split:
        - Train set: Built from train_sources (CIFAKE, GenImage, Deepfake_v1)
        - Test set: Built from unseen test_sources (New_Generators_Test)
        """
        print(f"[data_pipeline] Executing Source-Based Split...")
        print(f"                Train Sources: {train_sources}")
        print(f"                Test Sources (OOD New Generators): {test_sources}")

        train_records = [r for r in cleaned_records if r["source"] in train_sources]
        test_records = [r for r in cleaned_records if r["source"] in test_sources]

        os.makedirs(self.split_root, exist_ok=True)
        train_manifest = os.path.join(self.split_root, "train_manifest.json")
        test_manifest = os.path.join(self.split_root, "test_manifest.json")

        with open(train_manifest, "w", encoding="utf-8") as f:
            json.dump(train_records, f, indent=2)

        with open(test_manifest, "w", encoding="utf-8") as f:
            json.dump(test_records, f, indent=2)

        print(f"[data_pipeline] Source-Based Split Complete:")
        print(f"                Train samples: {len(train_records)} -> {train_manifest}")
        print(f"                Test samples:  {len(test_records)} -> {test_manifest}")
        return train_manifest, test_manifest


def main():
    pipeline = DataPipeline()
    raw_dir = pipeline.build_synthetic_multi_source_dataset(samples_per_source=25)
    cleaned = pipeline.process_data_cleaning_and_deduplication(raw_dir)
    train_manifest, test_manifest = pipeline.execute_source_based_split(cleaned)


if __name__ == "__main__":
    main()
