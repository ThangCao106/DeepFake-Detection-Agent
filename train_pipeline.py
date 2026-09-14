"""
Two-Stage Fine-Tuning Pipeline for ViT Deepfake Detection Model.
Stage 1: Linear Classifier Warmup (Frozen ViT Backbone)
Stage 2: Full Backbone Fine-Tuning with Cosine Annealing Learning Rate Decay
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup


class DeepfakeDataset(Dataset):
    def __init__(self, manifest_file, processor):
        with open(manifest_file, "r", encoding="utf-8") as f:
            self.records = json.load(f)
        self.processor = processor
        self.label2id = {"REAL": 0, "FAKE": 1}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["filepath"]).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        label_id = self.label2id[rec["label"]]
        return pixel_values, torch.tensor(label_id, dtype=torch.long)


class TwoStageTrainer:
    def __init__(self, base_model_name="dima806/deepfake_vs_real_image_detection", output_dir="custom_vit_model"):
        self.base_model_name = base_model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[train_pipeline] Initializing Trainer on device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(base_model_name)
        self.model = AutoModelForImageClassification.from_pretrained(base_model_name).to(self.device)

    def stage1_classifier_warmup(self, train_loader, epochs=3, lr=1e-3):
        """Stage 1: Freeze ViT backbone, fine-tune linear classification head only."""
        print("\n" + "="*60)
        print("=== STAGE 1: Linear Classifier Warmup (Frozen Backbone) ===")
        print("="*60)

        # Freeze ViT backbone
        for name, param in self.model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0.0, 0, 0
            for pixel_values, labels in train_loader:
                pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total if total > 0 else 0.0
            print(f"Stage 1 - Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc*100:.2f}%")

    def stage2_full_finetune(self, train_loader, epochs=5, lr=1e-5):
        """Stage 2: Unfreeze full backbone, fine-tune with Cosine Annealing scheduler."""
        print("\n" + "="*60)
        print("=== STAGE 2: Full Backbone Fine-Tuning (All Layers Unfrozen) ===")
        print("="*60)


        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss, correct, total = 0.0, 0, 0
            for pixel_values, labels in train_loader:
                pixel_values, labels = pixel_values.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = correct / total if total > 0 else 0.0
            print(f"Stage 2 - Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc*100:.2f}%")

        # Save custom model and processor
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"[train_pipeline] Custom ViT Model saved successfully to {self.output_dir}")


def main():
    data_dir = "dataset_pipeline"
    split_dir = os.path.join(data_dir, "source_split")
    train_manifest = os.path.join(split_dir, "train_manifest.json")

    if not os.path.exists(train_manifest):
        print("[train_pipeline] Running DataPipeline first to build dataset...")
        from data_pipeline import DataPipeline
        dp = DataPipeline(data_dir)
        raw_dir = dp.build_synthetic_multi_source_dataset(samples_per_source=25)
        cleaned = dp.process_data_cleaning_and_deduplication(raw_dir)
        train_manifest, _ = dp.execute_source_based_split(cleaned)

    trainer = TwoStageTrainer()
    dataset = DeepfakeDataset(train_manifest, trainer.processor)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    trainer.stage1_classifier_warmup(train_loader, epochs=3, lr=1e-3)
    trainer.stage2_full_finetune(train_loader, epochs=5, lr=1e-5)


if __name__ == "__main__":
    main()
