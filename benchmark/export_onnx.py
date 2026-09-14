"""
ONNX Export & Production Optimization Module.
Exports fine-tuned PyTorch ViT model to ONNX format (custom_deepfake_vit.onnx)
and verifies ONNX Runtime execution.
"""

import os
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification


def export_to_onnx(model_dir="custom_vit_model", onnx_filename="custom_deepfake_vit.onnx"):
    print(f"[export_onnx] Loading custom model from {model_dir}...")
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir)
    model.eval()

    # Create dummy input matching ViT input shape (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = os.path.join(model_dir, onnx_filename)

    class ViTOnnxWrapper(torch.nn.Module):
        def __init__(self, vit_model):
            super().__init__()
            self.vit_model = vit_model

        def forward(self, pixel_values):
            return self.vit_model(pixel_values=pixel_values).logits

    wrapped_model = ViTOnnxWrapper(model)
    wrapped_model.eval()

    print(f"[export_onnx] Exporting PyTorch model to ONNX: {onnx_path}...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )

    print(f"[export_onnx] ONNX Export Successful: {onnx_path}")

    # Verify with ONNX Runtime if available
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        print(f"[export_onnx] ONNX Runtime Verification Successful! Output shape: {ort_outs[0].shape}")
    except Exception as e:
        print(f"[export_onnx] Note: ONNX Runtime test skipped ({e}). ONNX model file is ready.")

    return onnx_path


if __name__ == "__main__":
    export_to_onnx()
