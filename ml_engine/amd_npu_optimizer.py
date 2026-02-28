"""
Gigit — AMD Ryzen AI NPU Optimizer (ONNX Export)
======================================================
This script loads the trained PyTorch model (``risk_model.pt``) and exports
it to the ONNX format (``risk_model.onnx``).

WHY ONNX?
---------
ONNX (Open Neural Network Exchange) is the standard intermediate
representation that AMD Ryzen AI Software uses to deploy models on the
Neural Processing Unit (NPU) built into AMD Ryzen AI processors.

AMD NPU INTEGRATION PATH (Production)
--------------------------------------
1. Export the model to ONNX using this script.         ← WE ARE HERE
2. Use the AMD Vitis AI / Ryzen AI SDK to quantise the ONNX model
   (e.g. INT8 quantisation) for optimal NPU throughput.
3. Load the quantised model via the ``onnxruntime`` VitisAI Execution
   Provider, which routes computation to the on-chip NPU:

       import onnxruntime as ort
       session = ort.InferenceSession(
           "risk_model_quantized.onnx",
           providers=["VitisAIExecutionProvider"],   # ← AMD NPU
       )

   This enables sub-millisecond inference on the edge with zero cloud
   dependency — ideal for secure, privacy-preserving credit scoring.

For this MVP prototype we run inference via the default CPU Execution
Provider.  The ONNX format ensures a seamless upgrade path to the NPU
when deploying on AMD Ryzen AI hardware.

Usage:
    python ml_engine/amd_npu_optimizer.py
"""

import os
import sys

import torch
import onnx

# We import the model class from our training script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_model import CreditRiskModel, INPUT_DIM  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PT_PATH = os.path.join(SCRIPT_DIR, "risk_model.pt")
MODEL_ONNX_PATH = os.path.join(SCRIPT_DIR, "risk_model.onnx")


def export_to_onnx() -> None:
    """Load the .pt weights and export to ONNX with dynamic batch size."""

    # 1. Verify the .pt file exists ------------------------------------------
    if not os.path.exists(MODEL_PT_PATH):
        print("[ERROR] risk_model.pt not found. Run: python ml_engine/train_model.py")
        sys.exit(1)

    # 2. Instantiate & load weights ------------------------------------------
    model = CreditRiskModel()
    model.load_state_dict(torch.load(MODEL_PT_PATH, map_location="cpu", weights_only=True))
    model.eval()
    print("[OK] Loaded risk_model.pt")

    # 3. Create a dummy input matching the expected shape --------------------
    #    Shape: (batch_size, 4)  — 4 behavioural features
    dummy_input = torch.randn(1, INPUT_DIM)

    # 4. Export to ONNX ------------------------------------------------------
    #    • opset_version 17 is broadly supported by ONNX Runtime and
    #      the AMD Vitis AI quantiser.
    #    • dynamic_axes lets us vary the batch dimension at inference time.
    torch.onnx.export(
        model,
        dummy_input,
        MODEL_ONNX_PATH,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features"],
        output_names=["risk_score"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "risk_score": {0: "batch_size"},
        },
    )
    print(f"[OK] Exported ONNX model -> {MODEL_ONNX_PATH}")

    # 5. Validate the exported model -----------------------------------------
    onnx_model = onnx.load(MODEL_ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model validation passed")

    # 6. Print model summary -------------------------------------------------
    print(f"\n--- Model Summary ---")
    print(f"    Input : features  - shape (batch, {INPUT_DIM})")
    print(f"    Output: risk_score - shape (batch, 1)  [0.0 = safe, 1.0 = risky]")
    print(f"    File  : {os.path.getsize(MODEL_ONNX_PATH) / 1024:.1f} KB")
    print(
        "\n[NEXT] Use AMD Ryzen AI SDK to quantise this ONNX model "
        "for NPU deployment."
    )


if __name__ == "__main__":
    export_to_onnx()
