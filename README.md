# Gigit — Dynamic Cash-Flow Credit Scoring for Gig Workers

> **AMD Slingshot 2026 Hackathon Entry**

Gigit is a B2B credit-scoring engine that evaluates gig workers using real-time behavioural cash-flow metrics instead of traditional credit history. It runs a lightweight neural network **entirely on-device** via ONNX Runtime, with a clear upgrade path to AMD Ryzen AI NPU acceleration for sub-millisecond, privacy-preserving inference.

---

## Key Features

| Feature | Description |
|---|---|
| **Cash-Flow Scoring** | Risk model built on income stability, spending discipline, tenure, and earning power — not CIBIL scores |
| **Local Inference** | All predictions run on-device via ONNX Runtime. No data leaves the machine |
| **NPU-Ready** | ONNX export is compatible with AMD Vitis AI / Ryzen AI SDK for INT8 quantised NPU deployment |
| **Underwriter Dashboard** | Interactive Streamlit UI with gauge dials, bar charts, and radar profiles |
| **Synthetic Data Pipeline** | Generates realistic, correlated mock data with a single command |

---

## Architecture

```
CrediGig_MVP/
├── data/
│   ├── generate_mock_data.py        # Synthetic gig worker data generator
│   └── mock_gig_worker_data.csv     # Generated dataset (500 rows)
├── ml_engine/
│   ├── train_model.py               # PyTorch FFNN training script
│   ├── amd_npu_optimizer.py         # ONNX export & validation
│   ├── risk_model.pt                # Trained PyTorch weights
│   ├── risk_model.onnx              # Exported ONNX model
│   └── scaler_params.json           # StandardScaler mean/scale for inference
├── frontend/
│   └── underwriter_dashboard.py     # Streamlit underwriter UI
├── utils/
│   └── feature_engineering.py       # Feature extraction & worker lookup
├── run.py                           # Quick launcher (starts the dashboard)
└── requirements.txt
```

---

## Tech Stack

- **Model** — PyTorch (training) → ONNX (inference)
- **Runtime** — ONNX Runtime (`CPUExecutionProvider`; swap to `VitisAIExecutionProvider` for AMD NPU)
- **Dashboard** — Streamlit + Plotly
- **Data** — pandas, NumPy, scikit-learn

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate mock data

```bash
python data/generate_mock_data.py
```

Creates `data/mock_gig_worker_data.csv` with 500 synthetic gig worker profiles. Each run uses a random seed for varied datasets.

### 3. Train the model

```bash
python ml_engine/train_model.py
```

Trains a 3-layer Feed-Forward Neural Network (4 → 32 → 16 → 1) for 150 epochs with mini-batch gradient descent. Outputs:

- `ml_engine/risk_model.pt` — trained weights
- `ml_engine/scaler_params.json` — feature scaler parameters

### 4. Export to ONNX

```bash
python ml_engine/amd_npu_optimizer.py
```

Exports the trained model to `ml_engine/risk_model.onnx` with dynamic batch support (opset 17). This is the AMD NPU-compatible format.

### 5. Launch the dashboard

```bash
python run.py
```

Opens the Streamlit underwriter dashboard at `http://localhost:8501`.

---

## Model Details

### Input Features

| Feature | Range | Risk Signal |
|---|---|---|
| `average_monthly_income` | ₹8,000 – ₹80,000 | Higher income → lower risk |
| `income_volatility_percentage` | 5% – 60% | Higher volatility → higher risk |
| `expense_to_income_ratio` | 0.30 – 1.10 | Higher ratio → higher risk |
| `work_tenure_months` | 1 – 120 | Longer tenure → lower risk |

### Output

A single risk score from **0** (safe) to **100** (risky), bucketed into:

- **Low Risk** (0–29) — Approve with standard terms
- **Medium Risk** (30–59) — Approve with enhanced monitoring
- **High Risk** (60–100) — Decline or require collateral

### Architecture

```
Input(4) → Linear(32) → ReLU → Dropout(0.3)
         → Linear(16) → ReLU → Dropout(0.2)
         → Linear(1)  → Sigmoid
```

---

## AMD Ryzen AI Integration Path

This MVP uses the CPU Execution Provider. To deploy on AMD Ryzen AI hardware:

1. **Quantise** the ONNX model using the AMD Vitis AI / Ryzen AI SDK (INT8)
2. **Swap** the execution provider in the dashboard:

```python
session = ort.InferenceSession(
    "risk_model_quantized.onnx",
    providers=["VitisAIExecutionProvider"],  # Routes to on-chip NPU
)
```

3. Achieve **sub-millisecond inference** with zero cloud dependency

---

## License

Built for the AMD Slingshot 2026 Hackathon.
