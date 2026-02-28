"""
Gigit — Model Training Script
====================================
Trains a lightweight Feed-Forward Neural Network (FFNN) on gig worker
behavioral metrics to predict default risk (0 → safe, 1 → risky).

The trained weights are saved as ``risk_model.pt`` and the feature scaler
parameters as ``scaler_params.json`` so inference can normalise new inputs
without re-fitting.

Usage:
    python ml_engine/train_model.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "mock_gig_worker_data.csv")
MODEL_PATH = os.path.join(SCRIPT_DIR, "risk_model.pt")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler_params.json")

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
INPUT_DIM = 4          # 4 behavioural features
HIDDEN_1 = 32
HIDDEN_2 = 16
OUTPUT_DIM = 1         # single sigmoid output (probability of default)
LEARNING_RATE = 1e-3
EPOCHS = 150
BATCH_SIZE = 64
TEST_SIZE = 0.20
SEED = 42

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class CreditRiskModel(nn.Module):
    """
    A compact 3-layer Feed-Forward Neural Network for binary credit-risk
    classification.

    Architecture:
        Input(4) → Linear(32) → ReLU → Dropout(0.3)
                 → Linear(16) → ReLU → Dropout(0.2)
                 → Linear(1)  → Sigmoid
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN_2, OUTPUT_DIM),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load data -----------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print("[ERROR] Data file not found. Run: python data/generate_mock_data.py")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "average_monthly_income",
        "income_volatility_percentage",
        "expense_to_income_ratio",
        "work_tenure_months",
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["historical_default_risk"].values.astype(np.float32).reshape(-1, 1)

    # 2. Train / test split --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # 3. Feature scaling (StandardScaler) ------------------------------------
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler params so the dashboard can re-use them at inference time
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(SCALER_PATH, "w") as f:
        json.dump(scaler_params, f, indent=2)
    print(f"[OK] Scaler parameters saved -> {SCALER_PATH}")

    # 4. Convert to tensors --------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # 5. Initialise model, loss, optimiser -----------------------------------
    model = CreditRiskModel()
    criterion = nn.BCELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training loop -------------------------------------------------------
    print(f"\n[*] Training CreditRiskModel for {EPOCHS} epochs...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # Mini-batch training
        perm = torch.randperm(X_train_t.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, X_train_t.size(0), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            xb, yb = X_train_t[idx], y_train_t[idx]

            preds = model(xb)
            loss = criterion(preds, yb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Log every 25 epochs
        if epoch % 25 == 0 or epoch == 1:
            avg_loss = epoch_loss / n_batches
            # Quick eval accuracy
            model.eval()
            with torch.no_grad():
                test_preds = (model(X_test_t) >= 0.5).float()
                acc = (test_preds == y_test_t).float().mean().item()
            print(f"  Epoch {epoch:>4d} │ loss {avg_loss:.4f} │ test acc {acc:.2%}")

    # 7. Final evaluation ----------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_preds = (model(X_test_t) >= 0.5).float()
        final_acc = (test_preds == y_test_t).float().mean().item()
    print(f"\n[OK] Final test accuracy: {final_acc:.2%}")

    # 8. Save model weights --------------------------------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[OK] Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
