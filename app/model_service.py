from pathlib import Path

import joblib
import pandas as pd

MODEL_VERSION = "v1"
FEATURE_ORDER = [
    "Car_Name",
    "Fuel_Type",
    "Seller_Type",
    "Transmission",
    "Owner",
    "Year",
    "Present_Price",
    "Kms_Driven",
]

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "vehicle_price_best.joblib"
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. Run scripts/train_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(payload: dict) -> dict:
    model = get_model()
    row = {k: payload[k] for k in FEATURE_ORDER if k in payload}
    # Present_Price no lo pide el usuario; se usa un valor fijo o 0.
    if "Present_Price" not in row:
        row["Present_Price"] = 0.0
    sample = pd.DataFrame([row], columns=FEATURE_ORDER)
    estimated_price = float(model.predict(sample)[0])
    return {
        "estimated_price": round(estimated_price, 4),
        "currency": "Lakhs INR",
        "model_version": MODEL_VERSION,
    }
