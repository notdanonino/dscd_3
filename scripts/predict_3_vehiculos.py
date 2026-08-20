import joblib
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parents[1]
model = joblib.load(root / "models" / "vehicle_price_best.joblib")

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

vehiculos = [
    {
        "Car_Name": "swift",
        "Year": 2015,
        "Kms_Driven": 40000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "Present_Price": 0.0,
    },
    {
        "Car_Name": "creta",
        "Year": 2017,
        "Kms_Driven": 25000,
        "Fuel_Type": "Diesel",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "Present_Price": 0.0,
    },
    {
        "Car_Name": "city",
        "Year": 2016,
        "Kms_Driven": 35000,
        "Fuel_Type": "Petrol",
        "Seller_Type": "Dealer",
        "Transmission": "Manual",
        "Owner": 0,
        "Present_Price": 0.0,
    },
]

df = pd.DataFrame(vehiculos, columns=FEATURE_ORDER)
pred = model.predict(df)

print("Predicciones para 3 vehiculos:")
print("-" * 60)
for v, p in zip(vehiculos, pred):
    print(f"{v['Car_Name']} {v['Year']} {v['Kms_Driven']} km -> {p:.4f} lakhs INR")