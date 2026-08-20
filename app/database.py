import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "price_predictions.db"


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                car_name TEXT NOT NULL,
                year INTEGER NOT NULL,
                kms_driven INTEGER NOT NULL,
                fuel_type TEXT NOT NULL,
                seller_type TEXT NOT NULL,
                transmission TEXT NOT NULL,
                owner INTEGER NOT NULL,
                estimated_price REAL NOT NULL,
                model_version TEXT NOT NULL
            )
            """
        )


def save_prediction(features: dict, result: dict) -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO price_predictions (
                car_name,
                year,
                kms_driven,
                fuel_type,
                seller_type,
                transmission,
                owner,
                estimated_price,
                model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                features["Car_Name"],
                features["Year"],
                features["Kms_Driven"],
                features["Fuel_Type"],
                features["Seller_Type"],
                features["Transmission"],
                features["Owner"],
                result["estimated_price"],
                result["model_version"],
            ),
        )


def list_recent_predictions(limit: int = 20) -> list[dict]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                id,
                created_at,
                car_name,
                year,
                kms_driven,
                fuel_type,
                seller_type,
                transmission,
                owner,
                estimated_price,
                model_version
            FROM price_predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
