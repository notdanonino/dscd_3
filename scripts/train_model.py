import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Selling_Price"

# Features base del dataset
CATEGORICAL_FEATURES = ["Car_Name", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]
NUMERIC_FEATURES = ["Year", "Present_Price", "Kms_Driven"]
FEATURE_ORDER = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_preprocessor(X, scale=False):
    """
    Preprocesador: imputa y hace One-Hot para categoricas.
    Si scale=True, agrego StandardScaler a las numericas.
    """
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), nums),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cats,
            ),
        ],
        remainder="drop",
    )


def add_features(df):
    """
    Experimento 2: Agregue variables nuevas:
    - Vehicle_Age: antiguedad del vehiculo (2020 - Year)
    - Log_Kms: log(km) para ver si captura mejor la relacion con el precio
    - Price_Ratio: Present_Price / Selling_Price como proxy de depreciacion
    """
    out = df.copy()
    out["Vehicle_Age"] = 2020 - out["Year"]
    out["Log_Kms"] = np.log1p(out["Kms_Driven"].clip(lower=0))
    out["Price_Ratio"] = out["Present_Price"] / out[TARGET].replace(0, np.nan)
    return out


def evaluate(name, estimator, X_train, X_test, y_train, y_test):
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)
    return {
        "experimento": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "R2": r2_score(y_test, pred),
        "pipeline": estimator,
    }


def train_and_save() -> None:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_path = project_root / "data" / "car-data.csv"
    df = pd.read_csv(data_path)
    df = df.drop_duplicates().copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET])

    # Filtre solo automoviles (para evitar motos que tienen otro rango de precios)
    is_car = ~df["Car_Name"].astype(str).str.match(
        r"^(Royal|UM |KTM|Bajaj|Hyosung|Mahindra|Honda|Yamaha|TVS|Hero|Activa|Suzuki)",
        case=False,
        na=False,
    )
    df = df[is_car].copy()

    X = df[FEATURE_ORDER]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # =========================================================================
    # BASELINE: Regresion Lineal con One-Hot (sin escalamiento)
    # Este es mi punto de partida: lo mas simple posible.
    # =========================================================================
    results.append(
        evaluate(
            "Baseline - LinearRegression",
            Pipeline(
                [("prep", make_preprocessor(X_train)), ("model", LinearRegression())]
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # =========================================================================
    # EXPERIMENTO 1: Ridge + escalamiento
    # Que cambie: Agregue StandardScaler y use Ridge en lugar de LinearRegression.
    # Por que: Lei que Ridge ayuda cuando hay muchas variables one-hot y posible
    #          multicolinealidad, asi que quise probar si mejoraba las metricas.
    # =========================================================================
    results.append(
        evaluate(
            "E1 - Ridge + scaling",
            Pipeline(
                [("prep", make_preprocessor(X_train, scale=True)), ("model", Ridge(alpha=10.0))]
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # =========================================================================
    # EXPERIMENTO 2: Feature engineering + Ridge
    # Que cambie: Agregue 3 variables nuevas:
    #   - Vehicle_Age (antiguedad)
    #   - Log_Kms (log de kilometros)
    #   - Price_Ratio (Present_Price / Selling_Price)
    # Por que: pense que la antiguedad y los kilometros en log podrian ayudar
    #          a capturar relaciones no lineales con el precio.
    # =========================================================================
    df2 = add_features(df)
    feat2 = [
        "Car_Name",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
        "Vehicle_Age",
        "Log_Kms",
        "Price_Ratio",
    ]
    X2 = df2[feat2]
    y2 = df2[TARGET]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=42
    )
    results.append(
        evaluate(
            "E2 - Feature engineering + Ridge",
            Pipeline(
                [("prep", make_preprocessor(X2_train, scale=True)), ("model", Ridge(alpha=10.0))]
            ),
            X2_train,
            X2_test,
            y2_train,
            y2_test,
        )
    )

    # =========================================================================
    # EXPERIMENTO 3: Random Forest
    # Que cambie: Cambie el algoritmo a RandomForestRegressor.
    # Por que: lei que Random Forest captura relaciones no lineales sin necesidad
    #          de transformar tanto las variables, asi que quise compararlo.
    # =========================================================================
    results.append(
        evaluate(
            "E3 - RandomForest",
            Pipeline(
                [
                    ("prep", make_preprocessor(X_train)),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=200,
                            min_samples_leaf=2,
                            max_features=0.8,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # =========================================================================
    # EXPERIMENTO 4: Gradient Boosting
    # Que cambie: Use GradientBoostingRegressor con learning_rate bajo y max_depth=3.
    # Por que: lei que Gradient Boosting suele funcionar bien en datos tabulares
    #          y quise ver si mejoraba respecto a Random Forest.
    # =========================================================================
    results.append(
        evaluate(
            "E4 - GradientBoosting",
            Pipeline(
                [
                    ("prep", make_preprocessor(X_train)),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    # =========================================================================
    # EXPERIMENTO 5: Gradient Boosting sin outliers en train
    # Que cambie: Elimine valores extremos (percentiles 1 y 99) de las variables
    #             numericas SOLO en el conjunto de entrenamiento.
    # Por que: pense que los outliers podrian estar afectando el aprendizaje,
    #          pero quise dejar el test intacto para evaluar en condiciones reales.
    # =========================================================================
    numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    mask = pd.Series(True, index=X_train.index)
    for c in numeric_cols:
        q1, q3 = X_train[c].quantile([0.01, 0.99])
        mask &= X_train[c].between(q1, q3) | X_train[c].isna()
    X5_train, y5_train = X_train.loc[mask], y_train.loc[mask]
    results.append(
        evaluate(
            "E5 - GradientBoosting sin outliers train",
            Pipeline(
                [
                    ("prep", make_preprocessor(X5_train)),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=200,
                            learning_rate=0.05,
                            max_depth=3,
                            random_state=42,
                        ),
                    ),
                ]
            ),
            X5_train,
            X_test,
            y5_train,
            y_test,
        )
    )

    # =========================================================================
    # TABLA DE RESULTADOS
    # =========================================================================
    table = pd.DataFrame(
        [
            {k: r[k] for k in ["experimento", "MAE", "RMSE", "R2"]}
            for r in results
        ]
    )
    baseline = table.iloc[0]
    table["mejora_MAE_%"] = (baseline["MAE"] - table["MAE"]) / baseline["MAE"] * 100
    table["mejora_RMSE_%"] = (baseline["RMSE"] - table["RMSE"]) / baseline["RMSE"] * 100

    table.to_csv(models_dir / "tabla_experimentos.csv", index=False)

    # Mejor modelo (menor RMSE)
    best_idx = table["RMSE"].idxmin()
    best = results[best_idx]

    joblib.dump(best["pipeline"], models_dir / "vehicle_price_best.joblib")

    metrics = {
        "best_experiment": best["experimento"],
        "MAE": round(float(table.loc[best_idx, "MAE"]), 4),
        "RMSE": round(float(table.loc[best_idx, "RMSE"]), 4),
        "R2": round(float(table.loc[best_idx, "R2"]), 4),
        "mejora_MAE_%": round(float(table.loc[best_idx, "mejora_MAE_%"]), 2),
        "mejora_RMSE_%": round(float(table.loc[best_idx, "mejora_RMSE_%"]), 2),
        "features": FEATURE_ORDER,
        "target": TARGET,
        "model_version": "v1",
        "tabla_experimentos": table.to_dict(orient="records"),
    }

    with open(models_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Modelo y metricas guardados en models/")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    train_and_save()