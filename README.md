# Precio de Vehiculos — Experimentos de ML + API + Frontend

Proyecto para la tarea de mejora de un modelo de prediccion de precios de vehiculos usados (dataset CarDekho).

## Objetivo

- Partir de un baseline (Regresion Lineal).
- Realizar al menos 5 experimentos modificando:
  - tratamiento de valores faltantes,
  - manejo de outliers,
  - creacion de nuevas variables (antiguedad, log-km, etc.),
  - seleccion de features,
  - transformacion o escalamiento,
  - cambio de algoritmo,
  - ajuste de hiperparametros.
- Seleccionar el mejor modelo segun MAE, RMSE y R2.
- Exponer el mejor modelo mediante una API FastAPI y un frontend simple.

## Dataset

- Archivo: `data/car-data.csv`
- Columnas principales:
  - `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`,
    `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`
- Target: `Selling_Price` (en lakhs de rupias).

## Estructura del repo

- `scripts/train_model.py`: entrena baseline + experimentos, guarda el mejor modelo y metricas.
- `app/`: servicio de inferencia (FastAPI).
- `app/static/`: frontend HTML+JS que consume la API.
- `models/`: modelo entrenado (`.joblib`) y `metrics.json`.

## Como ejecutarlo

```bash
# 1) Clonar y entrar al repo
git clone <tu-repo>
cd precio_vehiculos_experiments

# 2) Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Entrenar modelos y guardar el mejor
python scripts/train_model.py

# 5) Correr frontend + backend
bash scripts/run_site.sh
```

- Frontend: http://127.0.0.1:9010
- Backend: http://127.0.0.1:9011 (docs en `/docs`)

## Entregables generados

Despues de ejecutar `train_model.py` tendras en `models/`:

- `vehicle_price_best.joblib`: mejor pipeline.
- `metrics.json`: MAE, RMSE, R2 del mejor modelo y descripcion de experimentos.

El frontend permite capturar:

- `Car_Name`, `Year`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`

y devuelve el precio estimado en lakhs de rupias.
