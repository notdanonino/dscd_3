# DSCD-3 - Mejora de Modelo

Carlos Daniel Trujillo de Anda A00840360

Proyecto de mejora de un modelo de prediccion de precios de vehiculos usados, usando el dataset CarDekho de Kaggle.

## Objetivo

Partir de un baseline simple (Regresion Lineal) y hacer 5 experimentos para mejorar el desempeno del modelo.

## Dataset

- Archivo: `data/car-data.csv`
- Columnas: `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`
- Target: `Selling_Price` (en lakhs de rupias)
- Filtro: solo automoviles (se excluyeron motos)

## Estructura del proyecto

```
dscd_3/
├── data/
│   └── car-data.csv        # Dataset CarDekho
├── models/
│   ├── vehicle_price_best.joblib   # Mejor pipeline
│   ├── metrics.json                # Metricas de experimentos
│   └── tabla_experimentos.csv      # Tabla comparativa
├── scripts/
│   ├── train_model.py      # Entrenamiento + 5 experimentos
│   └── predict_3_vehiculos.py  # Prediccion de 3 vehiculos
├── requirements.txt
└── README.md
```

## Como ejecutar

```bash
# 1) Instalar dependencias
pip install -r requirements.txt

# 2) Entrenar modelos
python scripts/train_model.py

# 3) Predecir 3 vehiculos de ejemplo
python scripts/predict_3_vehiculos.py
```

## Experimentos realizados

| Experimento | MAE | RMSE | R2 | Mejora MAE | Mejora RMSE |
|---|---:|---:|---:|---:|---:|
| Baseline - LinearRegression | 1.2539 | 2.8403 | 0.7609 | 0% | 0% |
| **E1 - Ridge + scaling** | **1.2061** | **1.9607** | **0.8861** | **3.81%** | **30.97%** |
| E2 - Feature engineering + Ridge | 2.3095 | 5.1624 | 0.2103 | -84.18% | -81.75% |
| E3 - RandomForest | 1.1781 | 3.3307 | 0.6713 | 6.04% | -17.26% |
| E4 - GradientBoosting | 1.1410 | 2.8304 | 0.7626 | 9.01% | 0.35% |
| E5 - GradientBoosting sin outliers | 1.2850 | 3.6275 | 0.6101 | -2.48% | -27.71% |

### Descripcion de experimentos

1. **Baseline:** Regresion Lineal con One-Hot Encoding (sin escalamiento).
2. **E1:** Ridge + StandardScaler (regularizacion para controlar coeficientes).
3. **E2:** Feature engineering (Vehicle_Age, Log_Kms, Price_Ratio) + Ridge.
4. **E3:** Random Forest (cambio de algoritmo).
5. **E4:** Gradient Boosting (algoritmo mas potente).
6. **E5:** Gradient Boosting sin outliers en train (manejo de outliers).

## Mejor modelo

**E1 - Ridge + scaling** fue el mejor segun RMSE:

- MAE: 1.2061 lakhs INR
- RMSE: 1.9607 lakhs INR
- R2: 0.8861
- Mejora: 3.81% en MAE, 30.97% en RMSE

## Conclusiones

- **E1 (Ridge + scaling)** fue el mejor porque la regularizacion ayudo a controlar los coeficientes de las variables one-hot, y el escalamiento hizo el modelo mas estable (30.97% de mejora en RMSE).
- **E2 (Feature engineering)** empeoro porque `Price_Ratio` usa el target, causando data leakage.
- **E4 (Gradient Boosting)** tuvo mejor MAE pero no mejor RMSE, sugiriendo errores mas consistentes.
- **E5 (sin outliers)** empeoro porque se perdio informacion util al sacar datos del entrenamiento.

## Prediccion para 3 vehiculos

Con el mejor modelo (E1 - Ridge + scaling):

| Vehiculo | Año | Km | Precio Estimado |
|---|---:|---:|---:|
| swift | 2015 | 40,000 | 1.34 lakhs INR |
| creta | 2017 | 25,000 | 4.67 lakhs INR |
| city | 2016 | 35,000 | 2.24 lakhs INR |

## Entregables

- Script ejecutable: `scripts/train_model.py`
- Tabla de experimentos: `models/tabla_experimentos.csv`
- Mejor pipeline: `models/vehicle_price_best.joblib`
- Metricas: `models/metrics.json`
- Este README