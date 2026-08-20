# Precio de Vehiculos — Experimentos de ML + API + Frontend

En este proyecto trabaje en la mejora de un modelo de prediccion de precios de vehiculos usados, usando el dataset CarDekho. El objetivo fue partir de un baseline simple y hacer varios experimentos para ver como podia mejorar el desempeno.

## Objetivo

Mi objetivo fue:

- Empezar con un baseline sencillo (Regresion Lineal).
- Hacer 5 experimentos cambiando cosas concretas:
  - escalamiento y regularizacion,
  - creacion de nuevas variables (antiguedad, log-km, etc.),
  - cambio de algoritmo (Random Forest, Gradient Boosting),
  - manejo de outliers en el entrenamiento.
- Comparar los resultados con MAE, RMSE y R2.
- Elegir el mejor modelo y exponerlo con una API FastAPI y un frontend simple.

## Dataset

Use el archivo `data/car-data.csv`, que tiene las siguientes columnas principales:

- `Car_Name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`,
  `Fuel_Type`, `Seller_Type`, `Transmission`, `Owner`
- Target: `Selling_Price` (en lakhs de rupias).

Ademas, filtre solo los registros que corresponden a automoviles, para evitar que las motos metieran ruido en el modelo (tienen otro rango de precios).

## Estructura del repo

- `scripts/train_model.py`: aqui entreno el baseline + 5 experimentos y guardo el mejor modelo y las metricas.
- `app/`: servicio de inferencia con FastAPI.
- `app/static/`: frontend HTML+JS que consume la API.
- `models/`: modelo entrenado (`vehicle_price_best.joblib`), `metrics.json` y `tabla_experimentos.csv`.

## Como lo ejecute

Para correrlo localmente, hice lo siguiente:

```bash
# 1) Entrar a la carpeta del proyecto
cd dscd_3

# 2) Crear entorno virtual
python -m venv .venv
# Windows:
.venv\Scripts\activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Entrenar modelos y guardar el mejor
python scripts/train_model.py

# 5) Correr frontend + backend
# Backend (terminal 1)
uvicorn app.main:app --host 0.0.0.0 --port 9011 --reload

# Frontend (terminal 2)
cd app/static
python -m http.server 9010
```

- Frontend: http://127.0.0.1:9010
- Backend: http://127.0.0.1:9011 (docs en `/docs`)

## Experimentos que hice

Aqui describo que cambie en cada experimento, si mejoro o empeoro, y por que creo que paso eso.

### Baseline - Regresion Lineal

- **Que hice:** Use Regresion Lineal con One-Hot Encoding para las variables categoricas, sin escalar las numericas.
- **Por que:** Quise empezar con algo simple y rapido, solo para tener una referencia.
- **Resultado:** Este fue mi punto de partida; todos los demas experimentos los compare contra este.

### E1 - Ridge + escalamiento

- **Que hice:** Agregue StandardScaler a las variables numericas y cambie el modelo a Ridge (alpha=10).
- **Por que:** Lei que Ridge ayuda cuando hay muchas variables one-hot y posible multicolinealidad.
- **Resultado:** Las metricas mejoraron bastante respecto al baseline. El RMSE bajo de 2.84 a 1.96 (30.97% de mejora). Creo que la regularizacion ayudo a que los coeficientes no fueran tan inestables con tantas variables one-hot.

### E2 - Feature engineering + Ridge

- **Que hice:** Cree 3 variables nuevas:
  - `Vehicle_Age` (2020 - Year)
  - `Log_Kms` (log de los kilometros)
  - `Price_Ratio` (Present_Price / Selling_Price)
- **Por que:** pense que la antiguedad y los kilometros en log podrian capturar mejor la relacion no lineal con el precio.
- **Resultado:** Este experimento **empeoro** todas las metricas (MAE: 2.31, RMSE: 5.16, R2: 0.21). Creo que el problema fue que `Price_Ratio` usa el target (`Selling_Price`), lo cual causa data leakage y hace que el modelo no generalice bien en el test.

### E3 - Random Forest

- **Que hice:** Cambie el algoritmo a RandomForestRegressor (200 arboles).
- **Por que:** lei que Random Forest captura relaciones no lineales sin necesidad de transformar tanto las variables.
- **Resultado:** Mejoro el MAE respecto al baseline (1.18 vs 1.25, 6% de mejora), pero el RMSE empeoro (3.33 vs 2.84). Creo que el modelo comete errores mas consistentes pero tiene algunos errores grandes.

### E4 - Gradient Boosting

- **Que hice:** Use GradientBoostingRegressor con learning_rate=0.05 y max_depth=3.
- **Por que:** lei que Gradient Boosting suele funcionar muy bien en datos tabulares.
- **Resultado:** Este experimento tuvo el **mejor MAE** (1.14, 9% de mejora respecto al baseline) y un RMSE similar al baseline (2.83 vs 2.84). Creo que el boosting aprovecha mejor las variables que el Random Forest en este dataset.

### E5 - Gradient Boosting sin outliers en train

- **Que hice:** Elimine valores extremos (percentiles 1 y 99) de las variables numericas SOLO en el conjunto de entrenamiento.
- **Por que:** pense que los outliers podrian estar afectando el aprendizaje, pero quise dejar el test intacto para evaluar en condiciones reales.
- **Resultado:** Este experimento **empeoro** todas las metricas (MAE: 1.29, RMSE: 3.63, R2: 0.61). Creo que al sacar datos del entrenamiento, el modelo perdio informacion util y no aprendio tan bien.

## Resultados

Los resultados completos los guarde en:

- `models/tabla_experimentos.csv`
- `models/metrics.json`

### Tabla de experimentos

| Experimento | MAE | RMSE | R2 | Mejora MAE | Mejora RMSE |
|---|---:|---:|---:|---:|---:|
| Baseline - LinearRegression | 1.2539 | 2.8403 | 0.7609 | 0% | 0% |
| E1 - Ridge + scaling | 1.2061 | 1.9607 | 0.8861 | 3.81% | 30.97% |
| E2 - Feature engineering + Ridge | 2.3095 | 5.1624 | 0.2103 | -84.18% | -81.75% |
| E3 - RandomForest | 1.1781 | 3.3307 | 0.6713 | 6.04% | -17.26% |
| E4 - GradientBoosting | 1.1410 | 2.8304 | 0.7626 | 9.01% | 0.35% |
| E5 - GradientBoosting sin outliers | 1.2850 | 3.6275 | 0.6101 | -2.48% | -27.71% |

### Mejor modelo

El mejor modelo segun RMSE fue **E1 - Ridge + scaling** con:

- **MAE:** 1.2061 lakhs INR
- **RMSE:** 1.9607 lakhs INR
- **R2:** 0.8861
- **Mejora respecto al baseline:** 3.81% en MAE y 30.97% en RMSE

Aunque E4 (Gradient Boosting) tuvo mejor MAE (1.14), E1 tuvo mucho mejor RMSE y R2, asi que ese fue el que el script guardo como `vehicle_price_best.joblib`.

### Interpretacion

- **E1 (Ridge + scaling)** fue el mejor porque la regularizacion ayudo a controlar los coeficientes de las variables one-hot, y el escalamiento hizo que el modelo fuera mas estable numericamente.
- **E2 (Feature engineering)** empeoro porque `Price_Ratio` tiene informacion del target, lo cual causa data leakage.
- **E4 (Gradient Boosting)** tuvo el mejor MAE pero no el mejor RMSE, lo que sugiere que comete errores mas consistentes pero menos extremos.
- **E5 (sin outliers)** empeoro porque al sacar datos de entrenamiento, el modelo perdio informacion util.

## Conclusiones

- El mejor modelo fue: **E1 - Ridge + scaling** con:
  - MAE: 1.2061 lakhs INR
  - RMSE: 1.9607 lakhs INR
  - R2: 0.8861
- La mayor mejora respecto al baseline la obtuve con: **E1 - Ridge + scaling** (mejora de 3.81% en MAE y 30.97% en RMSE).
- Creo que esto paso porque la regularizacion Ridge ayudo a controlar los coeficientes de las variables one-hot, y el escalamiento hizo que el modelo fuera mas estable numericamente.
- Las metricas finales me parecen razonables para un primer modelo; un MAE de 1.2 lakhs representa aproximadamente 120,000 INR de error promedio, lo cual es aceptable para una estimacion inicial.
- El feature engineering (E2) no funciono como esperaba porque incluí´« una variable (`Price_Ratio`) que usa el target, lo cual causa data leakage. Si hubiera usado solo `Vehicle_Age` y `Log_Kms`, probablemente habria mejorado.
- Los modelos de boosting (E4) son prometedores, pero en este caso Ridge + scaling fue mejor porque el dataset no es tan grande y la regularizacion ayuda mas que la complejidad del boosting.

## Prediccion para 3 vehiculos

Con el mejor modelo (E1 - Ridge + scaling), estime el precio de 3 vehiculos como ejemplo:

1. `swift, 2015, 40000 km, Petrol, Dealer, Manual, 0`
2. `creta, 2017, 25000 km, Diesel, Dealer, Manual, 0`
3. `city, 2016, 35000 km, Petrol, Dealer, Manual, 0`

Para ver las predicciones exactas, corre:

```bash
python scripts/predict_3_vehiculos.py
```

O usa el frontend en http://127.0.0.1:9010 y captura los 3 vehiculos manualmente.

---

**Entregables:**

- Script ejecutable: `scripts/train_model.py`
- Tabla de experimentos: `models/tabla_experimentos.csv`
- Mejor pipeline: `models/vehicle_price_best.joblib`
- Metricas y conclusiones: `models/metrics.json` y este README.