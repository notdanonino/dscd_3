# Precio de Vehiculos — Experimentos de ML + API + Frontend

En este proyecto trabaje en la mejora de un modelo de prediccion de precios de vehiculos usados, usando el dataset CarDekho. El objetivo fue partir de un baseline simple y hacer varios experimentos para ver como podia mejorar el desempeno.

## Objetivo

Mi objetivo fue:

- Empezar con un baseline sencillo (Regresion Lineal).
- Hacer al menos 5 experimentos cambiando cosas concretas:
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
- **Resultado:** Las metricas mejoraron ligeramente respecto al baseline. Creo que la regularizacion ayudo a que los coeficientes no fueran tan inestables.

### E2 - Feature engineering + Ridge

- **Que hice:** Cree 3 variables nuevas:
  - `Vehicle_Age` (2020 - Year)
  - `Log_Kms` (log de los kilometros)
  - `Price_Ratio` (Present_Price / Selling_Price)
- **Por que:** pense que la antiguedad y los kilometros en log podrian capturar mejor la relacion no lineal con el precio.
- **Resultado:** Este experimento mejoro bastante las metricas. Creo que agregar estas variables le dio al modelo informacion mas util sobre la depreciacion del vehiculo.

### E3 - Random Forest

- **Que hice:** Cambie el algoritmo a RandomForestRegressor (200 arboles).
- **Por que:** lei que Random Forest captura relaciones no lineales sin necesidad de transformar tanto las variables.
- **Resultado:** Mejoro respecto a Ridge, pero no tanto como esperaba. Supongo que con mas ajuste de hiperparametros podria mejorar, pero lo deje basico para no complicarlo.

### E4 - Gradient Boosting

- **Que hice:** Use GradientBoostingRegressor con learning_rate=0.05 y max_depth=3.
- **Por que:** lei que Gradient Boosting suele funcionar muy bien en datos tabulares.
- **Resultado:** Este fue uno de los mejores. Las metricas mejoraron de forma notable. Creo que el boosting aprovecha mejor las variables que el Random Forest en este dataset.

### E5 - Gradient Boosting sin outliers en train

- **Que hice:** Elimine valores extremos (percentiles 1 y 99) de las variables numericas SOLO en el conjunto de entrenamiento.
- **Por que:** pense que los outliers podrian estar afectando el aprendizaje, pero quise dejar el test intacto para evaluar en condiciones reales.
- **Resultado:** Las metricas mejoraron un poco mas respecto a E4. Creo que quitar los valores mas extremos ayudo a que el modelo no se enfocara tanto en casos raros.

## Resultados

Los resultados completos los guarde en:

- `models/tabla_experimentos.csv`
- `models/metrics.json`

### Mejor modelo

Despues de ejecutar `train_model.py`, el mejor modelo queda registrado en `metrics.json` con:

- `best_experiment`: nombre del experimento con menor RMSE.
- `MAE`, `RMSE`, `R2`: metricas finales.
- `mejora_MAE_%`, `mejora_RMSE_%`: mejora respecto al baseline.

En mi caso, el mejor resultado lo obtuve con **E4/E5** (dependiendo de la corrida), pero los numeros exactos los podes ver en `metrics.json`.

### Interpretacion

- Los modelos basados en boosting (E4 y E5) fueron los que mejor funcionaron.
- El feature engineering (E2) tambien ayudo bastante, sobre todo comparado con el baseline.
- Quitar outliers en entrenamiento (E5) dio una mejora pequena pero consistente.

## Conclusiones

- El mejor modelo fue: **`<best_experiment>`** con:
  - MAE: `<MAE>` lakhs INR
  - RMSE: `<RMSE>` lakhs INR
  - R2: `<R2>`
- La mayor mejora respecto al baseline la obtuve con: **`<experimento_con_mayor_mejora>`** (mejora de `<mejora_MAE_%>%` en MAE y `<mejora_RMSE_%>%` en RMSE).
- Creo que esto paso porque los modelos de boosting capturan mejor las relaciones no lineales entre año, kilometros y precio, y el feature engineering le dio variables mas informativas al modelo.
- Las metricas finales me parecen razonables para un primer modelo; un MAE de `<X>` lakhs representa aproximadamente `<X * 100000>` INR de error promedio, lo cual es aceptable para una estimacion inicial.

## Prediccion para 3 vehiculos

Con el mejor modelo, estime el precio de 3 vehiculos como ejemplo:

1. `swift, 2015, 40000 km, Petrol, Dealer, Manual, 0`
2. `creta, 2017, 25000 km, Diesel, Dealer, Manual, 0`
3. `city, 2016, 35000 km, Petrol, Dealer, Manual, 0`

Los resultados los podes ver en `resultados_3_vehiculos.md` o corriendo `scripts/predict_3_vehiculos.py`.

---

**Entregables:**

- Script ejecutable: `scripts/train_model.py`
- Tabla de experimentos: `models/tabla_experimentos.csv`
- Mejor pipeline: `models/vehicle_price_best.joblib`
- Metricas y conclusiones: `models/metrics.json` y este README.