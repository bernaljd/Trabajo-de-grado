"""
M5. Evaluación de los modelos en el conjunto de prueba.

Entrada: dataset/processed/03_ventanas_mm4.csv          (salida de M3)
         dataset/processed/03_ventanas_xgboost.csv      (mismas filas; variables del gradient boosting)
         dataset/processed/03_ventanas_lstm.csv         (mismas filas)
         dataset/processed/04_entrenamiento.csv         (salida de M4: escala de la LSTM y modelo vigente)
         modelos/gradient_boosting.joblib, lstm.keras, arima.joblib y sarima.joblib   (salida de M4)
Salida:  dataset/processed/05_predicciones.csv          (una fila por plato y semana de prueba)
         dataset/processed/05_metricas.csv              (una fila por modelo, más la referencia «siempre 0»)
         dataset/processed/05_matriz_confusion.csv      (semanas con venta, por modelo)
Especificación: pipeline/DISENO_PIPELINE.md (sección M5) y pipeline/M5_evaluacion.md

Pasos:
  1. Ingesta y validación: leer las ventanas de M3, el resultado de M4 y los cuatro modelos.
  2. Conjunto de prueba y nivel: tomar las 26 semanas de prueba y calcular el nivel de cada fila
     igual que en M4.
  3. Predicciones: MM4, GradientBoosting, LSTM, ARIMA y SARIMA pronostican cada fila de prueba, en unidades.
  4. Métricas: MAE, RMSE y SMAPE (OE4); accuracy, precision, recall y F1 de las semanas con
     venta; y WAPE, MASE, MAE en semanas con venta y sesgo. Se agrega la referencia «siempre 0».
  5. Comparación con MM4: en cuántas de las 3 métricas del OE4 gana cada modelo (ML y estadísticos).
  6. Controles de calidad: comprobar filas, predicciones y métricas.
  7. Exportación: escribir 05_predicciones.csv, 05_metricas.csv y 05_matriz_confusion.csv.

Si un control falla, el script se detiene con un AssertionError y no escribe la salida.

Uso (desde code/src, después de M4):
  python3 m5_evaluacion.py
"""

import os
import joblib # type: ignore

# Quita los mensajes de arranque de TensorFlow, que no dicen nada de los datos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # type: ignore
import sklearn # type: ignore
import warnings
import statsmodels # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore
import tensorflow as tf # type: ignore

# Las mismas constantes de M4: si cambian allá, cambian aquí
HISTORIA_LSTM = 8
VARIABLES_LSTM = ['ur', 'precio', 'sin', 'cos']
NIVEL_MINIMO = 0.1
# Un pronóstico de 0,5 unidades o más cuenta como «predice venta»: redondeado da al menos 1 unidad
UMBRAL_VENTA = 0.5


def smape(y_real, y_pred):
    # SMAPE en %: 2|real - pred| / (|real| + |pred|). Cuando ambos son 0 el error es 0, no una división por 0.
    denominador = np.abs(y_real) + np.abs(y_pred)
    terminos = np.where(denominador == 0, 0, 2 * np.abs(y_real - y_pred) / np.where(denominador == 0, 1, denominador))
    return 100 * terminos.mean()


def a_cubo(tabla, columnas_fijas):
    # Igual que en M4: entrada de la LSTM con forma (muestras, 8 pasos, variables), de t-8 a t-1
    pasos = np.stack([tabla[[f'{variable}_t{k}' for k in range(HISTORIA_LSTM, 0, -1)]].to_numpy()
                      for variable in VARIABLES_LSTM], axis=2)
    # El plato y el nivel no cambian en la ventana: se repiten en los 8 pasos
    fijas = np.repeat(tabla[columnas_fijas].to_numpy()[:, np.newaxis, :], HISTORIA_LSTM, axis=1)
    return np.concatenate([pasos, fijas], axis=2).astype('float32')


# ---------------------------------------------------------------------------
# 1. Ingesta y validación
# ---------------------------------------------------------------------------

# El script se ejecuta desde code/src, así que los datos están dos carpetas más arriba
ruta_mm4 = '../../dataset/processed/03_ventanas_mm4.csv'
ruta_xgboost = '../../dataset/processed/03_ventanas_xgboost.csv'
ruta_lstm = '../../dataset/processed/03_ventanas_lstm.csv'
ruta_entrenamiento = '../../dataset/processed/04_entrenamiento.csv'
ruta_modelo_gb = '../../modelos/gradient_boosting.joblib'
ruta_modelo_lstm = '../../modelos/lstm.keras'
ruta_modelo_arima = '../../modelos/arima.joblib'
ruta_modelo_sarima = '../../modelos/sarima.joblib'
ruta_predicciones = '../../dataset/processed/05_predicciones.csv'
ruta_metricas = '../../dataset/processed/05_metricas.csv'
ruta_matriz = '../../dataset/processed/05_matriz_confusion.csv'

mm4 = pd.read_csv(ruta_mm4)
ventanas_xgboost = pd.read_csv(ruta_xgboost)
ventanas_lstm = pd.read_csv(ruta_lstm)
entrenamiento = pd.read_csv(ruta_entrenamiento)
# joblib.load devuelve el HistGradientBoostingRegressor tal como lo guardó M4
modelo_gb = joblib.load(ruta_modelo_gb)
modelo_lstm = tf.keras.models.load_model(ruta_modelo_lstm)
# ARIMA y SARIMA: diccionario con el orden, la parte estacional y los parámetros de cada plato (None si falló)
modelos_arima = {'ARIMA': joblib.load(ruta_modelo_arima), 'SARIMA': joblib.load(ruta_modelo_sarima)}

print('M5 Evaluación en prueba')
print(f'Versiones: pandas {pd.__version__}, scikit-learn {sklearn.__version__}, tensorflow {tf.__version__}, '
      f'statsmodels {statsmodels.__version__}')

columnas_comunes = ['fila_id', 'plato', 'semana_objetivo', 'conjunto', 'y_real']

# C1 los archivos no son vacíos
assert len(mm4) > 0 and len(ventanas_xgboost) > 0 and len(ventanas_lstm) > 0 and len(entrenamiento) > 0, \
    'Hay un archivo sin filas'
# C2 las ventanas empiezan con las columnas comunes de M3
for tabla in [mm4, ventanas_xgboost, ventanas_lstm]:
    assert list(tabla.columns[:5]) == columnas_comunes, 'Las columnas no son las de M3'
# C3 las tres ventanas tienen las mismas filas y el mismo conjunto (control C1 de M3)
assert mm4['fila_id'].equals(ventanas_xgboost['fila_id']) and mm4['fila_id'].equals(ventanas_lstm['fila_id']), \
    'Los archivos no tienen los mismos fila_id'
assert mm4['conjunto'].equals(ventanas_xgboost['conjunto']) and mm4['conjunto'].equals(ventanas_lstm['conjunto']), \
    'Los archivos no tienen el mismo conjunto'
# C4 sin vacíos en las ventanas
assert ventanas_xgboost.notna().all().all() and ventanas_lstm.notna().all().all(), 'Hay valores vacíos'
# C5 M4 dejó un solo modelo vigente y la escala de las 5 variables de la LSTM
assert (entrenamiento['vigente'] == 'sí').sum() == 1, 'M4 debe dejar un solo modelo vigente'
assert (entrenamiento['modelo'] == 'LSTM_escalado').sum() == 5, 'Falta la escala de la LSTM en 04_entrenamiento.csv'

# ---------------------------------------------------------------------------
# 2. Conjunto de prueba y nivel
# Reglas de la prueba
# P1: solo se evalúan las filas de 'prueba' (las 26 últimas semanas). Los modelos de M4 ya vieron
#     entrenamiento y validación al reentrenarse (R1 de M4), así que medirlos ahí no sería justo
# P2: la evaluación se hace una sola vez; nada de lo que sale aquí vuelve a cambiar los modelos
# Reglas del nivel (las mismas de M4, todo por plato)
# N1: el nivel de una fila es la media de las 8 semanas anteriores de ese plato más 0,1
# N2: la salida del gradient boosting y de la LSTM se multiplica por el nivel para tener unidades
# ---------------------------------------------------------------------------

# P1: filas de prueba
es_prueba = mm4['conjunto'] == 'prueba'

# N1: media de u_t1 a u_t8 (están en el archivo de la LSTM, que tiene las mismas filas)
media_8 = ventanas_lstm[[f'u_t{k}' for k in range(1, HISTORIA_LSTM + 1)]].mean(axis=1)
nivel = media_8 + NIVEL_MINIMO
nivel_prueba = nivel[es_prueba].to_numpy()
y_prueba = mm4.loc[es_prueba, 'y_real'].to_numpy()

# Tabla de predicciones: las columnas comunes de las filas de prueba
predicciones = mm4.loc[es_prueba, columnas_comunes].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 3. Predicciones
# Reglas de las predicciones
# F1: MM4 pronostica el promedio de las 4 semanas anteriores
# F2: el gradient boosting usa las mismas variables que en M4, más media_8, y en el mismo orden
# F3: la LSTM usa la escala 0-1 que guardó M4 (filas LSTM_escalado), calculada con entrenamiento
# F4: una predicción negativa se lleva a 0 unidades, como en M4
# F5: ARIMA y SARIMA usan los parámetros que dejó M4 (reentrenados con entrenamiento + validación) y pronostican
#     cada semana de prueba una semana adelante, con las unidades reales de las semanas anteriores del plato.
#     Un plato sin parámetros (la estimación falló en M4) usa MM4, como en M4
# ---------------------------------------------------------------------------

# F1: MM4
predicciones['pred_mm4'] = mm4.loc[es_prueba, ['u_t1', 'u_t2', 'u_t3', 'u_t4']].mean(axis=1).to_numpy()

# F2: las variables del gradient boosting, en el orden con el que se entrenó el modelo guardado
ventanas_xgboost['media_8'] = media_8
variables_gb = [columna for columna in ventanas_xgboost.columns if columna not in columnas_comunes]
assert list(modelo_gb.feature_names_in_) == variables_gb, \
    'Las variables del gradient boosting no son las del modelo guardado'
# N2: la salida relativa por el nivel da unidades
predicciones['pred_gradient_boosting'] = modelo_gb.predict(ventanas_xgboost.loc[es_prueba, variables_gb]) * nivel_prueba

# F3: lee el mínimo y el máximo de cada variable del texto 'variable=ur; minimo=0.0000; maximo=6.8966'
escala = {}
for texto in entrenamiento.loc[entrenamiento['modelo'] == 'LSTM_escalado', 'hiperparametros']:
    partes = dict(parte.split('=') for parte in texto.split('; '))
    escala[partes['variable']] = (float(partes['minimo']), float(partes['maximo']))

# N2: unidades de cada semana de la ventana divididas por el nivel, y el nivel como variable
lstm_escalada = ventanas_lstm[es_prueba].copy()
for k in range(1, HISTORIA_LSTM + 1):
    lstm_escalada[f'ur_t{k}'] = lstm_escalada[f'u_t{k}'] / nivel_prueba
lstm_escalada['nivel'] = nivel_prueba
# F3: cada variable pasa a 0-1 con la escala de entrenamiento
for variable, (minimo, maximo) in escala.items():
    columnas = ['nivel'] if variable == 'nivel' else [f'{variable}_t{k}' for k in range(1, HISTORIA_LSTM + 1)]
    lstm_escalada[columnas] = (lstm_escalada[columnas] - minimo) / (maximo - minimo)

columnas_plato = [columna for columna in ventanas_lstm.columns if columna.startswith('plato_')]
X_prueba_lstm = a_cubo(lstm_escalada, columnas_plato + ['nivel'])
# La salida sale de la escala 0-1, pasa a unidades relativas y luego a unidades (N2)
minimo_ur, maximo_ur = escala['ur']
relativa = modelo_lstm.predict(X_prueba_lstm, verbose=0).ravel() * (maximo_ur - minimo_ur) + minimo_ur
predicciones['pred_lstm'] = relativa * nivel_prueba

# F5: ARIMA y SARIMA, plato por plato. La serie de cada plato va hasta su última semana de prueba y el
# pronóstico de una semana adelante solo usa las semanas anteriores; los parámetros quedan fijos (filter)
for nombre_modelo, columna in [('ARIMA', 'pred_arima'), ('SARIMA', 'pred_sarima')]:
    guardado = modelos_arima[nombre_modelo]
    predicciones[columna] = np.nan
    pred_por_fila = pd.Series(np.nan, index=mm4['fila_id'])
    for plato, filas in mm4.sort_values(['plato', 'semana_objetivo']).groupby('plato'):
        parametros = guardado['parametros'].get(plato)
        if parametros is None:
            continue
        orden, estacional = guardado['orden'], guardado['estacional']
        tendencia = 'c' if orden[1] == 0 and estacional[1] == 0 else 'n'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pronostico = SARIMAX(filas['y_real'].to_numpy(dtype=float), order=orden, seasonal_order=estacional,
                                 trend=tendencia).filter(parametros).fittedvalues
        pred_por_fila[filas['fila_id'].to_numpy()] = pronostico
    predicciones[columna] = pred_por_fila[predicciones['fila_id'].to_numpy()].to_numpy()
    # Platos sin parámetros o con valores no finitos: MM4
    sin_valor = ~np.isfinite(predicciones[columna])
    predicciones.loc[sin_valor, columna] = predicciones.loc[sin_valor, 'pred_mm4']
    print(f'{nombre_modelo}: {sin_valor.sum()} filas de prueba con MM4 por falta de parámetros')

# F4: sin predicciones negativas
columnas_pred = {'MM4': 'pred_mm4', 'GradientBoosting': 'pred_gradient_boosting', 'LSTM': 'pred_lstm',
                 'ARIMA': 'pred_arima', 'SARIMA': 'pred_sarima'}
for columna in columnas_pred.values():
    predicciones[columna] = predicciones[columna].clip(lower=0)

# ---------------------------------------------------------------------------
# 4. Métricas
# Reglas de las métricas (todas en unidades reales, por plato y semana, sobre las filas de prueba)
# M1: MAE, RMSE y SMAPE, las tres del OE4. SMAPE vale 0 cuando real y predicción son 0
# M2: semana con venta: real si y_real > 0, pronosticada si la predicción es de al menos 0,5 unidades.
#     Con eso se calculan accuracy, precision, recall, F1 y la matriz de confusión (como en M4)
# M3: referencia «siempre 0»: pronosticar 0 en todas las filas. Con muchos ceros su accuracy es alta,
#     así que la mejora real de un modelo es su accuracy menos la de esta referencia
# M4: WAPE = suma de errores absolutos / suma de unidades reales, en %
# M5: MASE = MAE del modelo / MAE de repetir la semana anterior en entrenamiento + validación
# M6: MAE en semanas con venta (y_real > 0) y sesgo = (pronosticado - real) / real, en %
# ---------------------------------------------------------------------------

# M5: escala del MASE. u_t1 es la semana anterior, así que |y_real - u_t1| es el error del pronóstico ingenuo.
# Se usan las semanas con las que se entrenaron los modelos guardados (entrenamiento + validación).
no_prueba = ~es_prueba
escala_mase = (mm4.loc[no_prueba, 'y_real'] - mm4.loc[no_prueba, 'u_t1']).abs().mean()

# M2: semanas con venta reales
vende_real = y_prueba > 0

# M3: la referencia se evalúa igual que los modelos, con una predicción de 0 en cada fila
pronosticos = {nombre: predicciones[columna].to_numpy() for nombre, columna in columnas_pred.items()}
pronosticos['Siempre 0'] = np.zeros(len(y_prueba))

metricas, matrices = [], []
for nombre, pred in pronosticos.items():
    # M2: semanas en que el modelo dice «habrá venta»
    vende_pred = pred >= UMBRAL_VENTA
    # M2: VP dijo venta y hubo; FP dijo venta y no hubo; FN no dijo venta y hubo; VN no dijo venta y no hubo.
    # labels fija el orden, así sale bien aunque un modelo nunca diga «hay venta».
    vn, fp, fn, vp = confusion_matrix(vende_real, vende_pred, labels=[False, True]).ravel()
    matrices.append({'modelo': nombre, 'vp': vp, 'fp': fp, 'fn': fn, 'vn': vn})
    metricas.append({
        'modelo': nombre,
        'conjunto': 'prueba',
        'n': len(pred),
        # M1
        'mae': mean_absolute_error(y_prueba, pred),
        'rmse': np.sqrt(mean_squared_error(y_prueba, pred)),
        'smape': smape(y_prueba, pred),
        # M2: zero_division=0 evita un error si el modelo nunca dice «hay venta» (la referencia)
        'accuracy': accuracy_score(vende_real, vende_pred),
        'precision': precision_score(vende_real, vende_pred, zero_division=0),
        'recall': recall_score(vende_real, vende_pred, zero_division=0),
        'f1': f1_score(vende_real, vende_pred, zero_division=0),
        # M4
        'wape': 100 * np.abs(y_prueba - pred).sum() / y_prueba.sum(),
        # M5
        'mase': mean_absolute_error(y_prueba, pred) / escala_mase,
        # M6
        'mae_con_venta': mean_absolute_error(y_prueba[vende_real], pred[vende_real]),
        'unidades_reales': y_prueba.sum(),
        'unidades_pronosticadas': pred.sum(),
        'sesgo': 100 * (pred.sum() - y_prueba.sum()) / y_prueba.sum(),
    })
metricas = pd.DataFrame(metricas)
matriz_confusion = pd.DataFrame(matrices)

# M3: puntos de accuracy que cada modelo gana sobre «siempre 0»
accuracy_cero = metricas.loc[metricas['modelo'] == 'Siempre 0', 'accuracy'].iloc[0]
metricas['accuracy_sobre_cero'] = metricas['accuracy'] - accuracy_cero

# ---------------------------------------------------------------------------
# 5. Comparación con MM4
# Reglas de la comparación
# K1: un modelo (ML o estadístico) tiene éxito si supera a MM4 en al menos 2 de las 3 métricas del OE4
#     (MAE, RMSE y SMAPE) en prueba. Si no lo logra, se reporta tal cual
# K2: el vigente es el que eligió M4 con validación; M5 no lo cambia (P2)
# ---------------------------------------------------------------------------

fila_mm4 = metricas[metricas['modelo'] == 'MM4'].iloc[0]
es_ml = metricas['modelo'].isin(['GradientBoosting', 'LSTM', 'ARIMA', 'SARIMA'])
# K1: cuenta las métricas en que el modelo tiene menos error que MM4. Se cuenta en todos los modelos menos MM4 y la referencia.
metricas['supera_mm4'] = [sum(fila[m] < fila_mm4[m] for m in ['mae', 'rmse', 'smape']) if ml else None
                          for (_, fila), ml in zip(metricas.iterrows(), es_ml)]
metricas['exito'] = [('sí' if fila['supera_mm4'] >= 2 else 'no') if ml else None
                     for (_, fila), ml in zip(metricas.iterrows(), es_ml)]

# K2: en 04_entrenamiento.csv el vigente aparece como 'GradientBoosting' o 'LSTM_promedio'
vigente_m4 = entrenamiento.loc[entrenamiento['vigente'] == 'sí', 'modelo'].iloc[0].replace('_promedio', '')
metricas['vigente'] = np.where(metricas['modelo'] == vigente_m4, 'sí', 'no')

columnas_metricas = ['modelo', 'conjunto', 'n', 'mae', 'rmse', 'smape', 'accuracy', 'accuracy_sobre_cero',
                     'precision', 'recall', 'f1', 'wape', 'mase', 'mae_con_venta', 'unidades_reales',
                     'unidades_pronosticadas', 'sesgo', 'supera_mm4', 'exito', 'vigente']
metricas = metricas[columnas_metricas].reset_index(drop=True)
metricas['supera_mm4'] = metricas['supera_mm4'].astype('Int64')

columnas_salida = ['fila_id', 'plato', 'semana_objetivo', 'conjunto', 'y_real',
                   'pred_mm4', 'pred_gradient_boosting', 'pred_lstm', 'pred_arima', 'pred_sarima']
predicciones = predicciones[columnas_salida].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Controles de calidad
# ---------------------------------------------------------------------------

# C1: 26 semanas de prueba y una fila por plato y semana
assert predicciones['semana_objetivo'].nunique() == 26, 'La prueba no tiene 26 semanas'
assert not predicciones.duplicated(['plato', 'semana_objetivo']).any(), 'Hay platos repetidos en una semana'
# C2: todas las filas son de prueba
assert (predicciones['conjunto'] == 'prueba').all(), 'Hay filas que no son de prueba'
# C3: predicciones sin vacíos y no negativas
valores = predicciones[list(columnas_pred.values())]
assert np.isfinite(valores).all().all() and (valores >= 0).all().all(), 'Hay predicciones vacías o negativas'
# C4: una fila de métricas por modelo más la referencia, sin vacíos
assert list(metricas['modelo']) == ['MM4', 'GradientBoosting', 'LSTM', 'ARIMA', 'SARIMA', 'Siempre 0'], 'Faltan modelos en las métricas'
assert metricas.drop(columns=['supera_mm4', 'exito']).notna().all().all(), 'Hay métricas vacías'
# C5: cada matriz de confusión suma todas las filas de prueba
assert (matriz_confusion[['vp', 'fp', 'fn', 'vn']].sum(axis=1) == len(predicciones)).all(), \
    'La matriz de confusión no suma las filas de prueba'
# C6: la escala leída de M4 es la misma que sale de las filas de entrenamiento
es_train = mm4['conjunto'] == 'entrenamiento'
assert abs(nivel[es_train].max() - escala['nivel'][1]) < 0.001, 'La escala de M4 no coincide con entrenamiento'

# Avisos: no detienen el script, solo informan

# A1: cuántas métricas del OE4 le gana cada modelo a MM4 (criterio de éxito)
for _, fila in metricas[es_ml].iterrows():
    print(f"A1: {fila['modelo']}: MAE {fila['mae']:.3f} frente a MM4 {fila_mm4['mae']:.3f}; "
          f"le gana en {fila['supera_mm4']} de 3 métricas -> éxito: {fila['exito']}")

# A2: «siempre 0», para saber cuánto pesan los ceros en el MAE y en la accuracy
fila_cero = metricas[metricas['modelo'] == 'Siempre 0'].iloc[0]
print(f"A2: siempre 0 da MAE {fila_cero['mae']:.3f} y accuracy {fila_cero['accuracy']:.1%}; "
      f'{(~vende_real).mean():.1%} de las filas de prueba tiene 0 unidades')

# A3: la demanda de prueba frente a la de validación (M4 vio que cayó a la mitad)
unidades_validacion = mm4.loc[mm4['conjunto'] == 'validacion', 'y_real'].sum()
print(f'A3: prueba tiene {y_prueba.sum()} unidades frente a {unidades_validacion} de validación')

# A4: el pronóstico ingenuo en entrenamiento + validación (denominador del MASE) y en prueba.
# Como la demanda bajó, el MASE se ve mejor que la comparación con el ingenuo en prueba.
ingenua_prueba = mean_absolute_error(y_prueba, mm4.loc[es_prueba, 'u_t1'])
print(f'A4: repetir la semana anterior tiene MAE {escala_mase:.3f} en entrenamiento + validación '
      f'y {ingenua_prueba:.3f} en prueba')

print(matriz_confusion.to_string(index=False))
totales = metricas[['modelo', 'mae', 'rmse', 'smape', 'accuracy', 'precision', 'recall', 'f1', 'sesgo',
                    'supera_mm4', 'vigente']]
print(totales.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Exportación
# ---------------------------------------------------------------------------

# Cuatro decimales en predicciones y métricas
predicciones.to_csv(ruta_predicciones, index=False, float_format='%.4f')
metricas.to_csv(ruta_metricas, index=False, float_format='%.4f')
matriz_confusion.to_csv(ruta_matriz, index=False)
print(f'{len(predicciones)} filas -> {ruta_predicciones}; {len(metricas)} filas -> {ruta_metricas}; '
      f'matriz -> {ruta_matriz}')
