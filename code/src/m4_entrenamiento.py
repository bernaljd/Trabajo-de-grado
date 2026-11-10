"""
M4. Entrenamiento: búsqueda de hiperparámetros (grid search) sobre validación.

Entrada: dataset/processed/03_ventanas_mm4.csv        (salida de M3)
         dataset/processed/03_ventanas_xgboost.csv    (mismas filas)
         dataset/processed/03_ventanas_lstm.csv       (mismas filas)
Salida:  dataset/processed/04_entrenamiento.csv       (una fila por combinación probada)
         dataset/processed/04_matriz_confusion.csv    (semanas con venta: MM4 y los dos ganadores)
         modelos/gradient_boosting.joblib             (gradient boosting ganador, reentrenado)
         modelos/lstm.keras                           (LSTM ganadora, reentrenada)
Especificación: pipeline/DISENO_PIPELINE.md (sección M4) y pipeline/M4_entrenamiento.md

Pasos:
  1. Ingesta y validación: leer las tres ventanas de M3 y revisar que cumplen su contrato.
  2. Conjuntos y nivel: separar entrenamiento y validación (la prueba no se usa en M4) y
     calcular el nivel reciente de cada fila, la media de sus 8 semanas anteriores.
  3. MM4: medir el error de la media móvil en validación, como referencia.
  4. Gradient boosting (HistGradientBoostingRegressor): grid search de 144 combinaciones con parada temprana, prediciendo las ventas
     relativas al nivel reciente.
  5. LSTM: escalar a 0-1 con entrenamiento y grid search de 32 combinaciones con parada temprana.
     Las 3 mejores se repiten con 5 semillas y se comparan por su promedio.
  6. Selección: el menor MAE en validación gana en cada modelo, y entre GradientBoosting y LSTM.
  7. Reentrenamiento: los dos ganadores se entrenan con entrenamiento + validación.
  8. Controles de calidad: comprobar combinaciones, métricas y ganadores.
  9. Exportación: escribir 04_entrenamiento.csv y los dos modelos.

Si un control falla, el script se detiene con un AssertionError y no escribe la salida.

Uso (desde code/src, después de M3):
  python3 m4_entrenamiento.py
"""

import os
import joblib # type: ignore 

# Quita los mensajes de arranque de TensorFlow, que no dicen nada de los datos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd # type: ignore 
import numpy as np # type: ignore 
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # type: ignore 
from sklearn.model_selection import ParameterGrid # type: ignore 
from sklearn.ensemble import HistGradientBoostingRegressor # type: ignore 
import sklearn # type: ignore 
import tensorflow as tf # type: ignore 

# Hace que TensorFlow dé el mismo resultado en cada corrida con la misma semilla
tf.config.experimental.enable_op_determinism()

# Semilla de todas las ejecuciones
SEMILLA = 42
# Semillas con las que se repiten las mejores LSTM, porque su resultado cambia con la semilla
SEMILLAS_REPETICION = [42, 43, 44, 45, 46]
# Cuantas de las mejores combinaciones de la LSTM se repiten con las 5 semillas
MEJORES_LSTM = 3
# Define las ventanas hacia atras.
HISTORIA_LSTM = 8
VARIABLES_LSTM = ['ur', 'precio', 'sin', 'cos']
# Se suma al nivel para poder dividir cuando las 8 semanas anteriores fueron 0 (evita dividir entre 0)
NIVEL_MINIMO = 0.1
# Un pronóstico de 0,5 unidades o más cuenta como «predice venta»: redondeado da al menos 1 unidad.
# Con este umbral se calculan accuracy, precision, recall, F1 y la matriz de confusión.
UMBRAL_VENTA = 0.5
# Parada temprana: gradient boosting hasta 1.000 árboles y se detiene si 50 seguidos no mejoran.
# La LSTM hasta 200 épocas y se detiene si 10 seguidas no mejoran.
MAX_ARBOLES = 1000
PACIENCIA_ARBOLES = 50
MAX_EPOCAS = 200
PACIENCIA_EPOCAS = 10   

GRILLA_GB = {
    # la metrica matematica que el modelo quiere minimizar. (Ataca a MAE).
    'loss': ['absolute_error', 'squared_error', 'poisson'],
    # profundidad del arbol.
    'max_depth': [3, 4, 6],
    # Que tanto se aporta a la prediccion final por cada arbol.
    'learning_rate': [0.03, 0.1],
    # variabilidad para forzar variacion en los arboles.
    'max_features': [0.8, 1.0],
    # Cantidad minima de ejemplos que debe quedar en una rama.
    'min_samples_leaf': [5, 20],
    # Castigo matematico para mantener los arboles simples.
    'l2_regularization': [0.0, 1.0],
}
# Grilla de la LSTM: 2 valores por hiperparámetro, 2^5 = 32 combinaciones
GRILLA_LSTM = {
    # Define las capas de la red
    'capas': [1, 2],
    # Define las neuronas de la memoria
    'unidades': [16, 32],
    # Apaga algunas neuronas para no depender de una señal
    'dropout': [0.1, 0.3],
    'learning_rate': [0.001, 0.01],
    'batch_size': [32, 64],
}


def combinaciones(grilla):
    # Grid search: ParameterGrid de scikit-learn arma todas las combinaciones de la grilla,
    # cada una como un diccionario {hiperparámetro: valor}. Cada combinación se entrena con
    # 'entrenamiento' y se mide en 'validacion'.
    return list(ParameterGrid(grilla))


def a_texto(parametros):
    # Hiperparámetros en texto legible para el CSV, por ejemplo 'max_depth=4; learning_rate=0.03'
    return '; '.join(f'{nombre}={valor}' for nombre, valor in parametros.items())

# Symmetric Mean Absolute Percentage Error
def smape(y_real, y_pred):
    # SMAPE en %: 2|real - pred| / (|real| + |pred|). Cuando ambos son 0 el error es 0, no una división por 0.
    denominador = np.abs(y_real) + np.abs(y_pred)
    terminos = np.where(denominador == 0, 0, 2 * np.abs(y_real - y_pred) / np.where(denominador == 0, 1, denominador))
    return 100 * terminos.mean()


def metricas(y_real, y_pred):
    # Las tres métricas del diseño, en unidades reales. Una predicción negativa se lleva a 0 unidades.
    y_pred = np.clip(y_pred, 0, None)
    # «Semana con venta» (sí/no), real y pronosticada, para las métricas de clasificación que pidió el director
    vende_real = y_real > 0
    vende_pred = y_pred >= UMBRAL_VENTA
    return {'mae_validacion': mean_absolute_error(y_real, y_pred),
            'rmse_validacion': np.sqrt(mean_squared_error(y_real, y_pred)),
            'smape_validacion': smape(y_real, y_pred),
            # Accuracy: % de semanas en que acierta si hubo venta o no
            'accuracy_validacion': accuracy_score(vende_real, vende_pred),
            # Precision: de las semanas en que dijo «hay venta», cuántas la tuvieron.
            # zero_division=0 evita el error si el modelo nunca dice «hay venta».
            'precision_validacion': precision_score(vende_real, vende_pred, zero_division=0),
            # Recall: de las semanas que sí tuvieron venta, cuántas detectó
            'recall_validacion': recall_score(vende_real, vende_pred, zero_division=0),
            # F1: equilibrio entre precision y recall
            'f1_validacion': f1_score(vende_real, vende_pred, zero_division=0)}

# tridimensional para la lectura deade el LSTM
def a_cubo(tabla, columnas_fijas):
    # Arma la entrada de la LSTM con forma (muestras, 8 pasos, variables).
    # Para cada variable se toman sus columnas de t-8 a t-1, así el paso 0 es la semana más antigua.
    pasos = np.stack([tabla[[f'{variable}_t{k}' for k in range(HISTORIA_LSTM, 0, -1)]].to_numpy()
                      for variable in VARIABLES_LSTM], axis=2)
    # El plato y el nivel no cambian en la ventana: se repiten en los 8 pasos
    fijas = np.repeat(tabla[columnas_fijas].to_numpy()[:, np.newaxis, :], HISTORIA_LSTM, axis=1)
    return np.concatenate([pasos, fijas], axis=2).astype('float32') 


def crear_lstm(parametros, n_variables, semilla):
    # La semilla fija los pesos iniciales y el orden de los lotes, para que el resultado se repita
    tf.keras.utils.set_random_seed(semilla)
    modelo = tf.keras.Sequential([tf.keras.Input(shape=(HISTORIA_LSTM, n_variables))])
    # Con 2 capas, la primera entrega la secuencia completa a la segunda (return_sequences)
    for capa in range(parametros['capas']):
        ultima = capa == parametros['capas'] - 1
        modelo.add(tf.keras.layers.LSTM(parametros['unidades'], return_sequences=not ultima))
        modelo.add(tf.keras.layers.Dropout(parametros['dropout']))
    # Una neurona de salida: las unidades relativas de la semana t, en escala 0-1
    modelo.add(tf.keras.layers.Dense(1))
    # La pérdida es el error absoluto (MAE), la misma métrica con la que se elige al ganador
    modelo.compile(optimizer=tf.keras.optimizers.Adam(parametros['learning_rate']), loss='mae')
    return modelo


# ---------------------------------------------------------------------------
# 1. Ingesta y validación
# ---------------------------------------------------------------------------

# El script se ejecuta desde code/src, así que los datos están dos carpetas más arriba
ruta_mm4 = '../../dataset/processed/03_ventanas_mm4.csv'
ruta_xgboost = '../../dataset/processed/03_ventanas_xgboost.csv'
ruta_lstm = '../../dataset/processed/03_ventanas_lstm.csv'
ruta_salida = '../../dataset/processed/04_entrenamiento.csv'
ruta_matriz = '../../dataset/processed/04_matriz_confusion.csv'
ruta_modelo_gb = '../../modelos/gradient_boosting.joblib'
ruta_modelo_lstm = '../../modelos/lstm.keras'

mm4 = pd.read_csv(ruta_mm4)
ventanas_xgboost = pd.read_csv(ruta_xgboost)
ventanas_lstm = pd.read_csv(ruta_lstm)

print('M4 Entrenamiento')
print(f'Versiones: pandas {pd.__version__}, scikit-learn {sklearn.__version__}, tensorflow {tf.__version__}')

columnas_comunes = ['fila_id', 'plato', 'semana_objetivo', 'conjunto', 'y_real']

# C1 los archivos no son vacíos
assert len(mm4) > 0 and len(ventanas_xgboost) > 0 and len(ventanas_lstm) > 0, 'Hay un archivo sin filas'
# C2 los tres empiezan con las columnas comunes de M3
for tabla in [mm4, ventanas_xgboost, ventanas_lstm]:
    assert list(tabla.columns[:5]) == columnas_comunes, 'Las columnas no son las de M3'
# C3 los tres tienen las mismas filas y el mismo conjunto (control C1 de M3)
assert mm4['fila_id'].equals(ventanas_xgboost['fila_id']) and mm4['fila_id'].equals(ventanas_lstm['fila_id']), \
    'Los archivos no tienen los mismos fila_id'
assert mm4['conjunto'].equals(ventanas_xgboost['conjunto']) and mm4['conjunto'].equals(ventanas_lstm['conjunto']), \
    'Los archivos no tienen el mismo conjunto'
# C4 sin vacíos: el gradient boosting los aceptaría en silencio y la LSTM no
assert ventanas_xgboost.notna().all().all() and ventanas_lstm.notna().all().all(), 'Hay valores vacíos'

# ---------------------------------------------------------------------------
# 2. Conjuntos y nivel
# Reglas de los conjuntos
# S1: se entrena solo con 'entrenamiento' y se eligen hiperparámetros solo con 'validacion'
# S2: las filas de 'prueba' no se leen en M4; quedan intactas para M5
# Reglas del nivel (la demanda cayó a la mitad entre entrenamiento y validación)
# N1: el nivel de una fila es la media de sus 8 semanas anteriores más 0,1. Solo usa el pasado (V3 de M3)
# N2: GradientBoosting y la LSTM predicen y_real / nivel; la predicción vuelve a unidades multiplicando por el nivel
# ---------------------------------------------------------------------------

# S2: solo se marcan entrenamiento y validación; las filas de prueba no entran en ningún cálculo
es_train = mm4['conjunto'] == 'entrenamiento'
es_validacion = mm4['conjunto'] == 'validacion'

# N1: media de u_t1 a u_t8 (están en el archivo de la LSTM, que tiene las mismas filas)
media_8 = ventanas_lstm[[f'u_t{k}' for k in range(1, HISTORIA_LSTM + 1)]].mean(axis=1)
nivel = media_8 + NIVEL_MINIMO
# N2: la salida que aprenden los modelos. Un 2 significa «el doble de lo que viene vendiendo».
y_relativa = mm4['y_real'] / nivel

# Variables de GradientBoosting: todas las columnas que no son las comunes (u_t1..u_t4, media_4, calendario,
# precio, plato y categoría en 0/1), más la media de 8 semanas, para que el modelo conozca el nivel
ventanas_xgboost['media_8'] = media_8
variables_gb = [columna for columna in ventanas_xgboost.columns if columna not in columnas_comunes]
# S1: entradas (X) y salidas (y) de cada conjunto
X_train_gb = ventanas_xgboost.loc[es_train, variables_gb]
X_validacion_gb = ventanas_xgboost.loc[es_validacion, variables_gb]
y_validacion = mm4.loc[es_validacion, 'y_real'].to_numpy()
nivel_validacion = nivel[es_validacion].to_numpy()

# Lista de resultados: cada combinación probada agrega un diccionario y al final es el CSV
resultados = []

# ---------------------------------------------------------------------------
# 3. MM4 (modelo base)
# ---------------------------------------------------------------------------

# MM4 no se entrena: predice el promedio de las 4 semanas anteriores
pred_mm4 = mm4.loc[es_validacion, ['u_t1', 'u_t2', 'u_t3', 'u_t4']].mean(axis=1).to_numpy()
resultados.append({'modelo': 'MM4', 'combinacion': 1, 'hiperparametros': 'promedio de u_t1 a u_t4',
                   'semilla': None, **metricas(y_validacion, pred_mm4)})

# ---------------------------------------------------------------------------
# 4. Gradient boosting (HistGradientBoostingRegressor de scikit-learn)
# Reglas del gradient boosting
# X1: se prueban 144 combinaciones con ParameterGrid, todas con semilla 42
# X2: el número de árboles lo decide la parada temprana: hasta 1.000 y para si 50 seguidos no
#     bajan el MAE de validación (en unidades relativas). Ese número queda en el texto (max_iter)
# ---------------------------------------------------------------------------

# Predicciones en unidades y ajustes ganadores de cada combinación (para el aviso A4 y el paso 7)
predicciones = {}
ajustes_por_combinacion = {}
grilla_gb = combinaciones(GRILLA_GB)
for numero, ajustes in enumerate(grilla_gb, start=1):
    # La parada temprana interna de scikit-learn aparta filas al azar de entrenamiento; se apaga
    # (early_stopping=False) para que la parada temprana mire 'validacion' (X2)
    modelo = HistGradientBoostingRegressor(**ajustes, max_iter=MAX_ARBOLES, early_stopping=False,
                                           random_state=SEMILLA)
    # N2: aprende las ventas relativas al nivel
    modelo.fit(X_train_gb, y_relativa[es_train])
    # X2: staged_predict da la predicción de validación con 1 árbol, con 2, con 3...
    # Se guarda el mejor MAE y se para cuando 50 árboles seguidos no lo mejoran.
    mejor_mae, arboles = np.inf, 0
    for n_arboles, pred_relativa in enumerate(modelo.staged_predict(X_validacion_gb), start=1):
        mae = mean_absolute_error(y_relativa[es_validacion], pred_relativa)
        if mae < mejor_mae:
            mejor_mae, arboles, mejor_pred = mae, n_arboles, pred_relativa
        elif n_arboles - arboles >= PACIENCIA_ARBOLES:
            break
    ajustes = {**ajustes, 'max_iter': arboles}
    ajustes_por_combinacion[numero] = ajustes
    # N2: la predicción relativa por el nivel da unidades
    pred = mejor_pred * nivel_validacion
    predicciones[('GradientBoosting', numero)] = pred
    resultados.append({'modelo': 'GradientBoosting', 'combinacion': numero, 'hiperparametros': a_texto(ajustes),
                       'semilla': SEMILLA, **metricas(y_validacion, pred)})
print(f'GradientBoosting: {len(grilla_gb)} combinaciones probadas', flush=True)

# ---------------------------------------------------------------------------
# 5. LSTM
# Reglas de la LSTM
# L1: cada variable se escala a 0-1 con el mínimo y el máximo de 'entrenamiento', nunca de todo el
#     histórico (si no, la LSTM sabría de antemano los valores de validación y prueba)
# L2: la secuencia de cada fila es (8 semanas x variables). Las unidades entran relativas al nivel (N2),
#     y el plato 0/1 y el nivel se repiten en cada semana
# L3: se prueban 32 combinaciones con semilla 42; la parada temprana mira la pérdida de validación
#     y se queda con los pesos de la mejor época. Esa época queda en el texto (epocas)
# L4: las 3 mejores combinaciones se repiten con 5 semillas y compiten por su MAE promedio
#     (filas LSTM_promedio), porque con una sola semilla el ganador puede serlo por suerte
# ---------------------------------------------------------------------------

columnas_plato = [columna for columna in ventanas_lstm.columns if columna.startswith('plato_')]

# N2: unidades de cada semana de la ventana, divididas por el nivel de la fila
lstm_escalada = ventanas_lstm.copy()
for k in range(1, HISTORIA_LSTM + 1):
    lstm_escalada[f'ur_t{k}'] = lstm_escalada[f'u_t{k}'] / nivel
lstm_escalada['nivel'] = nivel
lstm_escalada['y_relativa'] = y_relativa

# L1: mínimo y máximo de cada variable en entrenamiento (las 8 semanas juntas) y del nivel
escala = {}
for variable in VARIABLES_LSTM:
    columnas = [f'{variable}_t{k}' for k in range(1, HISTORIA_LSTM + 1)]
    valores = lstm_escalada.loc[es_train, columnas].to_numpy()
    escala[variable] = (valores.min(), valores.max())
escala['nivel'] = (lstm_escalada.loc[es_train, 'nivel'].min(), lstm_escalada.loc[es_train, 'nivel'].max())

# L1: cada variable pasa a 0-1. 'ur' también escala la salida, porque son las mismas unidades relativas.
for variable, (minimo, maximo) in escala.items():
    columnas = ['nivel'] if variable == 'nivel' else [f'{variable}_t{k}' for k in range(1, HISTORIA_LSTM + 1)]
    lstm_escalada[columnas] = (lstm_escalada[columnas] - minimo) / (maximo - minimo)
minimo_ur, maximo_ur = escala['ur']
lstm_escalada['y_escalada'] = (lstm_escalada['y_relativa'] - minimo_ur) / (maximo_ur - minimo_ur)

# L2: los cubos de entrada y la salida escalada de cada conjunto
columnas_fijas = columnas_plato + ['nivel']
X_train_lstm = a_cubo(lstm_escalada[es_train], columnas_fijas)
X_validacion_lstm = a_cubo(lstm_escalada[es_validacion], columnas_fijas)
y_train_lstm = lstm_escalada.loc[es_train, 'y_escalada'].to_numpy()
y_validacion_lstm = lstm_escalada.loc[es_validacion, 'y_escalada'].to_numpy()
n_variables = X_train_lstm.shape[2]


def probar_lstm(parametros, semilla):
    # L3: entrena con entrenamiento, para cuando 10 épocas seguidas no bajan la pérdida de validación
    # y vuelve a los pesos de la mejor época
    modelo = crear_lstm(parametros, n_variables, semilla)
    parada = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PACIENCIA_EPOCAS,
                                              restore_best_weights=True)
    historia = modelo.fit(X_train_lstm, y_train_lstm, validation_data=(X_validacion_lstm, y_validacion_lstm),
                          epochs=MAX_EPOCAS, batch_size=parametros['batch_size'], callbacks=[parada], verbose=0)
    # Época con la menor pérdida de validación, contando desde 1
    epocas = int(np.argmin(historia.history['val_loss'])) + 1
    # La predicción sale de la escala 0-1 y luego se multiplica por el nivel para tener unidades (N2)
    relativa = modelo.predict(X_validacion_lstm, verbose=0).ravel() * (maximo_ur - minimo_ur) + minimo_ur
    return epocas, relativa * nivel_validacion


grilla_lstm = combinaciones(GRILLA_LSTM)
for numero, parametros in enumerate(grilla_lstm, start=1):
    epocas, pred = probar_lstm(parametros, SEMILLA)
    errores = metricas(y_validacion, pred)
    # Avance: la búsqueda de la LSTM tarda varios minutos
    print(f"LSTM {numero}/{len(grilla_lstm)}: {epocas} épocas, MAE {errores['mae_validacion']:.3f}", flush=True)
    resultados.append({'modelo': 'LSTM', 'combinacion': numero,
                       'hiperparametros': a_texto({**parametros, 'epocas': epocas}),
                       'semilla': SEMILLA, **errores})

# L4: las 3 combinaciones con menor MAE (semilla 42) se repiten con las otras 4 semillas
tabla_lstm = pd.DataFrame([fila for fila in resultados if fila['modelo'] == 'LSTM'])
mejores_lstm = tabla_lstm.nsmallest(MEJORES_LSTM, 'mae_validacion')['combinacion'].tolist()
epocas_por_combinacion = {}
for numero in mejores_lstm:
    parametros = grilla_lstm[numero - 1]
    errores_semillas, epocas_semillas, pred_semillas = [], [], []
    for semilla in SEMILLAS_REPETICION:
        epocas, pred = probar_lstm(parametros, semilla)
        errores = metricas(y_validacion, pred)
        errores_semillas.append(errores)
        epocas_semillas.append(epocas)
        pred_semillas.append(pred)
        # La semilla 42 ya está en la grilla; solo se agregan las otras 4
        if semilla != SEMILLA:
            resultados.append({'modelo': 'LSTM_semillas', 'combinacion': numero,
                               'hiperparametros': a_texto({**parametros, 'epocas': epocas}),
                               'semilla': semilla, **errores})
    # L4: el promedio de las 5 semillas es el resultado que compite. Las épocas son el promedio redondeado.
    epocas_por_combinacion[numero] = int(round(np.mean(epocas_semillas)))
    predicciones[('LSTM_promedio', numero)] = np.mean(pred_semillas, axis=0)
    resultados.append({'modelo': 'LSTM_promedio', 'combinacion': numero,
                       'hiperparametros': a_texto({**parametros, 'epocas': epocas_por_combinacion[numero]}),
                       'semilla': None, **pd.DataFrame(errores_semillas).mean().to_dict()})
    print(f'LSTM combinación {numero} con 5 semillas: MAE {pd.DataFrame(errores_semillas)["mae_validacion"].mean():.3f}',
          flush=True)

# ---------------------------------------------------------------------------
# 6. Selección
# Reglas de selección
# G1: en cada modelo gana la combinación con menor MAE en validación. En la LSTM compiten los
#     promedios de 5 semillas (L4)
# G2: entre GradientBoosting y LSTM, el de menor MAE en validación es el modelo vigente
# ---------------------------------------------------------------------------

entrenamiento = pd.DataFrame(resultados)

# G1: idxmin da la fila con el menor MAE dentro de cada modelo que compite
entrenamiento['mejor_del_modelo'] = 'no'
compiten = entrenamiento['modelo'].isin(['MM4', 'GradientBoosting', 'LSTM_promedio'])
mejores = entrenamiento[compiten].groupby('modelo')['mae_validacion'].idxmin()
entrenamiento.loc[mejores, 'mejor_del_modelo'] = 'sí'

# G2: el vigente se elige solo entre los dos modelos de ML; MM4 es la referencia
mejor_gb = entrenamiento.loc[mejores['GradientBoosting']]
mejor_lstm = entrenamiento.loc[mejores['LSTM_promedio']]
fila_vigente = mejores['GradientBoosting'] if mejor_gb['mae_validacion'] <= mejor_lstm['mae_validacion'] \
    else mejores['LSTM_promedio']
entrenamiento['vigente'] = 'no'
entrenamiento.loc[fila_vigente, 'vigente'] = 'sí'

# ---------------------------------------------------------------------------
# 7. Reentrenamiento
# Reglas del reentrenamiento
# R1: los dos ganadores se entrenan otra vez con entrenamiento + validación juntos, con semilla 42
# R2: sin validación no hay parada temprana: se usan los árboles y las épocas que encontró la búsqueda
# R3: la escala de la LSTM sigue siendo la de entrenamiento (la que queda en el CSV)
# ---------------------------------------------------------------------------

# R1: todas las filas que no son de prueba
es_train_validacion = es_train | es_validacion

# R2: gradient boosting con los hiperparámetros ganadores, incluido el número de árboles
modelo_gb = HistGradientBoostingRegressor(**ajustes_por_combinacion[int(mejor_gb['combinacion'])],
                                          early_stopping=False, random_state=SEMILLA)
modelo_gb.fit(ventanas_xgboost.loc[es_train_validacion, variables_gb], y_relativa[es_train_validacion])

# R2 y R3: LSTM con la combinación ganadora y el promedio de épocas de sus 5 semillas
combinacion_lstm = int(mejor_lstm['combinacion'])
parametros_lstm = grilla_lstm[combinacion_lstm - 1]
modelo_lstm = crear_lstm(parametros_lstm, n_variables, SEMILLA)
modelo_lstm.fit(a_cubo(lstm_escalada[es_train_validacion], columnas_fijas),
                lstm_escalada.loc[es_train_validacion, 'y_escalada'].to_numpy(),
                epochs=epocas_por_combinacion[combinacion_lstm], batch_size=parametros_lstm['batch_size'], verbose=0)

# R3: el mínimo y el máximo de cada variable van al final del CSV, para reproducir la escala.
# El nivel mínimo también queda, porque M5 y M6 lo necesitan para volver a unidades.
for variable, (minimo, maximo) in escala.items():
    entrenamiento.loc[len(entrenamiento)] = {
        'modelo': 'LSTM_escalado', 'combinacion': None,
        'hiperparametros': f'variable={variable}; minimo={minimo:.4f}; maximo={maximo:.4f}',
        'semilla': None, 'mejor_del_modelo': 'no', 'vigente': 'no'}
entrenamiento.loc[len(entrenamiento)] = {
    'modelo': 'nivel', 'combinacion': None,
    'hiperparametros': f'nivel = media de u_t1 a u_t8 + {NIVEL_MINIMO}; prediccion = salida del modelo x nivel',
    'semilla': None, 'mejor_del_modelo': 'no', 'vigente': 'no'}

columnas_salida = ['modelo', 'combinacion', 'hiperparametros', 'semilla', 'mae_validacion',
                   'rmse_validacion', 'smape_validacion', 'accuracy_validacion', 'precision_validacion',
                   'recall_validacion', 'f1_validacion', 'mejor_del_modelo', 'vigente']
entrenamiento = entrenamiento[columnas_salida].reset_index(drop=True)
# Int64 deja la combinación y la semilla como enteros aunque tengan vacíos
entrenamiento['combinacion'] = entrenamiento['combinacion'].astype('Int64')
entrenamiento['semilla'] = entrenamiento['semilla'].astype('Int64')

# ---------------------------------------------------------------------------
# 8. Controles de calidad
# ---------------------------------------------------------------------------

conteo = entrenamiento['modelo'].value_counts().to_dict()

# C1: se probaron todas las combinaciones del diseño
assert conteo == {'GradientBoosting': 144, 'LSTM': 32, 'LSTM_semillas': 12, 'LSTM_promedio': 3,
                  'LSTM_escalado': 5, 'MM4': 1, 'nivel': 1}, f'Combinaciones probadas: {conteo}'
# C2: las métricas existen y no son negativas en todas las filas de modelos
filas_modelo = ~entrenamiento['modelo'].isin(['LSTM_escalado', 'nivel'])
metricas_modelo = entrenamiento.loc[filas_modelo, ['mae_validacion', 'rmse_validacion', 'smape_validacion',
                                                   'accuracy_validacion', 'precision_validacion',
                                                   'recall_validacion', 'f1_validacion']]
assert metricas_modelo.notna().all().all() and (metricas_modelo >= 0).all().all(), 'Hay métricas vacías o negativas'
# C3: un solo mejor por modelo y un solo vigente, que es GradientBoosting o LSTM
assert sorted(entrenamiento.loc[entrenamiento['mejor_del_modelo'] == 'sí', 'modelo']) == \
    ['GradientBoosting', 'LSTM_promedio', 'MM4'], 'Debe haber un mejor por modelo'
assert (entrenamiento['vigente'] == 'sí').sum() == 1, 'Debe haber un solo modelo vigente de ML'
# C4: los modelos reentrenados predicen todas las filas sin vacíos
assert np.isfinite(modelo_gb.predict(X_validacion_gb)).all(), 'GradientBoosting reentrenado da valores vacíos'
assert np.isfinite(modelo_lstm.predict(X_validacion_lstm, verbose=0)).all(), 'LSTM reentrenada da valores vacíos'

# Avisos: no detienen el script, solo informan

# A1: si el mejor modelo de ML no le gana a MM4 en validación, M5 probablemente tampoco lo verá en prueba
fila_mm4 = entrenamiento[entrenamiento['modelo'] == 'MM4'].iloc[0]
for nombre, fila in [('GradientBoosting', mejor_gb), ('LSTM', mejor_lstm)]:
    gana = sum(fila[m] < fila_mm4[m] for m in ['mae_validacion', 'rmse_validacion', 'smape_validacion'])
    print(f"A1: {nombre}: MAE {fila['mae_validacion']:.3f} frente a MM4 {fila_mm4['mae_validacion']:.3f}; "
          f"le gana en {gana} de 3 métricas")

# A2: cuánto varía la LSTM ganadora con 5 semillas
variacion = entrenamiento.loc[entrenamiento['modelo'].isin(['LSTM', 'LSTM_semillas']) &
                              (entrenamiento['combinacion'] == combinacion_lstm), 'mae_validacion']
print(f'A2: LSTM con 5 semillas: MAE {variacion.mean():.3f} ± {variacion.std():.3f} '
      f'(de {variacion.min():.3f} a {variacion.max():.3f})')

# A3: un pronóstico de 0 unidades en todas las filas, para saber cuánto pesan los ceros en el MAE
ceros = metricas(y_validacion, np.zeros(len(y_validacion)))
print(f"A3: pronosticar siempre 0 da MAE {ceros['mae_validacion']:.3f} y accuracy {ceros['accuracy_validacion']:.1%}; "
      f"{(y_validacion == 0).mean():.1%} de las filas de validación tiene 0 unidades")

# A4: un modelo que pronostica casi siempre 0 tiene MAE bajo pero no sirve para planear compras.
# Se comparan las unidades pronosticadas con las reales y cuántas filas quedan por debajo de 0,05.
for nombre, fila in [('GradientBoosting', mejor_gb), ('LSTM', mejor_lstm)]:
    pred = np.clip(predicciones[(fila['modelo'], int(fila['combinacion']))], 0, None)
    print(f'A4: {nombre}: pronostica {pred.sum():.0f} unidades frente a {y_validacion.sum()} reales; '
          f'{(pred < 0.05).mean():.1%} de las filas con menos de 0,05')

# Matriz de confusión de «semana con venta» en validación, para MM4 y los dos ganadores.
# VP: dijo venta y hubo; FP: dijo venta y no hubo; FN: no dijo venta y hubo; VN: no dijo venta y no hubo.
# En la LSTM se usa el promedio de las predicciones de sus 5 semillas.
matrices = []
for nombre, pred in [('MM4', pred_mm4),
                     ('GradientBoosting', predicciones[('GradientBoosting', int(mejor_gb['combinacion']))]),
                     ('LSTM', predicciones[('LSTM_promedio', int(mejor_lstm['combinacion']))])]:
    vn, fp, fn, vp = confusion_matrix(y_validacion > 0, np.clip(pred, 0, None) >= UMBRAL_VENTA).ravel()
    matrices.append({'modelo': nombre, 'vp': vp, 'fp': fp, 'fn': fn, 'vn': vn})
matriz_confusion = pd.DataFrame(matrices)
print(matriz_confusion.to_string(index=False))

# Mejor combinación de cada modelo y el vigente
totales = entrenamiento.loc[entrenamiento['mejor_del_modelo'] == 'sí',
                            ['modelo', 'combinacion', 'mae_validacion', 'rmse_validacion',
                             'smape_validacion', 'accuracy_validacion', 'f1_validacion', 'vigente']]
print(totales.to_string(index=False))
print(f"Hiperparámetros GradientBoosting: {mejor_gb['hiperparametros']}")
print(f"Hiperparámetros LSTM: {mejor_lstm['hiperparametros']}")

# ---------------------------------------------------------------------------
# 9. Exportación
# ---------------------------------------------------------------------------

# Crea la carpeta modelos si no existe
os.makedirs('../../modelos', exist_ok=True)
# joblib guarda el modelo de scikit-learn tal cual, para cargarlo en M5 con joblib.load
joblib.dump(modelo_gb, ruta_modelo_gb)
modelo_lstm.save(ruta_modelo_lstm)
# Cuatro decimales en las métricas
entrenamiento.to_csv(ruta_salida, index=False, float_format='%.4f')
matriz_confusion.to_csv(ruta_matriz, index=False)
print(f'{len(entrenamiento)} filas -> {ruta_salida}; modelos -> {ruta_modelo_gb}, {ruta_modelo_lstm}')
