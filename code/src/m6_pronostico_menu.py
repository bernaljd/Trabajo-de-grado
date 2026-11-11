"""
M6. Pronóstico de la semana siguiente e ingeniería de menú.

Entrada: dataset/processed/01_ventas_limpias.csv      (salida de M1, para el reparto por día)
         dataset/processed/02_serie_semanal.csv       (salida de M2)
         dataset/processed/03_ventanas_mm4.csv        (salida de M3: serie de cada plato modelable)
         dataset/processed/03_ventanas_lstm.csv       (solo el orden de las columnas del plato)
         dataset/processed/03_platos_modelables.csv   (salida de M3: estado de cada plato)
         dataset/processed/04_entrenamiento.csv       (salida de M4: escala de la LSTM y el MAE de validación)
         modelos/gradient_boosting.joblib, lstm.keras, arima.joblib y sarima.joblib   (salida de M4)
Salida:  dataset/processed/06_pronostico.csv          (una fila por plato, para la semana siguiente a los datos)
         dataset/processed/06_ingenieria_menu.csv     (una fila por plato y escenario)
Especificación: pipeline/DISENO_PIPELINE.md (sección M6) y pipeline/M6_pronostico_menu.md

Pasos:
  1. Ingesta y validación: leer los CSV de M1 a M4 y los cuatro modelos.
  2. Semana siguiente y estado de los platos: fijar la semana a pronosticar y decidir qué platos se
     pronostican con modelo, con media móvil o no se pronostican.
  3. Modelo elegido: el de menor MAE en validación entre los cinco que dejó M4.
  4. Pronósticos: los cinco modelos pronostican la semana siguiente de cada plato modelable.
  5. Pronóstico final y reparto por día: el modelo elegido da el número de cada plato, que se reparte de lunes a sábado.
  6. Ingeniería de menú: matriz histórica (últimas 12 semanas) y proyectada (semana pronosticada).
  7. Controles de calidad: comprobar filas, sumas y clasificaciones.
  8. Exportación: escribir 06_pronostico.csv y 06_ingenieria_menu.csv.

Si un control falla, el script se detiene con un AssertionError y no escribe la salida.
Este script no mide el error de nada: el error de los modelos está en 04_entrenamiento.csv y 05_metricas.csv.

Uso (desde code/src, después de M5):
  python3 m6_pronostico_menu.py
"""

import os
import joblib # type: ignore
import warnings

# Quita los mensajes de arranque de TensorFlow, que no dicen nada de los datos
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd # type: ignore
import numpy as np # type: ignore
import sklearn # type: ignore
import statsmodels # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX # type: ignore
import tensorflow as tf # type: ignore

# Las mismas constantes de M4 y M5: si cambian allá, cambian aquí
HISTORIA_LSTM = 8
VARIABLES_LSTM = ['ur', 'precio', 'sin', 'cos']
NIVEL_MINIMO = 0.1
# Semanas del histórico de la matriz de menú y del reparto por día
SEMANAS_MENU = 12
SEMANAS_REPARTO = 26
# Un plato con menos de 20 unidades en esas semanas usa la participación diaria de su categoría
UNIDADES_MINIMAS_REPARTO = 20
# Un plato sin ventas en las últimas 26 semanas se considera retirado, igual que en M3
SEMANAS_RETIRO = 26
# La popularidad es alta si el plato vende al menos el 70 % de lo que vendería en un menú parejo (1 / N)
FACTOR_POPULARIDAD = 0.7
# Días que muestra el dashboard (el domingo no se muestra)
DIAS = ['lun', 'mar', 'mie', 'jue', 'vie', 'sab']
# Modelos que compiten: nombre en 04_entrenamiento.csv y columna de predicción
MODELOS = {'MM4': 'pred_mm4', 'GradientBoosting': 'pred_gradient_boosting', 'LSTM_promedio': 'pred_lstm',
           'ARIMA': 'pred_arima', 'SARIMA': 'pred_sarima'}
NOMBRE_METODO = {'MM4': 'MM4', 'GradientBoosting': 'GradientBoosting', 'LSTM_promedio': 'LSTM',
                 'ARIMA': 'ARIMA', 'SARIMA': 'SARIMA'}
ACCIONES = {'Estrella': 'Mantener precio, porción y visibilidad en la carta',
            'Caballo de arado': 'Revisar el precio o la porción, o combinarlo con platos de mayor margen',
            'Puzzle': 'Promoverlo, reubicarlo en la carta o recomendarlo en el servicio',
            'Perro': 'Evaluar retirarlo o reemplazarlo'}


def a_cubo(tabla, columnas_fijas):
    # Igual que en M4 y M5: entrada de la LSTM con forma (muestras, 8 pasos, variables), de t-8 a t-1
    pasos = np.stack([tabla[[f'{variable}_t{k}' for k in range(HISTORIA_LSTM, 0, -1)]].to_numpy()
                      for variable in VARIABLES_LSTM], axis=2)
    fijas = np.repeat(tabla[columnas_fijas].to_numpy()[:, np.newaxis, :], HISTORIA_LSTM, axis=1)
    return np.concatenate([pasos, fijas], axis=2).astype('float32')


def repartir(total, participaciones):
    # Reparte un total entero entre los días con el método del mayor residuo: cada día recibe la parte
    # entera de su cuota y las unidades que sobran van a los días con mayor decimal. La suma da el total exacto.
    cuotas = total * np.asarray(participaciones, dtype=float)
    base = np.floor(cuotas + 1e-9).astype(int)
    sobran = int(total - base.sum())
    orden = np.argsort(-(cuotas - base), kind='stable')
    base[orden[:sobran]] += 1
    return base


def clasificar(unidades, margen_unitario):
    # Kasavana y Smith: margen alto si es al menos el margen ponderado del menú; popularidad alta si es al
    # menos el 70 % de 1/N. Devuelve la tabla con popularidad, umbrales y clasificación.
    tabla = pd.DataFrame({'unidades': unidades, 'margen_unitario': margen_unitario})
    n = len(tabla)
    total = tabla['unidades'].sum()
    umbral_margen = (tabla['margen_unitario'] * tabla['unidades']).sum() / total
    umbral_popularidad = 100 * FACTOR_POPULARIDAD / n
    tabla['popularidad_pct'] = 100 * tabla['unidades'] / total
    tabla['umbral_margen'] = umbral_margen
    tabla['umbral_popularidad'] = umbral_popularidad
    margen_alto = tabla['margen_unitario'] >= umbral_margen
    popularidad_alta = tabla['popularidad_pct'] >= umbral_popularidad
    tabla['clasificacion'] = np.select(
        [margen_alto & popularidad_alta, ~margen_alto & popularidad_alta, margen_alto & ~popularidad_alta],
        ['Estrella', 'Caballo de arado', 'Puzzle'], default='Perro')
    return tabla


# ---------------------------------------------------------------------------
# 1. Ingesta y validación
# ---------------------------------------------------------------------------

# El script se ejecuta desde code/src, así que los datos están dos carpetas más arriba
ruta_ventas = '../../dataset/processed/01_ventas_limpias.csv'
ruta_serie = '../../dataset/processed/02_serie_semanal.csv'
ruta_mm4 = '../../dataset/processed/03_ventanas_mm4.csv'
ruta_lstm = '../../dataset/processed/03_ventanas_lstm.csv'
ruta_platos = '../../dataset/processed/03_platos_modelables.csv'
ruta_entrenamiento = '../../dataset/processed/04_entrenamiento.csv'
ruta_modelo_gb = '../../modelos/gradient_boosting.joblib'
ruta_modelo_lstm = '../../modelos/lstm.keras'
ruta_modelo_arima = '../../modelos/arima.joblib'
ruta_modelo_sarima = '../../modelos/sarima.joblib'
ruta_pronostico = '../../dataset/processed/06_pronostico.csv'
ruta_menu = '../../dataset/processed/06_ingenieria_menu.csv'

ventas = pd.read_csv(ruta_ventas, parse_dates=['fecha'])
serie = pd.read_csv(ruta_serie, parse_dates=['semana_inicio'])
mm4 = pd.read_csv(ruta_mm4, parse_dates=['semana_objetivo'])
# Solo los nombres de las columnas, para repetir el orden del plato 0/1 que usó la LSTM
columnas_plato = [c for c in pd.read_csv(ruta_lstm, nrows=0).columns if c.startswith('plato_')]
platos_m3 = pd.read_csv(ruta_platos)
entrenamiento = pd.read_csv(ruta_entrenamiento)
modelo_gb = joblib.load(ruta_modelo_gb)
modelo_lstm = tf.keras.models.load_model(ruta_modelo_lstm)
modelos_arima = {'ARIMA': joblib.load(ruta_modelo_arima), 'SARIMA': joblib.load(ruta_modelo_sarima)}

print('M6 Pronóstico e ingeniería de menú')
print(f'Versiones: pandas {pd.__version__}, scikit-learn {sklearn.__version__}, tensorflow {tf.__version__}, '
      f'statsmodels {statsmodels.__version__}')

columnas_serie = ['semana_inicio', 'plato', 'categoria', 'unidades', 'ventas', 'costo',
                  'margen', 'precio_unitario', 'activo']

# C1 los archivos no son vacíos
assert min(len(ventas), len(serie), len(mm4), len(platos_m3), len(entrenamiento)) > 0, 'Hay un archivo sin filas'
# C2 la serie tiene las columnas de M2 y una fila por plato y semana
assert list(serie.columns) == columnas_serie, 'Las columnas de la serie no son las de M2'
assert not serie.duplicated(['plato', 'semana_inicio']).any(), 'Hay pares plato-semana repetidos'
# C3 M4 dejó una combinación ganadora para cada uno de los cinco modelos
ganadores = entrenamiento[entrenamiento['mejor_del_modelo'] == 'sí']
assert sorted(ganadores['modelo']) == sorted(MODELOS), 'Faltan modelos ganadores en 04_entrenamiento.csv'
# C4 cada plato de la serie está en el archivo de estados de M3
assert set(serie['plato']) == set(platos_m3['plato']), 'Los platos de M2 y M3 no coinciden'

# ---------------------------------------------------------------------------
# 2. Semana siguiente y estado de los platos
# Reglas del pronóstico
# E1: la semana a pronosticar es la que sigue a la última semana de la serie
# E2: un plato sin ventas en las últimas 26 semanas es 'retirado' y no se pronostica (regla P2 de M3,
#     aplicada con todos los datos)
# E3: un plato que M3 dejó como 'modelable' y sigue vendiendo se pronostica con los modelos
# E4: un plato de 'historial_insuficiente' que sigue vendiendo se pronostica con la media de sus 4 semanas
#     anteriores (PRED-BASE)
# ---------------------------------------------------------------------------

ultima_semana = serie['semana_inicio'].max()
semana_pronostico = ultima_semana + pd.Timedelta(days=7)
# C5 la semana a pronosticar empieza en lunes
assert semana_pronostico.weekday() == 0, 'La semana a pronosticar no empieza en lunes'

semanas = sorted(serie['semana_inicio'].unique())
inicio_retiro = semanas[-SEMANAS_RETIRO]
# E2: platos con ventas en las últimas 26 semanas
vendio_reciente = serie[(serie['semana_inicio'] >= inicio_retiro) & (serie['unidades'] > 0)]['plato'].unique()

platos = serie.groupby('plato').agg(categoria=('categoria', 'first')).reset_index()
platos = platos.merge(platos_m3[['plato', 'estado_plato']].rename(columns={'estado_plato': 'estado_m3'}), on='plato')
platos['estado_plato'] = np.where(platos['plato'].isin(vendio_reciente), platos['estado_m3'], 'retirado')
platos = platos.drop(columns='estado_m3')

ultima = serie[serie['semana_inicio'] == ultima_semana].set_index('plato')
platos['unidades_ultima_semana'] = platos['plato'].map(ultima['unidades']).astype(int)
platos = platos.sort_values('plato').reset_index(drop=True)
modelables = platos.loc[platos['estado_plato'] == 'modelable', 'plato'].tolist()
# C6 queda al menos un plato modelable
assert len(modelables) > 0, 'Ningún plato modelable para pronosticar'

# ---------------------------------------------------------------------------
# 3. Modelo elegido
# Reglas de la elección
# G1: de los cinco modelos que dejó M4, el de menor MAE en validación es el elegido. Solo se mira la
#     validación: la prueba ya se usó en M5 y no vuelve a decidir nada
# G2: si hay empate, gana el primero de la lista MM4, GradientBoosting, LSTM, ARIMA, SARIMA
# ---------------------------------------------------------------------------

mae_validacion = ganadores.set_index('modelo')['mae_validacion'].reindex(list(MODELOS))
modelo_elegido = mae_validacion.idxmin()
print(f"Modelo elegido: {NOMBRE_METODO[modelo_elegido]} (MAE de validación {mae_validacion[modelo_elegido]:.3f})")

# ---------------------------------------------------------------------------
# 4. Pronósticos de los cinco modelos
# Reglas de los pronósticos (las mismas de M4 y M5)
# N1: nivel = media de las 8 semanas anteriores del plato + 0,1
# N2: GradientBoosting y LSTM predicen unidades relativas al nivel; la salida × nivel da unidades
# F1: MM4 = promedio de las 4 semanas anteriores
# F2: GradientBoosting recibe las mismas variables y en el mismo orden de M4
# F3: la LSTM usa la escala de entrenamiento guardada en 04_entrenamiento.csv
# F4: ARIMA y SARIMA usan los parámetros por plato de M4 y pronostican una semana adelante con la
#     serie completa del plato; un plato sin parámetros usa MM4
# F5: una predicción negativa se lleva a 0
# ---------------------------------------------------------------------------

# Las 8 semanas anteriores de cada plato modelable, de la más reciente (t1) a la más antigua (t8)
ventana = []
for plato in modelables:
    filas = serie[(serie['plato'] == plato) & (serie['activo'] == 1)].sort_values('semana_inicio').tail(HISTORIA_LSTM)
    # C7 la ventana son 8 semanas seguidas que terminan en la última semana de los datos
    assert len(filas) == HISTORIA_LSTM and filas['semana_inicio'].iloc[-1] == ultima_semana and \
        (filas['semana_inicio'].diff().dropna() == pd.Timedelta(days=7)).all(), f'{plato}: ventana incompleta'
    registro = {'plato': plato, 'categoria': filas['categoria'].iloc[0]}
    semana_anio = filas['semana_inicio'].dt.isocalendar().week.astype(int).to_numpy()
    for k in range(1, HISTORIA_LSTM + 1):
        # Posición -k: t1 es la semana más reciente
        registro[f'u_t{k}'] = int(filas['unidades'].iloc[-k])
        registro[f'precio_t{k}'] = float(filas['precio_unitario'].iloc[-k])
        registro[f'sin_t{k}'] = np.sin(2 * np.pi * semana_anio[-k] / 52)
        registro[f'cos_t{k}'] = np.cos(2 * np.pi * semana_anio[-k] / 52)
    ventana.append(registro)
ventana = pd.DataFrame(ventana)
# C8 sin vacíos en las ventanas (un plato activo siempre tiene precio)
assert ventana.notna().all().all(), 'Hay valores vacíos en las ventanas de la semana siguiente'

# N1: nivel de cada plato
media_8 = ventana[[f'u_t{k}' for k in range(1, HISTORIA_LSTM + 1)]].mean(axis=1)
nivel = (media_8 + NIVEL_MINIMO).to_numpy()
pronosticos = pd.DataFrame({'plato': ventana['plato']})

# F1: MM4
pronosticos['pred_mm4'] = ventana[[f'u_t{k}' for k in range(1, 5)]].mean(axis=1).to_numpy()

# F2: variables del gradient boosting, en el orden del modelo guardado. El calendario es el de la semana a pronosticar.
semana_anio_pronostico = int(semana_pronostico.isocalendar().week)
variables_gb = list(modelo_gb.feature_names_in_)
X_gb = pd.DataFrame(0.0, index=ventana.index, columns=variables_gb)
for k in range(1, 5):
    X_gb[f'u_t{k}'] = ventana[f'u_t{k}']
X_gb['media_4'] = pronosticos['pred_mm4']
X_gb['semana_sin'] = np.sin(2 * np.pi * semana_anio_pronostico / 52)
X_gb['semana_cos'] = np.cos(2 * np.pi * semana_anio_pronostico / 52)
X_gb['mes'] = semana_pronostico.month
X_gb['precio_t1'] = ventana['precio_t1']
X_gb['media_8'] = media_8
for indice, fila in ventana.iterrows():
    for columna in [f"plato_{fila['plato']}", f"categoria_{fila['categoria']}"]:
        assert columna in X_gb.columns, f'El modelo no conoce {columna}'
        X_gb.loc[indice, columna] = 1
# N2: salida relativa por el nivel
pronosticos['pred_gradient_boosting'] = modelo_gb.predict(X_gb) * nivel

# F3: escala de la LSTM, leída de 'variable=ur; minimo=0.0000; maximo=6.8966'
escala = {}
for texto in entrenamiento.loc[entrenamiento['modelo'] == 'LSTM_escalado', 'hiperparametros']:
    partes = dict(parte.split('=') for parte in texto.split('; '))
    escala[partes['variable']] = (float(partes['minimo']), float(partes['maximo']))
lstm_escalada = ventana.copy()
for k in range(1, HISTORIA_LSTM + 1):
    lstm_escalada[f'ur_t{k}'] = lstm_escalada[f'u_t{k}'] / nivel
lstm_escalada['nivel'] = nivel
for variable, (minimo, maximo) in escala.items():
    columnas = ['nivel'] if variable == 'nivel' else [f'{variable}_t{k}' for k in range(1, HISTORIA_LSTM + 1)]
    lstm_escalada[columnas] = (lstm_escalada[columnas] - minimo) / (maximo - minimo)
for columna in columnas_plato:
    lstm_escalada[columna] = (columna == 'plato_' + lstm_escalada['plato']).astype(int)
minimo_ur, maximo_ur = escala['ur']
relativa = modelo_lstm.predict(a_cubo(lstm_escalada, columnas_plato + ['nivel']), verbose=0).ravel() \
    * (maximo_ur - minimo_ur) + minimo_ur
pronosticos['pred_lstm'] = relativa * nivel

# F4: ARIMA y SARIMA, plato por plato. La serie es la que usó M4 (las filas de 03_ventanas_mm4.csv)
# hasta la última semana de los datos; los parámetros quedan fijos (filter) y forecast da la semana siguiente.
series_modelo = {plato: filas.sort_values('semana_objetivo')['y_real'].to_numpy(dtype=float)
                 for plato, filas in mm4.groupby('plato')}
sin_parametros = {}
for nombre_modelo, columna in [('ARIMA', 'pred_arima'), ('SARIMA', 'pred_sarima')]:
    guardado = modelos_arima[nombre_modelo]
    orden, estacional = guardado['orden'], guardado['estacional']
    tendencia = 'c' if orden[1] == 0 and estacional[1] == 0 else 'n'
    valores, usa_mm4 = [], 0
    for plato, valor_mm4 in zip(pronosticos['plato'], pronosticos['pred_mm4']):
        parametros = guardado['parametros'].get(plato)
        valor = np.nan
        if parametros is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    valor = float(SARIMAX(series_modelo[plato], order=orden, seasonal_order=estacional,
                                          trend=tendencia).filter(parametros).forecast(1)[0])
            except Exception:
                valor = np.nan
        if not np.isfinite(valor):
            # Sin parámetros o con un valor no finito: MM4
            valor, usa_mm4 = valor_mm4, usa_mm4 + 1
        valores.append(valor)
    pronosticos[columna] = valores
    sin_parametros[nombre_modelo] = usa_mm4

# F5: sin predicciones negativas
columnas_pred = list(MODELOS.values())
pronosticos[columnas_pred] = pronosticos[columnas_pred].clip(lower=0)

# ---------------------------------------------------------------------------
# 5. Pronóstico final y reparto por día
# Reglas del pronóstico final
# H1: modelable: el pronóstico es el del modelo elegido (PRED-ML); historial_insuficiente: la media de sus
#     4 semanas anteriores (PRED-BASE); retirado: sin pronóstico
# H2: el pronóstico semanal se redondea a unidades enteras (0,5 sube) y se reparte entre lunes y sábado según la
#     participación de cada día en las últimas 26 semanas del plato. Con menos de 20 unidades en ese periodo se usa
#     la participación de su categoría, y si la categoría tampoco tiene, la de todo el menú
# H3: el reparto usa el método del mayor residuo, así la suma de los días es exactamente la semana
# ---------------------------------------------------------------------------

resultado = platos.copy()
resultado['semana_pronostico'] = semana_pronostico
resultado = resultado.merge(pronosticos, on='plato', how='left')

# H1: la media móvil de los platos de historial insuficiente sale de sus últimas 4 semanas
ultimas_4 = (serie[serie['semana_inicio'] > ultima_semana - pd.Timedelta(days=28)]
             .groupby('plato')['unidades'].mean())
es_base = resultado['estado_plato'] == 'historial_insuficiente'
es_modelo = resultado['estado_plato'] == 'modelable'
resultado['unidades_pronosticadas'] = np.nan
resultado.loc[es_modelo, 'unidades_pronosticadas'] = resultado.loc[es_modelo, MODELOS[modelo_elegido]]
resultado.loc[es_base, 'unidades_pronosticadas'] = resultado.loc[es_base, 'plato'].map(ultimas_4)
resultado['metodo'] = np.select([es_modelo, es_base], [NOMBRE_METODO[modelo_elegido], 'media_movil'], default='ninguno')
# Los platos que no se pronostican con modelo no llevan predicciones de modelos
resultado.loc[~es_modelo, columnas_pred] = np.nan

# H2: unidades enteras (0,5 sube, no el redondeo a par de numpy)
enteras = np.floor(resultado['unidades_pronosticadas'] + 0.5)
resultado['unidades_redondeadas'] = enteras.astype('Int64')

# H2: participación de cada día en las últimas 26 semanas (lunes a sábado)
inicio_reparto = ultima_semana - pd.Timedelta(days=7 * (SEMANAS_REPARTO - 1))
periodo = ventas[ventas['fecha'] >= inicio_reparto].copy()
periodo['dia'] = periodo['fecha'].dt.weekday
periodo = periodo[periodo['dia'] <= 5]
por_plato_dia = periodo.pivot_table(index='plato', columns='dia', values='cantidad', aggfunc='sum', fill_value=0) \
    .reindex(columns=range(6), fill_value=0)
categoria_de = serie.groupby('plato')['categoria'].first()
por_categoria_dia = por_plato_dia.groupby(categoria_de.reindex(por_plato_dia.index)).sum()
total_dia = por_plato_dia.sum()


def participacion(plato, categoria):
    # Del plato si vendió lo suficiente en el periodo; si no, de su categoría; si no, de todo el menú
    for tabla, llave in [(por_plato_dia, plato), (por_categoria_dia, categoria)]:
        if llave in tabla.index and tabla.loc[llave].sum() >= UNIDADES_MINIMAS_REPARTO:
            fila = tabla.loc[llave]
            return (fila / fila.sum()).to_numpy(), 'plato' if tabla is por_plato_dia else 'categoria'
    return (total_dia / total_dia.sum()).to_numpy(), 'menu'


for dia in DIAS:
    resultado[dia] = pd.array([pd.NA] * len(resultado), dtype='Int64')
resultado['reparto_dia_segun'] = ''
for indice, fila in resultado.iterrows():
    if pd.isna(fila['unidades_redondeadas']):
        continue
    pesos, origen = participacion(fila['plato'], fila['categoria'])
    resultado.loc[indice, DIAS] = repartir(int(fila['unidades_redondeadas']), pesos)
    resultado.loc[indice, 'reparto_dia_segun'] = origen

# ---------------------------------------------------------------------------
# 6. Ingeniería de menú (Kasavana y Smith)
# Reglas de la matriz
# I1: histórico: las últimas 12 semanas de la serie; entran los platos con unidades > 0
# I2: proyectado: las unidades pronosticadas de la semana (con decimales: con 0 a 2 unidades por plato, redondear
#     borraría la mayoría de los platos), con el margen unitario del histórico; entran los platos con pronóstico mayor que 0
# I3: margen unitario = margen / unidades del plato en las últimas 12 semanas. Si el plato no vendió en ellas
#     se usan las últimas 26 semanas, y si tampoco, todo el historial
# I4: margen alto si el margen unitario es al menos el margen ponderado del menú (Σ margen × unidades / Σ unidades);
#     popularidad alta si la participación es al menos 70 % de 1/N, con N los platos de la matriz
# I5: cambia_de_categoria = sí si la clasificación proyectada del plato difiere de la histórica
# ---------------------------------------------------------------------------

semanas_menu = serie[serie['semana_inicio'] > ultima_semana - pd.Timedelta(days=7 * SEMANAS_MENU)]
historico = semanas_menu.groupby('plato').agg(unidades=('unidades', 'sum'), margen=('margen', 'sum'))
historico = historico[historico['unidades'] > 0]
historico['margen_unitario'] = historico['margen'] / historico['unidades']

# I3: margen unitario para los platos que no vendieron en las últimas 12 semanas
margen_unitario = historico['margen_unitario'].copy()
for semanas_atras in [SEMANAS_RETIRO, None]:
    base = serie if semanas_atras is None else serie[serie['semana_inicio'] > ultima_semana - pd.Timedelta(days=7 * semanas_atras)]
    suma = base.groupby('plato')[['unidades', 'margen']].sum()
    suma = suma[suma['unidades'] > 0]
    faltan = suma.index.difference(margen_unitario.index)
    margen_unitario = pd.concat([margen_unitario, suma.loc[faltan, 'margen'] / suma.loc[faltan, 'unidades']])

# C9 hay platos vendidos en el histórico para armar la matriz
assert len(historico) > 0, 'No hay ventas en las últimas 12 semanas'

matriz_historica = clasificar(historico['unidades'], historico['margen_unitario']).reset_index()
matriz_historica['escenario'] = 'historico'

proyectados = resultado[resultado['unidades_pronosticadas'].fillna(0) > 0].set_index('plato')['unidades_pronosticadas']
# Un plato proyectado sin ventas en todo el historial no tiene margen: no entra a la matriz
proyectados = proyectados[proyectados.index.isin(margen_unitario.index)]
if len(proyectados) > 0:
    matriz_proyectada = clasificar(proyectados, margen_unitario.reindex(proyectados.index)).reset_index()
    matriz_proyectada['escenario'] = 'proyectado'
    matriz = pd.concat([matriz_historica, matriz_proyectada], ignore_index=True)
else:
    matriz = matriz_historica

# I5: clasificación histórica de cada plato, para marcar los cambios
clase_historica = matriz_historica.set_index('plato')['clasificacion']
matriz['cambia_de_categoria'] = np.where(
    (matriz['escenario'] == 'proyectado') & matriz['plato'].map(clase_historica).notna() &
    (matriz['plato'].map(clase_historica) != matriz['clasificacion']), 'sí', 'no')
matriz['accion_sugerida'] = matriz['clasificacion'].map(ACCIONES)
columnas_menu = ['escenario', 'plato', 'unidades', 'margen_unitario', 'popularidad_pct', 'umbral_margen',
                 'umbral_popularidad', 'clasificacion', 'cambia_de_categoria', 'accion_sugerida']
matriz = matriz[columnas_menu]

# ---------------------------------------------------------------------------
# 7. Controles de calidad
# ---------------------------------------------------------------------------

columnas_pronostico = ['plato', 'categoria', 'estado_plato', 'semana_pronostico', 'unidades_ultima_semana',
                       'unidades_pronosticadas', 'unidades_redondeadas', 'metodo'] + columnas_pred + DIAS + \
                      ['reparto_dia_segun']
resultado = resultado[columnas_pronostico]

# C1: una fila por plato de la serie, sin repetidos
assert len(resultado) == serie['plato'].nunique() and not resultado['plato'].duplicated().any(), \
    'Debe haber una fila por plato'
# C2: pronóstico vacío solo en los retirados, y nunca negativo
assert resultado.loc[resultado['estado_plato'] != 'retirado', 'unidades_pronosticadas'].notna().all(), \
    'Hay platos sin pronóstico que no son retirados'
assert resultado.loc[resultado['estado_plato'] == 'retirado', 'unidades_pronosticadas'].isna().all(), \
    'Un plato retirado tiene pronóstico'
assert (resultado['unidades_pronosticadas'].dropna() >= 0).all(), 'Hay pronósticos negativos'
# C3: los cinco modelos dan valores finitos y no negativos en todos los platos modelables
valores = resultado.loc[resultado['estado_plato'] == 'modelable', columnas_pred]
assert np.isfinite(valores.to_numpy(dtype=float)).all() and (valores >= 0).all().all(), \
    'Hay pronósticos de modelos vacíos o negativos'
# C4: los días suman exactamente las unidades de la semana
con_reparto = resultado['unidades_redondeadas'].notna()
assert (resultado.loc[con_reparto, DIAS].sum(axis=1) == resultado.loc[con_reparto, 'unidades_redondeadas']).all(), \
    'Los días no suman la semana'
# C5: la matriz tiene las cuatro clasificaciones válidas, y la popularidad suma 100 % en cada escenario
assert set(matriz['clasificacion']) <= set(ACCIONES), 'Hay clasificaciones desconocidas'
assert (matriz.groupby('escenario')['popularidad_pct'].sum().round(6) == 100).all(), 'La popularidad no suma 100 %'
assert not matriz.duplicated(['escenario', 'plato']).any(), 'Hay platos repetidos en un escenario'
# C6: los márgenes unitarios son positivos (el costo no supera al precio)
assert (matriz['margen_unitario'] > 0).all(), 'Hay márgenes unitarios no positivos'
# C7: el margen ponderado de cada escenario se calcula igual por separado
for escenario, grupo in matriz.groupby('escenario'):
    ponderado = (grupo['margen_unitario'] * grupo['unidades']).sum() / grupo['unidades'].sum()
    assert np.allclose(grupo['umbral_margen'], ponderado), 'El umbral de margen no coincide'

# Avisos: no detienen el script, solo informan

# A1: estado de los platos y método con el que se pronostica cada uno
print(f"A1: platos: {resultado['estado_plato'].value_counts().to_dict()}; métodos: {resultado['metodo'].value_counts().to_dict()}")
# A2: MAE de validación de los cinco modelos, de donde sale el elegido
print('A2: MAE de validación: ' + '; '.join(f'{NOMBRE_METODO[m]} {v:.3f}' for m, v in mae_validacion.items()))
# A3: total pronosticado frente a la última semana y a la media de las últimas 4
print(f"A3: semana {semana_pronostico.date()}: {resultado['unidades_pronosticadas'].sum():.1f} unidades pronosticadas; "
      f"última semana {int(resultado['unidades_ultima_semana'].sum())}; "
      f"media de las últimas 4 semanas {serie[serie['semana_inicio'] > ultima_semana - pd.Timedelta(days=28)]['unidades'].sum() / 4:.1f}")
# A4: total de la semana pronosticado por cada modelo (platos modelables)
print('A4: total por modelo: ' + '; '.join(f'{NOMBRE_METODO[m]} {valores[c].sum():.1f}' for m, c in MODELOS.items()))
# A5: la última semana de los datos puede estar incompleta (los datos terminan a mitad de semana)
ultimo_dia = ventas['fecha'].max()
print(f"A5: los datos llegan al {ultimo_dia.date()} ({ultimo_dia.day_name()}); la última semana empieza el {ultima_semana.date()}")
# A6: cuántos platos tienen ARIMA o SARIMA reemplazado por MM4 y platos de la matriz por clasificación
print(f'A6: platos con MM4 en lugar de ARIMA/SARIMA: {sin_parametros}')
print(matriz.groupby(['escenario', 'clasificacion']).size().unstack(fill_value=0).to_string())
print(f"Cambian de categoría: {(matriz['cambia_de_categoria'] == 'sí').sum()}")

# ---------------------------------------------------------------------------
# 8. Exportación
# ---------------------------------------------------------------------------

resultado.to_csv(ruta_pronostico, index=False, date_format='%Y-%m-%d', float_format='%.4f')
matriz.to_csv(ruta_menu, index=False, float_format='%.4f')
print(f'{len(resultado)} filas -> {ruta_pronostico}; {len(matriz)} filas -> {ruta_menu}')
