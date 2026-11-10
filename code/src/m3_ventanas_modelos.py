"""
M3. Partición temporal y ventanas por modelo.

Entrada: dataset/processed/02_serie_semanal.csv        (salida de M2)
Salida:  dataset/processed/03_particion_semanas.csv    (una fila por semana, con su conjunto)
         dataset/processed/03_platos_modelables.csv    (una fila por plato, con su estado)
         dataset/processed/03_ventanas_mm4.csv         (una fila por plato y semana a predecir)
         dataset/processed/03_ventanas_xgboost.csv     (mismas filas)
         dataset/processed/03_ventanas_lstm.csv        (mismas filas)
Especificación: pipeline/DISENO_PIPELINE.md (sección M3)

Pasos:
  1. Ingesta y validación: leer la serie de M2 y revisar que cumple su contrato.
  2. Partición temporal: las últimas 26 semanas a prueba, las 26 anteriores a validación y
     el resto a entrenamiento, en orden cronológico, con TimeSeriesSplit de scikit-learn.
     Con el dataset de la tesis (157 semanas) da 105 / 26 / 26.
  3. Selección de platos: aplicar la regla de estado_plato con lo que se sabe al final
     del entrenamiento y quedarse con los modelables.
  4. Ventanas: para cada semana a predecir, tomar las 8 semanas activas anteriores.
  5. Archivos por modelo: elegir las columnas de MM4, XGBoost y LSTM.
  6. Controles de calidad: comprobar filas, partición y valores.
  7. Exportación: escribir los cinco CSV.

Si un control falla, el script se detiene con un AssertionError y no escribe la salida.

Uso (desde code/src, después de M2):
  python3 m3_ventanas_modelos.py
"""

import pandas as pd # type: ignore 
import numpy as np # type: ignore 
from sklearn.model_selection import TimeSeriesSplit # type: ignore 

# Tamaños de la partición, en semanas. Validación y prueba son fijos (un semestre cada uno);
# entrenamiento es lo que queda antes, así la partición sirve con cualquier largo de datos.
# Con el dataset de la tesis: 157 - 26 - 26 = 105 semanas de entrenamiento.
SEMANAS_VALIDACION = 26
SEMANAS_PRUEBA = 26
# Entrenamiento necesita al menos un año, para que cada semana del año (la estacionalidad
# que codifican semana_sin y semana_cos) aparezca al menos una vez antes de validar.
SEMANAS_MINIMAS_ENTRENAMIENTO = 52
# Semanas de historia que usa cada modelo. La LSTM usa la ventana más larga.
HISTORIA_MM4 = 4
HISTORIA_LSTM = 8
# Un plato necesita al menos 26 semanas de historia para entrar a los modelos.
SEMANAS_MINIMAS = 26

# Clasifica si un plato es modelable o no
def clasificar_plato(primera_venta, ultima_venta, fin_entrenamiento, inicio_ultimas):
    # Semanas desde la primera venta hasta el último lunes de entrenamiento (contando ambas).
    # Un plato que no vendió nada en entrenamiento queda con 0 semanas de historia.
    historia = ((fin_entrenamiento - primera_venta).dt.days // 7 + 1).fillna(0).astype(int)
    # Se parte de que todos son modelables y luego se aplican las dos excepciones.
    estado = pd.Series('modelable', index=primera_venta.index)
    # menos de 26 semanas de historia.
    estado[historia < SEMANAS_MINIMAS] = 'historial_insuficiente'
    # vendió alguna vez, pero nada en las últimas 26 semanas.   
    estado[ultima_venta < inicio_ultimas] = 'retirado'
    return historia, estado


# ---------------------------------------------------------------------------
# 1. Ingesta y validación
# ---------------------------------------------------------------------------

ruta_entrada = '../../dataset/processed/02_serie_semanal.csv'
ruta_particion = '../../dataset/processed/03_particion_semanas.csv'
ruta_platos = '../../dataset/processed/03_platos_modelables.csv'
ruta_mm4 = '../../dataset/processed/03_ventanas_mm4.csv'
ruta_xgboost = '../../dataset/processed/03_ventanas_xgboost.csv'
ruta_lstm = '../../dataset/processed/03_ventanas_lstm.csv'

# parse_dates convierte semana_inicio a fecha al leer, para poder ordenar y restar fechas
df = pd.read_csv(ruta_entrada, parse_dates=['semana_inicio'])

print('M3 Partición temporal y ventanas')
df.info()

columnas_entrada = ['semana_inicio', 'plato', 'categoria', 'unidades', 'ventas', 'costo',
                    'margen', 'precio_unitario', 'activo']

# C1 el archivo no es vacío
assert len(df) > 0, 'El archivo no tiene filas'
# C2 las columnas son las que deja M2
assert list(df.columns) == columnas_entrada, 'Las columnas no son las de M2'
# C3 no hay dos filas para el mismo plato y semana
assert not df.duplicated(['plato', 'semana_inicio']).any(), 'Hay pares plato-semana repetidos'
# C4 hay semanas suficientes: al menos un año de entrenamiento más validación y prueba
# (52 + 26 + 26 = 104 semanas). Con menos, los modelos no tendrían con qué aprender.
semanas = pd.Series(sorted(df['semana_inicio'].unique()))
semanas_minimas = SEMANAS_MINIMAS_ENTRENAMIENTO + SEMANAS_VALIDACION + SEMANAS_PRUEBA
assert len(semanas) >= semanas_minimas, f'Semanas: {len(semanas)}, se necesitan al menos {semanas_minimas}'
# Entrenamiento es lo que queda antes de validación y prueba (105 con el dataset de la tesis)
semanas_entrenamiento = len(semanas) - SEMANAS_VALIDACION - SEMANAS_PRUEBA

# ---------------------------------------------------------------------------
# 2. Partición temporal
# Reglas de partición
# T1: se parten las semanas, no las filas: una semana queda entera en un solo conjunto
# T2: el orden es cronológico: entrenamiento < validación < prueba
# T3: las últimas 26 semanas a prueba, las 26 anteriores a validación y el resto a entrenamiento
# ---------------------------------------------------------------------------

# T1 y T2: TimeSeriesSplit recibe una fila por semana y siempre deja el pasado para entrenar.
# Con 2 cortes de 26 semanas, el primero da (resto) + 26 y el segundo (resto + 26) + 26.
# Con el dataset de la tesis: 105 + 26 y 131 + 26.
cortes = list(TimeSeriesSplit(n_splits=2, test_size=SEMANAS_PRUEBA).split(semanas))
indices_train, indices_validacion = cortes[0]
indices_hasta_validacion, indices_prueba = cortes[1]

# T3: los cortes tienen el tamaño del diseño y la prueba empieza justo después de validación
assert len(indices_train) == semanas_entrenamiento, f'Entrenamiento: {len(indices_train)}'
assert len(indices_validacion) == SEMANAS_VALIDACION, f'Validación: {len(indices_validacion)}'
assert len(indices_prueba) == SEMANAS_PRUEBA, f'Prueba: {len(indices_prueba)}'
assert indices_hasta_validacion[-1] == indices_validacion[-1], 'La prueba no sigue a la validación'

# Tabla de una fila por semana con su conjunto. Las ventanas la consultan por semana_objetivo.
particion = pd.DataFrame({'semana_inicio': semanas, 'conjunto': 'entrenamiento'})
particion.loc[indices_validacion, 'conjunto'] = 'validacion'
particion.loc[indices_prueba, 'conjunto'] = 'prueba' 

semanas_train = semanas[indices_train]

# ---------------------------------------------------------------------------
# 3. Selección de platos
# Reglas de estado_plato (solo con semanas de entrenamiento, para no mirar el futuro)
# P1: historial_insuficiente si tiene menos de 26 semanas desde su primera venta hasta el fin
#     del entrenamiento (incluye los platos que todavía no se vendían en entrenamiento)
# P2: retirado si no vendió nada en las últimas 26 semanas de entrenamiento
# ---------------------------------------------------------------------------

"""  
1. Filtra las filas donde realmente hubo ventas (unidades > 0).
2. Agrupa los datos por plato.
3. Busca la fecha mínima (min) y máxima (max) para crear las columnas primera_venta y ultima_venta.
4. El paso clave (reindex): Fuerza a que la tabla resultante tenga exactamente todos los platos que 
existen en el dataset original completo (los 62 platos).
Si un plato no estaba en el entrenamiento, se le asignan valores de nan.
Esto es perfecto, porque así la función clasificar_plato detectará que tiene 0 semanas de historia 
y lo descartará correctamente.
"""
datos_train = df[df['semana_inicio'].isin(semanas_train)]
resumen = (datos_train[datos_train['unidades'] > 0]
           .groupby('plato')['semana_inicio']
           .agg(primera_venta='min', ultima_venta='max')
           .reindex(sorted(df['plato'].unique())))
resumen.index.name = 'plato'

# Referencias de tiempo para 'clasificar_plato'.
# la semana 80 se usa como las ultimas semanas de entrenamiento
fin_entrenamiento = semanas_train.iloc[-1]
inicio_ultimas = semanas_train.iloc[-SEMANAS_MINIMAS]
resumen['semanas_historia'], resumen['estado_plato'] = clasificar_plato(
    resumen['primera_venta'], resumen['ultima_venta'], fin_entrenamiento, inicio_ultimas)
resumen = resumen.reset_index()

# ubica todos los platos modelables
platos_modelables = resumen.loc[resumen['estado_plato'] == 'modelable', 'plato']

# ---------------------------------------------------------------------------
# 4. Ventanas
# Reglas de las ventanas
# V1: solo entran semanas activas de platos modelables (antes de su lanzamiento el plato no existía)
# V2: cada semana a predecir necesita 8 semanas activas antes, así los tres archivos tienen las mismas filas
# V3: las variables de la ventana usan solo semanas anteriores a la semana a predecir
# V4: el conjunto de cada fila es el de su semana a predecir
###### Importante en la documentación especificar el por qué se utilizan # de semanas especifico. ####
# ---------------------------------------------------------------------------

# V1: se quitan los platos no modelables y las semanas antes de la primera venta
filas = df[df['plato'].isin(platos_modelables) & (df['activo'] == 1)].copy()
# El orden por plato y semana es el que permite usar shift para mirar semanas anteriores
filas = filas.sort_values(['plato', 'semana_inicio']).reset_index(drop=True)

# La semana del año, en seno y coseno, hace que la semana 52 quede al lado de la semana 1
semana_anio = filas['semana_inicio'].dt.isocalendar().week.astype(int)
filas['sin'] = np.sin(2 * np.pi * semana_anio / 52)
filas['cos'] = np.cos(2 * np.pi * semana_anio / 52)

# V3: shift(k) dentro de cada plato trae el valor de k semanas antes. u_t1 es la semana anterior.
por_plato = filas.groupby('plato')
for k in range(1, HISTORIA_LSTM + 1):
    filas[f'u_t{k}'] = por_plato['unidades'].shift(k)
    filas[f'precio_t{k}'] = por_plato['precio_unitario'].shift(k)
    filas[f'sin_t{k}'] = por_plato['sin'].shift(k)
    filas[f'cos_t{k}'] = por_plato['cos'].shift(k)
# Semana de u_t8, para comprobar en C6 que la ventana es de semanas seguidas
filas['semana_t8'] = por_plato['semana_inicio'].shift(HISTORIA_LSTM)

# V2: si no hay valor 8 semanas atrás, la fila no tiene la ventana completa de la LSTM
filas = filas[filas[f'u_t{HISTORIA_LSTM}'].notna()].reset_index(drop=True)

# Las unidades vuelven a enteros: shift las había pasado a decimales por los vacíos
for k in range(1, HISTORIA_LSTM + 1):
    filas[f'u_t{k}'] = filas[f'u_t{k}'].astype(int)

# V4: cada fila toma el conjunto de su semana a predecir
filas = filas.merge(particion, on='semana_inicio', how='left')

# Columnas comunes del diseño. fila_id une las filas de los tres archivos en M4 y M5.
filas['fila_id'] = range(1, len(filas) + 1)
filas = filas.rename(columns={'semana_inicio': 'semana_objetivo', 'unidades': 'y_real'})

# ---------------------------------------------------------------------------
# 5. Archivos por modelo
# ---------------------------------------------------------------------------

columnas_comunes = ['fila_id', 'plato', 'semana_objetivo', 'conjunto', 'y_real']

# El plato y la categoría pasan a columnas 0/1 (one-hot) para XGBoost y LSTM
plato_0_1 = pd.get_dummies(filas['plato'], prefix='plato', dtype=int)
categoria_0_1 = pd.get_dummies(filas['categoria'], prefix='categoria', dtype=int)

# MM4: las 4 semanas anteriores. Su pronóstico en M4 es el promedio de u_t1 a u_t4.
columnas_mm4 = columnas_comunes + [f'u_t{k}' for k in range(1, HISTORIA_MM4 + 1)]
mm4 = filas[columnas_mm4].reset_index(drop=True)

# XGBoost: las 4 semanas, su media, el calendario de la semana a predecir y el último precio
filas['media_4'] = filas[[f'u_t{k}' for k in range(1, HISTORIA_MM4 + 1)]].mean(axis=1)
filas['semana_sin'] = filas['sin']
filas['semana_cos'] = filas['cos']
filas['mes'] = filas['semana_objetivo'].dt.month
columnas_xgboost = columnas_mm4 + ['media_4', 'semana_sin', 'semana_cos', 'mes', 'precio_t1']
xgboost = pd.concat([filas[columnas_xgboost], plato_0_1, categoria_0_1], axis=1)

# LSTM: la secuencia de t-8 a t-1. M4 la pasa a forma (muestras, 8 pasos, variables).
columnas_lstm = list(columnas_comunes)
for variable in ['u', 'precio', 'sin', 'cos']:
    columnas_lstm += [f'{variable}_t{k}' for k in range(HISTORIA_LSTM, 0, -1)]
lstm = pd.concat([filas[columnas_lstm], plato_0_1], axis=1)

# ---------------------------------------------------------------------------
# 6. Controles de calidad.
# ---------------------------------------------------------------------------

# C1: los tres archivos tienen las mismas filas y el mismo conjunto en cada fila
assert mm4['fila_id'].equals(xgboost['fila_id']) and mm4['fila_id'].equals(lstm['fila_id']), \
    'Los archivos no tienen los mismos fila_id'
assert mm4['conjunto'].equals(xgboost['conjunto']) and mm4['conjunto'].equals(lstm['conjunto']), \
    'Los archivos no tienen el mismo conjunto'

# C2: no hay dos filas para el mismo plato y semana a predecir
assert not mm4.duplicated(['plato', 'semana_objetivo']).any(), 'Hay semanas a predecir repetidas'

# C3: la partición tiene resto / 26 / 26 semanas y cada conjunto tiene filas
assert particion['conjunto'].value_counts().to_dict() == {
    'entrenamiento': semanas_entrenamiento, 'validacion': SEMANAS_VALIDACION,
    'prueba': SEMANAS_PRUEBA}, 'La partición no tiene los tamaños del diseño'
assert set(filas['conjunto']) == {'entrenamiento', 'validacion', 'prueba'}, 'Hay un conjunto sin filas'

# C4: todos los platos de M2 están en el archivo de platos, cada uno con un solo estado
assert len(resumen) == df['plato'].nunique(), 'Faltan platos en 03_platos_modelables.csv'

# C5: sin vacíos en ninguna columna que reciben los modelos
assert xgboost.notna().all().all(), 'Hay valores vacíos en XGBoost'
assert lstm.notna().all().all(), 'Hay valores vacíos en LSTM'

# C6: la ventana son las 8 semanas justo antes de la semana a predecir (V3)
assert ((filas['semana_objetivo'] - filas['semana_t8']).dt.days == 7 * HISTORIA_LSTM).all(), \
    'Hay ventanas con semanas saltadas o que miran hacia adelante'

# C7: la demanda no puede ser negativa
assert (mm4[['y_real'] + [f'u_t{k}' for k in range(1, HISTORIA_MM4 + 1)]] >= 0).all().all(), \
    'Hay unidades negativas'

# C8: queda al menos un plato modelable (si no, no hay nada que entrenar)
assert len(platos_modelables) > 0, 'Ningún plato cumple las reglas P1 y P2'

# Avisos: no detienen el script, solo informan

# A1: fechas de cada conjunto
for conjunto in ['entrenamiento', 'validacion', 'prueba']:
    fechas = particion.loc[particion['conjunto'] == conjunto, 'semana_inicio']
    print(f'A1: {conjunto}: {len(fechas)} semanas, {fechas.min().date()} a {fechas.max().date()}')

# A2: cuántos platos hay en cada estado y cuáles quedan fuera
print(f"A2: estado de los platos: {resumen['estado_plato'].value_counts().to_dict()}")
excluidos = resumen[resumen['estado_plato'] != 'modelable']
print(f"A2: fuera de los modelos: {', '.join(excluidos['plato'] + ' (' + excluidos['estado_plato'] + ')')}")

# A3: filas de la serie de M2 y filas que quedan para los modelos
print(f'A3: {len(df)} filas de M2 -> {len(filas)} filas con ventana completa')

# Filas y unidades reales a predecir en cada conjunto
totales = filas.groupby('conjunto')['y_real'].agg(filas='count', unidades='sum')
print(totales)

# ---------------------------------------------------------------------------
# 7. Exportación
# ---------------------------------------------------------------------------

# Cuatro decimales para que el seno y el coseno no pierdan precisión
particion.to_csv(ruta_particion, index=False, date_format='%Y-%m-%d')
resumen.to_csv(ruta_platos, index=False, date_format='%Y-%m-%d')
mm4.to_csv(ruta_mm4, index=False, date_format='%Y-%m-%d', float_format='%.4f')
xgboost.to_csv(ruta_xgboost, index=False, date_format='%Y-%m-%d', float_format='%.4f')
lstm.to_csv(ruta_lstm, index=False, date_format='%Y-%m-%d', float_format='%.4f')
print(f'{len(mm4)} filas, {mm4["plato"].nunique()} platos -> {ruta_mm4}, {ruta_xgboost}, {ruta_lstm}')
