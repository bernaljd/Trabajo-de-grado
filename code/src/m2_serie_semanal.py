"""
M2. Serie de tiempo semanal.

Entrada:
  dataset/processed/01_ventas_limpias.csv

Salida:
  dataset/processed/02_serie_semanal.csv

Objetivo:
  Transformar las líneas de pedido limpias de M1 en una serie temporal
  con una fila por plato y semana.

Reglas principales:
  S1. Las semanas comienzan el lunes y terminan el domingo.
  S2. Las unidades, ventas, costos y márgenes se suman por plato y semana.
  S3. Cada plato tiene una fila para cada semana del período completo.
  S4. Cada plato recibe una única categoría principal.
  S5. El precio semanal se calcula solo con líneas cobradas.
  S6. La columna activo distingue platos existentes de platos aún no lanzados.

Controles:
  Si un control falla, el script se detiene con AssertionError y no escribe
  una salida aparentemente válida.

Uso (desde code/src, después de M1):
  python(3) m2_serie_semanal.py
"""

import pandas as pd #type: ignore

# ---------------------------------------------------------------------------
# 1. Ingesta y validación de la entrada
# ---------------------------------------------------------------------------

# El script se ejecuta desde code/src. Por eso se sube dos niveles para llegar
# a la carpeta dataset y acceder al archivo generado por M1.
ruta_entrada = '../../dataset/processed/01_ventas_limpias.csv'
ruta_salida = '../../dataset/processed/02_serie_semanal.csv'

# M1 ya eliminó cancelaciones, cantidades inválidas y fechas ilegibles.
# M2 trabaja únicamente con esa salida validada.
df = pd.read_csv(ruta_entrada)

print('M2 Serie de tiempo semanal')
df.info()

# Este es el contrato de entrada entre M1 y M2. La comparación posterior
# también comprueba que las columnas estén en el orden esperado.
columnas_entrada = ['id_pedido', 'fecha', 'dia_semana', 'hora', 'plato', 'categoria', 'cantidad',
                    'precio_unitario', 'ventas_brutas', 'costo', 'margen']

# C1: el archivo de entrada debe contener al menos una fila.
assert len(df) > 0, 'El archivo no tiene filas'
# C2: M1 debe entregar exactamente las columnas definidas en el contrato.
assert list(df.columns) == columnas_entrada, 'Las columnas no son las de M1'

# La fecha se lee como texto desde el CSV. Se convierte a datetime para que
# pandas pueda agrupar por períodos y se usa como índice para resample().
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
df.set_index('fecha', inplace=True)
# El orden cronológico hace explícita la secuencia temporal antes de agrupar.
df.sort_index(inplace=True)

# ---------------------------------------------------------------------------
# 2. Agregación semanal
# S1: cada fecha pertenece a una semana que comienza el lunes.
# S2: las magnitudes económicas y de demanda se suman por plato y semana.
# ---------------------------------------------------------------------------

# 'W-MON' define períodos semanales con referencia al lunes. Con label='left'
# cada semana recibe como etiqueta la fecha de su lunes y con closed='left'
# se incluyen las fechas desde ese lunes hasta el domingo siguiente.
# Esta agregación global define las semanas del período del restaurante.
unidades_semana = df['cantidad'].resample('W-MON', label='left', closed='left').sum()

# Las líneas con precio cero sí representan unidades demandadas y por eso
# permanecen en cantidad. Sin embargo, no representan una venta cobrada y
# se excluyen de las variables auxiliares usadas para calcular el precio.
df['unidades_cobradas'] = df['cantidad'].where(df['precio_unitario'] > 0, 0)
df['ventas_cobradas'] = df['ventas_brutas'].where(df['precio_unitario'] > 0, 0)

# S2: se repite la agregación, esta vez separando cada plato. El resultado
# inicial solo contiene semanas dentro del intervalo de ventas de cada plato.
columnas_suma = ['cantidad', 'ventas_brutas', 'costo', 'margen', 'unidades_cobradas', 'ventas_cobradas']
semanal = df.groupby('plato')[columnas_suma].resample('W-MON', label='left', closed='left').sum()

# ---------------------------------------------------------------------------
# 3. Rejilla completa
# S3: cada plato debe tener una fila en cada semana del período completo.
# ---------------------------------------------------------------------------

# resample() solo crea semanas entre la primera y la última venta de cada
# plato. Para que todos los modelos reciban series con la misma longitud, se
# construye el producto completo de platos y semanas y se rellenan con cero
# las combinaciones que no aparecieron en la agregación inicial.
semanas = unidades_semana.index
platos = sorted(df['plato'].unique())
print(f'{len(platos)} platos x {len(semanas)} semanas ({semanas[0].date()} a {semanas[-1].date()})')

semanal.index.names = ['plato', 'semana_inicio']
indice_completo = pd.MultiIndex.from_product([platos, semanas], names=['plato', 'semana_inicio'])

# reindex() agrega las filas que faltan con cero y reset_index() convierte
# plato y semana_inicio de niveles del índice en columnas normales.
serie = semanal.reindex(indice_completo, fill_value=0).reset_index()
serie = serie.rename(columns={'cantidad': 'unidades', 'ventas_brutas': 'ventas'})

# ---------------------------------------------------------------------------
# 4. Columnas calculadas
# S4: la categoría principal es la que concentra más unidades del plato.
# S5: precio_unitario = ventas cobradas / unidades cobradas. Si no hay
#     ventas cobradas, se conserva el último precio conocido del plato.
# S6: activo = 1 desde la primera semana en que el plato tuvo ventas.
# ---------------------------------------------------------------------------

# S5: where() evita la división por cero al dejar vacío el denominador
# cuando no existen unidades cobradas.
serie['precio_unitario'] = serie['ventas_cobradas'] / serie['unidades_cobradas'].where(serie['unidades_cobradas'] > 0)
# ffill() copia el último precio conocido hacia adelante dentro de cada
# plato. El groupby evita que el precio de un plato pase a otro.
serie['precio_unitario'] = serie.groupby('plato')['precio_unitario'].ffill().round(2)
# Estas columnas solo fueron necesarias para calcular el precio y no forman
# parte de la salida final.
serie = serie.drop(columns=['unidades_cobradas', 'ventas_cobradas'])

# S4: se suman las unidades por plato y categoría. Se ordena de mayor a
# menor cantidad; en caso de empate, el orden alfabético hace la elección
# determinista.
unidades_categoria = df.groupby(['plato', 'categoria'])['cantidad'].sum().reset_index()
unidades_categoria = unidades_categoria.sort_values(['plato', 'cantidad', 'categoria'], ascending=[True, False, True])
# Después de ordenar, la primera fila de cada plato contiene su categoría
# principal.
categoria_principal = unidades_categoria.drop_duplicates('plato').set_index('plato')['categoria']
serie['categoria'] = serie['plato'].map(categoria_principal)

# S6: se obtiene el lunes de la primera semana con unidades vendidas.
# Antes de esa semana el plato todavía no estaba disponible, por lo que esas
# filas no deben interpretarse como semanas activas con demanda cero.
primera_venta = serie[serie['unidades'] > 0].groupby('plato')['semana_inicio'].min()
serie['activo'] = (serie['semana_inicio'] >= serie['plato'].map(primera_venta)).astype(int)

# ---------------------------------------------------------------------------
# 5. Columnas de salida
# M3 calculará la elegibilidad de los platos después del split temporal.
# M2 no crea estado_plato para evitar usar información futura en el modelado.
# ---------------------------------------------------------------------------

columnas_salida = ['semana_inicio', 'plato', 'categoria', 'unidades', 'ventas', 'costo',
                   'margen', 'precio_unitario', 'activo']

serie = serie.sort_values(['plato', 'semana_inicio'])
# Se seleccionan únicamente las columnas del contrato de salida y se
# reconstruye el índice para que el CSV no conserve el índice anterior.
serie = serie[columnas_salida].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Controles de calidad.
# ---------------------------------------------------------------------------

# Los controles revisan la estructura de la serie, no su tamaño: así M2 sirve con
# cualquier dataset que salga de M1.

# C1: todas las semanas comienzan en lunes y van seguidas, sin semanas faltantes
# entre la primera y la última (cada una está 7 días después de la anterior).
assert (serie['semana_inicio'].dt.dayofweek == 0).all(), 'Hay semanas que no empiezan en lunes'
assert (pd.Series(semanas).diff().dropna() == pd.Timedelta(days=7)).all(), 'Faltan semanas en el periodo'

# C2: la rejilla tiene una fila por cada plato y cada semana (platos x semanas filas)
# y no repite un mismo par plato-semana.
assert len(serie) == len(platos) * len(semanas), f'Filas: {len(serie)}, esperadas {len(platos) * len(semanas)}'
assert not serie.duplicated(['plato', 'semana_inicio']).any(), 'Hay pares plato-semana repetidos'

# C3: las sumas de M2 deben coincidir con las sumas de M1. Este control
# comprueba que la agregación no pierda ni duplique unidades o valores
# económicos (criterio CA-01).
totales = serie[['unidades', 'ventas', 'costo', 'margen']].sum().round(2)
totales_m1 = df[['cantidad', 'ventas_brutas', 'costo', 'margen']].sum().round(2)
assert (totales.values == totales_m1.values).all(), 'Las sumas no coinciden con M1'

# C4: las variables agregadas de demanda y dinero no pueden ser negativas.
assert (serie[['unidades', 'ventas', 'costo', 'margen']] >= 0).all().all(), 'Hay valores negativos'

# C5: cada plato debe conservar una única categoría principal en todas sus
# semanas.
assert (serie.groupby('plato')['categoria'].nunique() == 1).all(), 'Hay platos con más de una categoría'

# C6: toda semana activa debe tener precio conocido y no puede haber ventas
# antes de la primera semana activa del plato.
assert serie.loc[serie['activo'] == 1, 'precio_unitario'].notna().all(), 'Hay semanas activas sin precio'
assert not ((serie['activo'] == 0) & (serie['unidades'] > 0)).any(), 'Hay ventas en semanas inactivas'

# ---------------------------------------------------------------------------
# Avisos informativos
# Estos mensajes no detienen el pipeline; sirven para detectar situaciones
# que pueden requerir una revisión del contexto operativo.
# ---------------------------------------------------------------------------

# A1: semanas en las que el restaurante no registró unidades. Pueden indicar
# un cierre, una interrupción operativa o ausencia de registros.
semanas_vacias = unidades_semana[unidades_semana == 0].index
print(f"A1: {len(semanas_vacias)} semanas sin ninguna venta: {', '.join(str(s.date()) for s in semanas_vacias)}")

# A2: porcentaje de combinaciones plato-semana con ventas. La segunda cifra
# excluye las semanas anteriores a la primera venta de cada plato, porque
# esas semanas todavía no forman parte de su período activo.
activas = serie[serie['activo'] == 1]
print(f"A2: semanas con ventas: {(serie['unidades'] > 0).mean():.1%} de la rejilla, "
      f"{(activas['unidades'] > 0).mean():.1%} de las semanas activas")

# Los totales se imprimen para dejar una evidencia legible de la
# conciliación realizada antes de exportar.
print(totales)

# ---------------------------------------------------------------------------
# 7. Exportación
# ---------------------------------------------------------------------------

# Se exporta sin el índice interno de pandas, con fechas ISO y dos decimales
# para que el archivo sea reproducible y compatible con las siguientes etapas.
serie.to_csv(ruta_salida, index=False, date_format='%Y-%m-%d', float_format='%.2f')
print(f'{len(serie)} filas, {serie["plato"].nunique()} platos -> {ruta_salida}')