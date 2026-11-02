"""
M2. Serie de tiempo semanal.

Uso (desde code/src, después de M1):
  python(3) m2_serie_semanal.py
"""

import pandas as pd

# Semanas mínimas de historia, y semanas sin ventas al final para considerar un plato retirado.
SEMANAS_MINIMAS = 26

def clasificar_plato(primera_venta, ultima_venta, fin_periodo, inicio_ultimas_semanas):
    # Calcula la edad del plato
    semanas_historia = (fin_periodo - primera_venta).days // 7 + 1
    # un plato que ya no está en el menú no se pronostica
    if ultima_venta < inicio_ultimas_semanas:
        return 'retirado'
    # Un plato con poco historial tampoco
    if semanas_historia < SEMANAS_MINIMAS:
        return 'historial_insuficiente'
    return 'modelable'

# ---------------------------------------------------------------------------
# 1. Ingesta
# ---------------------------------------------------------------------------

# El script se ejecuta desde code/src, así que los datos están dos carpetas más arriba
ruta_entrada = '../../dataset/processed/01_ventas_limpias.csv'
ruta_salida = '../../dataset/processed/02_serie_semanal.csv'

df = pd.read_csv(ruta_entrada)

print('M2 Serie de tiempo semanal')
df.info()

columnas_entrada = ['id_pedido', 'fecha', 'dia_semana', 'hora', 'plato', 'categoria', 'cantidad',
                    'precio_unitario', 'ventas_brutas', 'costo', 'margen']

# C1 el archivo no es vacío
assert len(df) > 0, 'El archivo no tiene filas'
# C2 las columnas son las que deja M1
assert list(df.columns) == columnas_entrada, 'Las columnas no son las de M1'

# La fecha se lee como texto; se convierte y se pone como índice para poder usar resample
df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
df.set_index('fecha', inplace=True)

# ---------------------------------------------------------------------------
# 2. Agregación semanal
# Reglas de agregación
# S1: cada fecha va a la semana que empieza el lunes (semana_inicio = el lunes)
# S2: se suman unidades, ventas, costo y margen por plato y semana
# ---------------------------------------------------------------------------

# Agrupa de lunes a domingo. W-MON: W = week  y MON = Monday.
unidades_semana = df['cantidad'].resample('W-MON', label='left', closed='left').sum()

# Se descarta el plato Jolof que se vendio a 0. 
df['unidades_cobradas'] = df['cantidad'].where(df['precio_unitario'] > 0, 0)
df['ventas_cobradas'] = df['ventas_brutas'].where(df['precio_unitario'] > 0, 0)

# S2: la misma suma, pero separada por plato
columnas_suma = ['cantidad', 'ventas_brutas', 'costo', 'margen', 'unidades_cobradas', 'ventas_cobradas']
semanal = df.groupby('plato')[columnas_suma].resample('W-MON', label='left', closed='left').sum()

print(df)

# ---------------------------------------------------------------------------
# 3. Rejilla completa
# Reglas de rejilla
# S3: cada plato tiene una fila en cada una de las 157 semanas, con 0 si no vendió
# ---------------------------------------------------------------------------

# resample solo crea las semanas entre la primera y la última venta de cada plato.
# Se arma la lista completa de (plato, semana) y se rellena con 0 lo que falta.
semanas = unidades_semana.index
platos = sorted(df['plato'].unique())
print(f'{len(platos)} platos x {len(semanas)} semanas ({semanas[0].date()} a {semanas[-1].date()})')

semanal.index.names = ['plato', 'semana_inicio']
indice_completo = pd.MultiIndex.from_product([platos, semanas], names=['plato', 'semana_inicio'])

# reindex agrega las filas que faltan con 0; reset_index pasa plato y semana_inicio a columnas
serie = semanal.reindex(indice_completo, fill_value=0).reset_index()
serie = serie.rename(columns={'cantidad': 'unidades', 'ventas_brutas': 'ventas'})

# ---------------------------------------------------------------------------
# 4. Columnas calculadas
# Reglas de columnas calculadas
# S4: una sola categoría por plato: la de más unidades (Beef Suya queda en Apetisers)
# S5: precio_unitario = ventas cobradas / unidades cobradas. Sin ventas cobradas, el último precio
#     conocido del plato. Antes de la primera venta queda vacío.
# S6: activo = 1 desde la semana de la primera venta del plato
# ---------------------------------------------------------------------------

# S5: where deja vacías las semanas sin unidades cobradas para no dividir por 0
serie['precio_unitario'] = serie['ventas_cobradas'] / serie['unidades_cobradas'].where(serie['unidades_cobradas'] > 0)
# ffill copia hacia abajo el último precio conocido, sin pasar de un plato a otro
serie['precio_unitario'] = serie.groupby('plato')['precio_unitario'].ffill().round(2)
# Las columnas de cobradas ya no se necesitan
serie = serie.drop(columns=['unidades_cobradas', 'ventas_cobradas'])

# S4: unidades de cada plato en cada categoría. Se ordena de más a menos unidades y se deja
# la primera fila de cada plato.
unidades_categoria = df.groupby(['plato', 'categoria'])['cantidad'].sum().reset_index()
unidades_categoria = unidades_categoria.sort_values(['plato', 'cantidad', 'categoria'], ascending=[True, False, True])
categoria_principal = unidades_categoria.drop_duplicates('plato').set_index('plato')['categoria']
serie['categoria'] = serie['plato'].map(categoria_principal)

# S6: lunes de la primera semana con ventas de cada plato. Antes de esa semana el plato todavía
# no existía, así que esas semanas no cuentan como demanda 0.
primera_venta = serie[serie['unidades'] > 0].groupby('plato')['semana_inicio'].min()
serie['activo'] = (serie['semana_inicio'] >= serie['plato'].map(primera_venta)).astype(int)

# ---------------------------------------------------------------------------
# 5. Reglas de negocio
# Reglas de estado_plato (cap. 6, §6.1.2)
# S7: retirado: ninguna venta en las últimas 26 semanas del periodo (desde 2025-07-07)
#     historial_insuficiente: menos de 26 semanas entre la primera venta y el fin del periodo
#     (2025-12-29). Hoy ningún plato cumple esta regla, pero queda para datos futuros
#     modelable: todos los demás
# ---------------------------------------------------------------------------

# Lunes de la última semana del periodo, y lunes en que empiezan sus últimas 26 semanas
fin_periodo = semanas[-1]
inicio_ultimas_semanas = semanas[-SEMANAS_MINIMAS]

# Una fila por plato con su primera y última semana con ventas
platos_resumen = serie[serie['unidades'] > 0].groupby('plato')['semana_inicio'].agg(primera_venta='min', ultima_venta='max')

# S7: se aplica clasificar_plato a cada plato (axis=1 recorre las filas)
platos_resumen['estado_plato'] = platos_resumen.apply(
    lambda fila: clasificar_plato(fila['primera_venta'], fila['ultima_venta'], fin_periodo, inicio_ultimas_semanas), axis=1)

# Se copia el estado a todas las semanas del plato
serie['estado_plato'] = serie['plato'].map(platos_resumen['estado_plato'])

print(platos_resumen['estado_plato'].value_counts())

# Columnas de salida, ordenadas por plato y semana
columnas_salida = ['semana_inicio', 'plato', 'categoria', 'unidades', 'ventas', 'costo',
                   'margen', 'precio_unitario', 'activo', 'estado_plato']

serie = serie.sort_values(['plato', 'semana_inicio'])
# Sobrescribe la tabla conservando el nuevo orden y las columnas de 'columnas_salida'
serie = serie[columnas_salida].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 6. Controles de calidad.
# ---------------------------------------------------------------------------

# C1: 157 semanas, de 2023-01-02 a 2025-12-29, y todas empiezan en lunes
assert len(semanas) == 157, f'Semanas: {len(semanas)}'
assert str(semanas[0].date()) == '2023-01-02' and str(semanas[-1].date()) == '2025-12-29', 'El periodo no es el esperado'
assert (serie['semana_inicio'].dt.dayofweek == 0).all(), 'Hay semanas que no empiezan en lunes'

# C2: 62 platos x 157 semanas = 9.734 filas, sin pares (plato, semana) repetidos
assert serie['plato'].nunique() == 62, f"Platos: {serie['plato'].nunique()}"
assert len(serie) == 9734, f'Filas: {len(serie)}'
assert not serie.duplicated(['plato', 'semana_inicio']).any(), 'Hay pares plato-semana repetidos'

# C3: las sumas de la serie son iguales a las de M1 (criterio CA-01)
totales = serie[['unidades', 'ventas', 'costo', 'margen']].sum().round(2)
totales_m1 = df[['cantidad', 'ventas_brutas', 'costo', 'margen']].sum().round(2)
assert (totales.values == totales_m1.values).all(), 'Las sumas no coinciden con M1'
assert totales['unidades'] == 7118, f"Unidades: {totales['unidades']}"
assert totales['ventas'] == 38480.25, f"Ventas: {totales['ventas']}"

# C4: sin valores negativos
assert (serie[['unidades', 'ventas', 'costo', 'margen']] >= 0).all().all(), 'Hay valores negativos'

# C5: una sola categoría por plato
assert (serie.groupby('plato')['categoria'].nunique() == 1).all(), 'Hay platos con más de una categoría'

# C6: toda semana activa tiene precio, y no hay ventas en semanas inactivas
assert serie.loc[serie['activo'] == 1, 'precio_unitario'].notna().all(), 'Hay semanas activas sin precio'
assert not ((serie['activo'] == 0) & (serie['unidades'] > 0)).any(), 'Hay ventas en semanas inactivas'

# C7: 52 modelables, 0 con historial insuficiente y 10 retirados. get(..., 0) porque un estado
# sin platos no aparece en value_counts
conteo = platos_resumen['estado_plato'].value_counts()
assert conteo.get('modelable', 0) == 52, f"Modelables: {conteo.get('modelable', 0)}"
assert conteo.get('historial_insuficiente', 0) == 0, f"Historial insuficiente: {conteo.get('historial_insuficiente', 0)}"
assert conteo.get('retirado', 0) == 10, f"Retirados: {conteo.get('retirado', 0)}"

# Avisos: no detienen el script, solo informan

# A1: semanas en las que el restaurante no vendió nada (posible cierre)
semanas_vacias = unidades_semana[unidades_semana == 0].index
print(f"A1: {len(semanas_vacias)} semanas sin ninguna venta: {', '.join(str(s.date()) for s in semanas_vacias)}")

# A2: porcentaje de semanas con ventas
activas = serie[serie['activo'] == 1]
print(f"A2: semanas con ventas: {(serie['unidades'] > 0).mean():.1%} de la rejilla, "
      f"{(activas['unidades'] > 0).mean():.1%} de las semanas activas")

print(totales)

# ---------------------------------------------------------------------------
# 7. Exportación
# ---------------------------------------------------------------------------

serie.to_csv(ruta_salida, index=False, date_format='%Y-%m-%d', float_format='%.2f')
print(f'{len(serie)} filas, {serie["plato"].nunique()} platos -> {ruta_salida}')
