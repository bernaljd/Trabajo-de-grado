"""
M1. Ingesta y limpieza de datos.

Entrada: dataset/raw/ventas_originales.csv         (Kaggle, sin modificar)
Salida:  dataset/processed/01_ventas_limpias.csv   (una fila por línea de pedido válida)
Especificación: pipeline/M1_ingesta_limpieza.md

Pasos:
  1. Ingesta: leer el CSV crudo tal como viene.
  2. Validación de esquema: revisar filas, columnas y valores antes de tocar nada.
  3. Estandarización: renombrar columnas y unificar formatos. No quita filas.
  4. Filtrado: quitar las filas que no son ventas válidas.
  5. Controles de calidad: comprobar que los datos son coherentes (sirve con cualquier
     dataset que tenga las mismas columnas, sin importar cuántas filas o fechas tenga).
  6. Exportación: escribir 01_ventas_limpias.csv.

Si un control falla, el script se detiene con un AssertionError y no escribe la salida.

Uso (desde code/src):
  python3 m1_ingesta_limpieza.py
"""

import pandas as pd # type: ignore 

def filtrar(df, condicion, descripcion):
    # Cantidad de filas a quitar (negamos los valores asi que si hay 0 ventas es True y se suman esos True).
    quitadas = (~condicion).sum()
    print(f'{descripcion}: {quitadas} filas quitadas')
    # Retorna la tabla con las filas eliminadas.
    return df[condicion]

# ---------------------------------------------------------------------------
# 1. Ingesta
# ---------------------------------------------------------------------------

# El script se ejecuta desde code/src, así que los datos están dos carpetas más arriba
ruta_entrada = '../../dataset/raw/ventas_originales.csv'
ruta_salida = '../../dataset/processed/01_ventas_limpias.csv'

df = pd.read_csv(ruta_entrada)

print('M1 Ingesta y limpieza')
df.info()

# ---------------------------------------------------------------------------
# 2. Validación de esquema
# ---------------------------------------------------------------------------

columnas_originales = ['OrderID', 'Date', 'DayOfWeek', 'Time', 'Category', 'Line item name',
                       'Quantity', 'Price Per Item', 'Gross Sales', 'Est. Cost', 'Est. Profit',
                       'OrderType', 'Payment', 'Cancelled']

# C1 el archivo no es vacío
assert len(df) > 0, 'El archivo no tiene filas'
# C2 las columnas coinciden en el nombre exacto
assert list(df.columns) == columnas_originales, 'Las columnas no son las esperadas'
# C3: un valor nuevo en Cancelled haría que el filtro de cancelados falle en silencio
assert set(df['Cancelled'].unique()) <= {'Yes', 'No'}, 'Hay valores inesperados en Cancelled'

# ---------------------------------------------------------------------------
# 3. Estandarización 
# Reglas de estandarización (no quitan filas; L4 sí puede quitarlas)
# E1: fecha dd/mm/aaaa a datetime. Fechas ilegibles quedan NaT (L4 las quita)
# E2: categorías renombradas a los nombres del Cuadro 4.3 de la tesis
# E3: "PEPPERED TURKEY" pasa a "Peppered Turkey"
# E4: días de la semana en español
# E5: se quita el sufijo ".0" de id_pedido
# E6: columnas renombradas al español (Payment y Cancelled no pasan a la salida)
# ---------------------------------------------------------------------------

# E6: columnas en español. Payment, Cancelled y OrderType no son necesarios. 
df = df.rename(columns={
    'OrderID': 'id_pedido',
    'Date': 'fecha',
    'DayOfWeek': 'dia_semana',
    'Time': 'hora',
    'Line item name': 'plato',
    'Category': 'categoria',
    'Quantity': 'cantidad',
    'Price Per Item': 'precio_unitario',
    'Gross Sales': 'ventas_brutas',
    'Est. Cost': 'costo',
    'Est. Profit': 'margen'
})

# Quitar espacios sobrantes en los textos ' soup' -> 'soup'
for columna in df.select_dtypes(include=['object', 'string']).columns:
    df[columna] = df[columna].str.strip()

# E1: dd/mm/aaaa -> fecha. Las que no se puedan leer quedan vacías (NaT) y L4 las quita
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y', errors='coerce')

# E2: categorías con los nombres del Cuadro 4.3 de la tesis
categorias = {
    'EXTRAS': 'Extras',
    'MAIN COURSES': 'Main courses',
    'APETISERS': 'Apetisers',
    'SOUPS': 'Soups',
    'DRINKS': 'Drinks',
    'SPECIAL ORDERS': 'Special orders',
    'Restaurant DEALS': 'Restaurant deals',
    'Friday Specials': 'Friday specials',
}
df['categoria'] = df['categoria'].replace(categorias)

# E3: PEPPERED TURKEY es el único plato en mayúsculas
df['plato'] = df['plato'].replace({'PEPPERED TURKEY': 'Peppered Turkey'})

# E4: días en español. Se incluye el domingo: otro restaurante puede abrir ese día.
dias = {'Monday': 'lunes', 'Tuesday': 'martes', 'Wednesday': 'miercoles', 'Thursday': 'jueves',
        'Friday': 'viernes', 'Saturday': 'sabado', 'Sunday': 'domingo'}
df['dia_semana'] = df['dia_semana'].replace(dias)

# E5: RES_ORD_931.0 -> RES_ORD_931
df['id_pedido'] = df['id_pedido'].str.replace(r'\.0$', '', regex=True)

# ---------------------------------------------------------------------------
# 4. Filtrado (reglas L1 a L4)
# ---------------------------------------------------------------------------

cancelados = df[df['Cancelled'] == 'Yes']
print(f"Las {len(cancelados)} líneas canceladas son de {cancelados['id_pedido'].nunique()} pedidos")

df = filtrar(df, df['Cancelled'] == 'No', 'L1 canceladas')
df = filtrar(df, df['cantidad'] > 0, 'L2 cantidad <= 0')
df = filtrar(df, df['precio_unitario'] >= 0, 'L3 precio < 0')
df = filtrar(df, df['fecha'].notna(), 'L4 fecha inválida')

# Columnas de salida, ordenadas por fecha, hora y número de pedido
columnas_salida = ['id_pedido', 'fecha', 'dia_semana', 'hora', 'plato', 'categoria', 'cantidad',
                   'precio_unitario', 'ventas_brutas', 'costo', 'margen']

# RES_ORD_931 produce 931. Sirve para ordenar los pedidos 1, 2, ..., total de pedidos
df['numero_pedido'] = df['id_pedido'].str.extract(r'(\d+)$', expand=False).astype(int)
#print(df)
df = df.sort_values(['fecha', 'hora', 'numero_pedido'], kind='stable')
# Sobrescibe la tabla conservando el nuevo orden y conservando las columnas de 'columnas_salida'
df = df[columnas_salida].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 5. Controles de calidad.
# Los controles revisan que los datos sean coherentes, no cuántos son: así el pipeline
# corre con cualquier dataset que tenga las columnas de C2 (por ejemplo, el mismo dataset
# con más semanas o más pedidos).
# ---------------------------------------------------------------------------

# C1: Verifica que la fecha real coincida con la escrita en el csv.
dias_por_numero = ['lunes', 'martes', 'miercoles', 'jueves', 'viernes', 'sabado', 'domingo']
dia_real = df['fecha'].dt.dayofweek.map(lambda i: dias_por_numero[i])
assert (dia_real == df['dia_semana']).all(), 'Hay filas con dia_semana distinto al de la fecha'
# C2: Costo + margen = ventas en cada línea (si no cuadra, alguna columna viene mal)
assert ((df['costo'] + df['margen'] - df['ventas_brutas']).abs() <= 0.005).all(), \
    'Hay líneas donde costo + margen no es igual a ventas'

# C3: Sin vacíos
assert df.notna().all().all(), 'Hay valores vacíos'

# C4: después de filtrar sigue habiendo ventas (un archivo con todo cancelado no sirve)
assert len(df) > 0, 'No quedó ninguna venta válida después del filtrado'

# Avisos: no detienen el script, solo informan

# A1: rango de fechas
print(f"C10: las fechas van de {df['fecha'].min().date()} a {df['fecha'].max().date()}")

# A2: líneas con precio 0
print(f"C13: {(df['precio_unitario'] == 0).sum()} líneas con precio 0")

# Totales del archivo, para dejar evidencia de lo que se procesó
totales = df[['cantidad', 'ventas_brutas', 'costo', 'margen']].sum().round(2)
print(totales)

# ---------------------------------------------------------------------------
# 6. Exportación
# ---------------------------------------------------------------------------

df.to_csv(ruta_salida, index=False, date_format='%Y-%m-%d', float_format='%.2f')
print(f'{len(df)} filas, {df["plato"].nunique()} platos -> {ruta_salida}')