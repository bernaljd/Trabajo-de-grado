"""
M1. Ingesta y limpieza de datos.

Entrada: datos/ventas_originales.csv  (Kaggle, sin modificar)
Salida:  datos/01_ventas_limpias.csv  (una fila por línea de pedido válida)

Especificación: pipeline/M1_ingesta_limpieza.md

Pasos (nombre técnico entre paréntesis):
  1. cargar_datos        (ingesta)                  lee el CSV crudo tal como viene
  2. validar_esquema     (validación de esquema)    revisa filas, columnas y valores antes de tocar nada
  3. estandarizar        (estandarización)          renombra columnas y unifica formatos; no quita filas
  4. filtrar_registros   (filtrado)                 quita las filas que no son ventas válidas
  5. verificar_calidad   (controles de calidad)     comprueba que el resultado cuadra con la tesis
  6. exportar            (exportación)              escribe 01_ventas_limpias.csv

Si un control de tipo «Detiene» falla, el script se detiene con un error y no escribe la salida.

Uso:  python codigo/m1_ingesta_limpieza.py
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
ARCHIVO_ENTRADA = RAIZ / "datos" / "ventas_originales.csv"
ARCHIVO_SALIDA = RAIZ / "datos" / "01_ventas_limpias.csv"

COLUMNAS_ORIGINALES = [
    "OrderID", "Date", "DayOfWeek", "Time", "Category", "Line item name",
    "Quantity", "Price Per Item", "Gross Sales", "Est. Cost", "Est. Profit",
    "OrderType", "Payment", "Cancelled",
]

# Nombre original -> nombre en el pipeline (regla E6).
# Payment y Cancelled no pasan a la salida (sección 3 de la especificación).
RENOMBRAR = {
    "OrderID": "id_pedido",
    "Date": "fecha",
    "DayOfWeek": "dia_semana",
    "Time": "hora",
    "Line item name": "plato",
    "Category": "categoria",
    "Quantity": "cantidad",
    "Price Per Item": "precio_unitario",
    "Gross Sales": "ventas_brutas",
    "Est. Cost": "costo",
    "Est. Profit": "margen",
    "OrderType": "canal",
}
COLUMNAS_SALIDA = list(RENOMBRAR.values())

# Regla E2: categorías con los nombres del Cuadro 4.3 de la tesis.
CATEGORIAS = {
    "EXTRAS": "Extras",
    "MAIN COURSES": "Main courses",
    "APETISERS": "Apetisers",
    "SOUPS": "Soups",
    "DRINKS": "Drinks",
    "SPECIAL ORDERS": "Special orders",
    "Restaurant DEALS": "Restaurant deals",
    "Friday Specials": "Friday specials",
}

# Regla E3: nombres de platos que se corrigen.
PLATOS = {"PEPPERED TURKEY": "Peppered Turkey"}

# Regla E4: días en español.
DIAS = {
    "Monday": "lunes", "Tuesday": "martes", "Wednesday": "miercoles",
    "Thursday": "jueves", "Friday": "viernes", "Saturday": "sabado",
    "Sunday": "domingo",
}
DIAS_POR_NUMERO = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]

CANALES = {"Delivery", "Collection"}

# Cifras esperadas (capítulo 4 de la tesis). Si cambia el archivo de entrada,
# se actualizan solo aquí.
ESPERADO = {
    "filas_entrada": 4385,
    "filas_salida": 4361,
    "platos": 62,
    "unidades": 7118,
    "ventas": 38480.25,
    "costo": 13461.90,
    "margen": 25018.35,
    "fecha_min": "2023-01-03",
    "fecha_max": "2025-12-31",
    "lineas_precio_cero": 8,
}

TOLERANCIA_REDONDEO = 0.005  # libras por unidad (control C7)


class ErrorDeCalidad(Exception):
    """Un control de tipo «Detiene» falló."""


def miles(n, decimales=0):
    """Formatea un número al estilo de la tesis: 4.361 y 38.480,25."""
    texto = f"{n:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def detener_si(condicion, mensaje):
    if condicion:
        raise ErrorDeCalidad(mensaje)


# ---------------------------------------------------------------------------
# 1. Ingesta
# ---------------------------------------------------------------------------

def cargar_datos(ruta=ARCHIVO_ENTRADA):
    """Lee el CSV original como texto, sin interpretar nada todavía."""
    return pd.read_csv(ruta, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# 2. Validación de esquema (controles C1 a C3, sobre la entrada)
# ---------------------------------------------------------------------------

def validar_esquema(crudo):
    filas, columnas = crudo.shape

    # C1
    detener_si(
        filas != ESPERADO["filas_entrada"] or columnas != len(COLUMNAS_ORIGINALES),
        f"C1: se esperaban {ESPERADO['filas_entrada']} x {len(COLUMNAS_ORIGINALES)} "
        f"y llegaron {filas} x {columnas}",
    )

    # C2
    faltan = [c for c in COLUMNAS_ORIGINALES if c not in crudo.columns]
    detener_si(bool(faltan), f"C2: faltan columnas {faltan}")

    # C3: un valor nuevo en Cancelled haría que el filtro L1 falle en silencio
    valores = set(crudo["Cancelled"].str.strip())
    detener_si(not valores <= {"Yes", "No"}, f"C3: valores inesperados en Cancelled: {valores - {'Yes', 'No'}}")


# ---------------------------------------------------------------------------
# 3. Estandarización (reglas E1 a E7; no quita filas)
# ---------------------------------------------------------------------------

def estandarizar(crudo):
    df = crudo.copy()

    # Quitar espacios sobrantes en todos los textos
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Se guarda antes de descartar la columna, para el filtro L1
    df["_cancelado"] = df["Cancelled"].eq("Yes")

    df = df.rename(columns=RENOMBRAR)

    # E1: dd/mm/aaaa -> fecha (las que no se puedan leer quedan vacías y L4 las quita)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")

    # E2, E3, E4
    df["categoria"] = df["categoria"].map(CATEGORIAS).fillna(df["categoria"])
    df["plato"] = df["plato"].replace(PLATOS)
    df["dia_semana"] = df["dia_semana"].map(DIAS).fillna(df["dia_semana"])

    # E5: RES_ORD_931.0 -> RES_ORD_931
    df["id_pedido"] = df["id_pedido"].str.replace(r"\.0$", "", regex=True)

    # Números. E7: ventas, costo y margen se copian, no se recalculan.
    df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
    for col in ["precio_unitario", "ventas_brutas", "costo", "margen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 4. Filtrado (reglas L1 a L4)
# ---------------------------------------------------------------------------

def filtrar_registros(df):
    """Devuelve las filas válidas y un resumen de cuántas se quitaron por cada regla."""
    reglas = {
        "L1 canceladas": df["_cancelado"],
        "L2 cantidad <= 0": ~(df["cantidad"] > 0),
        "L3 precio < 0": df["precio_unitario"] < 0,
        "L4 fecha inválida": df["fecha"].isna(),
    }

    quitar = pd.Series(False, index=df.index)
    resumen = {}
    for nombre, condicion in reglas.items():
        # Cada fila se cuenta solo en la primera regla que la quita
        nuevas = condicion & ~quitar
        resumen[nombre] = int(nuevas.sum())
        quitar |= condicion

    pedidos_cancelados = df.loc[df["_cancelado"], "id_pedido"].nunique()

    # L3b: las filas con precio 0 se conservan (decisión del 27 de octubre)
    limpio = df.loc[~quitar].copy()
    limpio["cantidad"] = limpio["cantidad"].astype(int)
    limpio = limpio.drop(columns=["_cancelado"])

    # Orden de salida: fecha, hora y número de pedido
    limpio["_num"] = limpio["id_pedido"].str.extract(r"(\d+)$", expand=False).astype(float)
    limpio = limpio.sort_values(["fecha", "hora", "_num"], kind="stable").drop(columns="_num")
    limpio = limpio[COLUMNAS_SALIDA].reset_index(drop=True)

    return limpio, resumen, pedidos_cancelados


# ---------------------------------------------------------------------------
# 5. Controles de calidad (C4 a C13, sobre la salida)
# ---------------------------------------------------------------------------

def verificar_calidad(limpio):
    """Detiene el pipeline si falla un control crítico. Devuelve la lista de avisos."""
    avisos = []

    # C4
    detener_si(len(limpio) != ESPERADO["filas_salida"],
               f"C4: se esperaban {ESPERADO['filas_salida']} filas y hay {len(limpio)}")

    # C5
    platos = limpio["plato"].nunique()
    detener_si(platos != ESPERADO["platos"], f"C5: se esperaban {ESPERADO['platos']} platos y hay {platos}")

    # C6: el día coincide con la fecha y no hay domingos
    dia_real = limpio["fecha"].dt.dayofweek.map(lambda i: DIAS_POR_NUMERO[i])
    distintos = int((dia_real != limpio["dia_semana"]).sum())
    detener_si(distintos > 0, f"C6: {distintos} filas con dia_semana distinto al de la fecha")
    domingos = int((limpio["dia_semana"] == "domingo").sum())
    detener_si(domingos > 0, f"C6: hay {domingos} ventas en domingo")

    # C7: ventas ≈ cantidad × precio, con tolerancia por el redondeo del precio
    diferencia = (limpio["cantidad"] * limpio["precio_unitario"] - limpio["ventas_brutas"]).abs()
    fuera = int((diferencia > limpio["cantidad"] * TOLERANCIA_REDONDEO + 1e-9).sum())
    centavos = int((diferencia >= 0.01 - 1e-9).sum()) - fuera
    if fuera:
        avisos.append(f"C7: {fuera} filas con ventas_brutas lejos de cantidad x precio")
    if centavos:
        avisos.append(f"{centavos} diferencias de centavos por redondeo del precio")

    # C8: costo + margen = ventas (al centavo)
    descuadre = int(((limpio["costo"] + limpio["margen"] - limpio["ventas_brutas"]).abs() > 0.005).sum())
    detener_si(descuadre > 0, f"C8: {descuadre} filas donde costo + margen no da ventas_brutas")

    # C9: sumas totales
    totales = {
        "unidades": int(limpio["cantidad"].sum()),
        "ventas": round(limpio["ventas_brutas"].sum(), 2),
        "costo": round(limpio["costo"].sum(), 2),
        "margen": round(limpio["margen"].sum(), 2),
    }
    for clave, valor in totales.items():
        detener_si(abs(valor - ESPERADO[clave]) > 0.005,
                   f"C9: {clave} esperado {ESPERADO[clave]} y obtenido {valor}")

    # C10: rango de fechas
    fmin = limpio["fecha"].min().strftime("%Y-%m-%d")
    fmax = limpio["fecha"].max().strftime("%Y-%m-%d")
    if (fmin, fmax) != (ESPERADO["fecha_min"], ESPERADO["fecha_max"]):
        avisos.append(f"C10: las fechas van de {fmin} a {fmax}")

    # C11: sin vacíos
    textos = ["id_pedido", "dia_semana", "hora", "plato", "categoria", "canal"]
    vacios = int(limpio.isna().sum().sum() + (limpio[textos] == "").sum().sum())
    detener_si(vacios > 0, f"C11: hay {vacios} valores vacíos")

    # C12: categorías y canales conocidos
    cat_nuevas = set(limpio["categoria"]) - set(CATEGORIAS.values())
    detener_si(bool(cat_nuevas), f"C12: categorías sin nombre normalizado: {cat_nuevas}")
    canales_nuevos = set(limpio["canal"]) - CANALES
    detener_si(bool(canales_nuevos), f"C12: canales desconocidos: {canales_nuevos}")

    # C13: líneas con precio 0 (se conservan, solo se informan)
    cero = limpio[limpio["precio_unitario"] == 0]
    if len(cero):
        nombres = ", ".join(sorted(cero["plato"].unique()))
        avisos.append(f"{len(cero)} líneas con precio 0 ({nombres})")

    return totales, avisos


# ---------------------------------------------------------------------------
# 6. Exportación
# ---------------------------------------------------------------------------

def exportar(limpio, ruta=ARCHIVO_SALIDA):
    salida = limpio.copy()
    salida["fecha"] = salida["fecha"].dt.strftime("%Y-%m-%d")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(ruta, index=False, encoding="utf-8", float_format="%.2f")


# ---------------------------------------------------------------------------

def main():
    print("M1 Ingesta y limpieza")

    crudo = cargar_datos()
    validar_esquema(crudo)
    print(f"  Entrada: {miles(len(crudo))} filas, {crudo.shape[1]} columnas")

    df = estandarizar(crudo)
    limpio, quitadas, pedidos_cancelados = filtrar_registros(df)
    print(f"  Quitadas: {quitadas['L1 canceladas']} líneas canceladas ({pedidos_cancelados} pedidos), "
          f"{quitadas['L2 cantidad <= 0']} con cantidad <= 0,")
    print(f"            {quitadas['L3 precio < 0']} con precio < 0, "
          f"{quitadas['L4 fecha inválida']} con fecha inválida")

    totales, avisos = verificar_calidad(limpio)
    print(f"  Salida: {miles(len(limpio))} filas, {limpio['plato'].nunique()} platos, "
          f"{miles(totales['unidades'])} unidades, £{miles(totales['ventas'], 2)}")
    if avisos:
        print("  Avisos: " + "; ".join(avisos))

    exportar(limpio)
    print(f"  Controles OK -> {ARCHIVO_SALIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    try:
        main()
    except ErrorDeCalidad as error:
        print(f"  ERROR, el pipeline se detiene. {error}")
        raise SystemExit(1)
