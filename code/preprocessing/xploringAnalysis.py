"""
Authors: Juan David Bernal 
Date: 27/09/2026
This code generates data charts for chapter 4, 
section 4.2: "Analisis exploratorio".
These charts display data frequency such as daily, 
weekly, monthly, among others. 
"""

import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Read file
archiveLoc = '../../dataset/archive/Anonymized_Restaurant_Sales_Data.csv'
df = pd.read_csv(archiveLoc)

# Test
# print("Dimension de los datos: ", df.shape)
# Columns info
# print(df.info())

# Convert 'Date' format to DD/MM/YYYY
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 1. Extraer el nombre del día de la semana (Monday, Tuesday, etc.)
df['DayName'] = df['Date'].dt.day_name()

# 2. Definir tu orden personalizado: De Sábado a Viernes
orden_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', ]

# 3. Aplicar el orden convirtiendo la columna en una categoría
df['DayName'] = pd.Categorical(df['DayName'], categories=orden_dias, ordered=True)

# 4. Contar cuántos registros hay por día y ordenarlos según tu lista
conteo_semanal = df['DayName'].value_counts().sort_index()
print(conteo_semanal)
# 5. Crear el gráfico de barras
plt.figure(figsize=(10, 6))
conteo_semanal.plot(kind='bar', color='#2196F3')

# Configuración visual del gráfico
plt.title('Data Frequency', fontsize=16)
plt.xlabel('Day Of The Week', fontsize=12)
plt.ylabel('Data Ammount', fontsize=12)
plt.xticks(rotation=45, ha='right') # Rota los textos para que se lean bien
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

for i, valor in enumerate(conteo_semanal):
    plt.text(i, valor + 3, str(valor), ha='center', va='bottom', fontsize=8)

# Guardar la imagen sin pausar el programa
plt.savefig('images/daysChart.png')
plt.show()

####################################################

# 1. Leer el archivo
archiveLoc = '../../dataset/archive/Anonymized_Restaurant_Sales_Data.csv'
df = pd.read_csv(archiveLoc)

# 2. Convertir 'Date' a formato de tiempo
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 3. Extraer Año y Mes combinados (ej. '2023-01', '2023-02')
# El método strftime('%Y-%m') formatea la fecha para que solo deje el Año y el Mes
df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')

# 4. Contar la frecuencia y ordenar cronológicamente
conteo_mensual = df['YearMonth'].value_counts().sort_index()

# 5. Crear la figura (usamos un ancho mayor '14' para acomodar todos los meses)
plt.figure(figsize=(14, 6))
ax = conteo_mensual.plot(kind='bar', color='#FF9800')

# 6. Configuración visual
plt.title('Frecuencia de datos por Mes (2023 - 2025)', fontsize=16)
plt.xlabel('Año y Mes', fontsize=12)
plt.ylabel('Cantidad de datos', fontsize=12)

# Rotar las etiquetas del eje X a 45 grados para que no se sobrepongan
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Opcional: Agregar el número exacto sobre cada barra si lo deseas en tu documento
for i, valor in enumerate(conteo_mensual):
    plt.text(i, valor + 3, str(valor), ha='center', va='bottom', fontsize=8)

# Guardar la gráfica
plt.savefig('images/grafico_meses_años.png')
#print("¡Gráfico mensual generado exitosamente!")

########################## 12 months during 3 years

# 1. Leer el archivo y convertir fechas
archiveLoc = '../../dataset/archive/Anonymized_Restaurant_Sales_Data.csv'
df = pd.read_csv(archiveLoc)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 2. Extraer el nombre del mes de cada fecha
df['MonthName'] = df['Date'].dt.month_name()

# 3. Definir tu orden personalizado: De Enero a Diciembre
orden_meses = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']

# 4. Aplicar el orden convirtiendo la columna en una categoría
df['MonthName'] = pd.Categorical(df['MonthName'], categories=orden_meses, ordered=True)

# 5. Contar cuántos registros hay por mes
conteo_meses = df['MonthName'].value_counts().sort_index()

# 6. Crear el gráfico de barras
plt.figure(figsize=(10, 6))
ax = conteo_meses.plot(kind='bar', color='#E91E63') # Usamos un color diferente (rosado oscuro) para variar

# Configuración visual del gráfico
plt.title('Frecuencia global de datos por Mes del Año', fontsize=16)
plt.xlabel('Mes', fontsize=12)
plt.ylabel('Cantidad de datos', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# (Opcional) Agregar las etiquetas numéricas encima de cada barra
for i, valor in enumerate(conteo_meses):
    plt.text(i, valor + 5, str(valor), ha='center', va='bottom', fontsize=10)

# Guardar la imagen
plt.savefig('images/grafico_12_meses.png')
#print("¡Gráfico de 12 meses generado exitosamente!")