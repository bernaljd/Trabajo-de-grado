""" 
Author: Juan David Bernal
Date: 28/09/2026
"""

import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Read and open Dataset
archiveLoc = '../../dataset/archive/Anonymized_Restaurant_Sales_Data.csv'
df = pd.read_csv(archiveLoc)

#print(df)

df.info()

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# 3. Ahora sí puedes extraer la semana sin problemas
df['Periodo'] = df['Date'].dt.strftime('%G-W%V')
df['Semana_Numero'] = df['Date'].dt.isocalendar().week

#print(df[['Date', 'Periodo', 'Semana_Numero']].head(100))

df.info()