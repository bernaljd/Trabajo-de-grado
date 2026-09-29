import sys
import pandas as pd # type: ignore
import seaborn as sns # type: ignore
import matplotlib # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore
from statsmodels.tsa.seasonal import seasonal_decompose # type: ignore

archiveLoc = '../../dataset/archive/Anonymized_Restaurant_Sales_Data.csv'
df = pd.read_csv(archiveLoc)  # Aquí cargas los datos

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df[df["Cancelled"] == "No"]

# Serie semanal de demanda total
df = (
    df.groupby(pd.Grouper(key="Date", freq="W-SAT"))["Quantity"]
    .sum()
    .to_frame(name="Quantity") # Aquí df se "sobreescribe" con los datos agrupados
)

df.info()

# 1. Gráfico de demanda semanal con corrección de fechas
plt.figure()
plt.plot(df)
plt.ylabel("Unidades vendidas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/ts_demanda_semanal.png", dpi=150)
plt.close()

# 2. Análisis de tendencia estrictamente semanal (Media móvil de 4 semanas)
# Se reemplaza resample("ME") por rolling(4) para preservar la base semanal
tendencia_semanal = df.rolling(window=4).mean()
print(tendencia_semanal)
print(tendencia_semanal.pct_change())

plt.figure()
plt.plot(df, label="Demanda semanal")
plt.plot(tendencia_semanal, label="Tendencia (Media móvil 4 sem)")
plt.legend()
plt.ylabel("Unidades vendidas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("images/ts_demanda_vs_tendencia.png", dpi=150)
plt.close()

# 3. Descomposición estacional con corrección de fechas en subgráficos
decompose = seasonal_decompose(df["Quantity"].dropna(), model="additive", period=52)
fig = decompose.plot()
fig.autofmt_xdate(rotation=45)
plt.tight_layout()
plt.savefig("images/ts_descomposicion.png", dpi=150)
plt.close()

# 4. Gráficos de retardo
def plot_lag_analysis(df, time_column, value_column, lags=5):
    df_lag = pd.DataFrame(df[value_column])
    plt.figure(figsize=(12, 6 * lags))
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(f'Gráficos de Retardo para {value_column}', y=1.02)

    for i in range(1, lags + 1):
        df_lag[f'Lag_{i}'] = df[value_column].shift(i)
        
        plt.subplot(lags, 2, 2 * i - 1)
        sns.scatterplot(x=value_column, y=f'Lag_{i}', data=df_lag)
        plt.title(f'Retardo {i} vs {value_column}')
        plt.xlabel(f'Valor en t')
        plt.ylabel(f'Valor en t-{i}')
        plt.grid(True)

        plt.subplot(lags, 2, 2 * i)
        sns.kdeplot(df_lag[value_column].dropna(), label=f'Valor en t')
        sns.kdeplot(df_lag[f'Lag_{i}'].dropna(), label=f'Valor en t-{i}')
        plt.title(f'Distribución de {value_column} vs Retardo {i}')
        plt.xlabel('Valor')
        plt.ylabel('Densidad')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("images/ts_analisis_retardos.png", dpi=150)
    plt.close()

plot_lag_analysis(df.reset_index(), "Date", "Quantity", 25)