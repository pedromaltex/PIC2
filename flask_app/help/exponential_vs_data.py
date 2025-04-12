# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

# %%
# Função para obter os dados históricos do S&P 500
def get_data(symbol, period='40y', interval='1d'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data = data[['Close']].reset_index()
    return data

# %%
# Obter dados históricos do S&P 500
sp500_data = get_data('^GSPC')
sp500_data

# %%
# Criar um vetor de anos com base no número de dados
years = np.arange(len(sp500_data))
years
# %%
# Taxa de crescimento anual de 10%
growth_rate = 0.10


# %% 
sp500_data['Close'].iloc[0].item()

# %%
log_sp500 = np.log(sp500_data['Close'])
log_sp500

# %%
# Descobrir a de y = ax + b
a = (log_sp500.iloc[-1].item() - log_sp500.iloc[0].item()) / len(log_sp500)
a
# %%
linear_median_curve = np.arange(len(sp500_data))
linear_median_curve = a * linear_median_curve + log_sp500.iloc[0].item()

# %% 
# Calcular a curva de crescimento com base na fórmula P0 * (1 + r)^t
exp_curve = sp500_data['Close'].iloc[0].item() * (1 + growth_rate) ** (years/252)  # 252 é o número aproximado de dias úteis por ano
exp_curve

# %%
# Plotando os gráficos
plt.figure(figsize=(12, 6))
#plt.plot(sp500_data['Date'], sp500_data['Close'], label='S&P 500', color='blue')
#plt.plot(sp500_data['Date'], exp_curve, label='Crescimento Exponencial', linestyle='dashed', color='red')

plt.plot(sp500_data['Date'], linear_median_curve, label='Crescimento Exponencial', linestyle='dashed', color='red')
plt.plot(sp500_data['Date'], log_sp500, label='Crescimento Exponencial', linestyle='dashed', color='red')
# Melhorando visualmente
plt.xticks(rotation=45)
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.title('Comparação entre o S&P 500 e uma curva exponencial')
plt.legend()
plt.grid()
plt.show()
# %%
# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], sp500_data['Close'], label='S&P 500', color='blue')
plt.plot(sp500_data['Date'], exp_curve, label='Crescimento Exponencial', linestyle='dashed', color='red')

# Melhorando visualmente
plt.xticks(rotation=45)
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.title('Comparação entre o S&P 500 e uma curva exponencial')
plt.legend()
plt.grid()
plt.show()

# %%
'''
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Parâmetros da distribuição gaussiana
mean = 0      # Média
dev = 1       # Desvio padrão
x = np.linspace(-5, 5, 1000)  # Intervalo de x para o gráfico

# Função gaussiana
gaussian = (1 / (dev * np.sqrt(2 * pi))) * np.exp(-0.5 * ((x - mean) / dev) ** 2)

# Plotando a gaussiana
plt.plot(x, gaussian, label=f'Média = {mean}, Desvio padrão = {dev}')
plt.title('Distribuição Gaussiana')
plt.xlabel('x')
plt.ylabel('Densidade de Probabilidade')
plt.legend()
plt.grid(True)
plt.show()

'''
