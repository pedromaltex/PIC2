# LINKS IMPORTANTÍSSIMOS
# https://medium.com/data-hackers/implementando-regress%C3%A3o-linear-simples-em-python-91df53b920a8
# https://www.datacamp.com/pt/tutorial/linear-regression-in-python
# https://brains.dev/2022/pratica-regressao-linear-com-codigo-em-python/
# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

####################################
# Objetivo: Regressão linear SP500 #
####################################

# %%
# Função para obter os dados históricos do S&P 500
def get_data(symbol='^GSPC', period='120y', interval='1d'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data = data[['Close']].reset_index()
    return data

# %%
# sp500_data
# Obter dados históricos do S&P 500
sp500_data = get_data()
sp500_data


# %%
# unit_of_time
# Criar um vetor de anos com base no número de dados
unit_of_time = np.arange(len(sp500_data))
unit_of_time


# %%
# log_sp500
# Criar log do sp500
log_sp500 = np.log(sp500_data['Close'])
log_sp500


# %%
# Regressão linear simples
# Sample data
x = unit_of_time
y = log_sp500

# %%
# x,y
# Garantir que ambos são arrays numpy para evitar erros de broadcasting
x = np.array(x)
y = np.array(y)

x,y
# %%
# x_mean,y_mean
# Compute means
x_mean = np.mean(x)
y_mean = np.mean(y)

x_mean,y_mean


# %%
# Example of NumPy's polyfit
coef = np.polyfit(x, y, 1)
y_pred2 = np.polyval(coef, x)


# %%
## PORQUE ESTÁ ERRADO?????
# Compute slope (B1)
B1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

# Compute intercept (B0)
B0 = y_mean - B1 * x_mean

print(f"Slope (B1): {B1}")
print(f"Intercept (B0): {B0}")

# %%
# Predicting values
y_pred = B0 + B1 * x
print("Predicted values:", y_pred)

# %%
# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], y_pred2, label='Crescimento Exponencial', linestyle='dashed', color='red')
plt.plot(sp500_data['Date'], y_pred, label='Crescimento Exponencial', linestyle='dashed', color='red')
plt.plot(sp500_data['Date'], log_sp500, label='Crescimento Exponencial', linestyle='dashed', color='red')
# Melhorando visualmente
plt.xticks(rotation=45)
plt.xlabel('Ano')
plt.ylabel('Valor')
plt.title('Comparação entre o S&P 500 e uma curva exponencial')
plt.legend()
plt.grid()
plt.show()

