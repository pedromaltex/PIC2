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
def get_data(symbol='^GSPC', period='200y', interval='1mo'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data = data[['Close']].reset_index()
    return data

# %%
# sp500_data
# Obter dados históricos do S&P 500
sp500_data = get_data()
#sp500_data


# %%
# unit_of_time
# Criar um vetor de anos com base no número de dados
unit_of_time = np.arange(len(sp500_data))
#unit_of_time


# %%
# log_sp500
# Criar log do sp500
log_sp500 = np.log(sp500_data['Close'])
#log_sp500


# %%
# Regressão linear simples
# Sample data
x = unit_of_time
y_log = log_sp500
y = np.array(sp500_data['Close'])

# %%
# x,y
# Garantir que ambos são arrays numpy para evitar erros de broadcasting
x = np.array(x)
y_log = np.array(y_log)


#x, y_log, y
# %%
# x_mean,y_mean
# Compute means
x_mean = np.mean(x)
y_mean = np.mean(y)
y_mean_log = np.mean(y_log)

#x_mean,y_mean_log


# %%
# Example of NumPy's polyfit
coef_log = np.polyfit(x, y_log, 1)
y_pred_log = np.polyval(coef_log, x)

y_pred = np.exp(y_pred_log)
#y_pred


# %%
'''
## PORQUE ESTÁ ERRADO?????
# Compute slope (B1)
B1 = np.sum((x - x_mean) * (y_log - y_mean_log)) / np.sum((x - x_mean) ** 2)

# Compute intercept (B0)
B0 = y_mean_log - B1 * x_mean

print(f"Slope (B1): {B1}")
print(f"Intercept (B0): {B0}")
'''
# %%
'''
# Predicting values
y_pred = B0 + B1 * x
print("Predicted values:", y_pred)
'''

# %%
# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], y_pred_log, label='Exponential Growth', linestyle='dashdot', color='red')
plt.plot(sp500_data['Date'], log_sp500, label='S&P500', linestyle='solid', color='black')
# Melhorando visualmente
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Exponential vs S&P500 (Log Scale)')
plt.legend()
plt.grid()
plt.show()

# %%
# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], y_pred, label='Exponential Growth', linestyle='dashdot', color='red')
plt.plot(sp500_data['Date'], y, label='S&P500', linestyle='solid', color='black')
# Melhorando visualmente
plt.xticks(rotation=45)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Exponential vs S&P500')
plt.legend()
plt.grid()
plt.show()


# %%
# Calcular rendimento médio do sp500, apenas funciona com períodos em meses
y1 = y_pred[-11]
y2 = y_pred[-23]
percent = 100 * (y1 - y2) / y2
print(f"Preço teste: {y1}")
print(f"Preço teste 12 meses atrás: {y2}")

print(f"Rendimento médio do sp500: {percent}%")
# %%
