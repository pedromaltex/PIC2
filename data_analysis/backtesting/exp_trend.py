# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from plotter import plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10
from regression import log_linear_regression, get_data, get_file
####################################
# Objetivo: Regressão linear SP500 #
####################################
Monthly_investment = 10
Year = 2008
simulacoes = 100

sp500_data, name = get_file()[0], get_file()[1]

sp500_data = sp500_data[['Close']].reset_index()

dates, linear_average_log, y_log, name, x, y, coef_log = log_linear_regression(sp500_data, name)[0],log_linear_regression(sp500_data, name)[1],log_linear_regression(sp500_data, name)[2],log_linear_regression(sp500_data, name)[3],log_linear_regression(sp500_data, name)[4],log_linear_regression(sp500_data, name)[5], np.array([log_linear_regression(sp500_data, name)[7], log_linear_regression(sp500_data, name)[6]])

# Example of NumPy's polyfit
exp_average = np.exp(linear_average_log)

# SP500 - exponential (relative)
diference = 100 * (y - exp_average)/exp_average

# Plotando os gráficos
plot1(sp500_data['Date'], sp500_data['Close'], name)
plot2(
    log_linear_regression(sp500_data, name)[0],
    log_linear_regression(sp500_data, name)[1],
    log_linear_regression(sp500_data, name)[2],
    log_linear_regression(sp500_data, name)[3],
    log_linear_regression(sp500_data, name)[-2],
    log_linear_regression(sp500_data, name)[-1]
)
plot3(sp500_data['Date'], exp_average, y, name, coef_log[1], coef_log[0])
plot4(sp500_data['Date'], diference, name)


# A inclinação B1 representa o crescimento logarítmico por "unidade de tempo"
# Como estás a usar intervalos mensais:
growth_rate = np.exp(coef_log[0]) - 1
cagr = (1 + growth_rate)**12 - 1  # anualizado
# %%
print(f"Crescimento médio mensal: {growth_rate * 100:.2f}%")
print(f"Crescimento médio anual (CAGR): {cagr * 100:.2f}%")
# %%
# MÉTODO TESTE
total_invest = Monthly_investment * (np.arange(len(y)) + 1)

sp500_price = y / 10

stocks_owned = np.cumsum(Monthly_investment / sp500_price)

porfolio = stocks_owned * sp500_price # Calcular evolução portfolio

# %%
# Método de weighted buy
allocation = (Monthly_investment * (1 - 2.5 * diference/100)) # dinheiro investido mês a mês
total_allocation = np.zeros(len(allocation))
allocation = np.clip(allocation, Monthly_investment * 0.1, Monthly_investment * 2)
total_allocation = np.cumsum(allocation)

stocks_owned2 = allocation / sp500_price

stocks_owned2 = np.cumsum(allocation / sp500_price)

porfolio2 = stocks_owned2 * sp500_price
# %%
print(f"Totalidade de dinheiro alocado em Standart Investment: {total_invest[-1]}.")
print(f"Totalidade de carteira de investimento em Standart Investment: {porfolio[-1]}.")
print(f"Totalidade de dinheiro alocado: {total_allocation[-1]}.")
print(f"Totalidade de carteira de investimento: {porfolio2[-1]}.")
# %%
plot5(sp500_data['Date'], porfolio, porfolio2)
plot6(sp500_data['Date'], total_invest, total_allocation)

# %%
# Ver se a função Diference tem média 0 num espaço grande de tempo
print(f"Média do gráfico diference: {np.mean(diference)}%")

# %%

#############################################