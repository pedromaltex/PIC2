# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from plotter import plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10
from regression import log_linear_regression
####################################
# Objetivo: Regressão linear SP500 #
####################################
Monthly_investment = 10
Year = 2008
simulacoes = 100


# Função para obter os dados históricos do S&P 500
def get_data(symbol='^GSPC', period='80y', interval='1mo'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data.to_pickle("S&P500.pkl")
    data = data[['Close']].reset_index()
    return data

try:
    # Carregar de volta (sem perda de tipo)
    sp500_data = pd.read_pickle("/home/pedro-maltez-ubuntu/Documents/PIC2/flask_app/help/S&P500.pkl")
    name = 'S&P 500'
except:
    # Obter dados históricos do S&P 500
    name, periodo, intervalo = '^GSPC', '40y', '1mo'
    sp500_data = get_data(name, periodo, intervalo)
    name = yf.Ticker(name).info['longName']

sp500_data = sp500_data[['Close']].reset_index()

# Criar um vetor de anos com base no número de dados
unit_of_time = np.arange(len(sp500_data))

dates, linear_average_log, y_log, name, x, y, coef_log = log_linear_regression(sp500_data, name)[0],log_linear_regression(sp500_data, name)[1],log_linear_regression(sp500_data, name)[2],log_linear_regression(sp500_data, name)[3],log_linear_regression(sp500_data, name)[4],log_linear_regression(sp500_data, name)[5], np.array([log_linear_regression(sp500_data, name)[7], log_linear_regression(sp500_data, name)[6]])

# Compute means
x_mean = np.mean(x) #
y_mean = np.mean(y) #
y_mean_log = np.mean(y_log) #

# Example of NumPy's polyfit
exp_average = np.exp(linear_average_log) #

# SP500 - exponential (relative)
diference = 100 * (y - exp_average)/exp_average # 

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
year = Year
# Vamos fazer com dados até x estudar 
sp500_data_since_a_year = sp500_data[(sp500_data['Date'] <= '2025-12-31') & (sp500_data['Date'] >= f'{year}-01-01')]
days_year = len(sp500_data_since_a_year)
sp500_data_since_a_year['Close'].values[0][0]

# Obter as datas reais
datas = sp500_data_since_a_year['Date'].reset_index(drop=True)
time = len(datas)

# %%
# Parâmetros
preco_inicial = sp500_data_since_a_year['Close'].values[0]
# Retornos logarítmicos mensais
log_returns = np.log(sp500_data['Close'] / sp500_data['Close'].shift(1)).dropna()
# Desvio padrão mensal
sigma = log_returns.std()
mu = np.log(1 + cagr) / 12
# Matriz de preços
precos = np.zeros((time, simulacoes))
precos[0] = preco_inicial

########################################################################
################# FORMA EFICIENTE CHATGPT ##############################
########################################################################

# Log returns
log_returns = np.log(sp500_data['Close'] / sp500_data['Close'].shift(1)).dropna()

# Se for DataFrame por engano:
if isinstance(log_returns, pd.DataFrame):
    log_returns = log_returns.iloc[:, 0]  # pega a coluna certa

# Corrigir sigma para tipo float
sigma = float(log_returns.std())

# Drift e simulação vetorizada
rand_norm = np.random.normal(size=(time - 1, simulacoes))
drift = mu - 0.5 * sigma**2
steps = np.exp(drift + sigma * rand_norm)

# Preencher a matriz de preços
precos = np.zeros((time, simulacoes))
precos[0] = preco_inicial
precos[1:, :] = preco_inicial * steps.cumprod(axis=0)

########################################################################
########################################################################
########################################################################
# Converter em DataFrame com índice de datas
precos_df = pd.DataFrame(precos, index=datas)
# %%
##########################
# CHATGPT
# Colocar y_pred em 2024 até 2025
# Garante que a coluna 'Date' é datetime
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])

# Filtra os dados com date > 01/01/2024
mask = sp500_data['Date'] > pd.Timestamp(f'{year-1}-12-31')
dates_filtered = sp500_data.loc[mask, 'Date']
y_pred_filtered = y_pred[mask.values]  # y_pred deve ter mesmo comprimento que sp500_data

############################
plot7(precos_df, dates_filtered, y_pred_filtered)

# %%
# MÉTODO TESTE

total_invest = np.zeros(len(precos_df))
total_invest[0] = Monthly_investment
for i in range(1, len(total_invest)):
    total_invest[i] = total_invest[i-1] + Monthly_investment

sp500_price_monte = precos_df / 10

######################## Chat GPT
# Inicializar o array para guardar a evolução de ações compradas por simulação
stocks_owned_matrix = np.zeros_like(sp500_price_monte)

# Iterar por cada simulação (coluna)
stocks_owned_matrix = np.cumsum(Monthly_investment / sp500_price_monte.values, axis=0)

#######################
# %%
pd.DataFrame(stocks_owned_matrix)
porfolio = stocks_owned_matrix * sp500_price_monte # Calcular evolução portfolio
pd.DataFrame(porfolio)

diference = 100 * (precos_df - y_pred_filtered[:, np.newaxis]) / y_pred_filtered[:, np.newaxis] # Em percentagem


# Método de weighted buy
allocation = (Monthly_investment * (1 - 2.5 * diference/100)) # dinheiro investido mês a 

allocation = np.where(allocation < Monthly_investment*0.1, Monthly_investment*0.1, allocation)
total_allocation = np.cumsum(allocation, axis=0)

pd.DataFrame(allocation)
pd.DataFrame(total_allocation) 

stocks_owned2 = allocation / sp500_price_monte

stocks_owned2 = np.cumsum(stocks_owned2, axis=0)

porfolio2 = stocks_owned2 * sp500_price_monte

# Supondo que 'porfolio2' é o teu DataFrame com datas como índice e simulações nas colunas
final_values2 = porfolio2.iloc[-1]  # pega os valores da última data
final_values1 = porfolio.iloc[-1]

plot8(final_values1, final_values2)


# %%
final_allocation = total_allocation[-1, :]  # última linha (últimos valores de cada simulação)

plot9(total_allocation, final_allocation)

final_values2_0 = np.array(final_values2)

roi_maltez = 100*(final_values2 - final_allocation)/final_allocation

monthly_investment_array = np.ones(len(final_values1)) * Monthly_investment

roi_standart = 100*(final_values1 - total_invest[-1])/total_invest[-1]

# Define a largura dos bins
bin_width = roi_maltez.max()/50

# Define os limites globais
min_val = min(roi_standart.min(), roi_maltez.min())
max_val = max(roi_standart.max(), roi_maltez.max())

# Gera os bins com mesma largura
bins = np.arange(np.floor(min_val), np.ceil(max_val) + bin_width, bin_width)

plot10(roi_standart, roi_maltez, bins)

# %%

#############################################