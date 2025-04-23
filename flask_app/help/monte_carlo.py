# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

####################################
# Objetivo: Regressão linear SP500 #
####################################

# Função para obter os dados históricos do S&P 500
def get_data(symbol='^GSPC', period='80y', interval='1mo'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data = data[['Close']].reset_index()
    return data

# sp500_data
# Obter dados históricos do S&P 500
name, periodo, intervalo = '^GSPC', '40y', '1mo'
sp500_data = get_data(name, periodo, intervalo)
name = yf.Ticker(name).info['longName']

#sp500_data

# unit_of_time
# Criar um vetor de anos com base no número de dados
unit_of_time = np.arange(len(sp500_data))
#unit_of_time

# log_sp500
# Criar log do sp500
log_sp500 = np.log(sp500_data['Close'])
#log_sp500


# Regressão linear simples
# Sample data
x = unit_of_time
y_log = log_sp500
y = np.array(sp500_data['Close'])
y = y.flatten()
#y

# x,y
# Garantir que ambos são arrays numpy para evitar erros de broadcasting
x = np.array(x)
y_log = np.array(y_log)


#x, y_log, y

# x_mean,y_mean
# Compute means
x_mean = np.mean(x)
y_mean = np.mean(y)
y_mean_log = np.mean(y_log)

#x_mean,y_mean_log

# Example of NumPy's polyfit
coef_log = np.polyfit(x, y_log, 1)
y_pred_log = np.polyval(coef_log, x)
coef_log = coef_log.flatten()
y_pred = np.exp(y_pred_log)
#y_pred

# SP500 - exponential 

diference = 100* (y - y_pred)/y_pred
#diference

# Função que facilita o plot 
def improve_draw():
    # Melhorando visualmente
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()

# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], y_pred_log, label='Exponential Growth', linestyle='dashdot', color='red')
plt.plot(sp500_data['Date'], log_sp500, label=f'{name}', linestyle='solid', color='black')
plt.title(f'Exponential vs {name} (Log Scale)')
x_pos = sp500_data['Date'].iloc[-10]
y_pos = min(y_pred_log) * 1.05  # um pouco abaixo do topo
plt.text(x_pos, y_pos, rf'$y ={{{coef_log[1]:.4f} + {coef_log[0]:.4f} \cdot x}}$ (Exponential Growth)',
         fontsize=13,
         ha='right', va='bottom',
         color='red',
         bbox=dict(facecolor='white', alpha=0.6))

improve_draw()


# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], y_pred, label='Exponential Growth', linestyle='dashdot', color='red')
plt.plot(sp500_data['Date'], y, label=f'{name}', linestyle='solid', color='black')
plt.title(f'Exponential vs {name}')

x_pos = sp500_data['Date'].iloc[-10]
y_pos = min(y) * 1.05  # um pouco abaixo do topo
plt.text(x_pos,y_pos , rf'$y = e^{{{coef_log[1]:.4f} + {coef_log[0]:.4f} \cdot x}}$ (Exponential Growth)',
         fontsize=13,
         ha='right', va='bottom',
         color='red',
         bbox=dict(facecolor='white', alpha=0.6))

improve_draw()



# Plot SP500 - exponential 
# Plotando os gráficos
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], diference, label=f'{name}', linestyle='solid', color='black')
plt.title(f'Exponential vs {name} (Diference)')

improve_draw()


# Calcular rendimento médio do sp500, apenas funciona com períodos em meses
y1 = y_pred[-11]
y2 = y_pred[-23]
percent = 100 * (y1 - y2) / y2
print(f"Preço teste: {y1}")
print(f"Preço teste 12 meses atrás: {y2}")

print(f"Rendimento médio do {name}: {percent}%")

# A inclinação B1 representa o crescimento logarítmico por "unidade de tempo"
# Como estás a usar intervalos mensais:
growth_rate = np.exp(coef_log[0]) - 1
cagr = (1 + growth_rate)**12 - 1  # anualizado

print(f"Crescimento médio mensal: {growth_rate * 100:.2f}%")
print(f"Crescimento médio anual (CAGR): {cagr * 100:.2f}%")

# MÉTODO TESTE
monthly_investment = 30000/len(y)
total_invest = np.zeros(len(y))
total_invest[0] = monthly_investment
for i in range(1, len(total_invest)):
    total_invest[i] = total_invest[i-1] + monthly_investment
total_invest


sp500_price = y / 10

stocks_owned = monthly_investment / sp500_price # Quantas ações consigo comprar com 500 euros
for i in range(len(stocks_owned)-1):
    stocks_owned[i+1] += stocks_owned[i] # Tornar a função acumulativa



porfolio = stocks_owned * sp500_price # Calcular evolução portfolio


# Método de weighted buy
allocation = (monthly_investment * (1 - 2.5 * diference/100)) # dinheiro investido mês a mês
total_allocation = np.zeros(len(allocation))
for i in range(len(allocation)):
    if allocation[i] < 0:
        allocation[i] = 0 # Não retirar dinheiro para não pagar impostos
    total_allocation[i] = sum(allocation[:i+1])

total_allocation 

#allocation

stocks_owned2 = allocation / sp500_price
#stocks_owned2

for i in range(len(stocks_owned2)-1):
    stocks_owned2[i+1] += stocks_owned2[i]
#stocks_owned2
porfolio2 = stocks_owned2 * sp500_price
porfolio2

print(f"Totalidade de dinheiro alocado em Standart Investment: {total_invest[-1]}.")
print(f"Totalidade de carteira de investimento em Standart Investment: {porfolio[-1]}.")

print(f"Totalidade de dinheiro alocado: {total_allocation[-1]}.")
print(f"Totalidade de carteira de investimento: {porfolio2[-1]}.")

plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], porfolio, label='Standart Investment', linestyle='solid', color='red')
plt.plot(sp500_data['Date'], porfolio2, label="Maltez's way", linestyle='dotted', color='blue')
plt.title("Standart Investment vs Maltez's way")
improve_draw()
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], total_invest, label='Standart Investment (Allocation)', linestyle='solid', color='red')
plt.plot(sp500_data['Date'], total_allocation, label="Maltez's way (Allocation)", linestyle='dotted', color='blue')
plt.title("Allocation")
improve_draw()

# Ver se a função Diference tem média 0 num espaço grande de tempo
np.mean(diference) 
plt.figure(figsize=(12, 6))

# Dividir a linha em duas: positiva e negativa
positive_dates = sp500_data['Date'][diference >= 0]
positive_values = diference[diference >= 0]

negative_dates = sp500_data['Date'][diference < 0]
negative_values = diference[diference < 0]

# Desenhar linhas com cores diferentes
plt.plot(positive_dates, positive_values, color='red', label='Positive')
plt.plot(negative_dates, negative_values, color='green', label='Negative')

plt.title(f'Exponential vs {name} (Diference)')
improve_draw()


# %%
# Vamos fazer com dados até 2023 estudar 
sp500_data_since2017 = sp500_data[(sp500_data['Date'] <= '2025-12-31') & (sp500_data['Date'] >= '2017-01-01')]
days_year = len(sp500_data_since2017)
sp500_data_since2017['Close'].values[0][0]

# %%
# Obter as datas reais
datas = sp500_data_since2017['Date'].reset_index(drop=True)
dias = len(datas)

# %%
# Parâmetros
preco_inicial = sp500_data_since2017['Close'].values[0]
mu = 0.8 # Ver o porque de ser este o valor, não deveria ser 0.0749???
sigma = 0.2
simulacoes = 200

# %%
# Matriz de preços
precos = np.zeros((dias, simulacoes))
precos[0] = preco_inicial

# Simulação
for s in range(simulacoes):
    for t in range(1, dias):
        epsilon = np.random.normal()
        drift = (mu - 0.5 * sigma**2) * (1/dias)
        diffusion = sigma * np.sqrt(1/dias) * epsilon
        precos[t, s] = precos[t - 1, s] * np.exp(drift + diffusion)

# Converter em DataFrame com índice de datas
precos_df = pd.DataFrame(precos, index=datas)
precos_df
# %%
##########################
# CHATGPT
# Colocar y_pred em 2024 até 2025
# Garante que a coluna 'Date' é datetime
sp500_data['Date'] = pd.to_datetime(sp500_data['Date'])

# Filtra os dados com date > 01/01/2024
mask = sp500_data['Date'] > pd.Timestamp('2016-12-31')
dates_filtered = sp500_data.loc[mask, 'Date']
y_pred_filtered = y_pred[mask.values]  # y_pred deve ter mesmo comprimento que sp500_data

############################
# Plot com datas reais no eixo X
plt.figure(figsize=(12, 6))
plt.plot(precos_df, alpha=0.6)
plt.plot(dates_filtered, y_pred_filtered, label='Exponential Growth', linestyle='dashdot', color='black')
plt.title("Simulação de Monte Carlo - Movimento Browniano Geométrico (com datas)")
plt.xlabel("Data")
plt.ylabel("Preço simulado")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# MÉTODO TESTE
# monthly investment
monthly_investment =  45000 / len(y_pred_filtered)
monthly_investment = 500
monthly_investment
# %%
total_invest = np.zeros(len(precos_df))
total_invest[0] = monthly_investment
for i in range(1, len(total_invest)):
    total_invest[i] = total_invest[i-1] + monthly_investment
total_invest

# %%
precos_df
# %%
sp500_price_monte = precos_df / 10
sp500_price_monte
# %%
######################## Chat GPT
# Inicializar o array para guardar a evolução de ações compradas por simulação
stocks_owned_matrix = np.zeros_like(sp500_price_monte)

# Iterar por cada simulação (coluna)
for i in range(sp500_price_monte.shape[1]):
    prices = sp500_price_monte.iloc[:, i].values  # preços da simulação i
    stocks = np.zeros_like(prices)
    stocks[0] = monthly_investment / prices[0]
    
    for t in range(1, len(prices)):
        stocks[t] = stocks[t-1] + (monthly_investment / prices[t])
    
    stocks_owned_matrix[:, i] = stocks  # guardar resultado
#######################
# %%
pd.DataFrame(stocks_owned_matrix)
# %%
porfolio = stocks_owned_matrix * sp500_price_monte # Calcular evolução portfolio
pd.DataFrame(porfolio)

#####################################################################################
#Fazer a partir daqui
# %%
# Método de weighted buy
allocation = (monthly_investment * (1 - 2.5 * diference/100)) # dinheiro investido mês a mês
total_allocation = np.zeros(len(allocation))
for i in range(len(allocation)):
    if allocation[i] < 0:
        allocation[i] = 0 # Não retirar dinheiro para não pagar impostos
    total_allocation[i] = sum(allocation[:i+1])

total_allocation 

#allocation

stocks_owned2 = allocation / sp500_price
#stocks_owned2

for i in range(len(stocks_owned2)-1):
    stocks_owned2[i+1] += stocks_owned2[i]
#stocks_owned2
porfolio2 = stocks_owned2 * sp500_price
porfolio2

print(f"Totalidade de dinheiro alocado em Standart Investment: {total_invest[-1]}.")
print(f"Totalidade de carteira de investimento em Standart Investment: {porfolio[-1]}.")

print(f"Totalidade de dinheiro alocado: {total_allocation[-1]}.")
print(f"Totalidade de carteira de investimento: {porfolio2[-1]}.")

plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], porfolio, label='Standart Investment', linestyle='solid', color='red')
plt.plot(sp500_data['Date'], porfolio2, label="Maltez's way", linestyle='dotted', color='blue')
plt.title("Standart Investment vs Maltez's way")
improve_draw()
plt.figure(figsize=(12, 6))
plt.plot(sp500_data['Date'], total_invest, label='Standart Investment (Allocation)', linestyle='solid', color='red')
plt.plot(sp500_data['Date'], total_allocation, label="Maltez's way (Allocation)", linestyle='dotted', color='blue')
plt.title("Allocation")
improve_draw()
# %%
