# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from plotter import plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10

####################################
# Objetivo: Regressão linear SP500 #
####################################
Monthly_investment = 500
simulacoes = 10000
Future_Years = 5
future_months = Future_Years * 12


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

# Criar log do sp500
log_sp500 = np.log(sp500_data['Close'])


# Regressão linear simples
x = unit_of_time
y_log = log_sp500
y = np.array(sp500_data['Close'])
y = y.flatten()

# Garantir que ambos são arrays numpy para evitar erros de broadcasting
x = np.array(x)
y_log = np.array(y_log)

# Compute means
x_mean = np.mean(x)
y_mean = np.mean(y)
y_mean_log = np.mean(y_log)

# Example of NumPy's polyfit
coef_log = np.polyfit(x, y_log, 1)
y_pred_log = np.polyval(coef_log, x)
coef_log = coef_log.flatten()
y_pred = np.exp(y_pred_log)

# SP500 - exponential (relative)
diference = 100 * (y - y_pred)/y_pred

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
plot1(sp500_data['Date'], sp500_data['Close'], name)




#plot2(all_dates, y_pred_log, log_sp500, name, coef_log[1], coef_log[0])

# Número de meses futuros a prever
meses_futuros = Future_Years * 12

# Estender o vetor x para o futuro
x_future = np.arange(len(sp500_data), len(sp500_data) + meses_futuros)

# Prever o log do S&P 500 com base nos coeficientes
y_pred_log_future = np.polyval(coef_log, x_future)

# Aplicar exponencial para converter de log-voltar para escala original
y_pred_future = np.exp(y_pred_log_future)


# Criar datas futuras (mensais)
future_dates = pd.date_range(start=sp500_data['Date'].max() + pd.DateOffset(months=1), periods=meses_futuros, freq='MS')

# Juntar dados históricos com futuros para plotar
all_dates = pd.concat([sp500_data['Date'], pd.Series(future_dates)], ignore_index=True)
all_pred = np.concatenate([y_pred, y_pred_future])

def plot34(dates_pred, y_pred_log, y_log_real, name, intercept, slope):
    plt.figure(figsize=(12,6))

    # Real
    plt.plot(dates_pred[:len(y_log_real)], y_log_real, label='Log Real')
    plt.plot(all_dates, np.log(all_pred), label='Exponential Growth', linestyle='dashdot', color='red')

    # Previsão (pode ter mais datas)
    #plt.plot(dates_pred, y_pred_log, label='Previsão (Log)')
    
    plt.title(f'Regressão Logarítmica ({name})')
    plt.xlabel('Ano')
    plt.ylabel('Log(Valor)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot34(all_dates, y_pred_log, log_sp500, name, coef_log[1], coef_log[0])





# Retornos logarítmicos mensais
log_returns = np.log(sp500_data['Close'] / sp500_data['Close'].shift(1)).dropna()



# Corrigir sigma para tipo float
sigma = float(log_returns.std())


# %%
##########################################
# 17/05/2025
# sigma, last_price, y_pred_future, sp500_data['Date'], sp500_data['Close'], simulacoes, y_pred
def future_brownian(sigma, avg_expo, stock_data, years):

    last_price = float(stock_data['Close'].iloc[-1])

    expected_return = (avg_expo[-1] / last_price)**(1/(12*years))
    expected_return = (expected_return - 1)
    mu = np.log(1 + expected_return)  # anualizado

    # Data final dos teus dados reais
    last_date = stock_data['Date'].max()

    # Gerar datas mensais futuras
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods= 12 * years, freq='MS')


    rand_norm = np.random.normal(size=(12*years-1, simulacoes))
    drift = mu - 0.5 * sigma**2
    steps = np.exp(drift + sigma * rand_norm)

    # Inicializar matriz de preços
    precos = np.zeros((12*years, simulacoes))
    precos[0] = last_price
    pd.DataFrame(precos)

    # Corrigir multiplicação
    precos[1:, :] = last_price * np.array(steps.cumprod(axis=0))

    # Gerar DataFrame
    precos_df = pd.DataFrame(precos, index=future_dates)


    plot7(precos_df, future_dates, avg_expo)
    return precos_df
# %%
future_brownian(sigma, y_pred_future, sp500_data, Future_Years)

# %%
