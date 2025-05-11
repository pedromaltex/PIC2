import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from plotter import plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10

# Função para obter os dados históricos do S&P 500
def get_data(symbol='^GSPC', period='80y', interval='1mo'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    data.to_pickle("S&P500.pkl")
    data = data[['Close']].reset_index()
    return data

def get_file():
    try:
        # Carregar de volta (sem perda de tipo)
        data = pd.read_pickle("/home/pedro-maltez-ubuntu/Documents/PIC2/flask_app/help/S&P500.pkl")
        name = 'S&P 500'
    except:
        # Obter dados históricos do S&P 500
        name, periodo, intervalo = '^GSPC', '40y', '1mo'
        data = get_data(name, periodo, intervalo)
        name = yf.Ticker(name).info['longName']
    return data, name

def log_linear_regression(prices, name):

    y_log =  np.log(prices['Close'])
    x = np.arange(len(prices))
    y = np.array(prices['Close'])
    y = y.flatten()

    # Garantir que ambos são arrays numpy para evitar erros de broadcasting
    x = np.array(x)
    y_log = np.array(y_log)

    # Example of NumPy's polyfit
    coef_log = np.polyfit(x, y_log, 1)
    y_pred_log = np.polyval(coef_log, x)
    coef_log = coef_log.flatten()

    return prices['Date'], y_pred_log, y_log, name, x, y, coef_log[1], coef_log[0]
