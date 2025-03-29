import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
from datetime import datetime, timezone
from yahooquery import Screener


def dowl_data_return_dataset(target_data, market_data, start, end):
    # Baixar os dados históricos para os dois períodos
    data1 = yf.download(target_data, start=start, end=end)["Close"]
    data2 = yf.download(market_data, start=start, end=end)["Close"]

    # Criar um DataFrame único com os dados
    data = pd.concat([data1, data2], axis=1)

    return data

def calc_returns_daily(dataset):
    # Calcular retornos diários
    returns = dataset.pct_change().dropna()
    return returns

def calc_corr(dataset_returns):
    # Calcular a correlação
    correlation = dataset_returns.corr().iloc[0, 1]
    return correlation
