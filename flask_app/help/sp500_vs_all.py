# %%
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np

#########################################
# Objetivo: Comparar S&P500 com setores #
#########################################

list_of_tickers = ['TLT', 'GLD', 'SH', 'XLU', 'XLP', 'CL=F']

# %%
# Função para obter os dados históricos
def get_data(symbol='^GSPC', period='80y', interval='1mo'):
    data = yf.download(tickers=symbol, period=period, interval=interval)
    if data.empty:
        return pd.DataFrame()
    data = data[['Close']].reset_index()
    return data

# %%
# Parâmetros principais
name_sp500 = '^GSPC'
periodo, intervalo = '20y', '1mo'

# Obter dados históricos do S&P 500
sp500_data = get_data(name_sp500, periodo, intervalo)
sp500_prices = np.array(sp500_data['Close'])
name_sp500_full = yf.Ticker(name_sp500).info.get('longName', name_sp500)

# %%
# Loop pelos outros ativos
for ticker in list_of_tickers:
    try:
        compare_data = get_data(ticker, periodo, intervalo)
        if compare_data.empty:
            print(f'Sem dados para {ticker}')
            continue

        compare_prices = np.array(compare_data['Close'])
        name_compare = yf.Ticker(ticker).info.get('longName', ticker)

        plt.figure(figsize=(12, 6))

        # Plot S&P 500
        plt.plot(sp500_data['Date'], sp500_prices / sp500_prices[0],
                 label=f'{name_sp500_full} ({name_sp500})', linestyle='solid', color='red')

        # Plot ativo a comparar
        plt.plot(compare_data['Date'], compare_prices / compare_prices[0],
                 label=f'{name_compare} ({ticker})', linestyle='solid', color='blue')

        # Título e melhorias visuais
        plt.title(f'{name_sp500_full} vs {name_compare}')
        plt.xticks(rotation=45)
        plt.xlabel('Year')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f'Erro com o ticker {ticker}: {e}')

# %%
