import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# Empresa alvo
target_ticker = "AAPL"
# Índice de mercado
market_ticker = "TSLA"

day, month, year = datetime.now().day, datetime.now().month, datetime.now().year
start_date_1 = f"{year-1}-{month}-{day}"
end_date = f"{year}-{month}-{day}"

# Baixar os dados históricos para os dois períodos
data1 = yf.download(target_ticker, start=start_date_1, end=end_date)["Close"]
market_data = yf.download(market_ticker, start=start_date_1, end=end_date)["Close"]

# Criar um DataFrame único com os dados
data = pd.concat([data1, market_data], axis=1)

# Calcular retornos diários
returns = data.pct_change().dropna()

# Calcular a correlação
correlation = returns.corr().iloc[0, 1]

print(f"Correlação de {target_ticker} com {market_ticker}: {correlation:.4f}")
