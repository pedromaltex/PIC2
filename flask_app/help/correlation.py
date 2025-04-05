import yfinance as yf
import pandas as pd
from datetime import datetime, timezone

# Empresa alvo
target_ticker = "^GSPC"
# Índice de mercado
market_ticker = "GLD"

day, month, year = datetime.now().day, datetime.now().month, datetime.now().year
start_date_1 = f"{year-15}-{month}-{day}"
end_date = f"{year-10}-{month}-{day}"

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
'''
import yfinance as yf
import pandas as pd
from datetime import datetime

# Ativos para comparar (exemplo: Apple e Ouro)
ticker1 = "AAPL"
ticker2 = "GLD"  # SPDR Gold Trust (ETF de ouro)

# Baixar os dados históricos
day, month, year = datetime.now().day, datetime.now().month, datetime.now().year
data = yf.download([ticker1, ticker2], start=f"{year-20}-{month}-{day}", end=f"{year}-{month}-{day}")["Close"]

# Calcular retornos diários
returns = data.pct_change().dropna()

# Calcular a correlação
correlation = returns.corr().iloc[0, 1]

print(f"Correlação entre {ticker1} e {ticker2}: {correlation:.4f}")'''

