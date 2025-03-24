import yfinance as yf
import pandas as pd

# Baixar os dados da Amazon (AMZN)
data = yf.download("AMZN", start="1980-01-01", end="2025-02-09")
#print(data)

# Resetar o índice para transformar a coluna 'Date' em uma coluna normal
data.reset_index(inplace=True)
print(data)

# Caso o DataFrame tenha um MultiIndex, remover os níveis extras
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)  # Remove o primeiro nível do índice

# Selecionar apenas as colunas necessárias
data = data[['Date', 'Close']]

# Exibir os primeiros registros para conferir
print(data)
#


