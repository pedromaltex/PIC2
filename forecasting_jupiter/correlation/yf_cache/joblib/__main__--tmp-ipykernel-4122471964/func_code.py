# first line: 1
@memory.cache
def baixar_dados(ticker, period="1y"):
    print(f"Baixando dados de {ticker} do Yahoo Finance...")
    return yf.download(ticker, period=period)
