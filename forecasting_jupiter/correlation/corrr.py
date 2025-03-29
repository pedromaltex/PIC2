import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import help

# Lista de tickers das empresas do S&P 500 (Exemplo com alguns tickers - pode ser expandido)
sp500_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "NVDA", "V", "JNJ",  # Exemplos
    "PG", "DIS", "PYPL", "MA", "HD", "UNH", "VZ", "INTC", "CSCO", "ADBE"  # Outros exemplos
]

# Empresa alvo (ouro)
gold_ticker = "GLD"

# Baixar os dados históricos para as empresas do S&P 500 + GLD
day, month, year = datetime.now().day, datetime.now().month, datetime.now().year
start_date = f"{year-10}-{month}-{day}"  # Começo de 20 anos atrás
end_date = f"{year}-{month}-{day}"

# Baixar os dados
dataset = help.dowl_data_return_dataset(gold_ticker, sp500_tickers, start_date, end_date)

# Calcular as percentagens diárias
percentagens_dataset = help.calc_returns_daily(dataset)

# Calcular as correlações anuais
correlation_by_year = {}
for years_ago in range(11):
    # Filtrar os dados para o ano específico
    new_data = percentagens_dataset[percentagens_dataset.index.year == year - years_ago]
    
    # Calcular a correlação entre o ouro (GLD) e todas as empresas do S&P 500
    correlations = {}
    for ticker in sp500_tickers:
        corr = new_data[ticker].corr(new_data[gold_ticker])  # Correlação direta
        correlations[ticker] = corr
    
    correlation_by_year[f'{year - years_ago}'] = correlations

# Transformar as correlações em um DataFrame para visualização
correlation_df = pd.DataFrame(correlation_by_year)

# Gerar o heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=1, cbar=True)

# Personalizar o gráfico
plt.title('Correlação Anual: Ouro (GLD) vs Empresas do S&P 500')
plt.xlabel('Ano')
plt.ylabel('Empresas do S&P 500')

# Exibir o gráfico
plt.show()
