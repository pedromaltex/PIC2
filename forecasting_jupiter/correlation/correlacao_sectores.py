# %%
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from yahooquery import Screener

# %%
def get_companiesbysector(sector):  # Obtém 100 maiores empresas do determinado setor
    sector_mapping = {
        'basic_materials': 'ms_basic_materials',
        'communication_services': 'ms_communication_services',
        'consumer_cyclical': 'ms_consumer_cyclical',
        'consumer_defensive': 'ms_consumer_defensive',
        'energy': 'ms_energy',
        'financial_services': 'ms_financial_services',
        'healthcare': 'ms_healthcare',
        'industrials': 'ms_industrials',
        'real_estate': 'ms_real_estate',
        'technology': 'ms_technology',
        'utilities': 'ms_utilities'
    }

    if sector not in sector_mapping:
        raise ValueError(f"Setor inválido: {sector}. Escolha um dos seguintes: {list(sector_mapping.keys())}")

    screener_id = sector_mapping[sector]
    screener = Screener()
    data = screener.get_screeners(screener_id, count=100)

    # Verifica se os dados retornaram corretamente
    if screener_id not in data or 'quotes' not in data[screener_id]:
        raise ValueError(f"Não foi possível obter dados para o setor: {sector}")

    # Extraindo os símbolos das empresas
    sector_tickers = [stock['symbol'] for stock in data[screener_id]['quotes']]
    return sector_tickers

# %%
# Obtendo tickers dos setores desejados
ticker1 = get_companiesbysector('energy')
ticker2 = get_companiesbysector('communication_services')
ticker1.remove('WDS')
# %%
# Baixar os dados históricos para as empresas dos setores indicados
day, month, year = datetime.now().day, datetime.now().month, datetime.now().year
start_date = f"{year-40}-{month}-{day}"  # 40 anos atrás
end_date = f"{year}-{month}-{day}"

# Baixar os dados usando Yahoo Finance
def download_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)['Close']

dataset1 = download_data(ticker1, start_date, end_date)
dataset2 = download_data(ticker2, start_date, end_date)

# Concatenar os datasets em um único DataFrame
#dataset = pd.concat([dataset1, dataset2], axis=1)


# %%
# Preencher dados ausentes com o último valor disponível
dataset1 = dataset1.fillna(1)
dataset2 = dataset2.fillna(1)


# %%
# Calcular as percentagens diárias
returns1 = dataset1.pct_change().dropna()
returns2 = dataset2.pct_change().dropna()

# %%
correlation_by_company = {}
for company1 in returns1.columns:
    correlation_by_company[f'{company1}'] = {}
    for company2 in returns2.columns:
        merged_df = pd.concat([returns1[company1], returns2[company2]], axis=1)
        correlation_by_company[f'{company1}'][f'{company2}'] = merged_df[company1].corr(merged_df[company2])

# %%
# Converter o dicionário de correlações em um DataFrame
correlation_df = pd.DataFrame.from_dict(correlation_by_company, orient='index')

# %%
# Gerar o heatmap
plt.figure(figsize=(40, 40))
sns.heatmap(correlation_df, annot=False, cmap='coolwarm', center=0, linewidths=0.5, cbar=True)

# Personalizar o gráfico
plt.title('Correlação Anual: Empresas do Setor Financeiro vs. Energia')
plt.xlabel('Ano')
plt.ylabel('Empresas')

# Exibir o gráfico
plt.show()

# %%
