# %%
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from yahooquery import Screener

# %%
sector_mapping2 = [
        'basic_materials',
        'communication_services',
        'consumer_cyclical',
        'consumer_defensive',
        'energy',
        'financial_services',
        'healthcare',
        'industrials',
        'real_estate',
        'technology',
        'utilities']
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
# Função mãe
def mae(dataset1, dataset2):

    # Preencher dados ausentes com o último valor disponível
    dataset1 = dataset1.fillna(1)
    dataset2 = dataset2.fillna(1)


    # Calcular as percentagens diárias
    returns1 = dataset1.pct_change().dropna()
    returns2 = dataset2.pct_change().dropna()


    correlation_by_company = {}
    for company1 in returns1.columns:
        correlation_by_company[f'{company1}'] = {}
        for company2 in returns2.columns:
            merged_df = pd.concat([returns1[company1], returns2[company2]], axis=1)
            correlation_by_company[f'{company1}'][f'{company2}'] = merged_df[company1].corr(merged_df[company2])


    # Converter o dicionário de correlações em um DataFrame
    correlation_df = pd.DataFrame.from_dict(correlation_by_company, orient='index')


    # Gerar o heatmap
    plt.figure(figsize=(40, 40))
    sns.heatmap(correlation_df, annot=False, cmap='coolwarm', center=0, linewidths=0.5, cbar=True)

    # Personalizar o gráfico
    plt.title(f'Correlação Anual: Empresas do Setor {sector_mapping2[i]} vs. {sector_mapping2[j]}')
    plt.xlabel('Ano')
    plt.ylabel('Empresas')

    # Exibir o gráfico
    plt.show()

# %%
# Baixar os dados usando Yahoo Finance
def download_data(tickers, start, end):
    return yf.download(tickers, start=start, end=end)['Close']

# Baixar os dados históricos para as empresas dos setores indicados
day, month, year = datetime.now().day, datetime.now().month, datetime.now().year
start_date = f"{year-40}-{month}-{day}"  # 40 anos atrás
end_date = f"{year}-{month}-{day}"
# %%
datasetpro = []
for i in range(len(sector_mapping2)):
    ticker_pro = get_companiesbysector(sector_mapping2[i])
    datasetpro.append(download_data(ticker_pro, start_date, end_date))

# %%
for i in range(len(sector_mapping2)):
    # Obtendo tickers dos setores desejados
    dataset1 = datasetpro[i]
    for j in range(len(sector_mapping2)):
        if i != j:
            dataset2 = datasetpro[j]
            mae(dataset1, dataset2)