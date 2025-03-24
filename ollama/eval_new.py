from transformers import pipeline
import requests
from datetime import datetime

# Carrega modelo FinBERT
sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert", return_all_scores=True)

# API do Finnhub para buscar notícias
API_KEY_FINNHUB = "TUA_FINNHUB_API_KEY"

def get_news(symbol, max_news=5):
    """ Obtém notícias financeiras de uma empresa """
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to={datetime.today().strftime('%Y-%m-%d')}&token={API_KEY_FINNHUB}"
    response = requests.get(url)
    if response.status_code != 200:
        print("[ERROR] Falha ao buscar notícias")
        return []

    data = response.json()
    return [{"title": article["headline"], "content": article["summary"]} for article in data[:max_news]]

def analyze_sentiment_finbert(news):
    """ Usa FinBERT para retornar um número contínuo entre -1 e 1 """
    results = sentiment_pipeline(news)[0]
    
    # FinBERT retorna probabilidades para "positive", "negative" e "neutral"
    scores = {res["label"]: res["score"] for res in results}
    
    # Calculamos um score contínuo:
    sentiment_score = scores.get("positive", 0) - scores.get("negative", 0)

    return round(sentiment_score, 3)  # Arredonda para melhor visualização

# Teste com notícias da Apple (AAPL)
news_list = get_news("AAPL")
for i, news in enumerate(news_list):
    score = analyze_sentiment_finbert(news["content"])
    print(f"Notícia {i+1}: {news['title']} -> Score: {score}")
