import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime
import torch.nn.functional as F

# Baixa o modelo FinBERT
MODEL_NAME = "ProsusAI/finbert"

# Carrega o tokenizador e modelo FinBERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

# Usa GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# API do Finnhub para buscar notícias
API_KEY_FINNHUB = "cv3qp99r01ql2eusvo70cv3qp99r01ql2eusvo7g"

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
    """ Usa FinBERT para analisar o sentimento e retornar um score entre -1 e 1 """
    inputs = tokenizer(news, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Envia para GPU se disponível

    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    
    # FinBERT retorna três probabilidades: [negativo, neutro, positivo]
    sentiment_score = probs[2] - probs[0]  # Score entre -1 e 1
    return round(sentiment_score, 3)

# Teste com notícias da Apple (AAPL)
symbol = "AAPL"
news_list = get_news(symbol)

for i, news in enumerate(news_list):
    score = analyze_sentiment_finbert(news["content"])
    print(f"Notícia {i+1}: {news['title']} -> Score: {score}")
