import asyncio
from ollama import AsyncClient


import requests
from datetime import datetime, timezone

# Defina sua chave API do Finnhub
API_KEY = "cv3qp99r01ql2eusvo70cv3qp99r01ql2eusvo7g" 


def get_news(symbol, max_news=100):
    """Obtém notícias financeiras para um símbolo específico usando Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from=2024-01-01&to={datetime.today().strftime("%Y-%m-%d")}&token={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"[ERROR] Falha ao buscar notícias do Finnhub para {symbol}")
            return []

        data = response.json()

        news_list = []
        for article in data[:max_news]:  # Pega até 'max_news' notícias
            date_parsed = datetime.fromtimestamp(article["datetime"], tz=timezone.utc) if "datetime" in article else None

            news_list.append({
                "title": article.get("headline", "Sem título"),
                "link": article.get("url", "#"),
                "content": article.get("summary", "Sem descrição disponível"),
                "date": date_parsed,
                "publisher": article.get("source", "Desconhecido"),
                "image": article.get("image", None)  # Obtém a imagem da notícia
            })

        return news_list

    except Exception as e:
        print(f"[ERROR] Erro ao buscar notícias para {symbol}: {str(e)}")
        return []
    

news = get_news('AAPL')
mensagem = ''
for i in range(10):
   mensagem += f'Evaluate this new: {news[i]['content']}. Is this good to AAPL? Your answer should only be a rating from -1 to 1.\n'

async def chat():
  message = {'role': 'user', 'content': mensagem}
  async for part in await AsyncClient().chat(model='ratingllm', messages=[message], stream=True):
    print(part['message']['content'], end='', flush=True)

asyncio.run(chat())