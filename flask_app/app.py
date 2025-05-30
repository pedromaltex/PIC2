import os

from cs50 import SQL
from flask import Flask, flash, redirect, render_template, request, session
from flask_session import Session
from flask import jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime, timezone
import yfinance as yf
import pandas as pd
import json
import seaborn as sns
import matplotlib.pyplot as plt
from yahooquery import Screener


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from helpers import apology, login_required, lookup, usd, get_data, get_news, correlation, get_interval, get_data_percent, compare2, dowl_data_return_dataset, calc_returns_daily, calc_corr, get_companiesbysector, heatmap, download_data

# Configure application
app = Flask(__name__)

# Custom filter
app.jinja_env.filters["usd"] = usd

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Configure CS50 Library to use SQLite database
db = SQL("sqlite:///finance.db")

# Sets the time of the events
current_time = datetime.now()


@app.after_request
def after_request(response):
    """Ensure responses aren't cached"""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    # Deixar predefinido o gráfico de 1 ano
    selected_options = ['1y']
    stock_total = 0

    period = ['1d', '1wk', '1mo', 'YTD', '1y', '5y', '20y']
    interval = ['1m', '1h', '1h', '1h', '1d', '1wk', '1mo']

    # Processar a seleção do período quando o formulário é submetido
    if request.method == "POST":
        selected_options = request.form.getlist('selecao')

        for i in range(len(period)):
            if selected_options[0] == period[i]:
                data = get_data("^GSPC", period[i], interval[i])  # Sempre usar ^GSPC no index

        session['index_labels'] = data[0]
        session['index_values'] = data[1]
        session['index_selected_options'] = selected_options

        return redirect("/")

    # Obter dados do portfólio do utilizador
    portfolio = db.execute("SELECT * FROM portfolio WHERE user_id = ?", session["user_id"])
    cash = db.execute("SELECT cash FROM users WHERE id = ?", session["user_id"])[0]['cash']

    # Garantir que as variáveis têm os tipos corretos para cálculos no Jinja
    for stock in portfolio:
        stock["total"] = int(stock["shares"]) * float(stock["price"])
        stock_total += stock["total"]

    # Obter os dados do S&P 500 (por enquanto)
    data = get_data("^GSPC", period[4], interval[4])  # TODO: Substituir ^GSPC por PORTFOLIO quando a função estiver completa
    labels, values = data

    # Buscar os dados da sessão para GET requests
    labels = session.get('index_labels', [])
    values = session.get('index_values', [])
    selected_options = session.get('index_selected_options', [])

    #########################################################################################################################
    # TODO: O gráfico da página principal não está a ser do S&P 500 e sim do quoted, e não sei porque isto está a acontecer #
    #########################################################################################################################

    return render_template(
        "index.html",
        portfolio=portfolio,
        cash=float(cash),
        stock_total=float(stock_total),
        portfolio_json=json.dumps(portfolio),
        labels=labels,
        values=values,
        period=period,
        interval=interval,
        selected_options=selected_options
    )


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Get stock quote."""
    if request.method == "POST":
        if not request.form.get("symbol"):
            return apology("must provide symbol", 400)

        stock = lookup(request.form.get("symbol"))
        if stock is None:
            return apology("invalid symbol", 400)

        # Redirect to predicted page
        #TODO Falta fazer a predicted page
        return apology("You still have to do the integration with PIC")
        return render_template("predicted.html", stock=stock)
    return render_template("predict.html")


@app.route("/buy", methods=["GET", "POST"])
@login_required
def buy():
    """Buy shares of stock"""
    if request.method == "POST":

        # Ensure symbol was submitted
        symbol = request.form.get("symbol")
        if not symbol:
            apology("must provide symbol", 400)
        symbol = symbol.upper()
        if not lookup(symbol):
            apology("must provide a valid symbol", 400)

        # Require user's input of shares
        shares = request.form.get("shares")
        try:
            shares = int(shares)
            if shares <= 0:
                return apology("must provide a valid number of shares", 400)
        except ValueError:
            return apology("must provide a valid number of shares", 400)

        # Store stock price and total value of wallet in variables
        stock = lookup(symbol)
        if stock is None:
            return apology("invalid symbol", 400)

        # See current price of the selected stock
        current_price = float(stock["price"])

        # see the total cash in wallet
        total_cash_wallet = db.execute(
            "SELECT cash FROM users WHERE id = ?", session["user_id"]
        )
        total_cash_wallet = total_cash_wallet[0]["cash"]

        # conta quantas vezes há operações com aquele stock
        how_many_buys_stock = db.execute(
            "SELECT COUNT (*) FROM portfolio WHERE symbol = ? AND user_id = ?", symbol, session["user_id"])[0]['COUNT (*)']

        # STORE DATA TO KEEP TRACK OF PURCHASE. TABLES SQL
        if total_cash_wallet > current_price * int(shares):
            if how_many_buys_stock == 0:
                # Registra a compra na tabela portfolio da base de dados pela primeira vez
                db.execute(
                    "INSERT INTO portfolio (user_id, symbol, shares, price) VALUES (?,?,?,?)",
                    session["user_id"], symbol, shares, current_price
                )
            else:
                # Faz update no número de ações detidas pelo usuário
                db.execute(
                    "UPDATE portfolio SET shares = shares + ?, price = ? WHERE user_id = ? AND symbol = ?",
                    int(shares), current_price, session["user_id"], symbol)

            # Guardar no histórico de compras #TODO

            # Update no cash total da carteira
            cost = current_price * float(shares)
            db.execute("UPDATE users SET cash = cash - ? WHERE id = ?",
                       cost, session["user_id"]
                       )
            # Sets the time of the events formated already
            current_time = datetime.now()
            current_time_formatted = current_time.strftime("%Y-%m-%d %H:%M:%S")
            # Update in history
            db.execute(
                "INSERT INTO history (user_id, symbol, buysell, shares, price, time) VALUES (?, ?, 1, ?, ?, ?)",
                session["user_id"], symbol, shares, current_price, current_time_formatted
            )


            # Redirect to home page
            return redirect("/")
        # Render an apology
        return apology("Cash in wallet inferior to desired purchase. Cannot afford the number of shares.")
    else:
        return render_template("buy.html")


@app.route("/history")
@login_required
def history():
    """Show history of transactions"""
    portfolio = db.execute("SELECT * FROM portfolio WHERE user_id = ?",
                           session["user_id"])
    history = db.execute("SELECT * FROM history WHERE user_id = ? ORDER BY time DESC",
                         session["user_id"])

    return render_template("history.html", portfolio=portfolio, history=history)


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":
        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 403)

        # Ensure password was submitted
        elif not request.form.get("password"):
            return apology("must provide password", 403)

        # Query database for username
        rows = db.execute(
            "SELECT * FROM users WHERE username = ?", request.form.get("username")
        )

        # Ensure username exists and password is correct
        if len(rows) != 1 or not check_password_hash(
            rows[0]["hash"], request.form.get("password")
        ):
            return apology("invalid username and/or password", 403)

        # Remember which user has logged in
        session["user_id"] = rows[0]["id"]

        # Redirect user to home page
        return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")


@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare():
    if request.method == "POST": #TODO# Corrigir o facto de aceitar qualquer input nos símbolos de stock

        # In case the user don't put the enough information
        if not ((request.form.get("symbol1") and request.form.get("symbol2")) or \
                (request.form.get("sector1") and request.form.get("sector2"))):
            return apology("must provide symbol or sector", 400)
        
        # If Compare Stocks clicked
        if 'submit_stocks' in request.form:
            # Predefine years in case user don't put it
            try:
                year1 = int(request.form.get("stocks_year1"))
                year2 = int(request.form.get("stocks_year2"))
            except:
                return apology("must provide a valid set of years to analize each sector.", 400)
            # Predefine years in case user don't put it
            if not year1 or not (year1 >= 0 and year1 <= datetime.today().year-1):
                year1 = datetime.today().year - 60
            if not year2 or not (year2 >= 0 and year2 <= datetime.today().year):
                year2 = datetime.today().year
            
            # Empresa alvo
            ticker1 = request.form.get("symbol1")
            

            # Índice de mercado
            ticker2 = request.form.get("symbol2")

            day, month, year = datetime.now().day, datetime.now().month, year1
            start_date_1 = f"{year1}-{month}-{day}"
            end_date = f"{year2}-{month}-{day}"

            dataset = dowl_data_return_dataset(ticker1, ticker2, start_date_1, end_date)
            dataset.fillna(1)

            percentagens_dataset = calc_returns_daily(dataset)

            correlation_by_year = {}
            for years_ago in range(year2-year1+1):
                new_data = percentagens_dataset[percentagens_dataset.index.year == year2-years_ago]
                corr = calc_corr(new_data)
                correlation_by_year[f'{year2-years_ago}'] = corr

                # Criar um DataFrame a partir do dicionário de correlações
                correlation_df = pd.DataFrame(list(correlation_by_year.items()), columns=['Year', 'Correlation'])
                correlation_df.set_index('Year', inplace=True)
            
            # Gerar o heatmap
            plt.figure(figsize=(10, 4))
            sns.heatmap(correlation_df.T, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.1, cbar=True)

            # Personalizar o gráfico
            plt.title(f'Correlação Anual de Retornos: {ticker1.upper()} vs {ticker2.upper()}')
            plt.xlabel('Ano')
            plt.ylabel('Correlação')

            # Exibir o gráfico
            plt.show()

            return render_template("compared.html") # Colocar os dados do plot nesta página, e fazer plot em javascript e não em numpy


        # If Compare Sectors clicked
        elif 'submit_sectors' in request.form:
            # Predefine years in case user don't put it
            try:
                year1 = int(request.form.get("sectors_year1"))
                year2 = int(request.form.get("sectors_year2"))
            except:
                return apology("must provide a valid set of years to analize each sector.", 400)
            # Predefine years in case user don't put it
            if not year1 or not (year1 >= 0 and year1 <= datetime.today().year-1):
                year1 = 2025
            if not year2 or not (year2 >= 0 and year2 <= datetime.today().year):
                year2 = 2024
            
            # Baixar os dados históricos para as empresas dos setores indicados
            day, month = datetime.now().day, datetime.now().month
            start_date = f"{year1}-01-01"  # anos atrás
            end_date = f"{year2}-01-01"

            datasetpro = []
            #for i in range(len(sector_mapping.keys())):
            ticker_pro = get_companiesbysector(request.form.get('sector1'))
            ticker_pro2 = get_companiesbysector(request.form.get('sector2'))

            datasetpro.append(download_data(ticker_pro, start_date, end_date))
            datasetpro.append(download_data(ticker_pro2, start_date, end_date))

            heatmap(datasetpro[0], datasetpro[1])
        
            #TODO# Falta fazer gif para quando diferença de anos é superior a 1


        return render_template("compared.html")

    return render_template("compare.html")


@app.route("/quote", methods=["GET", "POST"])
@login_required
def quote():
    """Get stock quote."""
    if request.method == "POST":
        if not session['via_button']:
            if not request.form.get("symbol"):
                return apology("must provide symbol", 400)

            stock = lookup(request.form.get("symbol"))
            if stock is None:
                return apology("invalid symbol", 400)
            symbol = stock['symbol']

        try:
            stock = lookup(symbol)
        except UnboundLocalError:
            stock = lookup(session['symbol'])
        
        symbol = stock['symbol']

        selected_options = request.form.getlist('selecao')

        if selected_options == []:
            selected_options = ['1y']
        
        data = get_data(symbol, selected_options[0], get_interval(selected_options[0]))
        labels = data[0]
        values = data[1]

        # Store data in the session
        session['labels'] = labels
        session['values'] = values
        session['selected_options'] = selected_options
        session['symbol'] = symbol
        session['stock'] = stock

        # Redirect to quoted page
        session['via_button'] = True


        price_volume = {}
        stats = yf.Ticker(symbol)
        price_volume["marketcap"] = stats.info.get("marketCap")
        price_volume["pe_ratio"] = round(stats.info.get("trailingPE"), 4)
        price_volume["eps"] = round(stats.info.get("epsCurrentYear"), 4)
        price_volume["dividend_yield"] = round(stats.info.get("dividendYield") or 0, 4)
        price_volume["dividend_rate"] = round(stats.info.get("dividendRate") or 0, 4)

        results={}

        # Função auxiliar para evitar erros caso o índice não exista
        def get_financial_value(index_name):
            return stats.financials.loc[index_name].iloc[0] if index_name in stats.financials.index else 0

       # Obtendo valores diretamente do financials
        results["revenue"] = get_financial_value("Total Revenue")
        results["cost_of_revenue"] = - get_financial_value("Cost Of Revenue")
        results["gross_profit"] = get_financial_value("Gross Profit")
        results["earnings"] = get_financial_value("Net Income")
        results["other_expenses"] = results["earnings"] - results["gross_profit"]


        if request.form.get("symbol_compare"): #TODO# COLOCAR O COMPARE DOS HELPERS
            new_stock_symbol = request.form.get("symbol_compare")
            comparison = compare2(session['symbol'], new_stock_symbol, '1wk') #TODO# colocar timeline certa

            #Aqui a correlation está sempre a 5 anos, corrigir
            return render_template("quoted.html", stock=stock, new_stock_symbol=new_stock_symbol.upper(), labels=comparison['label1'], values=comparison['value1'], values2=comparison['value2'], \
                        selected_options=selected_options, correlation=correlation(symbol, new_stock_symbol, 5), \
                        price_volume=price_volume, results=results)


        return render_template("quoted.html", stock=stock, new_stock_symbol=None, labels=labels, values=values, values2=None, \
                        selected_options=selected_options, correlation=correlation(symbol, "^GSPC", 5), \
                        price_volume=price_volume, results=results)

    session['via_button'] = False
    return render_template("quote.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    if request.method == "POST":

        # Ensure username was submitted
        if not request.form.get("username"):
            return apology("must provide username", 400)
        # Ensure password was submitted
        elif not request.form.get("password") or not request.form.get("confirmation"):
            return apology("must provide password", 400)
        # Ensure confirmation password was submitted
        elif request.form.get("confirmation") != request.form.get("password"):
            return apology("password mismatched", 400)

        else:
            try:
                # Query database for username
                db.execute(
                    "INSERT INTO users (username, hash) VALUES (?, ?)", request.form.get(
                        "username"), generate_password_hash(request.form.get("password"))
                )
            except ValueError:
                return apology("user already exists", 400)

        # Redirect to login page
        return redirect("/login")

    # Redirect user to register page
    return render_template("register.html")


@app.route("/sell", methods=["GET", "POST"])
@login_required
def sell():
    """Sell shares of stock"""

    portfolio = db.execute("SELECT * FROM portfolio WHERE user_id = ?",
                           session["user_id"])
    if request.method == "POST":
        symbol = request.form.get("symbol")
        if not symbol:
            return apology("must provide symbol", 400)
        symbol = symbol.upper()
        how_many_buys_stock = db.execute(
            "SELECT COUNT (*) FROM portfolio WHERE symbol = ? AND user_id = ?", symbol, session["user_id"])[0]['COUNT (*)']
        if not lookup(symbol) or how_many_buys_stock == 0:
            return apology("must provide a valid symbol that you own", 400)

        # See the current price of the selected stock (IMPORTANTE PARA O HISTÓRICO)
        current_price = float(lookup(symbol)["price"])

        # Check how many shares the user wants to sell
        shares = int(request.form.get("shares"))

        # Check how many shares the user owns
        shares_owned = db.execute(
            "SELECT shares FROM portfolio WHERE symbol = ? AND user_id = ?", symbol, session["user_id"])[0]['shares']

        if type(shares) != int or shares <= 0 or shares > shares_owned:
            return apology("The number of shares is not supported. Remember, put a positive integer or don't try to sell stocks you don't have.")

        # Sets the time of the events formated already
        current_time = datetime.now()
        current_time_formatted = current_time.strftime("%Y-%m-%d %H:%M:%S")

        # Updates shares owned and adds the transaction to the history
        db.execute(
            "UPDATE portfolio SET shares = ? WHERE user_id = ? AND symbol = ?",
            (int(shares_owned) - int(shares)), session["user_id"], symbol)

        # Elimina da carteira caso se venda tudo
        if (int(shares_owned) - int(shares)) == 0:
            db.execute(
                "DELETE FROM portfolio WHERE user_id = ? AND symbol = ?",
                session["user_id"], symbol)

        db.execute(
            "INSERT INTO history (user_id, symbol, buysell, shares, price, time) VALUES (?, ?, -1, ?, ?, ?)",
            session["user_id"], symbol, shares, current_price, current_time_formatted)
        db.execute(
            "UPDATE users SET cash = cash + ? WHERE id = ?",
            shares * current_price, session["user_id"]
        )

        return redirect("/")
    return render_template("sell.html", portfolio=portfolio)


@app.route("/deposit", methods=["GET", "POST"])
@login_required
def deposit():
    if request.method == "POST":
        deposit = request.form.get("deposit")
        try:
            deposit = int(deposit)
            if deposit <= 0:
                return apology("must provide a valid deposit", 400)
        except ValueError:
            return apology("must provide a valid number of dollars", 400)
        db.execute("UPDATE users SET cash = cash + ? WHERE id = ?",
                   deposit, session["user_id"])
        render_template("deposit.html")

        # Redirect to home page
        return redirect("/")

    return render_template("deposit.html")


# GetPoint da news
@app.route("/news", methods=["GET", "POST"])
@login_required
def news():
    if request.method == "POST":
        symbol = request.form.get("symbol")  # Obtém o símbolo digitado pelo usuário

        if not symbol:
            return apology("Missing stock symbol", 400)

        try:
            news_data = get_news(symbol)  # Busca notícias

            nltk.download("vader_lexicon")
            sia = SentimentIntensityAnalyzer()

            content = []
            i = 0
            for new in news_data:
                print('DONE')
                sentiment = (sia.polarity_scores(new["title"])["compound"] + 
                             sia.polarity_scores(new["content"])["compound"]) / 2
                sentimento = "Neutral"
                if (sentiment >= -0.1) and (sentiment < 0.1):
                    sentimento = "Neutral"
                elif 0.1 < sentiment < 0.5:
                    sentimento = "Buy"
                elif 0.5 < sentiment <= 1:
                    sentimento = "Strong Buy"
                elif -0.5 <= sentiment < -0.1:
                    sentimento = "Sell"
                elif -1 <= sentiment < -0.5:
                    sentimento = "Strong Sell"
                print('DONE2')
                content.append({})
                content[i]['date'] = new['date']
                content[i]['image'] = new['image']
                content[i]['title'] = new['title']
                content[i]['content'] = new['content']
                content[i]['link'] = new['link']
                content[i]['sentiment'] = sentimento
                content[i]['sentiment_num'] = sentiment
                print('DONE3')

                print('*' * 50)
                print(content[0])
                print('*' * 50)
                print(news_data[0]['title'])
                print('*' * 50)
                i+=1


            return render_template("news.html", news=news_data, symbol=symbol, content=content)
        except Exception as e:
            return apology(f"Error fetching news: {str(e)}", 500)
        
        

    return render_template("news.html")  # Exibe a página com o formulário


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
