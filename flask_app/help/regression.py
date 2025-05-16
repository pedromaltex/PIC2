import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from plotter import plot1, plot2, plot3, plot4, plot5, plot6, plot7, plot8, plot9, plot10

def log_linear_regression(prices, name):

    y_log =  np.log(prices['Close'])
    x = np.arange(len(prices))
    y = np.array(prices['Close'])
    y = y.flatten()

    # Garantir que ambos são arrays numpy para evitar erros de broadcasting
    x = np.array(x)
    y_log = np.array(y_log)

    # Example of NumPy's polyfit
    coef_log = np.polyfit(x, y_log, 1)
    y_pred_log = np.polyval(coef_log, x)
    coef_log = coef_log.flatten()

    return prices['Date'], y_pred_log, y_log, name, x, y, coef_log[1], coef_log[0]
