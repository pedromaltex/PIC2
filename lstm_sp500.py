import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from PIL import Image
import os


# 1. Obter os dados históricos
data = yf.download('^GSPC', start='2010-01-01', end='2025-01-01')
close_prices = data['Close'].values.reshape(-1, 1)

# Pré-processamento
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(close_prices)

# Função para criar dados de treino e teste
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(scaled_prices, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Divisão em treino e teste
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 2. Criar e treinar o modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# Previsões
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 3. Criar imagens para o GIF
images = []
for i in range(len(y_test)):
    plt.figure()
    plt.plot(y_test[:i], label='Real', color='blue')
    plt.plot(y_pred[:i], label='Predicted', color='red')
    plt.legend()
    plt.title(f'Real Price vs Predicted (Frame {i+1})')
    plt.savefig(f'frame_{i}.png')
    plt.close()
    images.append(Image.open(f'frame_{i}.png'))

# 4. Criar um GIF
images[0].save('price_prediction.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

# Limpeza dos arquivos de frame
for i in range(len(y_test)):
    os.remove(f'frame_{i}.png')

print("GIF criado com sucesso!")
