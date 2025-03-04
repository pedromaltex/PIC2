# %% [code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy as dc

# Solicita o símbolo da ação e baixa os dados históricos
stock_symbol = str(input("Stock symbol: ")).upper()
data = yf.download(stock_symbol, start="1960-01-01", end="2025-02-09")
data.reset_index(inplace=True)

# Se houver MultiIndex, remove níveis extras
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.droplevel(1)

# Mantém somente as colunas 'Date' e 'Close'
data = data[['Date', 'Close']]

# Configura o dispositivo (GPU se disponível, caso contrário CPU)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Converte 'Date' para datetime e plota o preço de fechamento
data['Date'] = pd.to_datetime(data['Date'])
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Close'])
plt.title("Preço de Fechamento da Ação")
plt.savefig(f'{stock_symbol}_0.png')
plt.show()

# %% [code]
# Função para preparar o DataFrame com janelas de entrada (lags) e previsão futura
def prepare_dataframe_for_lstm(df, n_steps, forecast_days=365):
    df = dc(df)
    df.set_index('Date', inplace=True)
    # Cria colunas de lags (últimos n_steps dias)
    for i in range(1, n_steps+1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    # Cria colunas para os próximos forecast_days dias
    for i in range(1, forecast_days + 1):
        df[f'Close(t+{i})'] = df['Close'].shift(-i)
    df.dropna(inplace=True)
    return df

# Definições:
lookback = 7       # Últimos 7 dias para entrada
forecast_days = 365  # Prever 365 dias à frente

shifted_df = prepare_dataframe_for_lstm(data, lookback, forecast_days)
#print(shifted_df.head())

# %% [code]
# Converter para NumPy e aplicar escalonamento (scaling)
# A ordem das colunas em shifted_df é:
# ['Close', 'Close(t-1)', ..., 'Close(t-7)', 'Close(t+1)', ..., 'Close(t+365)']
shifted_df_as_np = shifted_df.to_numpy()

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(shifted_df_as_np)

# Definindo as entradas (x) e saídas (y):
# - x: utilizar as 7 colunas de lags (colunas 1 até lookback)
# - y: utilizar as 365 colunas de previsão (colunas lookback+1 até lookback+1+365)
x = scaled_data[:, 1:lookback+1]  
y = scaled_data[:, lookback+1:lookback+1+forecast_days]

print("x shape:", x.shape, "y shape:", y.shape)

# Opcional: inverter a ordem dos lags (se desejar que o dia mais recente venha primeiro)
x = dc(np.flip(x, axis=1))

# %% [code]
# Divide os dados em treinamento (95%) e teste (5%)
split_index = int(len(x) * 0.95)
x_train = x[:split_index]
x_test = x[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

print("Treino:", x_train.shape, y_train.shape)
print("Teste:", x_test.shape, y_test.shape)

# %% [code]
# Redimensiona x para (samples, timesteps, features)
x_train = x_train.reshape((-1, lookback, 1))
x_test = x_test.reshape((-1, lookback, 1))
# y já está no formato (samples, forecast_days)
print("x_train reshaped:", x_train.shape)
print("y_train reshaped:", y_train.shape)

# Converte para tensores do PyTorch
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

# %% [code]
# Cria as classes Dataset e DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]

train_dataset = TimeSeriesDataset(x_train, y_train)
test_dataset = TimeSeriesDataset(x_test, y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Visualiza um batch
for batch in train_loader:
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print("Batch x shape:", x_batch.shape, "Batch y shape:", y_batch.shape)
    break

# %% [code]
# Define o modelo LSTM modificado para saída com 365 dias
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Saída com forecast_days valores
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_size=1, hidden_size=1000, num_stacked_layers=1, output_size=forecast_days)
model.to(device)
print(model)

# %% [code]
# Funções de treinamento e validação
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_index + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {batch_index+1}, Loss: {running_loss/100:.3f}")
            running_loss = 0.0

def validate_one_epoch():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()
    avg_loss = running_loss / len(test_loader)
    print(f"Validation Loss: {avg_loss:.3f}")

learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    validate_one_epoch()

# %% [code]
# Após o treinamento, gera previsões (utilizando torch.no_grad() para desabilitar o cálculo de gradientes)
model.eval()
with torch.no_grad():
    predicted_test = model(x_test.to(device)).to('cpu').numpy()

# Para visualização, escolhe o primeiro exemplo do conjunto de teste
actual_forecast = y_test[0].cpu().numpy()      # shape: (365,)
predicted_forecast = predicted_test[0]           # shape: (365,)

# %% [code]
# Função para aplicar o inverso do escalonamento apenas na parte de forecast.
# O escalador foi ajustado em todas as colunas originais (total de 1 + lookback + forecast_days)
def inverse_transform_forecast(forecast_scaled):
    # Cria um array dummy com a mesma quantidade de colunas da matriz original
    dummy = np.zeros((1, scaled_data.shape[1]))
    # Preenche somente as colunas correspondentes à previsão:
    # As colunas: índice lookback+1 até lookback+forecast_days
    dummy[0, lookback+1:lookback+1+forecast_days] = forecast_scaled
    dummy_inv = scaler.inverse_transform(dummy)
    # Retorna somente a parte da previsão
    return dummy_inv[0, lookback+1:lookback+1+forecast_days]

actual_forecast_inv = inverse_transform_forecast(actual_forecast)
predicted_forecast_inv = inverse_transform_forecast(predicted_forecast)

# %% [code]
# Plot para comparar o forecast real e o previsto (para o primeiro exemplo do conjunto de teste)
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, forecast_days+1), actual_forecast_inv, label='Actual Close')
plt.plot(np.arange(1, forecast_days+1), predicted_forecast_inv, label='Predicted Close')
plt.xlabel('Dia')
plt.ylabel('Preço de Fechamento')
plt.title('Previsão para 365 Dias')
plt.legend()
plt.savefig(f'{stock_symbol}_forecast.png')
plt.show()
