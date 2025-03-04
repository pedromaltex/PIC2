import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Configurações
ticker = "IBM"
start_date = (datetime.today() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # Últimos 5 anos

# Baixar dados
stock_data = yf.download(ticker, start=start_date)
dates = stock_data.index.strftime('%Y-%m-%d').tolist()
close_prices = stock_data['Close'].values

# Normalização
scaler = StandardScaler()
normalized_prices = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

# Preparar dados para treino
def create_sequences(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(x), np.array(y)

window_size = 20
x_data, y_data = create_sequences(normalized_prices, window_size)

# Ajuste da dimensão para LSTM
x_data = np.expand_dims(x_data, axis=-1)  # Agora x_data tem shape (samples, window_size, 1)

# Divisão dos dados
test_size = int(len(y_data) * 0.2)
x_train, x_test = x_data[:-test_size], x_data[-test_size:]
y_train, y_test = y_data[:-test_size], y_data[-test_size:]

# Criar Dataset
class StockDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = StockDataset(x_train, y_train)
test_dataset = StockDataset(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = torch.relu(lstm_out[:, -1, :])  # ReLU para evitar vanishing gradient
        return self.fc(lstm_out)

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Treinar o modelo
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.view(x_batch.shape[0], x_batch.shape[1], -1)  # Garantindo formato correto
        optimizer.zero_grad()
        y_pred = model(x_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print(f"Loss: {loss.item()}")  # Printando a perda para monitoramento
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

# Fazer previsões
model.eval()
predictions = []
actual = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.view(x_batch.shape[0], x_batch.shape[1], -1)  # Garantindo formato correto
        preds = model(x_batch).squeeze().numpy()
        predictions.extend(preds)
        actual.extend(y_batch.numpy())

# Reverter normalização
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actual = scaler.inverse_transform(np.array(actual).reshape(-1, 1)).flatten()

# Plotar resultados
plt.figure(figsize=(15, 5))
plt.plot(dates[-len(actual):], actual, label="Actual Prices", color='blue')
plt.plot(dates[-len(predictions):], predictions, label="Predicted Prices", color='red')
plt.legend()
plt.title(f"{ticker} Stock Price Prediction")
plt.xticks(rotation=45)
plt.savefig("meu_anjo.png")
plt.show()