import matplotlib.pyplot as plt


def plot4(x, y1, y2, name, coef_log1, coef_log2):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y1, label='Exponential Growth', linestyle='dashdot', color='red')
    plt.plot(x, y2, label= name, linestyle='solid', color='black')
    plt.title(f'Exponential vs {name} (Log Scale)', fontsize=20)
    x_pos = x.iloc[-10]
    y_pos = min(y1) * 1.05  # um pouco abaixo do topo
    plt.text(x_pos, y_pos, rf'$y ={{{coef_log1:.4f} + {coef_log2:.4f} \cdot x}}$ (Exponential Growth)',
             fontsize=20,
             ha='right', va='bottom',
             color='red',
             bbox=dict(facecolor='white', alpha=0.6))
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()
def soma():
    return 5

def plot1(x, y, name):
    # Plotando os gráficos
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, label='Points', linestyle='solid', color='black')
    plt.title(f'{name} Historical Data', fontsize=20)
    plt.xticks(rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.show()





