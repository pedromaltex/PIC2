# PIC
My Graduation assignment

# S\&P 500 Investment Strategy Simulator

This project is a backtesting framework designed to analyze dynamic investment strategies based on historical data of the S\&P 500. The goal is to identify whether certain market conditions (e.g., undervaluation or overvaluation based on exponential moving averages) offer better opportunities for investing capital compared to traditional Buy & Hold strategies.

## 🚀 Features

* 📈 Historical simulation of the S\&P 500 using monthly investment contributions
* 🧠 Allocation strategies based on percentiles and valuation indicators
* 🔄 Dynamic rebalancing based on perceived market valuation
* 🎯 Monte Carlo simulations to evaluate strategy robustness
* 📊 Visualization of strategy performance and statistical distribution
* 🔍 Comparison of filtered strategies (e.g., only top 25%-75% performers)

## 📂 Project Structure

```
.
├── data/                  # Contains historical price data (e.g., from yfinance)
├── analysis/              # Core simulation scripts and strategy logic
├── plots/                 # Generated visualizations
├── venv/                  # Python virtual environment (optional)
├── main.py                # Main execution file
├── utils.py               # Helper functions for plotting, EMA calculations, etc.
└── README.md
```

## ⚙️ Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/sp500-investment-simulator.git
cd sp500-investment-simulator
```

2. (Optional) Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```
pip install -r requirements.txt
```

## 📊 How to Use

Make sure your data is stored in a format where each column represents a simulation and each row is a date/index.

Then run the core simulation:

```
python main.py
```

Or use the notebook directly:

```python
from analysis.simulation import run_simulation
run_simulation()
```

## 📈 Strategy Overview

The strategy evaluates whether the S\&P 500 is currently undervalued or overvalued by comparing the current price to an exponential moving average (EMA). Based on this, it allocates capital dynamically:

* 🟢 Undervalued → Invest more
* 🔴 Overvalued → Invest less

These strategies are tested across thousands of simulated investment periods using different market start times and durations.

## 🧪 Percentile Filtering

At the end of each simulation, the final portfolio value is used to compute performance percentiles. Simulations within the interquartile range (25th to 75th percentile) are filtered and plotted to analyze the core performance of the strategy, reducing the influence of outliers and extreme scenarios.

