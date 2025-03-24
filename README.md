# PIC
My Graduation assignment

# LSTM Model for S&P 500 Price Prediction

This repository contains a Long Short-Term Memory (LSTM) neural network model for predicting the next day's closing price of the S&P 500 index using historical data.


## Overview

The model is built using TensorFlow/Keras and leverages historical closing prices from the S&P 500 index to forecast future prices. It involves the following steps:

1. **Data Collection**: Historical data is fetched using the `yfinance` library.
2. **Data Preprocessing**: The data is normalized and structured into sequences suitable for LSTM input.
3. **Model Training**: An LSTM model is trained on the preprocessed data.
4. **Evaluation**: The model's predictions are compared to actual prices to assess performance.

## Prerequisites

- Python 3.7+
- TensorFlow
- Keras
- Pandas
- NumPy
- `yfinance`
- Scikit-learn

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/lstm-sp500-prediction.git
cd lstm-sp500-prediction
pip install -r requirements.txt
```

## Usage

1. **Run the Script**

   Execute the main script to train the LSTM model and generate predictions:

   ```bash
   python lstm_sp500.py
   ```

2. **Customization**

   - Modify the `start` and `end` dates in the `yfinance` data fetch section to change the time frame of the historical data.
   - Adjust hyperparameters such as `time_step`, `batch_size`, and `epochs` to optimize model performance.

## Model Structure

The LSTM model is structured as follows:

- Two LSTM layers with 50 units each.
- A final Dense layer for the output.
- Adam optimizer and Mean Squared Error (MSE) loss function.

## Evaluation

The model's performance is evaluated by comparing predicted prices against actual prices from the test dataset. Key metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) can be calculated for further insights.

## Results

Sample predictions vs. actual values:

| Date       | Predicted Price | Actual Price |
|------------|-----------------|--------------|
| 2025-01-02 | 4700.12         | 4720.34      |
| 2025-01-03 | 4715.23         | 4735.56      |

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for providing historical data.
- TensorFlow and Keras teams for the deep learning framework.

---

Feel free to explore, modify, and use this project as a starting point for your own financial predictions and machine learning projects.
