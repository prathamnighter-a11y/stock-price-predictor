# 📈 Stock Price Prediction & Paper Trading System

A machine learning project that predicts Indian stock prices using technical indicators and lets you track prediction accuracy through paper trading.

Built with Python, scikit-learn, Flask, and Yahoo Finance API.

---

## 🔍 What it does

- Downloads live stock data for any NSE-listed stock (e.g. TCS.NS, RELIANCE.NS)
- Calculates real trading indicators: Moving Averages (MA20/MA50), RSI, Volume signals, Momentum
- Trains a **Random Forest model** (200 decision trees) on historical data
- Predicts prices for the next **7 trading days**
- Generates a **BUY / SELL / HOLD signal** based on model output + indicators
- Tracks prediction accuracy through a **paper trading system**
- Serves everything through a **Flask web app** with interactive charts

---

## 🧠 ML Model

| Feature | Details |
|---|---|
| Algorithm | Random Forest Regressor |
| Look-back window | 30 trading days |
| Forecast horizon | 7 trading days |
| Train/test split | 80% / 20% |
| Accuracy metrics | MAE, RMSE, Direction Accuracy |

**Features used:**
- Close price, MA20, MA50, RSI (14-day)
- Volume vs Volume MA20 (high volume signal)
- 5-day and 10-day momentum
- Daily price range (High - Low)

---

## 📁 Project Structure

```
stock-predictor/
│
├── app.py               # Flask web app (main entry point)
├── predict_smart.py     # Upgraded ML model with technical indicators
├── predict_stock.py     # Basic linear regression model (Phase 1)
├── get_stock_data.py    # Downloads and saves stock data to CSV
├── paper_trader.py      # CLI tool to log and track predictions
├── index.html           # Frontend UI for the web app
│
├── TCS_NS_data.csv      # Sample data — TCS (Tata Consultancy Services)
├── RELIANCE_NS_data.csv # Sample data — Reliance Industries
├── paper_trades.csv     # Logged paper trades (auto-generated)
│
└── requirements.txt     # Python dependencies
```

---

## 🚀 How to run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the web app**
```bash
python app.py
```
Then open your browser at `http://localhost:5000`

**3. Run the prediction script directly (no web)**
```bash
python predict_smart.py
```

**4. Use the paper trading CLI**
```bash
python paper_trader.py
```

---

## 📊 Signal Logic

The model combines ML prediction with indicator analysis:

| Condition | Signal |
|---|---|
| 3+ bullish indicators + predicted gain > 1.5% | STRONG BUY |
| 2 bullish indicators + positive prediction | BUY |
| Predicted drop > 1.5% | SELL / AVOID |
| Mixed signals | HOLD |

Bullish indicators checked: predicted 7-day gain > 1.5%, price above MA20, price above MA50, RSI < 50.

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Always paper trade before using real money. Stock markets are unpredictable.

---

## 🛠️ Tech Stack

- **Python** — core language
- **scikit-learn** — Random Forest model
- **pandas / numpy** — data processing
- **yfinance** — live stock data from Yahoo Finance
- **matplotlib** — charts
- **Flask** — web API and server
- **HTML/CSS/JS** — frontend

---

## 👤 Author

Made by: [Vansh Singh]
College: Vivekananda Institute of Professional Studies, Delhi  
Branch: B.Tech AI & Data Science
