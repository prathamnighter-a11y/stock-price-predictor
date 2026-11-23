from flask import Flask, jsonify, request, send_from_directory
import pandas as pd
import numpy as np
import yfinance as yf
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

LOOK_BACK   = 30
FUTURE_DAYS = 7
TRADES_FILE = "paper_trades.csv"

# ── Serve index.html at root ───────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ── Download + run full prediction for any ticker ─────────────────
@app.route("/api/predict")
def predict():
    ticker = request.args.get("ticker", "TCS.NS").upper().strip()

    try:
        # Download 2 years of data fresh
        raw = yf.download(ticker, period="2y", progress=False)
        if raw.empty:
            return jsonify({"error": f"No data found for {ticker}. Check the ticker symbol."}), 404

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[['Close', 'Volume', 'High', 'Low']].dropna().copy()

        # ── Indicators ────────────────────────────────────────────
        df['MA20']       = df['Close'].rolling(20).mean()
        df['MA50']       = df['Close'].rolling(50).mean()
        delta            = df['Close'].diff()
        gain             = delta.clip(lower=0)
        loss             = -delta.clip(upper=0)
        avg_gain         = gain.rolling(14).mean()
        avg_loss         = loss.rolling(14).mean()
        rs               = avg_gain / avg_loss
        df['RSI']        = 100 - (100 / (1 + rs))
        df['Vol_MA20']   = df['Volume'].rolling(20).mean()
        df['High_Volume']= (df['Volume'] > df['Vol_MA20']).astype(int)
        df['Momentum5']  = df['Close'].pct_change(5)
        df['Momentum10'] = df['Close'].pct_change(10)
        df['DayRange']   = df['High'] - df['Low']
        df.dropna(inplace=True)

        feature_cols = ['Close','MA20','MA50','RSI',
                        'High_Volume','Momentum5','Momentum10','DayRange']

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(df[feature_cols])
        y_scaled = scaler_y.fit_transform(df[['Close']])

        X, y = [], []
        for i in range(LOOK_BACK, len(X_scaled)):
            X.append(X_scaled[i - LOOK_BACK:i].flatten())
            y.append(y_scaled[i, 0])

        X, y = np.array(X), np.array(y)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=200, max_depth=10,
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred_s = model.predict(X_test)
        y_pred   = scaler_y.inverse_transform(y_pred_s.reshape(-1,1)).flatten()
        y_real   = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()

        mae      = mean_absolute_error(y_real, y_pred)
        rmse     = np.sqrt(mean_squared_error(y_real, y_pred))
        avg_p    = df['Close'].mean()
        accuracy = (1 - mae / avg_p) * 100
        real_dir = np.diff(y_real) > 0
        pred_dir = np.diff(y_pred) > 0
        dir_acc  = float(np.mean(real_dir == pred_dir) * 100)

        # ── Future prediction ─────────────────────────────────────
        current_window = X_scaled[-LOOK_BACK:].copy()
        future_prices  = []
        for _ in range(FUTURE_DAYS):
            flat      = current_window.flatten().reshape(1, -1)
            pred_s    = model.predict(flat)[0]
            pred_price= scaler_y.inverse_transform([[pred_s]])[0][0]
            future_prices.append(float(pred_price))
            new_close_scaled = scaler_X.transform(
                [[pred_price if j == 0 else df[feature_cols[j]].iloc[-1]
                  for j in range(len(feature_cols))]]
            )[0]
            current_window = np.vstack([current_window[1:], new_close_scaled])

        future_dates = pd.bdate_range(
            start=df.index[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS
        )

        # ── Signal logic ──────────────────────────────────────────
        current_price = float(df['Close'].iloc[-1])
        pred_week     = future_prices[-1]
        chg_week      = ((pred_week - current_price) / current_price) * 100
        chg_tomorrow  = ((future_prices[0] - current_price) / current_price) * 100
        ma20          = float(df['MA20'].iloc[-1])
        ma50          = float(df['MA50'].iloc[-1])
        rsi           = float(df['RSI'].iloc[-1])
        above_ma20    = current_price > ma20
        above_ma50    = current_price > ma50
        bullish_count = sum([chg_week > 1.5, above_ma20, above_ma50, rsi < 50])

        if bullish_count >= 3:
            signal = "STRONG BUY"
        elif bullish_count == 2 and chg_week > 0:
            signal = "BUY"
        elif chg_week < -1.5:
            signal = "SELL / AVOID"
        else:
            signal = "HOLD"

        # Last 60 days for the chart
        hist_df    = df.tail(60)
        chart_data = {
            "dates":  [str(d.date()) for d in hist_df.index],
            "close":  [round(float(v), 2) for v in hist_df['Close']],
            "ma20":   [round(float(v), 2) for v in hist_df['MA20']],
            "ma50":   [round(float(v), 2) for v in hist_df['MA50']],
            "rsi":    [round(float(v), 2) for v in hist_df['RSI']],
            "volume": [int(v) for v in hist_df['Volume']],
        }
        future_data = [
            {"date": str(d.date()), "price": round(p, 2)}
            for d, p in zip(future_dates, future_prices)
        ]

        return jsonify({
            "ticker":        ticker,
            "current_price": round(current_price, 2),
            "ma20":          round(ma20, 2),
            "ma50":          round(ma50, 2),
            "rsi":           round(rsi, 1),
            "above_ma20":    above_ma20,
            "above_ma50":    above_ma50,
            "high_volume":   bool(df['High_Volume'].iloc[-1]),
            "pred_tomorrow": round(future_prices[0], 2),
            "pred_7day":     round(pred_week, 2),
            "chg_tomorrow":  round(chg_tomorrow, 2),
            "chg_week":      round(chg_week, 2),
            "signal":        signal,
            "dir_accuracy":  round(dir_acc, 1),
            "price_accuracy":round(accuracy, 1),
            "mae":           round(mae, 2),
            "rmse":          round(rmse, 2),
            "chart":         chart_data,
            "future":        future_data,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Paper trades API ──────────────────────────────────────────────
def load_trades():
    if os.path.exists(TRADES_FILE):
        return pd.read_csv(TRADES_FILE)
    return pd.DataFrame(columns=[
        'Date','Ticker','Signal','Price_At_Signal',
        'Predicted_7d','Check_Date','Actual_Price',
        'Predicted_Change_pct','Actual_Change_pct',
        'Direction_Correct','Result'
    ])

@app.route("/api/trades", methods=["GET"])
def get_trades():
    df = load_trades()
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/trades", methods=["POST"])
def add_trade():
    data   = request.json
    ticker = data.get("ticker", "").upper()
    signal = data.get("signal", "")
    price  = float(data.get("price", 0))
    pred   = float(data.get("predicted", 0))

    check_date = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
    pred_chg   = round(((pred - price) / price) * 100, 2)

    df = load_trades()
    new_row = {
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'Ticker': ticker, 'Signal': signal,
        'Price_At_Signal': price, 'Predicted_7d': pred,
        'Check_Date': check_date, 'Actual_Price': '',
        'Predicted_Change_pct': pred_chg,
        'Actual_Change_pct': '', 'Direction_Correct': '',
        'Result': 'PENDING'
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(TRADES_FILE, index=False)
    return jsonify({"status": "saved", "trade": new_row})

@app.route("/api/trades/check", methods=["POST"])
def check_trades():
    df = load_trades()
    if df.empty:
        return jsonify({"checked": 0})

    today    = datetime.now().date()
    checked  = 0
    for i, row in df.iterrows():
        if row['Result'] != 'PENDING':
            continue
        try:
            check_dt = datetime.strptime(str(row['Check_Date'])[:10], "%Y-%m-%d").date()
        except:
            continue
        if today >= check_dt:
            actual = None
            try:
                raw = yf.download(row['Ticker'], period="5d", progress=False)
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                actual = float(raw['Close'].iloc[-1])
            except:
                pass
            if actual:
                orig        = float(row['Price_At_Signal'])
                pred        = float(row['Predicted_7d'])
                actual_chg  = round(((actual - orig) / orig) * 100, 2)
                pred_chg    = float(row['Predicted_Change_pct'])
                dir_correct = (actual_chg > 0) == (pred_chg > 0)
                df.at[i, 'Actual_Price']       = round(actual, 2)
                df.at[i, 'Actual_Change_pct']  = actual_chg
                df.at[i, 'Direction_Correct']  = dir_correct
                df.at[i, 'Result']             = 'WIN' if dir_correct else 'LOSS'
                checked += 1

    df.to_csv(TRADES_FILE, index=False)
    return jsonify({"checked": checked, "trades": df.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
