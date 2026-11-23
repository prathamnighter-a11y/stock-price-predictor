# ── SMART STOCK PREDICTOR (Phase 4) ───────────────────────────────
# Upgrades: adds Moving Average, RSI, and Volume signals
# These are the same indicators real traders use
# ──────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── SETTINGS ──────────────────────────────────────────────────────
CSV_FILE    = "TCS_NS_data.csv"   # change to any stock CSV
TICKER      = "TCS.NS"
LOOK_BACK   = 30
FUTURE_DAYS = 7
# ──────────────────────────────────────────────────────────────────

print("=" * 55)
print("  SMART STOCK PREDICTOR  (Phase 4)")
print("=" * 55)

# ── STEP 1: LOAD DATA ─────────────────────────────────────────────
print("\n[1/6] Loading data...")
df = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0, parse_dates=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Close', 'Volume', 'High', 'Low']].dropna()
print(f"    Loaded {len(df)} trading days")

# ── STEP 2: BUILD INDICATORS ──────────────────────────────────────
print("\n[2/6] Calculating indicators...")

# --- Moving Averages ---
# Average price over last N days. When price > MA20, trend is up.
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

# --- RSI (Relative Strength Index) ---
# Measures if a stock is overbought (>70) or oversold (<30)
delta     = df['Close'].diff()
gain      = delta.clip(lower=0)
loss      = -delta.clip(upper=0)
avg_gain  = gain.rolling(window=14).mean()
avg_loss  = loss.rolling(window=14).mean()
rs        = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# --- Volume Signal ---
# Is today's volume higher than usual? (1 = yes, 0 = no)
df['Vol_MA20']      = df['Volume'].rolling(window=20).mean()
df['High_Volume']   = (df['Volume'] > df['Vol_MA20']).astype(int)

# --- Price momentum ---
# How much did price change over last 5 days?
df['Momentum5']  = df['Close'].pct_change(5)
df['Momentum10'] = df['Close'].pct_change(10)

# --- Candle body size ---
# Difference between high and low (volatility signal)
df['DayRange'] = df['High'] - df['Low']

# Drop rows with NaN (from rolling calculations)
df.dropna(inplace=True)
print(f"    MA20, MA50, RSI, Volume signal, Momentum — all calculated")
print(f"    Usable rows after indicator calculation: {len(df)}")

# Show current indicator values
latest = df.iloc[-1]
print(f"\n    Current indicators for {TICKER}:")
print(f"    Close     : ₹{latest['Close']:.2f}")
print(f"    MA20      : ₹{latest['MA20']:.2f}  ({'Price ABOVE MA20 — bullish' if latest['Close'] > latest['MA20'] else 'Price BELOW MA20 — bearish'})")
print(f"    MA50      : ₹{latest['MA50']:.2f}  ({'Price ABOVE MA50 — bullish' if latest['Close'] > latest['MA50'] else 'Price BELOW MA50 — bearish'})")
print(f"    RSI       : {latest['RSI']:.1f}  ({'Oversold — potential bounce' if latest['RSI'] < 30 else 'Overbought — potential drop' if latest['RSI'] > 70 else 'Neutral zone'})")
print(f"    High Vol  : {'Yes — strong move' if latest['High_Volume'] else 'No — average activity'}")

# ── STEP 3: BUILD FEATURES ────────────────────────────────────────
print("\n[3/6] Building feature matrix...")

# Features the model will learn from
feature_cols = ['Close', 'MA20', 'MA50', 'RSI',
                'High_Volume', 'Momentum5', 'Momentum10', 'DayRange']

# Scale all features to 0-1
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[feature_cols])
y_scaled = scaler_y.fit_transform(df[['Close']])

# Build X windows and y targets
X, y, dates = [], [], []
for i in range(LOOK_BACK, len(X_scaled)):
    X.append(X_scaled[i - LOOK_BACK:i].flatten())  # flatten 30 days × 8 features
    y.append(y_scaled[i, 0])
    dates.append(df.index[i])

X      = np.array(X)
y      = np.array(y)
dates  = np.array(dates)

# 80/20 split
split    = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"    Features per sample : {X.shape[1]}  (30 days × 8 indicators)")
print(f"    Training samples    : {len(X_train)}")
print(f"    Testing samples     : {len(X_test)}")

# ── STEP 4: TRAIN (Random Forest — smarter than Linear Regression) ─
print("\n[4/6] Training Random Forest model...")
print("    (This uses 200 decision trees — still fast!)")

model = RandomForestRegressor(
    n_estimators=200,    # 200 trees
    max_depth=10,        # how deep each tree can go
    random_state=42,     # makes results reproducible
    n_jobs=-1            # use all CPU cores
)
model.fit(X_train, y_train)
print("    Model trained!")

# ── STEP 5: TEST ACCURACY ─────────────────────────────────────────
print("\n[5/6] Testing accuracy...")

y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

mae       = mean_absolute_error(y_real, y_pred)
rmse      = np.sqrt(mean_squared_error(y_real, y_pred))
avg_price = df['Close'].mean()
accuracy  = (1 - mae / avg_price) * 100

# Direction accuracy: did model predict up/down correctly?
real_dir  = np.diff(y_real) > 0
pred_dir  = np.diff(y_pred) > 0
dir_acc   = np.mean(real_dir == pred_dir) * 100

print(f"    MAE  (avg ₹ error)  : ₹{mae:.2f}")
print(f"    RMSE                : ₹{rmse:.2f}")
print(f"    Price accuracy      : {accuracy:.1f}%")
print(f"    Direction accuracy  : {dir_acc:.1f}%  ← most important for trading!")

# ── STEP 6: PREDICT NEXT 7 DAYS ───────────────────────────────────
print("\n[6/6] Predicting next days...")

# Use last window of real data as starting point
last_window = X_scaled[-LOOK_BACK:]
future_prices = []
current_window = last_window.copy()

for day in range(FUTURE_DAYS):
    flat     = current_window.flatten().reshape(1, -1)
    pred_s   = model.predict(flat)[0]
    pred_price = scaler_y.inverse_transform([[pred_s]])[0][0]
    future_prices.append(pred_price)

    # Slide window: drop oldest day, add new predicted row
    # (carry forward last known indicator values, update Close)
    new_row = current_window[-1].copy()
    # Update Close position in feature space
    new_close_scaled = scaler_X.transform(
        [[pred_price if j == 0 else df[feature_cols[j]].iloc[-1]
          for j in range(len(feature_cols))]]
    )[0]
    current_window = np.vstack([current_window[1:], new_close_scaled])

future_prices = np.array(future_prices)
future_dates  = pd.bdate_range(
    start=df.index[-1] + pd.Timedelta(days=1), periods=FUTURE_DAYS
)

# ── SIGNAL SUMMARY ────────────────────────────────────────────────
current_price   = df['Close'].iloc[-1]
pred_tomorrow   = future_prices[0]
pred_week       = future_prices[-1]
chg_tomorrow    = ((pred_tomorrow - current_price) / current_price) * 100
chg_week        = ((pred_week     - current_price) / current_price) * 100

# Extra context from indicators
ma20       = df['MA20'].iloc[-1]
ma50       = df['MA50'].iloc[-1]
rsi        = df['RSI'].iloc[-1]
above_ma20 = current_price > ma20
above_ma50 = current_price > ma50

bullish_count = sum([chg_week > 1.5, above_ma20, above_ma50, rsi < 50])

if bullish_count >= 3:
    signal = "STRONG BUY"
    color  = "GREEN"
elif bullish_count == 2 and chg_week > 0:
    signal = "BUY"
    color  = "GREEN"
elif chg_week < -1.5:
    signal = "SELL / AVOID"
    color  = "RED"
else:
    signal = "HOLD"
    color  = "YELLOW"

print(f"\n{'='*55}")
print(f"  SIGNAL SUMMARY — {TICKER}")
print(f"{'='*55}")
print(f"  Current price       : ₹{current_price:.2f}")
print(f"  Predicted tomorrow  : ₹{pred_tomorrow:.2f}  ({chg_tomorrow:+.2f}%)")
print(f"  Predicted in 7 days : ₹{pred_week:.2f}  ({chg_week:+.2f}%)")
print(f"\n  Indicator checks:")
print(f"  Price vs MA20  : {'ABOVE (bullish)' if above_ma20 else 'BELOW (bearish)'}")
print(f"  Price vs MA50  : {'ABOVE (bullish)' if above_ma50 else 'BELOW (bearish)'}")
print(f"  RSI            : {rsi:.1f} — {'Oversold, bounce likely' if rsi < 30 else 'Overbought, drop possible' if rsi > 70 else 'Neutral'}")
print(f"  Direction acc  : {dir_acc:.1f}% on test data")
print(f"\n  *** SIGNAL : {signal} ***")
print(f"\n  !! Paper trade first. Not financial advice. !!")
print("=" * 55)

print(f"\n  Next {FUTURE_DAYS} trading days:")
print(f"  {'Date':<14} {'Predicted':>12}  {'Change':>8}")
print("  " + "-" * 38)
for d, p in zip(future_dates, future_prices):
    chg = ((p - current_price) / current_price) * 100
    arrow = "↑" if p > current_price else "↓"
    print(f"  {str(d.date()):<14} ₹{p:>10.2f}  {chg:>+7.2f}% {arrow}")

# ── CHART ─────────────────────────────────────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                      gridspec_kw={'height_ratios': [4, 1, 1]})
fig.suptitle(f"{TICKER} — Smart Prediction Model (Phase 4)", fontsize=14, fontweight='bold')

test_dates_arr = dates[split:]

# Top: price + predictions + MAs
ax1.plot(df.index, df['Close'], color='lightsteelblue', linewidth=1, label='Full history')
ax1.plot(test_dates_arr, y_real, color='steelblue', linewidth=1.5, label='Actual (test)')
ax1.plot(test_dates_arr, y_pred, color='orange', linewidth=1.5, linestyle='--', label='Model prediction')
ax1.plot(future_dates, future_prices, color='green', linewidth=2,
         linestyle='--', marker='o', markersize=6, label=f'Next {FUTURE_DAYS} days')
ax1.plot(df.index, df['MA20'], color='purple', linewidth=1, alpha=0.6, linestyle=':', label='MA20')
ax1.plot(df.index, df['MA50'], color='red',    linewidth=1, alpha=0.6, linestyle=':', label='MA50')
ax1.axvline(x=df.index[-1], color='gray', linestyle=':', linewidth=1)
ax1.set_ylabel("Price (₹)")
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Middle: RSI
ax2.plot(df.index, df['RSI'], color='purple', linewidth=1)
ax2.axhline(y=70, color='red',   linestyle='--', linewidth=0.8, alpha=0.7)
ax2.axhline(y=30, color='green', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.fill_between(df.index, df['RSI'], 70,
                 where=(df['RSI'] >= 70), alpha=0.2, color='red',   label='Overbought')
ax2.fill_between(df.index, df['RSI'], 30,
                 where=(df['RSI'] <= 30), alpha=0.2, color='green', label='Oversold')
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Bottom: Volume
ax3.bar(df.index, df['Volume'].values, color='gray', alpha=0.4, width=1)
ax3.plot(df.index, df['Vol_MA20'], color='blue', linewidth=1, label='Vol MA20')
ax3.set_ylabel("Volume")
ax3.set_xlabel("Date")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.tight_layout()
plt.show()