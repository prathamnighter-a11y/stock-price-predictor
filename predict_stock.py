# ── STOCK PRICE PREDICTOR ──────────────────────────────────────────
# File: predict_stock.py
# What this does: loads your CSV, trains a model on past prices,
# predicts future prices, and shows you a buy/sell signal
# ──────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ── SETTINGS (change these if you want) ───────────────────────────
CSV_FILE = "TCS_NS_data.csv"    # changed
TICKER   = "TCS.NS"             # changed
LOOK_BACK  = 30   # how many past days the model looks at to predict
FUTURE_DAYS = 7   # how many days ahead to predict
# ──────────────────────────────────────────────────────────────────

print("=" * 50)
print("  STOCK PRICE PREDICTOR")
print("=" * 50)

# ── STEP 1: LOAD YOUR DATA ─────────────────────────────────────────
print("\n[1/5] Loading data...")
df = pd.read_csv(CSV_FILE, header=[0, 1], index_col=0, parse_dates=True)

# Flatten multi-level columns (same fix as before)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Keep only the Close price, drop any missing values
df = df[['Close']].dropna()
prices = df['Close'].values.reshape(-1, 1)

print(f"    Loaded {len(df)} trading days of data")
print(f"    Price range: ₹{df['Close'].min():.2f} → ₹{df['Close'].max():.2f}")

# ── STEP 2: PREPARE THE DATA ───────────────────────────────────────
print("\n[2/5] Preparing data for the model...")

# Scale prices to 0-1 range (models learn better this way)
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(prices)

# Build X (inputs) and y (what we want to predict)
# X = last LOOK_BACK days of prices
# y = the price on the NEXT day
X, y = [], []
for i in range(LOOK_BACK, len(scaled_prices)):
    X.append(scaled_prices[i - LOOK_BACK:i, 0])  # window of past prices
    y.append(scaled_prices[i, 0])                 # next day's price

X = np.array(X)
y = np.array(y)

# Split: 80% for training, 20% for testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"    Training samples : {len(X_train)}")
print(f"    Testing samples  : {len(X_test)}")
print(f"    Look-back window : {LOOK_BACK} days")

# ── STEP 3: TRAIN THE MODEL ────────────────────────────────────────
print("\n[3/5] Training the model...")

model = LinearRegression()
model.fit(X_train, y_train)

print("    Model trained successfully!")

# ── STEP 4: TEST THE MODEL ─────────────────────────────────────────
print("\n[4/5] Testing accuracy...")

# Predict on the test set
y_pred_scaled = model.predict(X_test)

# Convert scaled predictions back to real ₹ prices
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_real = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Calculate accuracy metrics
mae  = mean_absolute_error(y_real, y_pred)
rmse = np.sqrt(mean_squared_error(y_real, y_pred))
avg_price = df['Close'].mean()
accuracy_pct = (1 - mae / avg_price) * 100

print(f"    MAE  (avg error)  : ₹{mae:.2f}")
print(f"    RMSE              : ₹{rmse:.2f}")
print(f"    Rough accuracy    : {accuracy_pct:.1f}%")

# ── STEP 5: PREDICT THE NEXT 7 DAYS ───────────────────────────────
print("\n[5/5] Predicting next days...")

# Start with the last LOOK_BACK days of real data
last_window = scaled_prices[-LOOK_BACK:].flatten().tolist()
future_predictions = []

for day in range(FUTURE_DAYS):
    window_array = np.array(last_window[-LOOK_BACK:]).reshape(1, -1)
    next_scaled  = model.predict(window_array)[0]
    future_predictions.append(next_scaled)
    last_window.append(next_scaled)   # slide the window forward

# Convert future predictions back to real ₹ prices
future_prices = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1)
).flatten()

# Generate future dates (skip weekends)
last_date = df.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                               periods=FUTURE_DAYS)

print(f"\n  Predicted prices for next {FUTURE_DAYS} trading days:")
print(f"  {'Date':<14} {'Predicted Price':>16}")
print("  " + "-" * 32)
for date, price in zip(future_dates, future_prices):
    print(f"  {str(date.date()):<14} ₹{price:>14.2f}")

# ── BUY / SELL SIGNAL ─────────────────────────────────────────────
current_price  = df['Close'].iloc[-1]
predicted_next = future_prices[0]
predicted_week = future_prices[-1]
change_tomorrow = ((predicted_next - current_price) / current_price) * 100
change_week     = ((predicted_week  - current_price) / current_price) * 100

print(f"\n{'='*50}")
print(f"  SIGNAL SUMMARY")
print(f"{'='*50}")
print(f"  Current price     : ₹{current_price:.2f}")
print(f"  Predicted tomorrow: ₹{predicted_next:.2f}  ({change_tomorrow:+.2f}%)")
print(f"  Predicted in {FUTURE_DAYS} days: ₹{predicted_week:.2f}  ({change_week:+.2f}%)")

if change_week > 1.5:
    signal = "BUY"
    reason = "Model expects meaningful upward movement"
elif change_week < -1.5:
    signal = "SELL / AVOID"
    reason = "Model expects meaningful downward movement"
else:
    signal = "HOLD"
    reason = "Model expects price to stay roughly flat"

print(f"\n  Signal : {signal}")
print(f"  Reason : {reason}")
print(f"\n  !! Always paper trade first. This is not financial advice. !!")
print("=" * 50)

# ── DRAW THE CHART ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle(f"{TICKER} — Prediction Model", fontsize=14, fontweight='bold')

# Test period dates (last 20% of data)
test_dates = df.index[split + LOOK_BACK:]

# Plot full historical price (light, for context)
ax.plot(df.index, df['Close'].values,
        color='lightsteelblue', linewidth=1, label='Full history', zorder=1)

# Plot actual prices in test window
ax.plot(test_dates, y_real,
        color='steelblue', linewidth=1.5, label='Actual price (test period)', zorder=2)

# Plot what the model predicted for that same window
ax.plot(test_dates, y_pred,
        color='orange', linewidth=1.5, linestyle='--', label='Model prediction', zorder=3)

# Plot future predictions
ax.plot(future_dates, future_prices,
        color='green', linewidth=2, linestyle='--',
        marker='o', markersize=5, label=f'Next {FUTURE_DAYS} days (forecast)', zorder=4)

# Mark the boundary between known and predicted
ax.axvline(x=df.index[-1], color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(df.index[-1], ax.get_ylim()[0], '  Today', fontsize=9,
        color='gray', va='bottom')

ax.set_ylabel("Price (₹)")
ax.set_xlabel("Date")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.tight_layout()
plt.show()