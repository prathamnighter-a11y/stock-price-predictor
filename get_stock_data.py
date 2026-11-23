# ── STOCK DATA DOWNLOADER (fixed for pandas 4.x) ──────────────────
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── CHANGE THIS to any stock you want ─────────────────────────────
TICKER = "TCS.NS"    # changed from RELIANCE.NS
PERIOD = "1y"        # keep this the same
# ──────────────────────────────────────────────────────────────────

print(f"Fetching data for: {TICKER}")
print("-" * 40)

# Download the data
df = yf.download(TICKER, period=PERIOD)

# FIX: newer yfinance returns multi-level columns like ('Close', 'RELIANCE.NS')
# This flattens it back to simple single-level columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Show the raw table
print(f"\nTotal trading days downloaded: {len(df)}")
print("\nSample of your data (last 5 days):")
print(df.tail())

# Key stats about the closing price
print("\nClosing price stats:")
print(f"  Highest price : ₹{df['Close'].max():.2f}")
print(f"  Lowest price  : ₹{df['Close'].min():.2f}")
print(f"  Average price : ₹{df['Close'].mean():.2f}")
print(f"  Latest price  : ₹{df['Close'].iloc[-1]:.2f}")

# Save to CSV for Phase 3
csv_filename = f"{TICKER.replace('.', '_')}_data.csv"
df.to_csv(csv_filename)
print(f"\nData saved to: {csv_filename}")

# ── DRAW THE CHART ─────────────────────────────────────────────────
close_prices = df['Close'].values          # FIX: .values gives a plain 1D array
close_min     = df['Close'].min()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7),
                                gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle(f"{TICKER} — Price History ({PERIOD})", fontsize=14, fontweight='bold')

# Top chart: closing price line + shaded area
ax1.plot(df.index, close_prices, color='steelblue', linewidth=1.5, label='Close Price')
ax1.fill_between(df.index, close_prices, close_min, alpha=0.1, color='steelblue')
ax1.set_ylabel("Price (₹)")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Bottom chart: volume bars
ax2.bar(df.index, df['Volume'].values, color='gray', alpha=0.5, width=1)
ax2.set_ylabel("Volume")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

plt.tight_layout()
plt.show()

print("\nPhase 2 complete! Your data is ready for the prediction model.")