# ── PAPER TRADING TRACKER (Phase 5) ───────────────────────────────
# Logs every signal your model gives, tracks if it was right,
# and shows your running score — all saved to a CSV file
# ──────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# ── SETTINGS ──────────────────────────────────────────────────────
TRADES_FILE = "paper_trades.csv"   # where your trades get saved
# ──────────────────────────────────────────────────────────────────

# ── HELPER: load or create the trades log ─────────────────────────
def load_trades():
    if os.path.exists(TRADES_FILE):
        return pd.read_csv(TRADES_FILE, parse_dates=['Date', 'Check_Date'])
    else:
        return pd.DataFrame(columns=[
            'Date', 'Ticker', 'Signal', 'Price_At_Signal',
            'Predicted_7d', 'Check_Date', 'Actual_Price',
            'Predicted_Change_pct', 'Actual_Change_pct',
            'Direction_Correct', 'Result'
        ])

def save_trades(df):
    df.to_csv(TRADES_FILE, index=False)
    print(f"    Saved to {TRADES_FILE}")

# ── HELPER: get current price of any stock ────────────────────────
def get_current_price(ticker):
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data['Close'].iloc[-1])
    except:
        return None

# ── MENU ──────────────────────────────────────────────────────────
def show_menu():
    print("\n" + "=" * 55)
    print("  PAPER TRADING TRACKER")
    print("=" * 55)
    print("  1. Log a new signal")
    print("  2. Check pending trades (see if model was right)")
    print("  3. View my scorecard")
    print("  4. Show performance chart")
    print("  5. Exit")
    print("=" * 55)
    return input("  Choose (1-5): ").strip()

# ── OPTION 1: Log a new signal ────────────────────────────────────
def log_signal(trades):
    print("\n--- Log New Signal ---")
    ticker   = input("  Ticker (e.g. TCS.NS): ").strip().upper()
    signal   = input("  Signal (BUY / SELL / HOLD): ").strip().upper()

    # Auto-fetch current price
    print(f"  Fetching current price for {ticker}...")
    price = get_current_price(ticker)
    if price:
        print(f"  Current price: ₹{price:.2f}")
        confirm = input("  Use this price? (y/n): ").strip().lower()
        if confirm != 'y':
            price = float(input("  Enter price manually: ₹"))
    else:
        price = float(input("  Could not fetch price. Enter manually: ₹"))

    predicted_7d = float(input(f"  Your model's 7-day predicted price: ₹"))
    pred_change  = ((predicted_7d - price) / price) * 100

    today      = datetime.today()
    check_date = today + timedelta(days=10)   # check after 10 calendar days (~7 trading days)

    new_row = {
        'Date'               : today.strftime('%Y-%m-%d'),
        'Ticker'             : ticker,
        'Signal'             : signal,
        'Price_At_Signal'    : round(price, 2),
        'Predicted_7d'       : round(predicted_7d, 2),
        'Check_Date'         : check_date.strftime('%Y-%m-%d'),
        'Actual_Price'       : None,
        'Predicted_Change_pct': round(pred_change, 2),
        'Actual_Change_pct'  : None,
        'Direction_Correct'  : None,
        'Result'             : 'PENDING'
    }

    trades = pd.concat([trades, pd.DataFrame([new_row])], ignore_index=True)
    save_trades(trades)

    print(f"\n  Logged! Signal: {signal} on {ticker} at ₹{price:.2f}")
    print(f"  Predicted 7-day price: ₹{predicted_7d:.2f} ({pred_change:+.2f}%)")
    print(f"  Check back on: {check_date.strftime('%d %b %Y')}")
    return trades

# ── OPTION 2: Check pending trades ────────────────────────────────
def check_pending(trades):
    print("\n--- Checking Pending Trades ---")

    today   = datetime.today()
    pending = trades[
        (trades['Result'] == 'PENDING') &
        (pd.to_datetime(trades['Check_Date']) <= today)
    ]

    if len(pending) == 0:
        print("  No trades are due for checking yet.")
        upcoming = trades[trades['Result'] == 'PENDING']
        if len(upcoming) > 0:
            print(f"  You have {len(upcoming)} pending trade(s) not due yet:")
            for _, row in upcoming.iterrows():
                print(f"    {row['Ticker']} — check on {row['Check_Date']}")
        return trades

    print(f"  {len(pending)} trade(s) ready to check!\n")

    for idx, row in pending.iterrows():
        ticker = row['Ticker']
        print(f"  Fetching current price for {ticker}...")
        actual = get_current_price(ticker)

        if actual is None:
            actual = float(input(f"  Could not fetch. Enter current price of {ticker}: ₹"))

        actual_change = ((actual - row['Price_At_Signal']) / row['Price_At_Signal']) * 100
        pred_change   = row['Predicted_Change_pct']

        # Direction correct = both predicted and actual move in same direction
        dir_correct = (pred_change > 0 and actual_change > 0) or \
                      (pred_change < 0 and actual_change < 0) or \
                      (abs(pred_change) < 1.5 and abs(actual_change) < 1.5)

        # Result: did the signal make money?
        if row['Signal'] == 'BUY' and actual_change > 0:
            result = 'WIN'
        elif row['Signal'] in ['SELL', 'SELL / AVOID'] and actual_change < 0:
            result = 'WIN'
        elif row['Signal'] == 'HOLD' and abs(actual_change) < 2:
            result = 'WIN'
        else:
            result = 'LOSS'

        trades.at[idx, 'Actual_Price']       = round(actual, 2)
        trades.at[idx, 'Actual_Change_pct']  = round(actual_change, 2)
        trades.at[idx, 'Direction_Correct']  = dir_correct
        trades.at[idx, 'Result']             = result

        print(f"  {ticker} | Signal: {row['Signal']}")
        print(f"    Entry price    : ₹{row['Price_At_Signal']:.2f}")
        print(f"    Current price  : ₹{actual:.2f}  ({actual_change:+.2f}%)")
        print(f"    Model predicted: ₹{row['Predicted_7d']:.2f}  ({pred_change:+.2f}%)")
        print(f"    Direction right: {'YES' if dir_correct else 'NO'}")
        print(f"    Result         : {result}")
        print()

    save_trades(trades)
    return trades

# ── OPTION 3: Scorecard ───────────────────────────────────────────
def show_scorecard(trades):
    print("\n--- My Scorecard ---")

    completed = trades[trades['Result'].isin(['WIN', 'LOSS'])]

    if len(completed) == 0:
        print("  No completed trades yet. Log some signals first!")
        print(f"  Total logged: {len(trades)}  |  Pending: {len(trades[trades['Result']=='PENDING'])}")
        return

    total   = len(completed)
    wins    = len(completed[completed['Result'] == 'WIN'])
    losses  = len(completed[completed['Result'] == 'LOSS'])
    win_pct = (wins / total) * 100

    dir_correct = completed['Direction_Correct'].sum()
    dir_pct     = (dir_correct / total) * 100

    avg_pred_err = (completed['Predicted_7d'] - completed['Actual_Price']).abs().mean()

    print(f"\n  Total completed trades : {total}")
    print(f"  Wins                   : {wins}  ({win_pct:.1f}%)")
    print(f"  Losses                 : {losses}")
    print(f"  Direction accuracy     : {dir_pct:.1f}%")
    print(f"  Avg prediction error   : ₹{avg_pred_err:.2f}")

    print(f"\n  {'Date':<12} {'Ticker':<14} {'Signal':<10} {'Entry':>8} {'Actual':>8} {'Change':>8} {'Result':<6}")
    print("  " + "-" * 72)
    for _, row in completed.iterrows():
        print(f"  {str(row['Date'])[:10]:<12} {row['Ticker']:<14} {row['Signal']:<10} "
              f"₹{row['Price_At_Signal']:>6.0f} ₹{row['Actual_Price']:>6.0f} "
              f"{row['Actual_Change_pct']:>+7.1f}% {row['Result']:<6}")

    # Grade
    print(f"\n  Model grade: ", end="")
    if win_pct >= 70:
        print("A — Excellent! Consider small real positions.")
    elif win_pct >= 55:
        print("B — Good. Keep paper trading for 2 more weeks.")
    elif win_pct >= 45:
        print("C — Average. Model needs more data or tuning.")
    else:
        print("D — Below random chance. Review your indicators.")

# ── OPTION 4: Performance Chart ───────────────────────────────────
def show_chart(trades):
    completed = trades[trades['Result'].isin(['WIN', 'LOSS'])].copy()

    if len(completed) < 2:
        print("\n  Need at least 2 completed trades to show a chart.")
        return

    completed['Date'] = pd.to_datetime(completed['Date'])
    completed = completed.sort_values('Date')
    completed['Cumulative_Wins'] = (completed['Result'] == 'WIN').cumsum()
    completed['Trade_Number']    = range(1, len(completed) + 1)
    completed['Win_Rate']        = (completed['Cumulative_Wins'] /
                                    completed['Trade_Number']) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    fig.suptitle("Paper Trading Performance", fontsize=14, fontweight='bold')

    # Top: actual vs predicted price per trade
    x = range(len(completed))
    ax1.plot(x, completed['Price_At_Signal'],  'o-', color='steelblue',
             label='Entry price', linewidth=1.5)
    ax1.plot(x, completed['Predicted_7d'],     's--', color='orange',
             label='Predicted', linewidth=1.5)
    ax1.plot(x, completed['Actual_Price'],     '^-', color='green',
             label='Actual price', linewidth=1.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{r['Ticker'][:6]}\n{str(r['Date'])[:10]}"
                         for _, r in completed.iterrows()], fontsize=8)
    ax1.set_ylabel("Price (₹)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: running win rate
    colors = ['green' if r == 'WIN' else 'red' for r in completed['Result']]
    ax2.bar(x, completed['Actual_Change_pct'], color=colors, alpha=0.7, width=0.6)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.plot(x, completed['Win_Rate'] - 50, color='purple',
             linewidth=2, label='Win rate (vs 50% baseline)')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"{r['Ticker'][:6]}" for _, r in completed.iterrows()],
                         fontsize=8)
    ax2.set_ylabel("Price change % / Win rate edge")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ── MAIN LOOP ─────────────────────────────────────────────────────
def main():
    print("\n  Loading your trade history...")
    trades = load_trades()
    print(f"  Found {len(trades)} existing trade(s)")

    while True:
        choice = show_menu()

        if choice == '1':
            trades = log_signal(trades)
        elif choice == '2':
            trades = check_pending(trades)
        elif choice == '3':
            show_scorecard(trades)
        elif choice == '4':
            show_chart(trades)
        elif choice == '5':
            print("\n  Goodbye! Keep paper trading before going real.\n")
            break
        else:
            print("  Invalid choice — pick 1 to 5")

if __name__ == "__main__":
    main()