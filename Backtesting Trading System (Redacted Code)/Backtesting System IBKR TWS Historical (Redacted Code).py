import ib_insync
from ib_insync import Future, Stock, util
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
import pandas_ta as ta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import os
from collections import defaultdict
import sys
# REMOVED IMPORT LINE
# functions2 contains all mathematical indicators required for point calculations.
# from functions2 import *
import traceback
from multiprocessing import Pool, cpu_count
from itertools import combinations_with_replacement, combinations
from collections import defaultdict
import traceback


clientId = 98

 
# TWS API Test System Information
clientId = 3
faGroupUsing = 1 
# host = REMOVED
# port = REMOVED
# account = REMOVED
faGroup = 'test1'
faMethod = 'NetLiq'

trend_categories = {
    "BUY-Trend Started (Long-Term)": [],
    "SELL-Trend Started (Long-Term)": [],
    "BUY-In Trend (Long-Term)": [],
    "SELL-In Trend (Long-Term)": [],
    "Do Nothing (Long-Term)": []
}

# sender_email = removed email
# sender_password = ENV Variable, Still Removed
# smtp_port = removed port
# smtp_server = removed server

#subscription_expiry_map = { Subscription map with expiration date }
today = datetime.now().date()

receiver_email_list = [
    email
    for email, expiry_str in subscription_expiry_map.items()
    if today <= datetime.strptime(expiry_str, "%Y-%m-%d").date()
]

print("Active subscribers:", receiver_email_list)

last_categorized_email_time = datetime.min
categorized_email_interval = timedelta(hours=1)
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

rsi_trend_map = {}
rsi_value_map = {}
price_map = {}
signal_history_map = {}

def compute_heikin_ashi(df):
    # Custom Heikin Ashi Calculation Method

# Evalutes an individual symbol with a collection of indicators and their individual weightage
def evaluate_symbol(symbol, df, indicator_1_pts, indicator_2_pts... indicator_k_pts):
    print(f"Symbol | {symbol}")
    trades = []
    position_counter = 0
    # df[Individual Dataframe] = Indicator For each Individual Indicator
    
    canClose = False
    sell_counter = 0
    buy_counter = 0
    signal = ""
    trailing_price = None
    entry_price = None
    entry_date = None
    
    start_idx = -min(len(df), 400)
    print(f"{symbol} — using {abs(start_idx)} bars of data (len={len(df)})")

    for i in range(start_idx, 0):
        if df.isnull().iloc[i].any():
            continue
        canClose = False
        sell_counter = 0
        buy_counter = 0

        # Removed Calculations for market trending
        if market_trending == 1:
            buy_counter += indicator_1_point_tally
        else:
            sell_counter += indicator_1_point_tally

        # This is the calculation set up used for each individual indicator which is backtested
        if calculation:
            sell_counter += indicator_1_point_tally
        else:
            buy_counter += indicator_1_point_tally

        price = round(df['close'].iloc[i], 2)
        timestamp = df.index[i]
        timestamp = str(df.index[i])[:10]
        
        if position_counter != 0:
            canClose = True

        # Signal Conversion to Action Conversion
        if position_counter == 0 and buy_counter >= max_points:
            signal = "BUY TREND STARTED"
            trailing_price = price
                
        if position_counter == 1 and buy_counter >= max_points:
            signal = "BUY IN-TREND"

        if position_counter == 1 and buy_counter < max_points:
            signal = "HOLD LONG POSITION"

        if position_counter == 0 and sell_counter >= max_points:
            signal = "SELL TREND STARTED"
            trailing_price = price

        if position_counter == -1 and sell_counter >= max_points:
                signal = "SELL IN-TREND"

        if position_counter == -1 and sell_counter < max_points:
                signal = "HOLD SHORT POSITION"        

        # Signal Conversion when there is a long or short position open
        if canClose:
            if position_counter == 1 and sell_counter >= max_points:
                signal = "CLOSE LONG POSITION"
            if position_counter == -1 and buy_counter >= max_points:
                signal = "CLOSE SHORT POSITION"

        profit_target_pct    = 0.01 
        stoploss_pct_long    = 0.99 
        stoploss_pct_short   = 1.01
          
        # Calculation if stoploss and profit taker have been hit inside of the backtest
        if position_counter == 1 and entry_price is not None and trailing_price is not None:
            if price >= entry_price * (1 + profit_target_pct):
                signal = "CLOSE LONG POSITION"
                print(f"[{timestamp}] PROFIT-TAKER hit for LONG @ {price:.2f}")

            if price < trailing_price * stoploss_pct_long:
                signal = "CLOSE LONG POSITION"
                print(f"[{timestamp}] STOP-LOSS triggered for LONG @ {price:.2f} "
                    f"(threshold {trailing_price * stoploss_pct_long:.2f})")

            if price > trailing_price:
                trailing_price = price

        elif position_counter == -1 and entry_price is not None and trailing_price is not None:
            if price <= entry_price * (1 - profit_target_pct):
                signal = "CLOSE SHORT POSITION"
                print(f"[{timestamp}] PROFIT-TAKER hit for SHORT @ {price:.2f}")

            if price > trailing_price * stoploss_pct_short:
                signal = "CLOSE SHORT POSITION"
                print(f"[{timestamp}] STOP-LOSS triggered for SHORT @ {price:.2f} "
                    f"(threshold {trailing_price * stoploss_pct_short:.2f})")

            if price < trailing_price:
                trailing_price = price
                
        # Logging and transaction model for all open and closing transactions in the backtesting.
        if signal == "SELL TREND STARTED":
            entry_price = price
            entry_date = timestamp
            position_counter = -1

        elif signal == "BUY TREND STARTED":
            entry_price = price
            entry_date = timestamp
            position_counter = 1
            
        if signal == "CLOSE LONG POSITION" and entry_price is not None:
            trades.append({
                "symbol":     symbol,
                "type":       "CLOSE LONG",
                "entry_price": entry_price,
                "exit_price":  price,
                "entry_date":  entry_date,
                "exit_date":   timestamp,
                "gain":        round((price - entry_price) / entry_price * 100, 2),
            })
            position_counter = 0
            entry_price     = None
            entry_date      = None
            trailing_price  = None

        elif signal == "CLOSE SHORT POSITION" and entry_price is not None:
            trades.append({
                "symbol":     symbol,
                "type":       "CLOSE SHORT",
                "entry_price": entry_price,
                "exit_price":  price,
                "entry_date":  entry_date,
                "exit_date":   timestamp,
                "gain":        round((entry_price - price) / entry_price * 100, 2),
            })
            position_counter = 0
            entry_price     = None
            entry_date      = None
            trailing_price  = None

        if i == -1 and entry_price is not None and entry_date is not None:
            current_price = round(df['close'].iloc[-1], 2)
            direction = "LONG" if position_counter == 1 else "SHORT"
            gain = (
                ((current_price - entry_price) / entry_price) * 100
                if direction == "LONG"
                else ((entry_price - current_price) / entry_price) * 100
            )
            trades.append({
                "symbol": symbol,
                "type": f"{direction} (OPEN)",
                "entry_price": entry_price,
                "exit_price": current_price,
                "entry_date": entry_date,
                "exit_date": str(df.index[i])[:10],
                "gain": round(gain, 2),
            })

    # Console print statement
    print(f"\n{symbol} — All Trades Finished:")

    # Sorts all trades inside of the entry_date when completed (important due to parallel processing)
    trades_sorted = sorted(trades, key=lambda x: x['entry_date'])
    
    # Prints out the best n
    if trades_sorted:
        last = trades_sorted[-1]
        print(
            f"{last['entry_date']} → {last['exit_date']} | {last['type']:13} | "
            f"Entry: ${last['entry_price']} → Exit: ${last['exit_price']} | "
            f"Gain: {last['gain']:+.2f}% | " # amount of points
        )
    else:
        print("No trades to summarize.")

    initial_capital = 7000
    capital = initial_capital

    for trade in trades_sorted[::-1]:
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        trade_type = trade['type'].ljust(13)
        gain = trade['gain']
        
        capital *= (1 + gain / 100)

    return_pct = (capital - 7000) / 7000 * 100
    print(f"Final Capital: ${capital:.2f} (Start: $7,000.00, Return: {return_pct:+.2f}%)")

    return capital, trades

# Generates the unique combos which are going to be backtested, the list of points is then tested using other methods. 
def generate_combos(indicators, total_points):
    combos = []
    for num_active in range(1, len(indicators) + 1):
        for active_inds in combinations(indicators, num_active):
            for points in combinations_with_replacement(range(1, total_points + 1), num_active):
                if sum(points) != total_points:
                    continue
                combo_dict = {ind: 0 for ind in indicators}
                for ind, val in zip(active_inds, points):
                    combo_dict[ind] = val
                combos.append(tuple(combo_dict[ind] for ind in indicators))
    return combos

def process_combo(args):
    symbol, combo, df = args
    #individual_indicator1, individual_indicator2, individual_indicatork = combo

    print(f"\nTesting {symbol} | Combo: list of points and their allocations")

    try:
        capital, trades = evaluate_symbol(symbol, df, individual_indicator1, individual_indicator2, individual_indicatork)

        if trades:
            print(f"\n{symbol} — All Completed Trades:")
            for trade in trades:
                print(f"{trade['entry_date']} → {trade['exit_date']} | {trade['type'].ljust(13)} | "
                    f"Entry: ${trade['entry_price']:>8} → Exit: ${trade['exit_price']:>8} | "
                    f"Gain: {trade['gain']:+.2f}% | "
                    f"Pts: ")

            return_pct = round((capital - 7000) / 7000 * 100, 2)
            print(f"💰 Final Capital: ${capital:,.2f} (Start: $7,000.00, Return: {return_pct:+.2f}%)")
        else:
            print(f"\n{symbol} — No Trades Completed for | Combo: list of points and their allocations")

    except Exception as e:
        print(f"Error in {symbol} | {e}")
        traceback.print_exc() 
        capital, trades = 0, []

    return {
        "symbol": symbol,
        # Point Weights for each indicator
        "Final Capital": capital,  # 
        "Return (%)": round((capital - 7000) / 7000 * 100, 2),
        "Total Trades": len(trades),
        "trades": trades  
    }


if __name__ == "__main__":
    from ib_insync import IB, util
    from ib_insync import Future

    ib = IB()
    # connect to the IBKR system with host, port, and the clientID.
    all_results = []

    contract_NQ_test = Future(symbol="NQ", exchange="CME", currency="USD", lastTradeDateOrContractMonth="202509")
    contract_ES_test = Future(symbol="ES", exchange="CME", currency="USD", lastTradeDateOrContractMonth="202509")
    contract_GC_test = Future(symbol="GC", exchange="COMEX", currency="USD", lastTradeDateOrContractMonth="202507")

    futures_contracts = {
        "NQ": contract_NQ_test,
        "ES": contract_ES_test,
        "GC": contract_GC_test
    }

    stocks = ['FNGU', 'TQQQ', 'SPXL', 'SOXL', 'QQQ', 'SPY', 'FNGD', 'SQQQ', 'SPXS', 'SOXS',
            'AAPL', 'AMZN', 'AMD', 'GOOGL', 'MSFT', 'META', 'NFLX', 'NVDA', 'CRWD', 'AVGO',
            'NOW', 'SNOW', 'TSLA', 'SHOP', 'AMAT', 'QCOM', 'KLAC', 'MPWR', 'MU', 'TSM',
            'TXN', 'INTC', 'LMT', 'MSTR', 'ORCL', 'SPOT', 'HIMS', 'ADBE', 'MDB', 'NVDU',
            'NVDQ', 'TSLL', 'TSLQ', 'GGLL', 'GGLS', 'NFXL', 'NFXS', 'AVL', 'AVS', 'PLTR']

    stocks = []
    symbols = list(futures_contracts.keys()) + stocks
    all_1d = []
    
    for symbol in symbols:
        if symbol.upper() == 'NQ':
            months = [
                "202512",
                "202603",
                "202606"
            ]
            
            for month in months:
                
                contract = Future(
                    symbol="NQ",
                    exchange="CME",
                    currency="USD",
                    lastTradeDateOrContractMonth=month,
                    includeExpired=True
                )

                bars_1d = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr='150 D',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=False,
                    formatDate=1
                )

                df_1d = pd.DataFrame(
                    [[bar.date, bar.open, bar.high, bar.low, bar.close] for bar in bars_1d],
                    columns=['date', 'open', 'high', 'low', 'close']
                )
                print(f"{month} → Loaded {len(df_1d)} rows")
                all_1d.append(df_1d)
            
        elif symbol.upper() == 'ES':
            contract = contract_ES_test
        elif symbol.upper() == 'GC':
            contract = contract_GC_test
        else:
            contract = Stock(symbol, 'SMART', 'USD', primaryExchange='NASDAQ')
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='60 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )
            
            df_1d = util.df(bars)

        df_1d = pd.concat(all_1d).drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
        earliest_date = df_1d['date'].min()
        latest_date = df_1d['date'].max()
        price_at_earliest = df_1d.loc[df_1d['date'] == earliest_date, 'close'].values[0]

        print(f"NQ Data Range: {earliest_date} → {latest_date} | Total rows: {len(df_1d)}")
        print(f"Price on {earliest_date}: {price_at_earliest}")

        df_data = df_1d

        if df_data is None or df_data.empty:
            continue

        if 'date' in df_data.columns:
            df_data.index = pd.to_datetime(df_data['date'])

        if 'date' in df_data.columns:
            df_data.index = pd.to_datetime(df_data['date'])

        # indicators = list of indicators
        combos = generate_combos(indicators, total_points=max_length_points)
        print(f"Total combinations generated: {len(combos)}")

        # Cap to amount of combos
        max_combos_to_process = 10000
        args_list = [(symbol, combo, df_data) for combo in combos]
        args_list = args_list[:max_combos_to_process]

        # Parrelel processing using the cpu_count which was allocated to processing. 
        with Pool(cpu_count()) as pool:
            results = pool.map(process_combo, args_list)

        for row in results:
            row['Symbol'] = symbol
            all_results.append(row)

    df_results = pd.DataFrame(all_results)
    df_results = df_results[['Symbol'] + [col for col in df_results.columns if col != 'Symbol']]
    df_results = df_results[pd.to_numeric(df_results['Final Capital'], errors='coerce').notnull()]
    df_results['Final Capital'] = df_results['Final Capital'].astype(float)
    df_results.sort_values(by='Final Capital', ascending=False, inplace=True)


    summary_csv_path = os.path.join(
        # Path for the file to be saved
        f'email_conditions_backtest_inverse_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    
    all_trades = []
    worst_result = min(all_results, key=lambda x: x.get("Final Capital", float("inf")))
    worst_trades = worst_result.get("trades", [])

    if worst_trades:
        df_worst = pd.DataFrame(worst_trades)
        
        df_worst["symbol"] = worst_result["symbol"]
        df_worst["indicator_pts_1"] = worst_result["indicator_pts_1"]
        df_worst["indicator_pts_2"] = worst_result["indicator_pts_2"]
        ...
        df_worst["indicator_pts_k"] = worst_result["indicator_pts_k"]

        worst_csv_path = os.path.join(
            # Path for the file to be saved
            f'worst_combo_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        df_worst.to_csv(worst_csv_path, index=False)
        print(f"Worst combo trades saved to: {worst_csv_path}")
    else:
        print("No trades found for worst-performing combo.")

    best_result = max(all_results, key=lambda x: x.get("Final Capital", float("-inf")))
    best_trades = best_result.get("trades", [])

    if best_trades:
        df_best = pd.DataFrame(best_trades)
        df_best["symbol"]   = best_result["symbol"]
        df_best["indicator_pts_1"]  = best_result["indicator_pts_1"]
        df_best["indicator_pts_2"]  = best_result["indicator_pts_2"]
        ...
        df_best["indicator_pts_k"] = best_result["indicator_pts_k"]

        best_csv_path = os.path.join(
            # Path for the file to be saved
            f'best_combo_trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        df_best.to_csv(best_csv_path, index=False)
        print(f"Best combo trades saved to: {best_csv_path}")
    else:
        print("No trades found for best-performing combo.")

        
    df_results.to_csv(summary_csv_path, index=False)
    print(f"\nResults saved to: {summary_csv_path}")

mountain_tz = pytz.timezone("America/Edmonton")

