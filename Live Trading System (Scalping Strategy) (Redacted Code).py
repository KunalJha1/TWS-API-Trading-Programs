from ib_insync import *
import numpy as np
from numpy import nan as npNaN
import pandas as pd
from datetime import datetime, time as dt_time
import pytz
import pandas_ta as ta
import math
from ta.volume import VolumeWeightedAveragePrice
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from threading import Timer
import smtplib
from email.message import EmailMessage
import sys
# REMOVED IMPORT LINE
# functions2 contains all mathematical indicators required for point calculations.
# from functions2 import *
import time             
import os
import csv

# TWS API Test System Information
clientId = 3
faGroupUsing = 1 
# host = REMOVED
# port = REMOVED
# account = REMOVED
faGroup = 'test1'
faMethod = 'NetLiq'
entry_price = None

# Creation of IB Object
ib = IB()

# Contract and Traded Equity Information
orderType = "MKT"
symbol = "NQ"
currency = "USD"
exchange = "CME"
expiry = "202509"
total_quantity = 1
trailing_stop_loss_price = None
trailing_price = 0.0

# Purpose of method is to log trades to a CSV, in case of failure roll back to a txt file for logging full trade. 
def log_trade_to_csv(data_dict):
    # log_dir = REMOVED LOCAL PATH 
    log_file = os.path.join(log_dir, "trade_log_NQ_exits.csv")

    os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        print(f"⚠️ Skipping trade log due to error (LIKELY CURRENTLY OPEN): {e}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_filename = os.path.join(log_dir, f"FAILED_LOGGING_NQ_{timestamp}.txt")
        with open(error_filename, 'w') as f:
            f.write(f"⚠️ Failed to log trade at {timestamp}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Transaction Data:\n")
            for k, v in data_dict.items():
                f.write(f"{k}: {v}\n")

# Purpose of method is to log entry trades to a CSV in case of failure it rolls to a txt file and logs the entry information.
def log_entry_to_csv(data_dict):
    # log_dir = REMOVED LOCAL PATH 
    log_file = os.path.join(log_dir, "entry_log_NQ_entry.csv")

    os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.isfile(log_file)

    try:
        with open(log_file, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=data_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_dict)
    except Exception as e:
        print(f"⚠️ Skipping entry log due to error (LIKELY CURRENTLY OPEN): {e}")
        # Save to TXT fallback
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_filename = os.path.join(log_dir, f"FAILED_LOGGING_NQ_{timestamp}.txt")
        with open(error_filename, 'w') as f:
            f.write(f"⚠️ Failed to log entry at {timestamp}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write("Transaction Data:\n")
            for k, v in data_dict.items():
                f.write(f"{k}: {v}\n")


# Purpose of this method is to get the IB information about the open position and get the average cost_per_unit.
def get_avg_cost_per_unit(symbol: str):
    positions = ib.reqPositions()

    for pos in positions:
        contract = pos.contract
        if contract.symbol.upper() == symbol.upper():
            avg_cost = pos.avgCost
            multiplier = float(contract.multiplier or 1)
            normalized_cost = avg_cost / multiplier
            return normalized_cost

    return None  # Symbol not found

# Purpose of this method is to send an email to a list of receivers in case of failure or open positions.
def send_email_alert(subject, body):
    # sender_email = REMOVED EMAIL
    # sender_password = REMOVED PASSWORD
    # smtp_port = REMOVED PORT
    # smtp_server = REMOVE SMTP SERVER
    # receiver_email_list = REMOVED RECEIVER EMAIL LIST

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = ", ".join(receiver_email_list)
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("📧 Email alert sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

# Purpose of this method is to get from the TWS API if there is a currently open NQ position.
def get_nq_position(symbol):
    total_nq_position = 0
    positions = ib.reqPositions()

    for pos in positions:
        contract = pos.contract
        
        if (
            contract.symbol.upper() == symbol
            and contract.secType.upper() == "FUT"
            and contract.lastTradeDateOrContractMonth.startswith(expiry)
        ):
            total_nq_position += pos.position

    if total_nq_position > 0:
        return 1
    elif total_nq_position < 0:
        return -1
    else:
        return 0

# Sends an email alert alerting of TWS API disconnection so program can be restarted after notification to user.
def send_disconnect_alert():
    # sender_email = REMOVED EMAIL
    # sender_password = REMOVED PASSWORD
    # smtp_port = REMOVED PORT
    # smtp_server = REMOVE SMTP SERVER
    # receiver_email_list = REMOVED RECEIVER EMAIL LIST

    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = "🚨 IBKR Disconnected Alert"
    msg.set_content(f"IBKR disconnection detected. Attempting to reconnect in 5 minutes. Clinet ID | {clientId} | Symbol : {symbol}")

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("📧 Disconnect alert sent.")
    except Exception as e:
        print(f"❌ Failed to send disconnect alert: {e}")

# Purpose of this method is to reconnect to the IB object incase of disconnection from TWS API. Will try every 2 minutes during disconnection.
# Method is requried due to frequent disconnection from 9:00PM MST - 12:00AM MST
def connect_with_retry():
    while True:
        try:
            ib.connect(host, port, clientId)
            if ib.isConnected():
                print("✅ Connected to IBKR.")
                ib.reqPositions()  # ✅ refresh live position cache
                return
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            send_disconnect_alert()
            print("⏳ Retrying in 2 minutes...")
            time.sleep(150)


connect_with_retry()

contract_nq = Future(symbol=symbol, lastTradeDateOrContractMonth=expiry, exchange=exchange, currency=currency)
ib.qualifyContracts(contract_nq)
order_pending = False
position_just_opened = False

while True:
    time.sleep(7)
    os.system('cls' if os.name == 'nt' else 'clear')

    try:
        if order_pending:
            time.sleep(10)
            open_position_nq = get_nq_position(symbol)
            if open_position_nq != 0:
                order_pending = False
            else:
                continue

        if position_just_opened:
            open_position_nq = get_nq_position(symbol)
            if open_position_nq != 0:
                position_just_opened = False
                order_pending = False
            else:
                time.sleep(10)
                continue

        open_position_nq = get_nq_position(symbol)
        
        if not ib.isConnected():
            raise ConnectionError("Lost connection to IBKR.")
        
        mountain_tz = pytz.timezone('America/Edmonton')
        current_time = datetime.now(mountain_tz).strftime('%I:%M:%S %p')

        # Pull 2 years of Daily Data (Requests all avaiable data for the current NQ Contract due to limited data.)
        bars = ib.reqHistoricalData(contract_nq, '', '2 Y', '1 day', 'TRADES', useRTH=False, formatDate = 1)
        df = util.df(bars)
        current_price = df['close'].iloc[-1]
        
        if open_position_nq != 0 and trailing_price == 0.0:
            entry_price = get_avg_cost_per_unit(symbol)
            trailing_price = current_price
            mountain_tz = pytz.timezone('America/Edmonton')
            entry_time = datetime.now(mountain_tz)
            
        earliest_date = df['date'].min()
        latest_date = df['date'].max()
        price_at_earliest = df.loc[df['date'] == earliest_date, 'close'].values[0]
        
        df['RSI'] = df.ta.rsi(length=14)
        df['ADX_14'] = df.ta.adx(length=14)['ADX_14']
        adx_now = df['ADX_14'].iat[-1]
        
        #market_trending = REMOVED CODE

        rsi_now = df['RSI'].iloc[-1]
        rsi_prev = df['RSI'].iloc[-1 - 1]

        buy_signal_counter = 0
        sell_signal_counter = 0

        # Removed Buy Sell Conditions and Math Calculations

        total_buy_conditions = 5
        total_sell_conditions = 5
        
        print("\n===================== FINAL DECISION SUMMARY =====================")
        print(f"ADX NOW {adx_now}")
        print(f"📈 Buy Signal Score (Daily Chart):  {buy_signal_counter}/{total_buy_conditions} " +
              ("🟢 STRONG" if buy_signal_counter == 5 else "🟨 WEAK"))
        print(f"📉 Sell Signal Score (Daily Chart): {sell_signal_counter}/{total_sell_conditions} " +
              ("🔴 STRONG" if sell_signal_counter == 5 else "🟨 WEAK"))

        if buy_signal_counter >= 5:
            print("✅ ACTION: Enter LONG — Trend, Confirmation, and Signals Aligned")
        elif sell_signal_counter >= 5:
            print("✅ ACTION: Enter SHORT — Trend, Confirmation, and Signals Aligned")
        else:
            print("⏸️ ACTION: No Entry — Conditions Not Fully Met")
        print("==================================================================\n")

        # Stop Loss and Profit Taker Percentiles
        stoploss_pct       = 0.9925
        stoploss_pct_short = 1.0075
        profit_taker_pct_long = 1.0100
        profit_taker_pct_short = 0.9900


        # Determines the price target of profit_taker and trailing_price for a long position.
        if open_position_nq == 1:
            
            if current_price >= (entry_price * (profit_taker_pct_long)):
                signal = "CLOSE LONG POSITION"
                print(f"[{current_time}] PROFIT-TAKER hit for LONG @ {current_price:.2f}")

            if current_price < (trailing_price * stoploss_pct):
                signal = "CLOSE LONG POSITION"
                print(f"[{current_time}] STOP-LOSS triggered for LONG @ {current_price:.2f}  "
                    f"(threshold {trailing_price * stoploss_pct:.2f})")

            if current_price > trailing_price:
                trailing_price = current_price
                
    
                
        # Determines the price target of profit_taker and trailing_price for a short position.
        elif open_position_nq == -1:
         
            if current_price <= (entry_price * (profit_taker_pct_short)):
                signal = "CLOSE SHORT POSITION"
                print(f"[{current_time}] PROFIT-TAKER hit for SHORT @ {current_price:.2f}")

            if current_price > trailing_price * stoploss_pct_short:
                signal = "CLOSE SHORT POSITION"
                print(f"[{current_time}] STOP-LOSS triggered for SHORT @ {current_price:.2f}  "
                    f"(threshold {trailing_price * stoploss_pct_short:.2f})")
                
            if current_price < trailing_price:
                trailing_price = current_price
                
   
        # Converts the stop-loss to actionable items.
        if signal == "CLOSE SHORT POSITION":
            buy_signal_counter = 5
        if signal == "CLOSE LONG POSITION":
            sell_signal_counter = 5

        #  Logging and Closing the open-position due to to a profit-taker or stop-loss.
        if open_position_nq == 1 and sell_signal_counter == 5:
            reason = "PROFIT-TAKER" if current_price >= entry_price * (profit_taker_pct_long) else "STOP-LOSS"
            gain_pct = round(((current_price - entry_price) / entry_price) * 100, 2)

            log_trade_to_csv( {
                "symbol": symbol,
                "direction": "LONG",
                "entry_price": round(entry_price, 2),
                "exit_price": round(current_price, 2),
                "entry_time": entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                "exit_time": datetime.now(mountain_tz).strftime('%Y-%m-%d %H:%M:%S'),
                "reason": reason,
                "gain_pct": gain_pct
            })

            open_position_nq = 0
            order = Order(action="SELL", orderType=orderType, totalQuantity=total_quantity)
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]

            if faGroupUsing:
                order.faGroup = faGroup
                order.faMethod = faMethod
            else:
                order.account = account

            ib.placeOrder(contract_nq, order)
            print("📌 Market Order Selected.")
            print("✅ CLOSED LONG POSITION (NQ)")
            send_email_alert(
                subject="📤  NQ LONG CLOSED | KUNAL",
                body=f"Closed long position on NQ.\nSell Price: {current_price}\nTime: {current_time}"
            )
            signal = None
            buy_trend_ended_counter = 0
            trailing_stop_loss_price = None
            time.sleep(7200)
            continue
        
        # Logging and closing the open-position due to to a profit-taker or stop-loss.
        if open_position_nq == -1 and buy_signal_counter == 5:
            reason = "PROFIT-TAKER" if current_price <= entry_price * (profit_taker_pct_short) else "STOP-LOSS"
            gain_pct = round(((entry_price - current_price) / entry_price) * 100, 2)

            log_trade_to_csv({
                "symbol": symbol,
                "direction": "SHORT",
                "entry_price": round(entry_price, 2),
                "exit_price": round(current_price, 2),
                "entry_time": entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                "exit_time": datetime.now(mountain_tz).strftime('%Y-%m-%d %H:%M:%S'),
                "reason": reason,
                "gain_pct": gain_pct
            })

            open_position_nq = 0
            order = Order(action="BUY", orderType=orderType, totalQuantity=total_quantity)
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]

            if faGroupUsing:
                order.faGroup = faGroup
                order.faMethod = faMethod
            else:
                order.account = account

            ib.placeOrder(contract_nq, order)
            print("📌 Market Order Selected.")
            print("✅ CLOSED SHORT POSITION (NQ)")
            send_email_alert(
                subject="📤  NQ SHORT CLOSED | KUNAL",
                body=f"Closed Short position on NQ.\nBuy Price: {current_price}\nTime: {current_time}"
            )   
            signal = None
            sell_trend_ended_counter = 0
            trailing_stop_loss_price = None
            time.sleep(7200)
            continue

        # Opens the long-position according to logical market-conditions.
        if buy_signal_counter == 5 and open_position_nq == 0:
            entry_time = datetime.now(mountain_tz)
            order = Order(action="BUY", orderType=orderType, totalQuantity=total_quantity)
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]

            if faGroupUsing:
                order.faGroup = faGroup
                order.faMethod = faMethod
            else:
                order.account = account

            ib.placeOrder(contract_nq, order)
            print(f"🟢 Executed LONG on new 5-min candle | Entry Price: {current_price}")
            send_email_alert(
                    subject="📤  NQ LONG OPENED | KUNAL",
                    body=f"Opened long position on  NQ.\nBuy Price: {current_price}\nTime: {current_time}"
            )

            # ✅ Correct — initialize trailing to a safe placeholder
            trailing_stop_loss_price = None
            open_position_nq = 1
            order_pending = True
            position_just_opened = True
            trailing_price = current_price
            entry_price = current_price
            
            log_entry_to_csv({
                "symbol": symbol,
                "position": "LONG",
                "entry_price": round(current_price, 2),
                "entry_time": entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                "reason": "Signal Match (Buy Score = 5)",
                "timeframe": "1D",
                "adx": round(adx_now, 2),
                "rsi": round(rsi_now, 2),
                "buy_score": buy_signal_counter,
                "sell_score": sell_signal_counter
            })
            
            signal = None
            
            continue

        # Opens a short-position according to logical market-conditions.
        if sell_signal_counter == 5 and open_position_nq == 0:
            entry_time = datetime.now(mountain_tz)
            order = Order(action="SELL", orderType=orderType, totalQuantity=total_quantity)
            order.algoStrategy = "Adaptive"
            order.algoParams = [TagValue("adaptivePriority", "Normal")]

            if faGroupUsing:
                order.faGroup = faGroup
                order.faMethod = faMethod
            else:
                order.account = account

            ib.placeOrder(contract_nq, order)
            print(f"🔴 Executed SHORT on new 5-min candle | Entry Price: {current_price}")
            send_email_alert(
                    subject="📤  NQ SHORT OPENED | KUNAL" ,
                    body=f"Opened Short position on  NQ.\nSell Price: {current_price}\nTime: {current_time}"
            )
            
            # ✅ Replace with:
            trailing_stop_loss_price = True
            open_position_nq = -1
            order_pending = True
            position_just_opened = True
            trailing_price = current_price
            entry_price = current_price
            
            log_entry_to_csv({
                "symbol": symbol,
                "position": "SHORT",
                "entry_price": round(current_price, 2),
                "entry_time": entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                "reason": "Signal Match (Sell Score = 5)",
                "timeframe": "1D",
                "adx": round(adx_now, 2),
                "rsi": round(rsi_now, 2),
                "buy_score": buy_signal_counter,
                "sell_score": sell_signal_counter
            })

            signal = None
            
            time.sleep(10)

            continue

        print()
        # Print information for console overview when running trading script.
        if open_position_nq == 1:
            print(f"[{current_time}] 💰 Current Price (LONG): {current_price:.2f}")
            print(f"[{current_time}] 🟢 Entry Price (LONG): {entry_price:.2f}")

            profit_taker_price = entry_price * (profit_taker_pct_long)
            profit_pct = ((profit_taker_price - current_price) / profit_taker_price) * 100
            print(f"[{current_time}] 🎯 Profit-Taker (LONG): {profit_taker_price:.2f} ({profit_pct:+.2f}% to target)")

            trailing_stop_loss_price = trailing_price * stoploss_pct
            stop_pct = ((trailing_stop_loss_price - current_price) / current_price) * 100
            print(f"[{current_time}] 📍 Trailing Stop (LONG): {trailing_stop_loss_price:.2f} ({stop_pct:+.2f}%)")

        elif open_position_nq == -1:
            print(f"[{current_time}] 💰 Current Price (SHORT): {current_price:.2f}")
            print(f"[{current_time}] 🔴 Entry Price (SHORT): {entry_price:.2f}")

            profit_taker_price = entry_price * (profit_taker_pct_short)
            profit_pct = ((current_price - profit_taker_price) / profit_taker_price) * 100
            print(f"[{current_time}] 🎯 Profit-Taker (SHORT): {profit_taker_price:.2f} ({profit_pct:+.2f}% to target)")

            trailing_stop_loss_price = trailing_price * stoploss_pct_short
            stop_pct = ((current_price - trailing_stop_loss_price) / current_price) * 100
            print(f"[{current_time}] 📍 Trailing Stop (SHORT): {trailing_stop_loss_price:.2f} ({stop_pct:.2f}%)")


        # Current Position counter.
        if open_position_nq == 1:
            print("Current Position: +1 Long NQ 🟢")
        elif open_position_nq == -1:
            print("Current Position: -1 Short NQ 🔴")
        else:
            print("Current Position: No Position 🟨")
        print()
        print(f"Current Time: {current_time}")
        print()
        print(f"Client ID : {clientId} | Symbol : {symbol}\n")
        print("Sleeping for 10 second before the next iteration...")
        print("-" * 50)
        time.sleep(1)

    except Exception as e:
        import traceback
        traceback.print_exc()        # <-- full traceback to stderr
        print(f"⚠️ Disconnected or error: {e}")
        ib.disconnect()

        print("⏳ Waiting 2 minutes before reconnect attempt...")
        time.sleep(150)  # <-- Add fixed sleep to avoid email spam

        connect_with_retry()





