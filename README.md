# 📊 IBKR Trading Bot & Multi-Timeframe Trading Dashboard

A streamlined set of Python based trading systems which combine:

- **Automated NQ futures trading** 
- **Real-time Technical Analysis GUI which includes a multi-timeframe indicator interface** with parallel processing & caching**  
- **Automated options volume & unusual activity scanner** with daily reports
  
---

## 🚀 Features

### 🔹 Technical Analysis GUI (CustomTkinter)
Full source code: **[Technical Analysis GUI](https://github.com/KunalJha1/TWS-API-Trading-Programs/tree/main/LIVE%20GUI%20Technical%20Analysis%20System%20(Redacted%20Code))**

![GUI Project](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/LIVE%20GUI%20Technical%20Analysis%20System%20(Redacted%20Code)/GUI%20Project.png)

![GUI Project Indicators](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/LIVE%20GUI%20Technical%20Analysis%20System%20(Redacted%20Code)/GUI%20Project%20Indicators.png)

- **Multi-timeframe indicator engine:**
  • 1m / 3m / 5m / 15m / 30m / 1h / 4h / 1d
  • GUI Built using CustomTkinter
- **20+ indicators including:**
  • EMA (6/21, 9/14), MACD Histogram, ATR  
  • RSI, Momentum, Stoch, Williams %R  
  • ADX + DI+/DI-, OBV, Volume Spike  
  • Heikin Ashi, Parabolic SAR, Pivot Points  
- **Summary per timeframe:**
  • BUY / SELL / NEUTRAL counts  
  • Weighted Overall signal  
- **Watchlist with:**
  • Live pricing using the TWS API  
- **Additional features:**
  • Built-in caching  
  • Async historical data fetching  
  • CPU/RAM usage metrics  

### 🔹 NQ Futures Scalping Bot
Full source code: [Live Scalping Strategy](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/Live%20Trading%20System%20(Scalping%20Strategy)%20(Redacted%20Code).py)

![Open Position](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/Live%20Trading%20System%20Scalping%20(Redacted%20Code)/Open%20Position.png)

![Dynamic Trailing Stop Loss and Profit Taker](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/Live%20Trading%20System%20Scalping%20(Redacted%20Code)/Dynamic%20Trailing%20Stop%20Loss%20and%20Profit%20Taker.png)

- Connects to **IBKR TWS API**
- Uses **2 years of data** for calculations
- Executes automated **long and short** trades when conditions align  
- Adaptive order entry using IBKR Adaptive Algo
- Dynamic **trailing stop-loss** + **profit-taker** attached to all executed trades
- Email alerting system for, entries, exits (profit-taker / stop-loss), disconnect/reconnect events.  
- Logs every trade to CSV (entry/exit, gain %, timestamps)

### 🔹 Options Volume & Unusual Activity Scanner
Full source code: **[Options Scanner](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/Option%20Volume%20Data%20Retrival%20(Redacted%20Code)/Option%20Volume%20Data%20Retrival%20and%20Analysis%20(Redacted%20Code).py)**

![Option Volume Read Me](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/Option%20Volume%20Data%20Retrival%20(Redacted%20Code)/Option%20Volume%20Read%20Me.png)

- Scans **stocks, ETFs, and futures options** (NQ/ES/GC) using the IBKR API  
- Automatic Scheduler to run the program at **12:30 PM MT** every day  
- Filters for **high-volume** and **unusual-volume** contracts  
- Extracts greeks: Δ / Γ / Θ / Vega / IV  
- Generates **4 CSV reports**: full dataset, Top 10 volume, Unusual volume, PCR  
- Parallel processing for fast daily scans of option volume.
- Sends a detailed **email summary + attachments** after every run of the program to clients. 

If you have anything you would like to talk about regarding this project please email me at kunal.jha@uwaterloo.ca. Please not all code inside of this repo will be complete as sections are redacted to keep trading strategies as secure as possible. 

## Disclaimer
This project is for educational and research purposes only. Futures, equities, and leveraged ETFs involve significant risk and may not be suitable for all investors. Nothing in this repository constitutes financial advice. Always test thoroughly using paper trading before deploying with real capital, and trade at your own risk.


