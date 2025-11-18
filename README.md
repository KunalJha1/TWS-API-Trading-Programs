# 📊 IBKR Trading Bot & Multi-Timeframe Trading Dashboard

A streamlined set of Python based trading systems which combine:

- **Automated NQ futures trading** 
- **Real-time Technical Analysis GUI**
- **Multi-timeframe indicator engine** with parallel processing & caching  

---

## 🚀 Features

### 🔹 Technical Analysis GUI (CustomTkinter)
Full source code: **[Technical Analysis GUI](https://github.com/KunalJha1/TWS-API-Trading-Programs/tree/main/LIVE%20GUI%20Technical%20Analysis%20System%20(Redacted%20Code))**

![GUI Project](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/LIVE%20GUI%20Technical%20Analysis%20System%20(Redacted%20Code)/GUI%20Project.png)
![GUI Project Indicators](https://github.com/KunalJha1/TWS-API-Trading-Programs/blob/main/LIVE%20GUI%20Technical%20Analysis%20System%20(Redacted%20Code)/GUI%20Project%20Indicators.png)

- **Multi-timeframe indicator engine:**
  • 1m / 3m / 5m / 15m / 30m / 1h / 4h / 1d  
- **20+ indicators including:**
  • EMA (6/21, 9/14), MACD Histogram, ATR  
  • RSI, Momentum, Stoch, Williams %R  
  • ADX + DI+/DI-, OBV, Volume Spike  
  • Heikin Ashi, Parabolic SAR, Pivot Points  

- **Summary per timeframe:**
  • BUY / SELL / NEUTRAL counts  
  • Weighted “Overall” signal  
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

- Connects to **IBKR TWS / Gateway**
- Uses **2 years of data** for indicator calculations
- Executes automated **LONG/SHORT** trades when conditions align  
- Adaptive entry orders
- Dynamic **trailing stop-loss** + **profit-taker**
- Email alerts for:  
  - Entries  
  - Exits (profit-taker / stop-loss)  
  - Disconnect/reconnect events  
- Logs every trade to CSV (entry/exit, gain %, timestamps)



If you have anything you would like to talk about regarding this project please email me at kunal.jha@uwaterloo.ca. Please not all code inside of this repo will be complete as sections are redacted to keep trading strategies as secure as possible. 

## Disclaimer
This project is for educational and research purposes only. Futures, equities, and leveraged ETFs involve significant risk and may not be suitable for all investors. Nothing in this repository constitutes financial advice. Always test thoroughly using paper trading before deploying with real capital, and trade at your own risk.


