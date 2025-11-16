import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import time
import csv
import os
import ib_insync
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from ib_insync import Future
from ib_insync import *
from ibapi.contract import Contract
import random
from datetime import datetime, time as dt_time, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pandas as pd
from ib_insync import Stock, util
import time
import threading
import queue 
import multiprocessing
from functools import lru_cache
import psutil

# Global Settings and Resource Allocation
cpu_count = multiprocessing.cpu_count()
max_workers = min(cpu_count, 8)
executor = ThreadPoolExecutor(max_workers=max_workers)

indicator_cache = {}
cache_lock = threading.Lock()
CACHE_DURATION = 60
PERFORMANCE_MONITORING = True
FAST_UPDATE_MODE = True 
DEBUG_INDICATORS = True
FORCE_DATA_REFRESH = False 

# Accurately Determine the Duration for Futures Contracts such as NQ, MNQ, ES.
def get_futures_contract_month():
    """Get the appropriate contract month for futures (quarterly rollover on Sunday at 16:00)"""
    now = datetime.now()
    current_month = now.month
    current_year = now.year
    current_weekday = now.weekday()
    current_hour = now.hour
    quarterly_months = [3, 6, 9, 12]
    
    is_sunday_after_4pm = (current_weekday == 6 and current_hour >= 16)
    next_contract_month = None
    next_contract_year = current_year
    
    if is_sunday_after_4pm:
        for month in quarterly_months:
            if month > current_month:
                next_contract_month = month
                break
        if next_contract_month is None:
            next_contract_month = quarterly_months[0]
            next_contract_year = current_year + 1
    else:
        if current_month in quarterly_months:
            next_contract_month = current_month
        else:
            for month in quarterly_months:
                if month > current_month:
                    next_contract_month = month
                    break
            
            if next_contract_month is None:
                next_contract_month = quarterly_months[0]
                next_contract_year = current_year + 1
    
    return f"{next_contract_year}{next_contract_month:02d}"

# Gives the total future contracts the GUI system can handel
def is_futures_contract(symbol):
    """Check if a symbol is a futures contract"""
    futures_symbols = ['NQ', 'MNQ', 'ES']
    return symbol.upper() in futures_symbols

# Provides the symbols which can have % changes before 2:00AM on Monday. For this case it should only be futures contract.
def should_allow_weekend_change(symbol):
    """Check if change/percentage change should be allowed on weekends for a symbol"""
    if not is_futures_contract(symbol):
        return False
    
    now = datetime.now()
    current_weekday = now.weekday()
    current_hour = now.hour
    if current_weekday == 6 and current_hour >= 16:
        return True
    
    return False


class BasicTraderGUI:
    # Creates the GUI itself for where the user can see technical indicator information and select specific symbols. 
    def __init__(self):
        self.last_refresh_time = time.time()
        self.indicator_widgets = {}
        # Determines how fast the GUI updates and correlates to the resource load. 
        self.refresh_interval = 20 if FAST_UPDATE_MODE else 60
        
        self.after_job = None
        self.indicator_update_job = None
        self.indicator_container = None
        self.last_updated_label = None
        self.signal_tally_label = None
        self.indicator_queue = queue.Queue()
        self.first_refresh_done = False
        
        # Resource and Thread management
        self.indicator_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_lock = threading.Lock()
        self.active_tasks = set()
        
        # Memory management
        self.memory_monitor_active = True
        
        # TWS Connection for Live Market Information, uses random clientID to decrease using a clientID used by another script or instance of the program.
        self.ib = IB()
        # REMOVED CODE
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Stock Trading GUI")
        self.root.geometry("1000x700")
        self.root.configure(fg_color="#f8f9fa")
        
        self.start_memory_monitor()

        # Creates the Database which logs all technical indicator sums into a database folder inside of the local directory.
        self.db_dir = "database"
        os.makedirs(self.db_dir, exist_ok=True)
        self.csv_path = os.path.join(self.db_dir, "tickers.csv")
        # Used to save the capital so it updates the quantity on every use of the program.
        self.capital_path = os.path.join(self.db_dir, "capital.txt")

        # Load list of tickers and capital into the GUI.
        self.ticker_list = self.load_tickers_from_csv()
        self.capital = self.load_capital_from_file()
        self.create_gui()
        self.schedule_all_indicator_loops()

        # Keep a updated time on the GUI Screen to the second. 
        self.update_clock()

    # Starts parallel processing of indicators for all indictators for all symbols inside of the list. 
    def schedule_all_indicator_loops(self):
        """Optimized parallel processing of all indicators"""
        # Tracks the amount of time that processing all of the indicators takes. 
        start_time = time.time()
        
        with self.processing_lock:
            self.active_tasks.clear()
        
        data_start = time.time()
        self.fetch_all_historical_data()
        self.log_performance_metrics("Data fetching", time.time() - data_start)
        
        # Submit indicator calculations for parallel processing
        futures = []
        for symbol in self.ticker_list:
            future = self.indicator_executor.submit(self.process_symbol_indicators, symbol)
            futures.append(future)
            with self.processing_lock:
                self.active_tasks.add(future)
        
        # Log to console the amount of time which it takes to process indicators. 
        self.monitor_indicator_completion(futures)
        self.log_performance_metrics("Total indicator processing", time.time() - start_time)
    
    # Purpose of this method is to grab the historical data for all of the tiem frames for a specific symbol. 
    def fetch_all_historical_data(self):
        """Fetch historical data for all symbols and timeframes on the main thread"""
        import asyncio
        
        async def fetch_data_async():
            all_timeframes = ["1 Minute", "2 Minutes", "3 Minutes", "5 Minutes", "10 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "4 Hour", "1 Day"]
            
            # It will make it such that 3 Minutes is the timeframe the program automatically selects.
            minimal_timeframe = ["3 Minutes"]

            selected_symbol = self.ticker_var.get().upper() if hasattr(self, 'ticker_var') and self.ticker_var.get() else "AAPL"

            print(f"📡 Fetching historical data...")
            print(f"📍 Selected symbol: {selected_symbol}")

            for symbol in self.ticker_list:
                timeframes = all_timeframes if symbol.upper() == selected_symbol else minimal_timeframe
                for timeframe in timeframes:
                    try:
                        await self.fetch_historical_data_async(symbol, timeframe)
                    except Exception as e:
                        print(f"Failed to fetch data for {symbol} {timeframe}: {e}")
            
        
        # Run the async function synchronously on the main thread
        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            loop.run_until_complete(fetch_data_async())
        except Exception as e:
            try:
                asyncio.run(fetch_data_async())
            except Exception as e2:
                print(f"Fallback also failed.")
    
    # The map of information about what the duration minutes means, and how much data is required to pull. 
    async def fetch_historical_data_async(self, symbol, timeframe):
        """Async method to fetch historical data for a symbol and timeframe"""
        duration_map = {
            "1 Minute": ("2 D", "1 min"),
            "2 Minute": ("2 D", "2 mins"),
            "3 Minutes": ("2 D", "3 mins"),
            "5 Minutes": ("3 D", "5 mins"),
            "10 Minutes": ("3 D", "10 mins"),
            "15 Minutes": ("3 D", "15 mins"),
            "30 Minutes": ("5 D", "30 mins"),
            "1 Hour": ("3 D", "1 hour"),
            "4 Hour": ("10 D", "4 hours"),
            "1 Day": ("180 D", "1 day")
        }
        
        duration, bar_size = duration_map.get(timeframe, ("2 D", "5 mins"))
        
        # Creates the contract for whichever symbol is being processed.
        try:
            symbol_upper = symbol.upper()
            
            if symbol_upper in ["NQ", "MNQ", "ES"]:
                contract_month = get_futures_contract_month()
                if symbol_upper == "ES":
                    contract = ib_insync.Future(
                        symbol="ES", 
                        exchange="CME", 
                        currency="USD", 
                        lastTradeDateOrContractMonth=contract_month
                    )
                else:  # NQ or MNQ
                    contract = ib_insync.Future(
                        symbol=symbol_upper, 
                        exchange="CME", 
                        currency="USD", 
                        lastTradeDateOrContractMonth=contract_month
                    )
            else:
                contract = ib_insync.Stock(symbol, 'SMART', 'USD', primaryExchange='NASDAQ')
            
            await self.ib.qualifyContractsAsync(contract)
            bars = await self.ib.reqHistoricalDataAsync(contract, '', duration, bar_size, 'TRADES', useRTH=False, formatDate=1)
            
            if not bars:
                cache_key = f"{symbol}_{timeframe}"
                with cache_lock:
                    indicator_cache[cache_key] = (None, time.time())
                return
            
            df = util.df(bars)
            if df.empty:
                cache_key = f"{symbol}_{timeframe}"
                with cache_lock:
                    indicator_cache[cache_key] = (None, time.time())
                return
            
            df.set_index('date', inplace=True)
            df = df.sort_index()
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None)
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                cache_key = f"{symbol}_{timeframe}"
                with cache_lock:
                    indicator_cache[cache_key] = (None, time.time())
                return
            
            # Check for sufficient data points
            if len(df) < 30:
                print(f"Insufficient data ({len(df)}) for {symbol} {timeframe} - need at least 30")
            
            cache_key = f"{symbol}_{timeframe}"
            with cache_lock:
                indicator_cache[cache_key] = (df, time.time())
            
        except Exception as e:
            print(f"Error fetching data for {symbol} {timeframe}: {e}")
            cache_key = f"{symbol}_{timeframe}"
            with cache_lock:
                indicator_cache[cache_key] = (None, time.time())
    
    
    def process_symbol_indicators(self, symbol):
        """Process indicators for a single symbol with error handling"""
        try:
            start_time = time.time()
            self.log_all_indicators_to_csv(symbol)
            processing_time = time.time() - start_time
            print(f"✅ {symbol}: Processed in {processing_time:.2f}s")
            return symbol, True
        except Exception as e:
            return symbol, False
    
    def monitor_indicator_completion(self, futures):
        """Monitor the completion of indicator processing tasks"""
        def check_completion():
            completed = 0
            failed = 0
            
            for future in futures[:]:
                if future.done():
                    try:
                        symbol, success = future.result(timeout=0.1)
                        if success:
                            completed += 1
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        print(f"Task failed: {e}")
                    
                    with self.processing_lock:
                        self.active_tasks.discard(future)
            
            total = len(futures)
            if completed + failed == total:

                def update_colors_after_delay():
                    self.update_button_colors()

                    if not self.first_refresh_done:
                        self.first_refresh_done = True

                self.root.after(2000, update_colors_after_delay)
                self.root.after(self.refresh_interval * 1000, self.schedule_all_indicator_loops)

            else:
                # Check again in 1 second
                self.root.after(1000, check_completion)
        
        check_completion()

    def start_memory_monitor(self):
        """Monitor memory usage and clean cache when needed"""
        def monitor():
            if not self.memory_monitor_active:
                return
                
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                
                if memory_mb > 500:
                    self.cleanup_cache()
                
                self.root.after(60000, monitor)
                
            except Exception as e:
                print(f"Memory monitor error: {e}")
                self.root.after(60000, monitor)
        
        monitor()
    
    def log_performance_metrics(self, operation, duration):
        """Log performance metrics for monitoring"""
        if PERFORMANCE_MONITORING:
            print(f"{operation}: {duration:.2f}s")
    
    def cleanup_cache(self):
        """Clean old cache entries to free memory - optimized for performance"""
        current_time = time.time()
        with cache_lock:
            if len(indicator_cache) > 150:
                expired_keys = [
                    key for key, (_, cache_time) in indicator_cache.items()
                    if current_time - cache_time > CACHE_DURATION * 3
                ]
                for key in expired_keys:
                    del indicator_cache[key]
                
                if len(indicator_cache) > 100:
                    sorted_items = sorted(
                        indicator_cache.items(),
                        key=lambda x: x[1][1]
                    )
                    for key, _ in sorted_items[:-75]:
                        del indicator_cache[key]
    
    # Fixes a error that the refresh label will flash when updating the GUI
    def flash_refresh_label(self):
        original_color = "gray"
        self.refresh_timer_label.configure(text_color="green")
        self.root.after(500, lambda: self.refresh_timer_label.configure(text_color=original_color))

    # Loads the capital into the GUI unless there is no data in which case the base case is 10,000.
    def load_capital_from_file(self):
        try:
            if os.path.exists(self.capital_path):
                with open(self.capital_path, "r") as f:
                    return float(f.read().strip())
        except:
            pass
        return 10000.0 

    # Saves the capital from the GUI to the required file. 
    def save_capital_to_file(self):
        try:
            with open(self.capital_path, "w") as f:
                f.write(str(self.capital_var.get()))
        except Exception as e:
            print(f"Error saving capital: {e}")

    # Loads the tickers from the CSV, if this file doesn't exist it falls back on the list provided.
    def load_tickers_from_csv(self):
        default = ["AAPL", "MSFT", "GOOGL", "TSLA", "META", "QQQ", "TQQQ", "SPY", "SQQQ", "FNGU", "FNGD", "SHOP", "HIMS"]
        if os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, newline='') as file:
                    return [row[0] for row in csv.reader(file) if row]
            except:
                return default
        else:
            self.save_tickers_to_csv(default)
            return default

    # Saves the tickers from the GUI into the CSV so it can be used every time.
    def save_tickers_to_csv(self, ticker_list):
        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for ticker in ticker_list:
                writer.writerow([ticker])

    # Force refreshes the price section of the GUI. 
    def refresh_price_section(self):
        if hasattr(self, "price_section") and self.price_section:
            self.price_section.destroy()
        selected_symbol = self.ticker_var.get()
        self.create_price_section(self.price_section_container, selected_symbol)

    # Resets the GUI.
    def reset_gui(self):
        
        if self.after_job:
            self.root.after_cancel(self.after_job)
            self.after_job = None
            
        if self.indicator_update_job:
            self.root.after_cancel(self.indicator_update_job)
            self.indicator_update_job = None

        # Destroy all widgets inside root
        for widget in self.root.winfo_children():
            widget.destroy()

        # Clear cache to ensure fresh data
        with cache_lock:
            indicator_cache.clear()
        
        self.capital = self.load_capital_from_file()
        
        self.create_gui()
        
        self.schedule_all_indicator_loops()
        

    def log_all_indicators_to_csv(self, symbol):
        """
        Logs BUY/SELL/NEUTRAL count and majority signal for each timeframe to a CSV.
        Example output:
        Timestamp | Symbol | Timeframe | BUY | SELL | NEUTRAL | Overall
        """
        import os
        import csv
        from datetime import datetime
        from collections import Counter

        start_time = time.time()
        timeframes = ["1 Minute", "2 Minutes", "3 Minutes", "5 Minutes", "10 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "4 Hour", "1 Day"]
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows_to_write = []

        for tf in timeframes:
            indicators = self.get_indicators_for_timeframe(symbol, tf)

            # Count BUY / SELL / NEUTRAL signals
            signal_counts = Counter(signal for _, signal in indicators.values())
            buy_count = signal_counts["BUY"]
            sell_count = signal_counts["SELL"]
            neutral_count = signal_counts["NEUTRAL"]

            # Determine majority
            majority_signal = max(signal_counts, key=lambda x: signal_counts[x]) if signal_counts else "NEUTRAL"

            rows_to_write.append({
                "Timestamp": now,
                "Symbol": symbol,
                "Timeframe": tf,
                "BUY": buy_count,
                "SELL": sell_count,
                "NEUTRAL": neutral_count,
                "Overall": majority_signal
            })

        # Save to database folder alongside other ticker files
        folder = self.db_dir
        filename = os.path.join(folder, f"{symbol.lower()}_indicator_summary.csv")

        file_exists = os.path.isfile(filename)
        with open(filename, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Symbol", "Timeframe", "BUY", "SELL", "NEUTRAL", "Overall"])
            writer.writeheader()
            writer.writerows(rows_to_write)

        processing_time = time.time() - start_time
        print(f"Summary written for {symbol} → {filename} (took {processing_time:.2f}s)")

    # As all timeframes and their points have been calculated, it gets the data for a specific timeframe. 
    def get_latest_overall_by_timeframe(self, symbol):
        """
        Reads the most recent summary row per timeframe from the symbol's summary CSV
        and returns a dictionary like: {"3 Minutes": "BUY", "5 Minutes": "SELL", ...}
        """
        import os
        import csv
        from collections import defaultdict

        filename = os.path.join(self.db_dir, f"{symbol.lower()}_indicator_summary.csv")

        if not os.path.exists(filename):
            print(f"Summary file not found: {filename}")
            return self.get_signals_from_cache(symbol)

        latest_overall_map = {}

        try:
            with open(filename, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timeframe = row.get("Timeframe")
                    overall = row.get("Overall")
                    if timeframe and overall and overall != "NEUTRAL":
                        latest_overall_map[timeframe] = overall
                
                if not latest_overall_map:
                    print(f"CSV file empty or all NEUTRAL for {symbol}, trying cache...")
                    return self.get_signals_from_cache(symbol)
                    
        except Exception as e:
            print(f"[!] Error reading CSV for {symbol}: {e}")
            return self.get_signals_from_cache(symbol)

        return latest_overall_map
    
    # Gets signal from the cache which is built if the CSV is not available.
    def get_signals_from_cache(self, symbol):
        """
        Fallback method to get signals from cache when CSV is not available
        """
        timeframes = ["1 Minute", "3 Minutes", "5 Minutes", "10 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "4 Hour", "1 Day"]
        signals = {}
        
        for timeframe in timeframes:
            cache_key = f"{symbol}_{timeframe}"
            with cache_lock:
                if cache_key in indicator_cache:
                    cached_data, cache_time = indicator_cache[cache_key]
                    if isinstance(cached_data, dict) and time.time() - cache_time < CACHE_DURATION:
                        # Calculate overall signal from cached indicators
                        overall_signal = self.calculate_overall_signal(cached_data)
                        if overall_signal != "NEUTRAL":
                            signals[timeframe] = overall_signal
                            print(f"📊 Using cache signal for {symbol} {timeframe}: {overall_signal}")
        
        return signals
    
    # Calculates the overall symbol which is used for the price section. 
    def calculate_overall_signal(self, indicators_dict):
        """
        Calculate overall signal from individual indicators
        """
        if not indicators_dict:
            return "NEUTRAL"
            
        buy_count = 0
        sell_count = 0
        total_count = 0
        
        
        for indicator_name, (value, signal) in indicators_dict.items():
            weight = 3 if indicator_name in {"MACD HISTOGRAM", "Heikin Ashi"} else 1
            
            if signal in ["BUY", "SELL"]:
                total_count += weight
                if signal == "BUY":
                    buy_count += weight
                elif signal == "SELL":
                    sell_count += weight
        
        buy_ratio = buy_count / total_count
        sell_ratio = sell_count / total_count
        
        if buy_ratio > 0.6:
            return "BUY"
        elif sell_ratio > 0.6:
            return "SELL"
        else:
            return "NEUTRAL"

    # Instead of using the live data, use the cached data when it is not available. 
    def get_indicators_for_timeframe(self, symbol, timeframe):
        """Calculate indicators from cached data (no ib_insync calls)"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas_ta as ta

        # Check cache first
        cache_key = f"{symbol}_{timeframe}"
        current_time = time.time()
        df = None 
        
        with cache_lock:
            if cache_key in indicator_cache:
                cached_data, cache_time = indicator_cache[cache_key]
                if current_time - cache_time < CACHE_DURATION and not FORCE_DATA_REFRESH:
                    if isinstance(cached_data, dict):
                        if DEBUG_INDICATORS:
                            print(f"Using cached indicators for {symbol} {timeframe}")
                        return cached_data
                    elif isinstance(cached_data, pd.DataFrame):
                        df = cached_data
                        if DEBUG_INDICATORS:
                            print(f"Found raw DataFrame data for {symbol} {timeframe} - calculating indicators (DataFrame shape: {df.shape})")
                    else:
                        print(f"Invalid cached data type for {symbol} {timeframe}: {type(cached_data)}")
                        return {}
                else:
                    if DEBUG_INDICATORS:
                        print(f"Cache expired or force refresh for {symbol} {timeframe}")
                    del indicator_cache[cache_key]

        # If we have a DataFrame validate it has sufficient data for calculations
        if df is None or df.empty:
            import time as time_module
            for attempt in range(3): 
                time_module.sleep(0.5) 
                with cache_lock:
                    if cache_key in indicator_cache:
                        cached_data, cache_time = indicator_cache[cache_key]
                        if isinstance(cached_data, dict):
                            if DEBUG_INDICATORS:
                                print(f"Found processed indicators after waiting for {symbol} {timeframe}")
                            return cached_data
                        elif isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                            df = cached_data
                            if DEBUG_INDICATORS:
                                print(f"Found DataFrame after waiting for {symbol} {timeframe}")
                            break
            
            return self._get_placeholder_indicators()
        
        # Validate DataFrame has data for technical indicators
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            print(f"DataFrame missing required columns for {symbol} {timeframe}. Available: {df.columns.tolist()}")
            return self._get_placeholder_indicators()
        
        # Check if we have enough data points for calculations (need at least 30 for most indicators)
        if len(df) < 30:
            print(f"Insufficient data points ({len(df)}) for {symbol} {timeframe} - need at least 30")
            import time as time_module
            for attempt in range(2):
                time_module.sleep(1.0)
                with cache_lock:
                    if cache_key in indicator_cache:
                        cached_data, cache_time = indicator_cache[cache_key]
                        if isinstance(cached_data, pd.DataFrame) and len(cached_data) >= 30:
                            df = cached_data
                            if DEBUG_INDICATORS:
                                print(f"Found sufficient data after waiting for {symbol} {timeframe}")
                            break
                        elif isinstance(cached_data, dict):
                            if DEBUG_INDICATORS:
                                print(f"Found processed indicators after waiting for {symbol} {timeframe}")
                            return cached_data
            
            if len(df) < 30:
                print(f"Insufficient data ({len(df)}) for {symbol} {timeframe} - using placeholders")
                return self._get_placeholder_indicators()

        def compute_rsi():
            # Removed Information
            
        def compute_macd():
            # Removed Information

        def compute_volume():
            # Removed Information

        def compute_vwap():
            # Removed Information

        def compute_cci():
            # Removed Information

        def compute_adx():
            # Removed Information

        def compute_stoch():
            # Removed Information

        def compute_momentum():
            # Removed Information

        def compute_williams():
            # Removed Information

        def compute_heikin():
            # Removed Information
        
        def compute_ema_9_14():
            try:
                ema9 = ta.ema(df['close'], length=9)
                ema14 = ta.ema(df['close'], length=14)
                if ema9.iloc[-1] > ema14.iloc[-1]:
                    signal = "BUY"
                elif ema9.iloc[-1] < ema14.iloc[-1]:
                    signal = "SELL"
                else:
                    signal = "NEUTRAL"
                return ("EMA 9 vs 14", (f"{ema9.iloc[-1]:.2f}/{ema14.iloc[-1]:.2f}", signal))
            except Exception as e:
                print(f"Error computing EMA 9 vs 14 for {symbol} {timeframe}: {e}")
                return ("EMA 9 vs 14", ("N/A", "NEUTRAL"))
        
        def compute_ema_6_21():
            try:
                ema5 = ta.ema(df['close'], length=6)
                ema20 = ta.ema(df['close'], length=21)
                if ema5.iloc[-1] > ema20.iloc[-1]:
                    signal = "BUY"
                elif ema5.iloc[-1] < ema20.iloc[-1]:
                    signal = "SELL"
                else:
                    signal = "NEUTRAL"
                return ("EMA 6 vs 21", (f"{ema5.iloc[-1]:.2f}/{ema20.iloc[-1]:.2f}", signal))
            except Exception as e:
                print(f"Error computing EMA 6 vs 21 for {symbol} {timeframe}: {e}")
                return ("EMA 6 vs 21", ("N/A", "NEUTRAL"))
        
        def compute_atr():
            # Removed Information

        def compute_bollinger():
            # Removed Information

        def compute_obv():
            # Removed Information
            
        def compute_parabolic_sar():
            # Removed Information

        def compute_dema():
            # Removed Information

        def compute_tema():
            # Removed Information

        def compute_pivot_point():
            # Removed Information

        funcs = [
            compute_rsi, compute_macd, compute_volume, compute_vwap, compute_cci,
            compute_adx, compute_stoch, compute_momentum, compute_williams,
            compute_heikin, compute_ema_9_14, compute_ema_6_21,
            compute_atr, compute_bollinger, compute_obv,
            compute_parabolic_sar, compute_dema, compute_tema,
            compute_pivot_point
        ]

        # Optimized parallel processing with better error handling
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(f) for f in funcs]
            for future in as_completed(futures):
                try:
                    k, v = future.result(timeout=10)
                    results[k] = v
                except Exception as e:
                    print(f"Error in indicator computation for {symbol} {timeframe}: {e}")

        if not results:
            print(f"No indicators calculated successfully for {symbol} {timeframe}")
            import time as time_module
            for attempt in range(2):
                time_module.sleep(1.0)
                with cache_lock:
                    if cache_key in indicator_cache:
                        cached_data, cache_time = indicator_cache[cache_key]
                        if isinstance(cached_data, dict) and len(cached_data) > 0:
                            if DEBUG_INDICATORS:
                                print(f"Found processed indicators after waiting for {symbol} {timeframe}")
                            return cached_data
            
            return self._get_placeholder_indicators()

        with cache_lock:
            indicator_cache[cache_key] = (results, current_time)

        if DEBUG_INDICATORS:
            print(f"Successfully calculated {len(results)} indicators for {symbol} {timeframe}")
        return results

    # This is for the first run before a live update can be calculated for all indicators.
    # Allows for the GUI to load much faster.
    def _get_placeholder_indicators(self):
        """Return placeholder indicators when data is not available"""
        return {
            "RSI": ("N/A", "NEUTRAL"),
            "MACD HISTOGRAM": ("N/A", "NEUTRAL"),
            "Heikin Ashi": ("N/A", "NEUTRAL"),
            "EMA 9 vs 14": ("N/A", "NEUTRAL"),
            "ADX and DI+/-": ("N/A", "NEUTRAL"),
            "CCI": ("N/A", "NEUTRAL"),
            "Stochastic": ("N/A", "NEUTRAL"),
            "Momentum": ("N/A", "NEUTRAL"),
            "Williams %R": ("N/A", "NEUTRAL"),
            "Volume Spike": ("N/A", "NEUTRAL"),
            "VWAP": ("N/A", "NEUTRAL"),
            "EMA 6 vs 21": ("N/A", "NEUTRAL"),
            "ATR": ("N/A", "NEUTRAL"),
            "Bollinger Bands": ("N/A", "NEUTRAL"),
            "OBV": ("N/A", "NEUTRAL"),
            "Parabolic SAR": ("N/A", "NEUTRAL"),
            "DEMA 21": ("N/A", "NEUTRAL"),
            "TEMA 21": ("N/A", "NEUTRAL"),
            "Pivot Point": ("N/A", "NEUTRAL")
        }

    # Renders the indicators onto the GUI. 
    def render_indicators(self, tf, container):
        try:
            if not container.winfo_exists():
                print("Container no longer exists - skipping render")
                return
        except Exception as e:
            print(f"Exception checking container validity: {e}")
            return

        if not hasattr(self, 'indicator_widgets'):
            self.indicator_widgets = {}

        self.current_indicator_tf = tf
        self.indicator_container = container

        buy_count = 0
        sell_count = 0
        neutral_count = 0

        symbol = self.ticker_var.get()
        print(f"Rendering indicators for {symbol} ({tf})")

        indicators = self.get_indicators_for_timeframe(symbol, tf)
        if not indicators:
            print(f"No indicators available for {symbol} {tf} - showing placeholder")
            placeholder_frame = ctk.CTkFrame(container, fg_color="white", border_width=1, border_color="#e5e7eb", corner_radius=8)
            placeholder_frame.grid(row=0, column=0, columnspan=5, padx=10, pady=10, sticky="ew")

            placeholder_label = ctk.CTkLabel(
                placeholder_frame,
                text=f"Loading indicators for {symbol} {tf}...",
                font=("Arial", 14, "bold"),
                text_color="#6b7280"
            )
            placeholder_label.pack(pady=20)
            return

        leading_indicators = {
            "RSI", "Stochastic", "Momentum", "Williams %R", "Heikin Ashi",
            "EMA 9 vs 14", "Volume Spike", "VWAP", "CCI", "MACD HISTOGRAM", "EMA 6 vs 21"
        }

        lagging_indicators = {
            "ADX and DI+/-", "ATR", "Bollinger Bands", "OBV",
            "Parabolic SAR", "DEMA 21", "TEMA 21", "Pivot Point"
        }

        indicator_order = [
            "MACD HISTOGRAM", "Heikin Ashi", "EMA 9 vs 14", "RSI", "VWAP", "CCI",
            "Stochastic", "Momentum", "Williams %R",
            "EMA 6 vs 21", "Volume Spike", "ADX and DI+/-",
            "ATR", "Bollinger Bands", "OBV",
            "Parabolic SAR", "DEMA 21", "TEMA 21", "Pivot Point"
        ]

        shown_indicators = [(name, indicators[name]) for name in indicator_order if name in indicators]

        if not hasattr(self, 'legend_frame'):
            self.legend_frame = ctk.CTkFrame(container, fg_color="white", border_width=1, border_color="#e5e7eb", corner_radius=8)
            self.legend_frame.grid(row=0, column=0, columnspan=5, padx=10, pady=(10, 5), sticky="ew")

            legend_label = ctk.CTkLabel(
                self.legend_frame,
                text="Technical Indicators - Green (BUY) | Red (SELL) | Orange (NEUTRAL) - Blue (Leading) | Purple (Lagging)",
                font=("Arial", 12, "bold"),
                text_color="#030213"
            )
            legend_label.pack(pady=10)

        for i, (name, (value, signal)) in enumerate(shown_indicators):
            row = (i // 5)
            col = i % 5
            key = (symbol, tf, name)

            if signal == "BUY":
                box_color = "#10b981"
            elif signal == "SELL":
                box_color = "#ef4444"
            else:
                box_color = "#f59e0b"

            border_color = box_color
            indicator_type = "Leading" if name in leading_indicators else "Lagging"
            type_box_color = "#1e40af" if indicator_type == "Leading" else "#7c3aed"

            weight = 3 if name in {"MACD HISTOGRAM", "Heikin Ashi"} else 1

            if signal == "BUY":
                buy_count += weight
            elif signal == "SELL":
                sell_count += weight
            else:
                neutral_count += weight
            

            if key in self.indicator_widgets:
                widgets = self.indicator_widgets[key]
                widgets["value_label"].configure(text=value)
                widgets["signal_label"].configure(text=signal)
                widgets["box"].configure(fg_color=box_color, border_color=border_color)
            else:
                box = ctk.CTkFrame(container, fg_color=box_color, border_width=2, border_color=border_color, corner_radius=8)
                box.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

                name_with_type = f"{name} ({indicator_type})"
                name_label = ctk.CTkLabel(box, text=name_with_type, font=("Arial", 12, "bold"), text_color="white")
                name_label.pack(pady=(5, 0))

                value_label = ctk.CTkLabel(box, text=value, font=("Arial", 12), text_color="white")
                value_label.pack(pady=(0, 2))

                type_box = ctk.CTkFrame(box, fg_color=type_box_color, corner_radius=4)
                type_box.pack(pady=(0, 2))
                type_label = ctk.CTkLabel(type_box, text=indicator_type, font=("Arial", 10, "bold"), text_color="white")
                type_label.pack(padx=8, pady=2)

                signal_label = ctk.CTkLabel(box, text=signal, font=("Arial", 12, "bold"), text_color="white")
                signal_label.pack(pady=(0, 5))

                self.indicator_widgets[key] = {
                    "box": box,
                    "value_label": value_label,
                    "signal_label": signal_label
                }

        if hasattr(self, "signal_tally_label") and self.signal_tally_label:
            summary_text = f"Tally: BUY = {buy_count}, SELL = {sell_count}, NEUTRAL = {neutral_count}"
            if buy_count > sell_count:
                summary_color = "#10b981"
            elif sell_count > buy_count:
                summary_color = "#ef4444"
            else:
                summary_color = "#6b7280"
            self.signal_tally_label.configure(text=summary_text, text_color=summary_color)

        for col in range(5):
            container.grid_columnconfigure(col, weight=1)



    # Updates the selected timeframe. 
    def update_selected(self, tf):
        """Update the selected timeframe and render indicators"""
        self.selected_timeframe.set(tf)
        
        self.update_button_colors()

        if self.indicator_update_job:
            self.root.after_cancel(self.indicator_update_job)

        self.render_indicators(tf, self.indicator_container)
        self.schedule_indicator_refresh()
        self.flash_refresh_label()

    def create_technical_indicators_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color="white", corner_radius=12)
        section.pack(fill="x", pady=(0, 20), padx=20)

        ctk.CTkLabel(section, text="Technical Indicators", font=("Arial", 15, "bold"), text_color="#030213").pack(anchor="w", pady=10, padx=20)

        refresh_button = ctk.CTkButton(
            section, 
            text="🔄 Refresh Indicators", 
            command=self.manual_refresh_indicators,
            fg_color="#3b82f6", 
            text_color="white", 
            hover_color="#2563eb",
            font=("Arial", 12), 
            corner_radius=8, 
            width=150
        )
        refresh_button.pack(anchor="w", padx=20, pady=(0, 10))

        self.refresh_timer_label = ctk.CTkLabel(section, text="⏳ Next update in some seconds", font=("Calibri", 12), text_color="gray")
        self.refresh_timer_label.pack(anchor="w", padx=20)
        self.signal_tally_label = ctk.CTkLabel(section, text="", font=("Calibri", 12), text_color="gray")
        self.signal_tally_label.pack(anchor="w", padx=20, pady=(0, 10))


        self.last_updated_label = ctk.CTkLabel(section, text="", font=("Calibri", 11), text_color="gray")
        self.last_updated_label.pack(anchor="w", padx=20)

        self.start_refresh_timer_countdown()

        button_frame = ctk.CTkFrame(section, fg_color="#f4f4f4", corner_radius=8)
        button_frame.pack(fill="x", padx=20, pady=(0, 10))

        timeframes = ["1 Minute", "3 Minutes", "5 Minutes", "10 Minutes", "15 Minutes", "30 Minutes", "1 Hour", "4 Hour", "1 Day"]
        self.selected_timeframe = tk.StringVar(value="3 Minutes")

        self.timeframe_buttons = {}

        for tf in timeframes:
            btn = ctk.CTkButton(
                button_frame, text=tf, command=lambda tf=tf: self.update_selected(tf),
                fg_color="#e5e7eb", text_color="#030213", hover_color="#d1d5db",
                font=("Arial", 13), corner_radius=8, width=120
            )
            btn.pack(side="left", padx=5, pady=5)
            self.timeframe_buttons[tf] = btn

        self.indicator_container = ctk.CTkFrame(section, fg_color="white")
        self.indicator_container.pack(fill="x", padx=20, pady=5)

        self.root.after(500, self.update_selected, "3 Minutes")
        
        def update_colors_with_retry():
            current_symbol = self.ticker_var.get().strip().upper() if hasattr(self, 'ticker_var') else "AAPL"
            symbol_signals = self.get_latest_overall_by_timeframe(current_symbol)
            
            non_neutral_count = sum(1 for signal in symbol_signals.values() if signal != "NEUTRAL")
            
            if non_neutral_count > 0:
                self.update_button_colors()
            else:
                retry_count = getattr(self, '_color_retry_count', 0) + 1
                self._color_retry_count = retry_count
                
                if retry_count <= 10:
                    delay = retry_count * 3
                    self.root.after(delay * 1000, update_colors_with_retry)
                else:
                    self.update_button_colors()
        
        self.root.after(3000, update_colors_with_retry) 

    def manual_refresh_indicators(self):
        """Manually refresh indicators for debugging"""
        try:
            current_symbol = self.ticker_var.get().strip().upper() if hasattr(self, 'ticker_var') else "AAPL"
            current_tf = self.selected_timeframe.get() if hasattr(self, 'selected_timeframe') else "3 Minutes"
            
            print(f"🔄 Manual refresh triggered for {current_symbol} ({current_tf})")
            
            print(f"📡 Force refreshing data for {current_symbol} {current_tf}")
            self.root.after(0, lambda: self._refresh_data_for_symbol(current_symbol, current_tf))
            
            cache_key = f"{current_symbol}_{current_tf}"
            with cache_lock:
                if cache_key in indicator_cache:
                    del indicator_cache[cache_key]
                    print(f"🗑️ Cleared cache for {cache_key}")
            
            self.root.after(3000, lambda: self.render_indicators(current_tf, self.indicator_container))
            
            self.root.after(4000, self.update_button_colors)
            
            now_str = datetime.now().strftime("%H:%M:%S")
            if hasattr(self, 'last_updated_label') and self.last_updated_label:
                self.last_updated_label.configure(text=f"🕒 Last manually updated at {now_str}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh indicators: {e}")

    def schedule_indicator_refresh(self):
        try:
            if not self.first_refresh_done:
                self.first_refresh_done = True
                self.root.after(7000, self.schedule_indicator_refresh)
                return
            
            if hasattr(self, 'indicator_container') and self.indicator_container and hasattr(self, 'selected_timeframe'):
                current_tf = self.selected_timeframe.get()
                current_symbol = self.ticker_var.get().strip().upper() if hasattr(self, 'ticker_var') else "AAPL"
                

                if FORCE_DATA_REFRESH:
                    self.root.after(0, lambda: self._refresh_data_for_symbol(current_symbol, current_tf))
                
                self.render_indicators(current_tf, self.indicator_container)
                self.last_refresh_time = time.time()

                now_str = datetime.now().strftime("%H:%M:%S")
                if hasattr(self, 'last_updated_label') and self.last_updated_label:
                    self.last_updated_label.configure(text=f"Last updated at {now_str}")

                # Flash "Updating…" text for 1 second
                if hasattr(self, 'refresh_timer_label') and self.refresh_timer_label:
                    self.refresh_timer_label.configure(text="Updating...", text_color="blue")
                    self.root.after(1000, lambda: self.refresh_timer_label.configure(text="Next update in 60s", text_color="gray"))

                self.update_button_colors()

                # Schedule next refresh
                self.indicator_update_job = self.root.after(self.refresh_interval * 1000, self.schedule_indicator_refresh)
            else:
                print("Missing indicator container or selected timeframe - skipping refresh")
        except Exception as e:
            print(f"Error during indicator refresh: {e}")
            
            self.indicator_update_job = self.root.after(self.refresh_interval * 1000, self.schedule_indicator_refresh)

    def _refresh_data_for_symbol(self, symbol, timeframe):
        """Refresh data for a specific symbol and timeframe"""
        try:
            print(f"Refreshing data for {symbol} {timeframe}")
            # Use asyncio to fetch fresh data
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.fetch_historical_data_async(symbol, timeframe))
            finally:
                loop.close()
        except Exception as e:
            print(f"Error refreshing data for {symbol} {timeframe}: {e}")

    def start_refresh_timer_countdown(self):
        def update_countdown():
            now = time.time()
            remaining = max(0, int(self.refresh_interval - (now - self.last_refresh_time)))

            # Update countdown label
            minutes, seconds = divmod(remaining, 60)
            self.refresh_timer_label.configure(
                text=f"Next update in {seconds:02d}s",
                text_color="red" if remaining <= 5 else "gray"
            )

            self.root.after(1000, update_countdown)  # schedule next tick

        update_countdown()

    # Creates the GUI itself with all sections as seperate.
    def create_gui(self):
        self.watchlist_frame = None
        self.stat_widgets = {}

        # Only one root-packed container: scrollable_frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self.root, fg_color="white")
        self.scrollable_frame.pack(fill="both", expand=True, padx=20, pady=10)

        main_frame = self.scrollable_frame  # Use this for all GUI content
        self.price_section_container = main_frame  # Save for price-only refresh
        # Build UI components inside scrollable_frame
        self.create_top_bar(main_frame)
        self.refresh_price_section()
        self.create_watchlist_and_portfolio_section(main_frame)
        self.create_technical_indicators_section(main_frame)
           
    # Creates the "Add Ticker Dialouge Menu"
    def show_add_ticker_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Add Ticker")
        dialog.geometry("300x150")
        dialog.configure(fg_color="white")
        dialog.resizable(False, False)

        label = ctk.CTkLabel(dialog, text="Enter Ticker:", font=("Arial", 13, "bold"))
        label.pack(pady=10)

        entry = ctk.CTkEntry(dialog, width=200)
        entry.pack(pady=5)
        entry.focus()

        def add():
            ticker = entry.get().strip().upper()
            if ticker and ticker not in self.ticker_list:
                self.ticker_list.append(ticker)
                self.save_tickers_to_csv(self.ticker_list)
                messagebox.showinfo("Success", f"{ticker} added to CSV!")
                dialog.destroy()
                self.reset_gui()
            else:
                messagebox.showerror("Error", "Invalid or duplicate ticker.")

        button_frame = ctk.CTkFrame(dialog, fg_color="white")
        button_frame.pack(pady=10)

        ctk.CTkButton(button_frame, text="Add", command=add, fg_color="#10b981").pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, fg_color="#f8f9fa", text_color="#030213").pack(side="left", padx=5)

    # Updates the overall symbol label.
    def update_signal_button(self):
        """
        Updates the BUY/SELL/NEUTRAL signal button every 60 seconds.
        """
        try:
            signal = self.get_latest_3min_overall_signal() or "NEUTRAL"
        except Exception as e:
            print(f"Failed to get signal: {e}")
            signal = "NEUTRAL"

        # Colors based on signal
        signal_colors = {
            "BUY": ("#10b981", "#059669"),
            "SELL": ("#ef4444", "#b91c1c"),
            "NEUTRAL": ("#f59e0b", "#d97706")
        }
        fg_color, hover_color = signal_colors.get(signal.upper(), ("#6b7280", "#4b5563"))

        self.signal_button.configure(
            text=signal.upper(),
            fg_color=fg_color,
            hover_color=hover_color
        )

        self.root.after(60_000, self.update_signal_button)

    # Creates the Top Bar Itself.
    def create_top_bar(self, parent):
        frame = ctk.CTkFrame(parent, fg_color="white", corner_radius=12, height=60)
        frame.pack(fill="x", pady=(0, 20))
        frame.pack_propagate(False)
        
        try:
            signal = self.get_latest_3min_overall_signal() or "NEUTRAL"
        except Exception as e:
            print(f"⚠️ Failed to get signal: {e}")
            signal = "NEUTRAL"

        # Map signal to color
        signal_colors = {
            "BUY": ("#10b981", "#059669"),   
            "SELL": ("#ef4444", "#b91c1c"),   
            "NEUTRAL": ("#f59e0b", "#d97706")    
        }
        fg_color, hover_color = signal_colors.get(signal.upper(), ("#6b7280", "#4b5563"))

        self.signal_button = ctk.CTkButton(
            frame,
            text=signal.upper(),
            fg_color=fg_color,
            hover_color=hover_color,
            text_color="white",
            font=("Arial", 13, "bold"),
            width=80,
            height=36,
            corner_radius=8
        )
        self.signal_button.pack(side="left", padx=20)
        self.update_signal_button()
        
        ticker_label = ctk.CTkLabel(
            frame,
            text="Ticker:",
            font=("Arial", 14, "bold"),
            text_color="#030213"
        )
        ticker_label.pack(side="left", padx=(10, 5))

        dropdown_frame = ctk.CTkFrame(
            frame,
            fg_color="#f8f9fa",
            corner_radius=8,
            border_width=1,
            border_color="#e0e0e0"
        )
        dropdown_frame.pack(side="left", padx=(0, 20))

        self.ticker_var = tk.StringVar(value=self.ticker_list[0] if self.ticker_list else "")
        if not hasattr(self, 'ticker_traced'):
            self.ticker_var.trace_add("write", lambda *args: self.root.after(100, self.on_ticker_change))


            self.ticker_traced = True

        self.ticker_combo = ctk.CTkComboBox(
            dropdown_frame,
            values=self.ticker_list,
            variable=self.ticker_var,
            width=150,
            height=36,
            font=("Arial", 13),
            corner_radius=8,
            border_color="#e0e0e0",
            button_color="#d1d5db",
            button_hover_color="#9ca3af",
            fg_color="#f8f9fa",
            text_color="#030213",
            state="readonly"
        )
        self.ticker_combo.pack(padx=1, pady=1)

        self.ticker_combo.bind("<Enter>", lambda e: dropdown_frame.configure(fg_color="#e9ecef"))
        self.ticker_combo.bind("<Leave>", lambda e: dropdown_frame.configure(fg_color="#f8f9fa"))

        add_button = ctk.CTkButton(
            frame,
            text="+ Add",
            width=80,
            height=36,
            font=("Arial", 13, "bold"),
            corner_radius=8,
            fg_color="#030213",
            hover_color="#1a1a1a",
            text_color="white",
            command=self.show_add_ticker_dialog 
        )
        add_button.pack(side="left")

        self.resources_label = ctk.CTkLabel(
            frame,
            text="CPU: --% | RAM: --MB",
            font=("Arial", 12),
            text_color="#6b7280"
        )
        self.resources_label.pack(side="right", padx=(0, 10))
        
        self.clock_label = ctk.CTkLabel(
            frame,
            text="--:--:--",
            font=("Arial", 14),
            text_color="#6b7280"
        )
        self.clock_label.pack(side="right", padx=20)

    # Creates the price information section at the top of the GUI.
    def create_price_section(self, parent, symbol):
        section = ctk.CTkFrame(parent, fg_color="white", corner_radius=12)
        section.pack(fill="x", pady=10, padx=10)
        section.pack_propagate(False)
        self.price_section = section 

        self.price_title_label = ctk.CTkLabel(
            section,
            text=f"{symbol} - Price Information",
            font=("Arial", 16, "bold"),
            text_color="#030213"
        )
        self.price_title_label.pack(anchor="w", padx=20, pady=(20, 10))

        self.stats_frame = ctk.CTkFrame(section, fg_color="white")
        self.stats_frame.pack(fill="x", padx=20, pady=(0, 20))

        # Create IB connection with random clientId
        self.ib = IB()
        client_id = random.randint(1000, 9999)
        # Removed

        # Contract setup
        symbol_upper = symbol.upper()
            
        if symbol_upper in ["NQ", "MNQ", "ES"]:
            # Futures contracts
            contract_month = get_futures_contract_month()
            if symbol_upper == "ES":
                self.contract = ib_insync.Future(
                    symbol="ES", 
                    exchange="CME", 
                    currency="USD", 
                    lastTradeDateOrContractMonth=contract_month
                )
            else:  # NQ or MNQ
                self.contract = ib_insync.Future(
                    symbol=symbol_upper, 
                    exchange="CME", 
                    currency="USD", 
                    lastTradeDateOrContractMonth=contract_month
                )
            print(f"📈 Created futures contract for {symbol_upper} with month {contract_month}")
        else:
            # Stock contract
            self.contract = ib_insync.Stock(symbol, 'SMART', 'USD', primaryExchange='NASDAQ')
            print(f"📊 3 Created stock contract for {symbol}")

        # Start update loop
        self.update_price_data()

    def create_watchlist_and_portfolio_section(self, parent):
        outer_frame = ctk.CTkFrame(parent, fg_color="#f8f9fa")
        outer_frame.pack(fill="both", expand=True, padx=20, pady=20)

        outer_frame.grid_columnconfigure(0, weight=1, uniform="half")
        outer_frame.grid_columnconfigure(1, weight=1, uniform="half")

        watchlist_box = ctk.CTkFrame(outer_frame, corner_radius=12, fg_color="white")
        watchlist_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=5)
        
        self.create_watchlist_section(watchlist_box)

        portfolio_box = ctk.CTkFrame(outer_frame, corner_radius=12, fg_color="white")
        portfolio_box.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=5)
        
        self.create_portfolio_section(portfolio_box)


    def create_portfolio_section(self, parent):
        port_title = ctk.CTkLabel(parent, text="Portfolio Management", font=("Arial", 14, "bold"), text_color="#030213")
        port_title.pack(anchor="w", padx=10, pady=(10, 5))

        placeholder_label = ctk.CTkLabel(parent, text="(Coming soon...)", font=("Arial", 12), text_color="#6b7280")
        placeholder_label.pack(anchor="w", padx=10, pady=5)

    # Gets the latest like standing of the 3 minute for all of the symbols. Uses this for the overall indicator on ticker list.
    def get_latest_3min_overall_signal(self):
        """
        Reads the latest '3 Minutes' timeframe signal from the summary CSV
        for the symbol currently selected in the dropdown (or AAPL by default).
        """
        import os
        import csv

        symbol = self.ticker_var.get().strip().upper() if hasattr(self, 'ticker_var') else "AAPL"
        filename = os.path.join(self.db_dir, f"{symbol.lower()}_indicator_summary.csv")

        if not os.path.exists(filename):
            print(f"⚠️ File not found for {symbol}: {filename}")
            return None

        latest_row = None

        with open(filename, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Timeframe") == "3 Minutes":
                    latest_row = row
                    

        return latest_row.get("Overall", None) if latest_row else None

    # Runs and gets the 3 min like technical status.
    def get_3min_signal_for_symbol(self, symbol):
        """
        Reads the latest '3 Minutes' timeframe signal for the given symbol from the summary CSV.
        """
        import os
        import csv
        from datetime import datetime

        filename = os.path.join(self.db_dir, f"{symbol.lower()}_indicator_summary.csv")
        if not os.path.exists(filename):
            return "WAIT"

        rows = []
        try:
            with open(filename, mode='r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Timeframe") == "3 Minutes":
                        try:
                            row["ParsedTimestamp"] = datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M:%S")
                            rows.append(row)
                        except:
                            pass
        except Exception as e:
            print(f"Error reading {symbol} 3m signal: {e}")
            return "WAIT"

        if not rows:
            return "WAIT"

        latest = sorted(rows, key=lambda r: r["ParsedTimestamp"], reverse=True)[0]
        return latest.get("Overall", "WAIT")


    def get_overall_signals_by_timeframe(self):
        """
        Gets the overall signal for each timeframe across all symbols.
        Returns a dictionary like: {"3 Minutes": "BUY", "5 Minutes": "SELL", ...}
        """
        from collections import Counter
        
        timeframes = ["3 Minutes", "5 Minutes","10 Minutes" ,"15 Minutes", "1 Hour", "4 Hour", "1 Day"]
        timeframe_signals = {tf: [] for tf in timeframes}
        
        for symbol in self.ticker_list:
            symbol_signals = self.get_latest_overall_by_timeframe(symbol)
            for tf in timeframes:
                if tf in symbol_signals:
                    timeframe_signals[tf].append(symbol_signals[tf])
        
        overall_signals = {}
        for tf in timeframes:
            signals = timeframe_signals[tf]
            if signals:
                signal_counts = Counter(signals)
                majority_signal = signal_counts.most_common(1)[0][0]
                overall_signals[tf] = majority_signal
            else:
                overall_signals[tf] = "NEUTRAL"
        
        return overall_signals

    def get_button_color_for_signal(self, signal):
        """
        Returns the appropriate color for a button based on the signal.
        """
        if signal == "BUY":
            colors = "#10b981", "white", "#059669" 
            return colors
        elif signal == "SELL":
            colors = "#ef4444", "white", "#dc2626"
            return colors
        else:  # NEUTRAL
            colors = "#e5e7eb", "#030213", "#d1d5db"
            return colors

    def update_button_colors(self):
        """
        Updates all timeframe button colors based on the current symbol's indicator data from CSV.
        """
        if not hasattr(self, 'timeframe_buttons'):
            print("🔍 No timeframe_buttons found")
            return
            
        current_symbol = self.ticker_var.get().strip().upper() if hasattr(self, 'ticker_var') else "AAPL"
        print(f"Updating button colors for symbol: {current_symbol}")
        
        symbol_signals = self.get_latest_overall_by_timeframe(current_symbol)
        current_selection = self.selected_timeframe.get()
        print(f"Symbol signals: {symbol_signals}")
        print(f"Current selection: {current_selection}")
        
        non_neutral_count = sum(1 for signal in symbol_signals.values() if signal != "NEUTRAL")
        print(f"Found {non_neutral_count} non-NEUTRAL signals out of {len(symbol_signals)} total")
        
        if not symbol_signals or non_neutral_count == 0:
            print("No signals or all NEUTRAL - retrying after 3 seconds...")
            self.root.after(3000, self.update_button_colors)
            return
        
        for name, btn in self.timeframe_buttons.items():
            signal = symbol_signals.get(name, "NEUTRAL")
            fg_color, text_color, hover_color = self.get_button_color_for_signal(signal)
            
            btn.configure(fg_color=fg_color, text_color=text_color, hover_color=hover_color)
            
            if name == current_selection:
                btn.configure(border_width=2, border_color="#3b82f6") 
            else:
                btn.configure(border_width=0) 

    # Creates the watchlist widget for the GUI.
    def create_watchlist_section(self, parent):
        self.watchlist_frame = parent 
        top_row = ctk.CTkFrame(parent, fg_color="white")
        top_row.pack(fill="x", padx=10, pady=(10, 0))

        title = ctk.CTkLabel(top_row, text="Watchlist", font=("Arial", 14, "bold"), text_color="#030213")
        title.pack(side="left")

        save_button = ctk.CTkButton(top_row, text="💾", width=30, height=28, command=self.handle_save_and_refresh)
        save_button.pack(side="right", padx=(0, 10))

        update_capital_button = ctk.CTkButton(
            top_row, 
            text="🔄", 
            width=30, 
            height=28, 
            command=self.handle_capital_update,
            fg_color="#3b82f6",
            hover_color="#2563eb"
        )
        update_capital_button.pack(side="right", padx=(0, 10))

        self.capital_var = tk.StringVar(value=str(self.capital))
        self.capital_entry = ctk.CTkEntry(top_row, textvariable=self.capital_var, width=100, font=("Arial", 14))
        self.capital_entry.pack(side="right", padx=(0, 5))
        
        self.capital_var.trace_add("write", self.on_capital_entry_change)

        capital_label = ctk.CTkLabel(top_row, text="Total Capital:", font=("Arial", 14), text_color="#030213")
        capital_label.pack(side="right", padx=(10, 5))
        table_frame = ctk.CTkFrame(parent, fg_color="white")
        table_frame.pack(fill="both", expand=True, padx=10, pady=10)

        headers = ["Symbol", "Price", "Change", "% Change", "Max Qty", "Signal 3m", "Action"]
        for idx, header in enumerate(headers):
            label = ctk.CTkLabel(table_frame, text=header, font=("Arial", 12, "bold"), text_color="#030213")
            label.grid(row=0, column=idx, padx=5, pady=5, sticky="ew")

        # Set column weights
        table_frame.grid_columnconfigure(0, weight=1) 
        table_frame.grid_columnconfigure(1, weight=1)
        table_frame.grid_columnconfigure(2, weight=1) 
        table_frame.grid_columnconfigure(3, weight=1) 
        table_frame.grid_columnconfigure(4, weight=1) 
        table_frame.grid_columnconfigure(5, weight=1)

        for row, symbol in enumerate(self.ticker_list, start=1):
            try:
                symbol_upper = symbol.upper()
            
                if symbol_upper in ["NQ", "MNQ", "ES"]:
                    contract_month = get_futures_contract_month()
                    if symbol_upper == "ES":
                        contract = ib_insync.Future(
                            symbol="ES", 
                            exchange="CME", 
                            currency="USD", 
                            lastTradeDateOrContractMonth=contract_month
                        )
                    else: 
                        contract = ib_insync.Future(
                            symbol=symbol_upper, 
                            exchange="CME", 
                            currency="USD", 
                            lastTradeDateOrContractMonth=contract_month
                        )
                    print(f"📈 Created futures contract for {symbol_upper} with month {contract_month}")
                else:
                    contract = ib_insync.Stock(symbol, 'SMART', 'USD', primaryExchange='NASDAQ')
                
                
                
                self.ib.qualifyContracts(contract)
                bars_daily = self.ib.reqHistoricalData(
                    contract, '', '2 D', '1 day', 'TRADES', useRTH=False, formatDate=1
                )
                df_daily = util.df(bars_daily)

                bars_hourly = self.ib.reqHistoricalData(
                    contract, '', '2 D', '1 hour', 'TRADES', useRTH=False, formatDate=1
                )
                df_hourly = util.df(bars_hourly)

                if not df_daily.empty:
                    price = df_daily['close'].iloc[-1]
                else:
                    price = 0.0

                match_16 = df_hourly[df_hourly['date'].astype(str).str.contains("16:00:00")]
                if not match_16.empty:
                    open_16 = match_16['open'].iloc[-1]
                else:
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    today_bars = df_hourly[df_hourly['date'].astype(str).str.contains(today_str)]
                    open_16 = today_bars['open'].iloc[0] if not today_bars.empty else None


                if price and open_16:
                    weekday = datetime.now().weekday()
                    if weekday in [5, 6]:
                        if should_allow_weekend_change(symbol):
                            change = price - open_16
                            percent = (change / open_16) * 100
                        else:
                            change = percent = 0.0
                    else:
                        change = price - open_16
                        percent = (change / open_16) * 100
                else:
                    change = percent = 0.0

            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                price = change = percent = 0.0

            try:
                current_capital = float(self.capital_var.get())
            except:
                current_capital = 10000.0



            max_qty = int(current_capital // price) if price > 0 else 0
            signal = self.get_3min_signal_for_symbol(symbol)

            signal_color = {
                "BUY": "#10b981",
                "SELL": "#ef4444",
                "NEUTRAL": "#f59e0b",
                "WAIT": "#f59e0b",
            }.get(signal, "#6b7280")


            change_color = "#10b981" if change > 0 else "#ef4444"
            percent_color = change_color
            if change == 0:
                change_color = percent_color = "#030213"
            change_icon = "📈" if change > 0 else ("📉" if change < 0 else "⏺️")

            ctk.CTkLabel(table_frame, text=symbol, font=("Arial", 12)).grid(row=row, column=0, padx=5)
            ctk.CTkLabel(table_frame, text=f"${price:.2f}", font=("Arial", 12)).grid(row=row, column=1, padx=5)
            ctk.CTkLabel(table_frame, text=f"{change_icon} ${abs(change):.2f}", font=("Arial", 12), text_color=change_color).grid(row=row, column=2, padx=5)
            ctk.CTkLabel(table_frame, text=f"{percent:+.2f}%", font=("Arial", 12), text_color=percent_color).grid(row=row, column=3, padx=5)
            ctk.CTkLabel(table_frame, text=str(max_qty), font=("Arial", 12)).grid(row=row, column=4, padx=5)
            signal_btn = ctk.CTkButton(table_frame, text=signal, fg_color=signal_color, text_color="white", width=40, height=24, font=("Arial", 11), corner_radius=6)
            signal_btn.grid(row=row, column=5, padx=5)
            delete_button = ctk.CTkButton(
                table_frame,
                text="🗑️",
                font=("Arial", 12),
                width=24,
                height=24,
                fg_color="#e11d48",
                hover_color="#be123c",
                text_color="white",
                corner_radius=6,
                command=lambda sym=symbol: self.delete_symbol_from_watchlist(sym)
            )
            delete_button.grid(row=row, column=6, padx=5, pady=5, sticky="nsew")
            if (symbol == self.ticker_var.get().strip().upper()):
                self.log_all_indicators_to_csv(symbol)
            
        self.schedule_watchlist_update()

    # Method to delete a symbol from the watchlist
    def delete_symbol_from_watchlist(self, symbol):
        if symbol in self.ticker_list:
            self.ticker_list.remove(symbol)
            with open(self.csv_path, "w") as f:
                for ticker in self.ticker_list:
                    f.write(ticker + "\n")

            # Refresh GUI
            self.handle_save_and_refresh()
            
    # Refreshes the watchlist section itself
    def refresh_watchlist_table(self):
        for widget in self.watchlist_frame.winfo_children():
            widget.destroy()
        self.create_watchlist_section(self.watchlist_frame)


    # Saves all of the indicators to the CSV and then resets the GUI and renders those elements. 
    def handle_save_and_refresh(self):
        try:
            new_capital = float(self.capital_var.get())
            self.capital = new_capital
            self.save_capital_to_file()

            selected_symbol = self.ticker_var.get().strip().upper()

            print(f"Saving indicator data to CSV for selected symbol: {selected_symbol}")
            try:
                self.log_all_indicators_to_csv(selected_symbol)
            except Exception as e:
                print(f"Error saving indicators for {selected_symbol}: {e}")

            print("Data saved successfully!")

            messagebox.showinfo("Success", f"Data saved for {selected_symbol} and GUI restarted successfully!")

            self.reset_gui()

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number.")
        except Exception as e:
            print(f"Error in save and refresh: {e}")
            messagebox.showerror("Error", f"Failed to save and refresh: {e}")


    def handle_capital_update(self):
        """Handle capital updates by pulling from capital.txt and resetting GUI"""
        try:
            new_capital = self.load_capital_from_file()
            print(f"Capital updated from file: ${new_capital:,.2f}")
            self.capital = new_capital
            messagebox.showinfo("Capital Updated", f"Capital updated to ${new_capital:,.2f} from capital.txt\nGUI will restart...")
            self.reset_gui()
            
        except Exception as e:
            print(f"Error updating capital: {e}")
            messagebox.showerror("Error", f"Failed to update capital: {e}")

    def on_capital_entry_change(self, *args):
        """Handle manual changes to the capital entry field"""
        try:
            new_value = self.capital_var.get()
            if new_value: 
                new_capital = float(new_value)
                self.capital = new_capital
                print(f"Capital manually updated to: ${new_capital:,.2f}")
        except ValueError:
            pass
        except Exception as e:
            print(f"Error in capital entry change: {e}")

    
    # Refresh watchlist method 
    def refresh_watchlist(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.create_gui()

    # Schedule Watclist Updates on an interval.
    def schedule_watchlist_update(self):
        self.update_watchlist_data()
        self.root.after(50000, self.schedule_watchlist_update)
    
    # Update the watchlist data
    def update_watchlist_data(self):
        if not hasattr(self, 'watchlist_frame') or self.watchlist_frame is None:
            return

        table_frame = self.watchlist_frame.winfo_children()[-1]
        rows = table_frame.winfo_children()

        for i, symbol in enumerate(self.ticker_list):
            try:
                symbol_upper = symbol.upper()
            
                if symbol_upper in ["NQ", "MNQ", "ES"]:
                    contract_month = get_futures_contract_month()
                    if symbol_upper == "ES":
                        contract = ib_insync.Future(
                            symbol="ES", 
                            exchange="CME", 
                            currency="USD", 
                            lastTradeDateOrContractMonth=contract_month
                        )
                    else: 
                        contract = ib_insync.Future(
                            symbol=symbol_upper, 
                            exchange="CME", 
                            currency="USD", 
                            lastTradeDateOrContractMonth=contract_month
                        )
                    print(f"📈 Created futures contract for {symbol_upper} with month {contract_month}")
                else:
                    contract = ib_insync.Stock(symbol, 'SMART', 'USD', primaryExchange='NASDAQ')
                    print(f"5 Created stock contract for {symbol}")
                
                self.ib.qualifyContracts(contract)
                bars_hourly = self.ib.reqHistoricalData(contract, '', '2 D', '1 hour', 'TRADES', useRTH=False, formatDate=1)
                df_hourly = util.df(bars_hourly)

                price = df_hourly['close'].iloc[-1] if not df_hourly.empty else 0.0
                match_16 = df_hourly[df_hourly['date'].astype(str).str.contains("16:00:00")]
                open_16 = match_16['open'].iloc[-1] if not match_16.empty else None


                if price and open_16:
                    weekday = datetime.now().weekday()
                    if weekday in [5, 6]:
                        print(f"DEBUG: Watchlist Update - Symbol: '{symbol}', Weekday: {weekday}, Should allow: {should_allow_weekend_change(symbol)}")
                        if should_allow_weekend_change(symbol):
                            change = price - open_16
                            percent = (change / open_16) * 100
                        else:
                            change = percent = 0.0
                    else:
                        change = price - open_16
                        percent = (change / open_16) * 100
                else:
                    change = percent = 0.0

            except Exception as e:
                print(f"Error updating {symbol}: {e}")
                price = change = percent = 0.0

            try:
                current_capital = float(self.capital_var.get())
            except:
                current_capital = 10000.0

            max_qty = int(current_capital // price) if price > 0 else 0
            signal = self.get_3min_signal_for_symbol(symbol)

            signal_color = {
                "BUY": "#10b981",
                "SELL": "#ef4444",
                "NEUTRAL": "#f59e0b",
                "WAIT": "#f59e0b", 
            }.get(signal, "#6b7280")

            base_index = (i + 1) * 7

            if base_index + 6 >= len(rows):
                continue

            rows[base_index + 1].configure(text=f"${price:.2f}")
            icon = "📈" if change > 0 else "📉" if change < 0 else "⏺️"
            change_color = "#10b981" if change > 0 else "#ef4444" if change < 0 else "#030213"
            rows[base_index + 2].configure(text=f"{icon} ${abs(change):.2f}", text_color=change_color)
            rows[base_index + 3].configure(text=f"{percent:+.2f}%", text_color=change_color) 
            rows[base_index + 4].configure(text=str(max_qty)) 
            rows[base_index + 5].configure(text=signal, fg_color=signal_color) 

    # If there is new data update the price data.
    def update_price_data(self):
        if self.after_job is not None:
            self.root.after_cancel(self.after_job)
        
        bars = self.ib.reqHistoricalData(
            self.contract,
            '',
            '2 D',
            '1 day',
            'TRADES',
            useRTH=False,
            formatDate=1
        )
        df = util.df(bars)

        bars4h = self.ib.reqHistoricalData(
            self.contract,
            '',
            '2 D',
            '1 hour',
            'TRADES',
            useRTH=False,
            formatDate=1
        )
        df4h = util.df(bars4h)

        current_price_val = df['close'].iloc[-1]
        current_price = f"${current_price_val:.2f}" if current_price_val is not None else "N/A"
        day_high = f"${df['high'].iloc[-1]:.2f}"
        day_low = f"${df['low'].iloc[-1]:.2f}"

        match_16 = df4h[df4h['date'].astype(str).str.contains("16:00:00")]
        if not match_16.empty:
            open_16_val = match_16['open'].iloc[-1]
        else:
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_bars = df4h[df4h['date'].astype(str).str.contains(today_str)]
            open_16_val = today_bars['open'].iloc[0] if not today_bars.empty else None

        if current_price_val is not None and open_16_val is not None:
            today = datetime.now().weekday()
            if today in [5, 6]:
                current_symbol = self.ticker_var.get().strip().upper()
                if should_allow_weekend_change(current_symbol):
                    change_val = current_price_val - open_16_val
                    percent_val = (change_val / open_16_val) * 100
                else:
                    change_val = 0.0
                    percent_val = 0.0
            else:
                change_val = current_price_val - open_16_val
                percent_val = (change_val / open_16_val) * 100
            if change_val == 0:
                daily_change_text = "$0.00"
                percent_change_text = "0.00%"
                change_icon = ""
                change_color = "#030213"
            else:
                daily_change_text = f"{'+' if change_val > 0 else ''}${change_val:.2f}"
                percent_change_text = f"{'+' if percent_val > 0 else ''}{percent_val:.2f}%"
                change_icon = "📈" if change_val > 0 else "📉"
                change_color = "#10b981" if change_val > 0 else "#ef4444"

        else:
            daily_change_text = "N/A"
            percent_change_text = "N/A"
            change_icon = ""
            change_color = "#6b7280"

        if not hasattr(self, 'stat_widgets') or "price" not in self.stat_widgets:
            self.stat_widgets = {} 


            def create_stat(parent, label, key):
                frame = ctk.CTkFrame(parent, fg_color="white")
                frame.pack(side="left", expand=True, fill="both", padx=10)

                title = ctk.CTkLabel(frame, text=label, font=("Arial", 13, "bold"), text_color="#6b7280")
                title.pack()
                value = ctk.CTkLabel(frame, text="--", font=("Arial", 18, "bold"))
                value.pack()

                self.stat_widgets[key] = value

            create_stat(self.stats_frame, "Current Price", "price")
            create_stat(self.stats_frame, "Daily Change", "daily")
            create_stat(self.stats_frame, "% Change", "percent")
            create_stat(self.stats_frame, "Day High", "high")
            create_stat(self.stats_frame, "Day Low", "low")

        self.stat_widgets["price"].configure(text=current_price, text_color="#030213")
        self.stat_widgets["daily"].configure(text=f"{change_icon} {daily_change_text}", text_color=change_color)
        self.stat_widgets["percent"].configure(text=percent_change_text, text_color=change_color)
        self.stat_widgets["high"].configure(text=day_high, text_color="#10b981")
        self.stat_widgets["low"].configure(text=day_low, text_color="#ef4444")

        self.root.after(3000, self.update_price_data)

    # Keeps the live clock inside of the top section of the GUI and the CPU metrics.
    def update_clock(self):
        current_time = datetime.now().strftime("%H:%M:%S")
        self.clock_label.configure(text=current_time)
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            
            resources_text = f"CPU: {cpu_percent:.1f}% | RAM: {memory_mb:.0f}MB"
            self.resources_label.configure(text=resources_text)
        except Exception as e:
            print(f"Error updating resources: {e}")
        
        self.root.after(1000, self.update_clock)
        
    # If the ticker changes update all other information to relate to that. 
    def on_ticker_change(self, *args):
        symbol = self.ticker_var.get().strip().upper()
        
        if not symbol:
            print("Empty symbol selected — ignoring.")
            return

        print(f"Ticker changed to: {symbol}")

        try:
            
            symbol_upper = symbol.upper()
            
            if symbol_upper in ["NQ", "MNQ", "ES"]:
                contract_month = get_futures_contract_month()
                if symbol_upper == "ES":
                    self.contract = ib_insync.Future(
                        symbol="ES", 
                        exchange="CME", 
                        currency="USD", 
                        lastTradeDateOrContractMonth=contract_month
                    )
                else: 
                    self.contract = ib_insync.Future(
                        symbol=symbol_upper, 
                        exchange="CME", 
                        currency="USD", 
                        lastTradeDateOrContractMonth=contract_month
                    )
                print(f"Created futures contract for {symbol_upper} with month {contract_month}")
            else:
                self.contract = ib_insync.Stock(symbol, 'SMART', 'USD', primaryExchange='NASDAQ')
                print(f"Created stock contract for {symbol}")
                
            
            self.ib.qualifyContracts(self.contract)

            if hasattr(self, "price_title_label"):
                self.price_title_label.configure(text=f"{symbol} - Price Information")

            self.update_price_data()

            if hasattr(self, "current_indicator_tf") and hasattr(self, "indicator_container"):
                print(f"🔄 Refreshing indicators for new ticker: {symbol} ({self.current_indicator_tf})")
                
                if FORCE_DATA_REFRESH:
                    print(f"🔄 Force refreshing data for new ticker: {symbol}")
                    self.root.after(0, lambda: self._refresh_data_for_symbol(symbol, self.current_indicator_tf))
                    
                    self.log_all_indicators_to_csv(symbol=self.current_indicator_tf)
                self.root.after(2000, lambda: self.render_indicators(self.current_indicator_tf, self.indicator_container))
                
            self.root.after(3000, self.update_button_colors)

        except Exception as e:
            print(f"Failed to update price data for symbol '{symbol}': {e}")
            messagebox.showerror("Error", f"Could not load data for symbol: {symbol}")

    # Runs the application.
    def run(self):
        try:
            self.root.mainloop()
        finally:
            self.cleanup()
    
    # Helps close the application more quickly.
    def cleanup(self):
        """Clean up resources when application closes"""
        
        self.memory_monitor_active = False
        
        if hasattr(self, 'indicator_executor'):
            self.indicator_executor.shutdown(wait=True)
        
        with cache_lock:
            indicator_cache.clear()
        
        if hasattr(self, 'active_tasks'):
            for task in self.active_tasks:
                if not task.done():
                    task.cancel()


if __name__ == "__main__":
    app = BasicTraderGUI()
    app.run()
