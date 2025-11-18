from ib_insync import *
import pandas as pd
import time
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from zoneinfo import ZoneInfo
from datetime import time as dt_time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import pytz

# Port and connection information.
clientId = 1
host = "127.0.0.1"
port = 7497
# account = removed

# sender_email = removed email
# sender_password = ENV Variable, Still Removed
# smtp_port = removed port
# smtp_server = removed server
# receiver_email_list = removed list

# Paths & run settings
# directory = rremoved file directory
os.makedirs(directory, exist_ok=True)

# Expiriy Date for NQ and Volume Threshold for Data Retrival
EXPIRY = '20251226
GC_FUTURE_EXPIRY = '202512'
NQ_ES_FUTURE_EXPIRY = '202512'
VOLUME_THRESHOLD = 1000

# Resource Management
PARALLEL = True
CPU_FRACTION = 0.20 
MAX_WORKERS_CAP = 3 
BASE_CLIENT_ID = 2000 

# Database of all tickers option data will be analyzed for
stocks = [
    'NQ', 'ES', 'GC', 'TQQQ', 'SPXL', 'SOXL', 'QQQ', 'SPY',
    'AAPL', 'AMZN', 'AMD', 'GOOGL', 'MSFT', 'META', 'NFLX', 'NVDA', 'CRWD', 'AVGO', 'NOW', 'SNOW',
    'TSLA', 'SHOP', 'AMAT', 'QCOM', 'KLAC', 'MPWR', 'MU', 'TSM', 'TXN', 'INTC', 'LMT', 'MSTR',
    'ORCL', 'SPOT', 'HIMS', 'ADBE', 'MDB', 'NVDU', 'NVDQ', 'TSLL', 'GGLL', 'GGLS', 'NFXL',
    'NFXS', 'AVL', 'AVS', 'PLTR', 'HOOD', 'UPS', 'LULU', 'MSFU','CRWL','NOWL','SHPU','HOOG','HIMZ','PLTU'
]
FOP_SYMBOLS = ['NQ', 'ES', 'GC']
STOCK_SYMBOLS = [s for s in stocks if s not in FOP_SYMBOLS]


# Method to check if it is inside of market hours. This is vital as option data is only available on TWS during open market hours.
def is_us_equity_rth_mt(now_utc=None) -> bool:
    """RTH in MT: 07:30–14:00, Mon–Fri."""
    now_utc = now_utc or datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
    now_mt = now_utc.astimezone(ZoneInfo("America/Edmonton"))
    if now_mt.weekday() >= 5:
        return False
    t = now_mt.time()
    return dt_time(7, 30) <= t <= dt_time(14, 0)

def get_previous_trading_day():
    prev_day = datetime.now() - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    return prev_day.strftime("%Y%m%d")

# This method is used to send a email to multiple email addresses using the information which is defined globally.
def send_multi(subject, body, paths):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = ', '.join(receiver_email_list)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        for p in paths:
            with open(p, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(p))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(p)}"'
            msg.attach(part)
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email_list, msg.as_string())
        server.quit()
        print("Email sent.")
    except Exception as e:
        print(f"Email failed: {e}")

# Gets the greek information for option derivative contracts
def _extract_greeks(t: Ticker):
    """Pick bid/ask mid; fallback to whichever exists; then modelGreeks."""
    def _pick(field):
        bid = getattr(getattr(t, 'bidGreeks', None), field, None)
        ask = getattr(getattr(t, 'askGreeks', None), field, None)
        mdl = getattr(getattr(t, 'modelGreeks', None), field, None)
        if (bid is not None) and (ask is not None):
            return (bid + ask) / 2.0
        return bid if (bid is not None) else (ask if (ask is not None) else mdl)

    und_bid = getattr(getattr(t, 'bidGreeks', None), 'undPrice', None)
    und_ask = getattr(getattr(t, 'askGreeks', None), 'undPrice', None)
    und_mdl = getattr(getattr(t, 'modelGreeks', None), 'undPrice', None)
    if (und_bid is not None) and (und_ask is not None):
        und_price = (und_bid + und_ask) / 2.0
    else:
        und_price = und_bid if (und_bid is not None) else (und_ask if (und_ask is not None) else und_mdl)

    return {
        'delta': _pick('delta'),
        'gamma': _pick('gamma'),
        'vega':  _pick('vega'),
        'theta': _pick('theta'),
        'iv':    _pick('impliedVol'),
        'undPrice': und_price
    }


# A single interation of the program for a specific symbol. 
def check_high_volume(ib: IB, symbol: str, expiry: str, right: str = 'C'):
    """
    Returns a list[dict] with rows for this symbol/right.
    Handles FUT options (NQ/ES on CME; GC on COMEX) and stock/ETF options on SMART.
    """
    rows = []

    if symbol in FOP_SYMBOLS:
        print(f"\n{symbol} {right} (FOP) near LTP — no equity volume threshold (target expiry {expiry})...")
    else:
        print(f"\n{symbol} {right} options near LTP with volume ≥ {VOLUME_THRESHOLD} (expiry {expiry})...")

    fop_req_expiry = None
    if symbol == 'NQ':
        fop_req_expiry = NQ_ES_FUTURE_EXPIRY
    elif symbol == 'ES':
        fop_req_expiry = NQ_ES_FUTURE_EXPIRY
    elif symbol == 'GC':
        fop_req_expiry = GC_FUTURE_EXPIRY

    # Attempts to make a contract depending on the type of equity or contract which has its data pulled
    try:
        if symbol == 'NQ':
            contract_list = ib.reqContractDetails(Future(symbol='NQ', exchange='CME', lastTradeDateOrContractMonth=fop_req_expiry))
            if not contract_list:
                print(f"No futures contracts found for {symbol}")
                return rows
            underlying = contract_list[0].contract
            increment = 50
        elif symbol == 'ES':
            contract_list = ib.reqContractDetails(Future(symbol='ES', exchange='CME', lastTradeDateOrContractMonth=fop_req_expiry))
            if not contract_list:
                print(f"No futures contracts found for {symbol}")
                return rows
            underlying = contract_list[0].contract
            increment = 5
        elif symbol == 'GC':
            contract_list = ib.reqContractDetails(Future(symbol='GC', exchange='COMEX', lastTradeDateOrContractMonth=fop_req_expiry))
            if not contract_list:
                print(f"No futures contracts found for {symbol}")
                return rows
            underlying = contract_list[0].contract
            increment = 10
        else:
            underlying = Stock(symbol, 'SMART', 'USD')

        ib.qualifyContracts(underlying)
        stock_ticker = ib.reqMktData(underlying, '', False, False)
        ib.sleep(2)

        ltp = stock_ticker.last
        if ltp is None or (isinstance(ltp, float) and (ltp != ltp)):
            ltp = stock_ticker.close
            if ltp is None or (isinstance(ltp, float) and (ltp != ltp)):
                bid = stock_ticker.bid
                ask = stock_ticker.ask
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    ltp = (bid + ask) / 2.0
                elif bid:
                    ltp = bid
                elif ask:
                    ltp = ask
        if not ltp or (isinstance(ltp, float) and (ltp != ltp)) or ltp == 0:
            print(f"Could not fetch valid LTP for {symbol}")
            ib.cancelMktData(stock_ticker)
            return rows

        # If a symbol a one of the futures inside of the list. 
        if symbol in FOP_SYMBOLS:
            base = round(ltp / increment) * increment
            strikes = [base + i * increment for i in range(-2, 3)]
            exch = 'CME' if symbol in ['NQ', 'ES'] else 'COMEX'

            sec_def = ib.reqSecDefOptParams(underlying.symbol, exch, underlying.secType, underlying.conId)
            opt_expiries = sorted(sec_def[0].expirations) if sec_def else []
            print(f"Available expiries for {symbol} ({exch}): {opt_expiries}")

            valid_expiry = next((d for d in opt_expiries if d == fop_req_expiry), None)
            if not valid_expiry:
                valid_expiry = next((d for d in opt_expiries if d > fop_req_expiry), None)
            if not valid_expiry:
                print(f"No valid FOP expiry found for {symbol} in or after {fop_req_expiry}")
                ib.cancelMktData(stock_ticker)
                return rows

            contracts = [
                FuturesOption(
                    symbol=symbol,
                    lastTradeDateOrContractMonth=valid_expiry,
                    strike=s,
                    right=right,
                    exchange=exch
                ) for s in strikes
            ]
        else:
            chains = ib.reqSecDefOptParams(symbol, '', 'STK', underlying.conId)
            chain = next((c for c in chains if c.exchange == 'SMART'), None)
            if not chain:
                print(f"No option chain for {symbol}")
                ib.cancelMktData(stock_ticker)
                return rows

            expiries = sorted(set(chain.expirations)) if getattr(chain, 'expirations', None) else []
            strikes_all = sorted(set(chain.strikes)) if getattr(chain, 'strikes', None) else []
            if not expiries or not strikes_all:
                print(f"No expiries/strikes in chain for {symbol}")
                ib.cancelMktData(stock_ticker)
                return rows

            target = expiry
            if target in expiries:
                valid_expiry = target
            else:
                later = [d for d in expiries if d > target]
                earlier = [d for d in expiries if d < target]
                valid_expiry = later[0] if later else (earlier[-1] if earlier else expiries[-1])
                print(f"{symbol}: requested {target} not in chain; using {valid_expiry} instead.")

            # Choose strikes near the last traded price
            lower = sorted([s for s in strikes_all if s < ltp], reverse=True)[:10]
            upper = sorted([s for s in strikes_all if s >= ltp])[:10]
            selected = sorted(lower + upper)
            if not selected:
                print(f"No strikes near LTP for {symbol} @ {valid_expiry}")
                ib.cancelMktData(stock_ticker)
                return rows

            # Build option contracts with the validated expiry date
            contracts = [Option(symbol, valid_expiry, strike, right, 'SMART') for strike in selected]


        qualified = ib.qualifyContracts(*contracts)

        if symbol in FOP_SYMBOLS:
            tickers = [ib.reqMktData(c, '100,101,104,106', False, False) for c in qualified]
        else:
            tickers = [ib.reqMktData(c, '100,101,106', False, False) for c in qualified]

        ib.sleep(8) 

        tmp_rows = []
        
        for c, t in zip(qualified, tickers):
            vol_today = getattr(t, 'optionVolume', None)
            if vol_today is None:
                vol_today = getattr(t, 'volume', 0) 

            oi_val = getattr(t, 'openInterest', None)
            if oi_val is None:
                oi_val = 'N/A'

            # Futures options bypass threshold due to high cost of contracts, stocks need to clear a threshold for volume
            pass_threshold = (symbol in FOP_SYMBOLS) or (vol_today >= VOLUME_THRESHOLD)
            if not pass_threshold:
                continue
            # Add symbol and volume to export on the basis that it clears the volume threshold
            g = _extract_greeks(t)

            tmp_rows.append({
                'Symbol': symbol,
                'Right': right,
                'Strike': c.strike,
                'LTP': ltp,
                'Delta': g['delta'],
                'Gamma': g['gamma'],
                'Vega':  g['vega'],
                'Theta': g['theta'],
                'IV':    g['iv'],
                'UndPrice(model)': g['undPrice'],
                'VolumeToday': vol_today,
                'OpenInterest': oi_val,
                'Contract': c
            })

        #Previous day total volume
        prev_day = get_previous_trading_day()
        for r in tmp_rows:
            try:
                bars = ib.reqHistoricalData(
                    r['Contract'],
                    endDateTime=prev_day + " 23:59:59",
                    durationStr="1 D",
                    barSizeSetting="5 mins",
                    whatToShow="TRADES",
                    useRTH=False,
                    formatDate=1
                )
                if bars:
                    r['VolumePrevDay'] = sum(b.volume for b in bars if getattr(b, 'volume', 0))
                else:
                    r['VolumePrevDay'] = 0
            except Exception:
                r['VolumePrevDay'] = 'N/A'
            time.sleep(0.2)

        for r in tmp_rows:
            r.pop('Contract', None)

        rows.extend(tmp_rows)

        try:
            if stock_ticker:
                ib.cancelMktData(stock_ticker) 
        except Exception:
            pass

        try:
            for tkr in tickers:
                ib.cancelMktData(tkr)
        except Exception:
            pass
            
        return rows

    except Exception as e:
        print(f"Error processing {symbol} {right}: {e}")
        return rows

# Utilizes individual IB connections for individual symbols. 
def worker_scan(symbols_chunk, expiry, volume_threshold, host, port, client_id):
    """
    Each worker opens its own IB connection and scans its chunk.
    Returns list[dict].
    """
    rows = []
    ib = IB()
    try:
        ib.connect(host, port, client_id, timeout=15)
        ib.reqMarketDataType(1)
        for sym in symbols_chunk:
            rows.extend(check_high_volume(ib, sym, expiry, 'C')) # CALLS
            rows.extend(check_high_volume(ib, sym, expiry, 'P')) # PUTS
            time.sleep(0.3)
    except Exception as e:
        print(f"Worker {client_id} failure: {e}")
    finally:
        if ib.isConnected():
            ib.disconnect()
    return rows


def run_scan_serial(ib, symbols_to_run):
    results = []
    for symbol in symbols_to_run:
        results.extend(check_high_volume(ib, symbol, EXPIRY, right='C'))
        results.extend(check_high_volume(ib, symbol, EXPIRY, right='P'))
    return results

def run_scan_parallel(symbols_to_run):
    cpu_count = os.cpu_count() or 2
    want = max(1, math.floor(cpu_count * CPU_FRACTION))
    max_workers = min(want, MAX_WORKERS_CAP)
    if max_workers <= 1:
        ib_local = IB()
        try:
            ib_local.connect(host, port, clientId + 99, timeout=10)
            ib_local.reqMarketDataType(1)
            return run_scan_serial(ib_local, symbols_to_run)
        finally:
            if ib_local.isConnected():
                ib_local.disconnect()

    chunks = [[] for _ in range(max_workers)]
    for i, sym in enumerate(symbols_to_run):
        chunks[i % max_workers].append(sym)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for idx, chunk in enumerate(chunks):
            if not chunk:
                continue
            futs.append(
                ex.submit(
                    worker_scan,
                    chunk,
                    EXPIRY,
                    VOLUME_THRESHOLD,
                    host,
                    port,
                    BASE_CLIENT_ID + idx
                )
            )
        for f in as_completed(futs):
            try:
                results.extend(f.result())
            except Exception as e:
                print(f"Worker future error: {e}")
    return results

def analytics_and_email(all_results):
    if not all_results:
        return

    df_all = pd.DataFrame(all_results)

    df_all['VolumePrevDay'] = pd.to_numeric(df_all['VolumePrevDay'], errors='coerce').fillna(0)
    def _uf(r):
        if r['VolumePrevDay'] > 0:
            return r['VolumeToday'] / r['VolumePrevDay']
        return float('inf') if r['VolumeToday'] > 0 else 0
    df_all['UnusualFactor'] = df_all.apply(_uf, axis=1)

    pcr = (
        df_all.groupby(['Symbol', 'Right'])['VolumeToday']
              .sum()
              .unstack(fill_value=0)
    )
    for col in ['P', 'C']:
        if col not in pcr.columns:
            pcr[col] = 0
    pcr['PCR'] = pcr['P'] / pcr['C'].replace(0, float('inf'))
    pcr = pcr.reset_index()

    top10 = df_all.sort_values('VolumeToday', ascending=False).head(10)[
        ['Symbol', 'Right', 'Strike', 'LTP', 'VolumeToday', 'OpenInterest', 'Delta', 'IV', 'UnusualFactor']
    ]
    unusual = df_all[(df_all['UnusualFactor'].replace([float('inf')], 9999) > 3) & (df_all['VolumeToday'] >= 50)]
    unusual = unusual.sort_values('UnusualFactor', ascending=False).head(10)[
        ['Symbol', 'Right', 'Strike', 'LTP', 'VolumeToday', 'VolumePrevDay', 'UnusualFactor', 'OpenInterest', 'Delta', 'IV']
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(directory, f"Consolidated_Options_Report_{ts}.csv")
    top10_path = os.path.join(directory, f"Top10_{ts}.csv")
    unusual_path = os.path.join(directory, f"Unusual_{ts}.csv")
    pcr_path = os.path.join(directory, f"PCR_{ts}.csv")

    df_all.to_csv(full_path, index=False)
    top10.to_csv(top10_path, index=False)
    unusual.to_csv(unusual_path, index=False)
    pcr.to_csv(pcr_path, index=False)

    summary_lines = []
    if not top10.empty:
        summary_lines.append("Top 5 by Option Volume Today:")
        for _, r in top10.head(5).iterrows():
            uf = '∞' if r['UnusualFactor'] == float('inf') else round(r['UnusualFactor'], 1)
            dlt = None if pd.isna(r['Delta']) else round(r['Delta'], 3)
            ivv = None if pd.isna(r['IV']) else round(r['IV'], 3)
            summary_lines.append(f"  • {r['Symbol']} {r['Right']}{int(r['Strike'])}  Vol={int(r['VolumeToday'])}  OI={r['OpenInterest']}  Δ={dlt}  IV={ivv}  UF={uf}")

    if not unusual.empty:
        summary_lines.append("\nMost Unusual (VolToday / VolPrevDay):")
        for _, r in unusual.head(5).iterrows():
            uf = '∞' if r['UnusualFactor'] == float('inf') else round(r['UnusualFactor'], 1)
            summary_lines.append(f"  • {r['Symbol']} {r['Right']}{int(r['Strike'])}  UF={uf}  Vol={int(r['VolumeToday'])} vs {int(r['VolumePrevDay'])}")

    if not pcr.empty:
        summary_lines.append("\nPut/Call Ratio (by Symbol):")
        for _, r in pcr.sort_values(by='PCR', ascending=False).head(10).iterrows():
            summary_lines.append(f"  • {r['Symbol']}: PCR={round(r['PCR'],2)}  (P={int(r['P'])}, C={int(r['C'])})")

    subject = f"Consolidated Options Volume Report ({ts})"
    body = f"Attached: full report + summaries for equity expiry {EXPIRY}.\n\n" + "\n".join(summary_lines)

    send_multi(subject, body, [full_path, top10_path, unusual_path, pcr_path])


# Schdueler to consistely produce option data on this time each day for the week. 
time_to_run = ("12", "30") 
mountain_tz = pytz.timezone("America/Edmonton")

def _next_1230_mt(now_mt: datetime) -> datetime:
    """Return the next occurrence of 12:30 MT (today if still upcoming, else tomorrow)."""
    target = now_mt.replace(hour=int(time_to_run[0]), minute=int(time_to_run[1]),
                            second=0, microsecond=0)
    if now_mt >= target:
        target = (target + timedelta(days=1)).replace(second=0, microsecond=0)
    return target

def main_loop():
    while True:
        try:
            now_mt = datetime.now(mountain_tz)
            next_run = _next_1230_mt(now_mt)
            wait_secs = max(1, int((next_run - now_mt).total_seconds()))
            print(f"Next run scheduled at {next_run.strftime('%Y-%m-%d %H:%M %Z')}")

            # Live countdown timer
            for remaining in range(wait_secs, 0, -1):
                hrs, rem = divmod(remaining, 3600)
                mins, secs = divmod(rem, 60)
                print(f"\rSleeping {hrs:02d}h {mins:02d}m {secs:02d}s until next run...", end="", flush=True)
                time.sleep(1)
            print() 

            ib = IB()
            while True:
                try:
                    ib.connect(host, port, clientId, timeout=10)
                    ib.reqMarketDataType(1)
                    print("\n✅ Connected (LIVE).")
                    break
                except Exception as e:
                    print(f"Initial connection failed: {e}")
                    time.sleep(5)

            try:
                if is_us_equity_rth_mt():
                    symbols_to_run = STOCK_SYMBOLS
                    print("Market hours: scanning STOCK/ETF options (single daily run).")
                    if PARALLEL:
                        print("Parallel mode (up to 20% CPU).")
                        all_results = run_scan_parallel(symbols_to_run)
                    else:
                        all_results = run_scan_serial(ib, symbols_to_run)
                else:
                    symbols_to_run = FOP_SYMBOLS
                    print("Outside equity RTH: scanning futures options (NQ/ES/GC).")
                    all_results = run_scan_serial(ib, symbols_to_run)

                analytics_and_email(all_results)

            finally:
                if ib.isConnected():
                    ib.disconnect()
                    print("Disconnected.")

            print("Daily run complete. Sleeping for 24 hours...")

            for remaining in range(24 * 3600, 0, -1):
                hrs, rem = divmod(remaining, 3600)
                mins, secs = divmod(rem, 60)
                print(f"\rNext scan in {hrs:02d}h {mins:02d}m {secs:02d}s...", end="", flush=True)
                time.sleep(1)
            print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nUnexpected error in main loop: {e}")
            time.sleep(30)



if __name__ == "__main__":
    main_loop()
