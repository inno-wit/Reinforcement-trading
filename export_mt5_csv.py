"""Export M1 OHLCV history from a running MT5 terminal into the exact CSV
format data_loader.load_mt_ohlcv_csv() expects: a single 'Time (EET)' column
formatted as YYYY.MM.DD HH:MM:SS, plus Open/High/Low/Close/Volume.

Requires the MT5 terminal to already be installed and logged in
(same account as Wit-Hedge-fund's .env on this box). Run with the repo venv:

    .venv\\Scripts\\python.exe export_mt5_csv.py --symbol XAUUSD --start 2020-01-01

Pulls as much history as the broker's server actually retains for that
symbol/timeframe -- MT5 silently returns fewer bars than requested if the
server doesn't have them, it does not error. Check the printed row count
and date range against what you expect.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

# copy_rates_range chokes ("Invalid params") on ranges much above ~30k M1 bars
# even when the terminal has the data cached -- observed empirically against
# MetaQuotes-Demo (27k/1mo succeeded, 150k/3.5mo failed). Chunk requests
# instead of relying on the single-call limit.
CHUNK_DAYS = 25


def _fetch_chunked(symbol: str, start: datetime, end: datetime, timeframe) -> pd.DataFrame:
    frames = []
    cursor = start
    step = timedelta(days=CHUNK_DAYS)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        rates = mt5.copy_rates_range(symbol, timeframe, cursor, chunk_end)
        if rates is not None and len(rates) > 0:
            frames.append(pd.DataFrame(rates))
            print(f"  {cursor.date()} -> {chunk_end.date()}: {len(rates)} rows")
        else:
            print(f"  {cursor.date()} -> {chunk_end.date()}: 0 rows (err={mt5.last_error()})")
        cursor = chunk_end

    if not frames:
        raise RuntimeError(f"No rates returned for {symbol} across {start.date()}..{end.date()}: {mt5.last_error()}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="time", keep="last").sort_values("time")
    return df


def export(symbol: str, start: datetime, end: datetime, out_path: Path, timeframe=mt5.TIMEFRAME_M1) -> None:
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

    try:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select({symbol!r}) failed: {mt5.last_error()}")

        df = _fetch_chunked(symbol, start, end, timeframe)
        # mt5 'time' is epoch seconds in the terminal/server timezone already
        # (matches source_tz assumption in config.py -- verify against
        # ProjectConfig.source_tz for your broker before trusting long runs).
        ts = pd.to_datetime(df["time"], unit="s")

        out = pd.DataFrame({
            "Time (EET)": ts.dt.strftime("%Y.%m.%d %H:%M:%S"),
            "Open": df["open"],
            "High": df["high"],
            "Low": df["low"],
            "Close": df["close"],
            "Volume": df["tick_volume"],
        })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)

        print(f"Wrote {len(out)} rows to {out_path}")
        print(f"Range: {ts.min()} -> {ts.max()}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="XAUUSD")
    p.add_argument("--start", required=True, help="YYYY-MM-DD, earliest date to request")
    p.add_argument("--end", default=None, help="YYYY-MM-DD, defaults to now")
    p.add_argument("--out", default=None, help="defaults to data/<symbol>_1 Min_MT5_<start>_<end>.csv")
    args = p.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now(timezone.utc).replace(tzinfo=None)

    out = Path(args.out) if args.out else Path("data") / f"{args.symbol}_1 Min_MT5_{start:%Y.%m.%d}_{end:%Y.%m.%d}.csv"
    export(args.symbol, start, end, out)
