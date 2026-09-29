"""Binance public spot data (https://data.binance.vision): aggregated trades -> 1-minute bars.

Daily `aggTrades` files are header-less CSVs with the columns

    agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time,
    is_buyer_maker, is_best_match

`transact_time` is in milliseconds before 2025 and in microseconds from 2025 on; both are
accepted. `is_buyer_maker` true means the buyer posted the resting order, so the *taker* sold.

Everything here is a pure function of its inputs (no network access; downloading is
`experiments/fetch_binance.py`).
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AGG_TRADE_COLUMNS = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
    "is_best_match",
)
BAR_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "vwap",
    "trade_count",
    "buy_volume",
    "signed_volume",
    "half_spread_bps",
    "spread_pairs",
)
# Timestamps above this are microseconds (1e14 ms would be the year 5138).
_MICROSECOND_THRESHOLD = 10**14
SPREAD_MAX_GAP_MS = 100.0
SPREAD_MIN_PAIRS = 3


def parse_checksum(text: str) -> str:
    """SHA-256 hex digest from a Binance `.CHECKSUM` file (`<digest>  <file name>`)."""
    parts = text.split()
    if not parts or len(parts[0]) != 64:
        raise ValueError(f"not a SHA-256 checksum line: {text[:80]!r}")
    return parts[0].lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_agg_trades(source: str | Path | io.BytesIO) -> pd.DataFrame:
    """Aggregated trades as a DataFrame sorted by time.

    `source` is a CSV path, a `.zip` path containing one CSV, or a file object. A header
    row, if present, is skipped. The `time` column is a timezone-naive UTC timestamp.
    """
    if isinstance(source, (str, Path)) and str(source).endswith(".zip"):
        with zipfile.ZipFile(source) as archive:
            names = [n for n in archive.namelist() if n.endswith(".csv")]
            if len(names) != 1:
                raise ValueError(f"expected one CSV in {source}, found {names}")
            with archive.open(names[0]) as fh:
                return read_agg_trades(io.BytesIO(fh.read()))
    df = pd.read_csv(source, header=None, names=list(AGG_TRADE_COLUMNS), dtype=str)
    if not df.empty and not df.iloc[0]["agg_trade_id"].isdigit():
        df = df.iloc[1:]
    out = pd.DataFrame(
        {
            "agg_trade_id": df["agg_trade_id"].astype(np.int64).to_numpy(),
            "price": df["price"].astype(np.float64).to_numpy(),
            "quantity": df["quantity"].astype(np.float64).to_numpy(),
            "first_trade_id": df["first_trade_id"].astype(np.int64).to_numpy(),
            "last_trade_id": df["last_trade_id"].astype(np.int64).to_numpy(),
            "is_buyer_maker": df["is_buyer_maker"].str.strip().str.lower().eq("true").to_numpy(),
        }
    )
    raw_time = df["transact_time"].astype(np.int64).to_numpy()
    per_second = 10**6 if raw_time.size and raw_time.max() > _MICROSECOND_THRESHOLD else 10**3
    out["time"] = (raw_time * (10**9 // per_second)).astype("datetime64[ns]")
    return out.sort_values(["time", "agg_trade_id"], kind="stable").reset_index(drop=True)


def opposite_side_gaps(trades: pd.DataFrame, max_gap_ms: float = SPREAD_MAX_GAP_MS) -> pd.DataFrame:
    """Consecutive trade pairs of opposite taker side at most `max_gap_ms` apart.

    Returns the later trade's time and the absolute price difference in bps of the mean
    price of the pair: a trade-sign proxy for the quoted spread (a buyer-initiated trade
    prints at the ask, a seller-initiated one at the bid).
    """
    if len(trades) < 2:
        return pd.DataFrame({"time": pd.Series([], dtype="datetime64[ns]"), "gap_bps": []})
    price = trades["price"].to_numpy()
    side = trades["is_buyer_maker"].to_numpy()
    dt_ms = np.diff(trades["time"].to_numpy()).astype("timedelta64[us]").astype(np.float64) / 1e3
    mask = (side[1:] != side[:-1]) & (dt_ms <= max_gap_ms)
    mean_price = (price[1:] + price[:-1]) / 2
    gap = np.abs(price[1:] - price[:-1]) / mean_price * 1e4
    return pd.DataFrame({"time": trades["time"].to_numpy()[1:][mask], "gap_bps": gap[mask]})


def minute_bars(
    trades: pd.DataFrame,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    *,
    max_gap_ms: float = SPREAD_MAX_GAP_MS,
    min_pairs: int = SPREAD_MIN_PAIRS,
) -> pd.DataFrame:
    """1-minute bars on the full grid [start, end) (default: the trades' own minutes).

    Columns (`BAR_COLUMNS`): OHLC of trade prices; base `volume` and `quote_volume`;
    `vwap` = quote / base volume; `trade_count` (underlying trades, summed over aggregated
    trades); `buy_volume` (taker buys); `signed_volume` = taker buy - taker sell;
    `half_spread_bps` = half the median opposite-side price gap of the bar
    (`opposite_side_gaps`), NaN with fewer than `min_pairs` pairs; `spread_pairs`.
    A minute without trades has zero volume and carries the previous close in OHLC and
    VWAP (NaN before the first trade).
    """
    minute = trades["time"].dt.floor("min")
    if start is None:
        start = minute.min() if len(trades) else pd.Timestamp(0)
    if end is None:
        end = (minute.max() + pd.Timedelta(minutes=1)) if len(trades) else start
    grid = pd.date_range(start, end, freq="min", inclusive="left")

    qty = trades["quantity"]
    taker_buy = ~trades["is_buyer_maker"]
    frame = pd.DataFrame(
        {
            "minute": minute,
            "price": trades["price"],
            "quantity": qty,
            "quote": trades["price"] * qty,
            "count": trades["last_trade_id"] - trades["first_trade_id"] + 1,
            "buy": qty.where(taker_buy, 0.0),
        }
    )
    grouped = frame.groupby("minute", sort=True)
    bars = pd.DataFrame(
        {
            "open": grouped["price"].first(),
            "high": grouped["price"].max(),
            "low": grouped["price"].min(),
            "close": grouped["price"].last(),
            "volume": grouped["quantity"].sum(),
            "quote_volume": grouped["quote"].sum(),
            "trade_count": grouped["count"].sum(),
            "buy_volume": grouped["buy"].sum(),
        }
    ).reindex(grid)

    for col in ("volume", "quote_volume", "trade_count", "buy_volume"):
        bars[col] = bars[col].fillna(0.0)
    bars["trade_count"] = bars["trade_count"].astype(np.int64)
    traded = bars["volume"] > 0
    bars["vwap"] = (bars["quote_volume"] / bars["volume"]).where(traded)
    bars["close"] = bars["close"].ffill()
    for col in ("open", "high", "low", "vwap"):
        bars[col] = bars[col].fillna(bars["close"])
    bars["signed_volume"] = 2 * bars["buy_volume"] - bars["volume"]

    gaps = opposite_side_gaps(trades, max_gap_ms)
    by_minute = gaps.groupby(gaps["time"].dt.floor("min"))["gap_bps"]
    pairs = by_minute.size().reindex(grid, fill_value=0)
    half = (by_minute.median() / 2).reindex(grid)
    bars["half_spread_bps"] = half.where(pairs >= min_pairs)
    bars["spread_pairs"] = pairs.astype(np.int64)

    bars.index.name = None
    bars.insert(0, "timestamp", grid)
    return bars.reset_index(drop=True)[list(BAR_COLUMNS)]


def day_bars(zip_path: str | Path, day: str | pd.Timestamp) -> pd.DataFrame:
    """1,440 one-minute bars of UTC day `day` from its daily aggTrades zip."""
    start = pd.Timestamp(day).normalize()
    trades = read_agg_trades(Path(zip_path))
    return minute_bars(trades, start, start + pd.Timedelta(days=1))
