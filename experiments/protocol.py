"""Constants of the pre-registered Phase 7 protocol (docs/PROTOCOL.md).

Changing any value here is a protocol deviation and must be recorded in docs/PROTOCOL.md.
"""

from pathlib import Path

import pandas as pd

SYMBOLS = ("BTCUSDT", "LINKUSDT")
DEV_START, DEV_END = "2026-07-22", "2026-08-06"
TEST_START, TEST_END = "2026-08-07", "2026-08-18"

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "binance" / "aggTrades"
BARS_DIR = DATA_DIR / "binance" / "bars"

START_HOURS = (0, 4, 8, 12, 16, 20)
BUY_HOURS = frozenset({0, 8, 16})
SIZES_PCT_ADV = (0.1, 0.5)
HORIZONS_MIN = (60, 240)
PARTICIPATION_CAP = 0.25
BUCKET_MINUTES = 15
MIN_LOTS_SMALLEST_ORDER = 10_000
IMPACT_MULTIPLES = (0.5, 1.0, 2.0)


def days(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


DEV_DAYS = tuple(days(DEV_START, DEV_END))
TEST_DAYS = tuple(days(TEST_START, TEST_END))
ALL_DAYS = DEV_DAYS + TEST_DAYS


def split_days(split: str) -> tuple[str, ...]:
    if split == "dev":
        return DEV_DAYS
    if split == "test":
        return TEST_DAYS
    raise ValueError(f"unknown split {split!r}; expected 'dev' or 'test'")


def zip_name(symbol: str, day: str) -> str:
    return f"{symbol}-aggTrades-{day}.zip"


def raw_path(symbol: str, day: str, root: Path = RAW_DIR) -> Path:
    return root / symbol / zip_name(symbol, day)


def bars_path(symbol: str, day: str, root: Path = BARS_DIR) -> Path:
    return root / symbol / f"{symbol}-{day}.csv.gz"


def load_bars(symbol: str, day_list: tuple[str, ...] | list[str], root: Path = BARS_DIR):
    """Concatenated 1-minute bars of `day_list` (built by experiments.fetch_binance)."""
    frames = []
    for day in day_list:
        path = bars_path(symbol, day, root)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `make data`")
        frames.append(pd.read_csv(path, parse_dates=["timestamp"]))
    return pd.concat(frames, ignore_index=True)
