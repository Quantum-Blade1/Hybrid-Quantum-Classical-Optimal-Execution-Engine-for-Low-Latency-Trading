"""Tick-data loading from CSV files with `timestamp, price[, volume]` columns."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_COLUMN_ALIASES = {
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "last": "price",
    "close": "price",
    "vol": "volume",
    "qty": "volume",
    "quantity": "volume",
}
_DEFAULT_TICK_VOLUME = 100


@dataclass(frozen=True)
class TickData:
    """Time-sorted tick series for one symbol."""

    symbol: str
    data: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp


class DataLoader:
    """Loads exchange tick files into a normalised `TickData`."""

    @staticmethod
    def load_csv(filepath: str | Path, symbol: str = "UNKNOWN") -> TickData:
        """Load a tick CSV; common column aliases (e.g. `close`, `qty`) are normalised."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df.rename(columns=_COLUMN_ALIASES)

        if "timestamp" not in df.columns or "price" not in df.columns:
            raise ValueError(
                f"CSV must contain 'timestamp' and 'price' columns. Found: {list(df.columns)}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        if "volume" not in df.columns:
            df["volume"] = _DEFAULT_TICK_VOLUME
        df = df.ffill().bfill()

        return TickData(
            symbol=symbol,
            data=df,
            start_time=df["timestamp"].iloc[0],
            end_time=df["timestamp"].iloc[-1],
        )
