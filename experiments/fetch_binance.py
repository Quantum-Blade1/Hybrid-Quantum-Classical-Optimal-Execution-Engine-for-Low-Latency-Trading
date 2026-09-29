"""Download the protocol's Binance aggTrades days, verify SHA-256, and build 1-minute bars.

Idempotent: a zip already on disk whose SHA-256 matches its published `.CHECKSUM` is not
downloaded again, and bars are rebuilt only when missing. No API keys; public files only.
Raw zips go to data/binance/aggTrades/<SYMBOL>/, bars to data/binance/bars/<SYMBOL>/
(both git-ignored).

Usage:
    python -m experiments.fetch_binance [--symbols BTCUSDT ...] [--days 2026-07-22 ...]
                                        [--no-bars]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from experiments.protocol import (
    ALL_DAYS,
    BARS_DIR,
    BASE_URL,
    DATA_DIR,
    RAW_DIR,
    SYMBOLS,
    bars_path,
    raw_path,
    zip_name,
)
from qexec.market.binance import day_bars, parse_checksum, sha256_file

RETRIES = 3
TIMEOUT_S = 120


def _get(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
                data: bytes = response.read()
                return data
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def fetch_day(symbol: str, day: str, raw_dir: Path = RAW_DIR) -> dict[str, object]:
    """Ensure the verified zip of (symbol, day) is on disk; returns its record."""
    path = raw_path(symbol, day, raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{symbol}/{zip_name(symbol, day)}"
    checksum_path = path.with_suffix(".zip.CHECKSUM")
    if not checksum_path.exists():
        checksum_path.write_bytes(_get(url + ".CHECKSUM"))
    expected = parse_checksum(checksum_path.read_text())
    downloaded = False
    if not path.exists() or sha256_file(path) != expected:
        part = path.with_suffix(".zip.part")
        part.write_bytes(_get(url))
        actual = sha256_file(part)
        if actual != expected:
            part.unlink()
            raise RuntimeError(f"checksum mismatch for {url}: {actual} != {expected}")
        part.replace(path)
        downloaded = True
    return {
        "symbol": symbol,
        "day": day,
        "url": url,
        "sha256": expected,
        "bytes": path.stat().st_size,
        "downloaded_now": downloaded,
    }


def build_bars(symbol: str, day: str, raw_dir: Path = RAW_DIR, bars_dir: Path = BARS_DIR) -> Path:
    out = bars_path(symbol, day, bars_dir)
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        bars = day_bars(raw_path(symbol, day, raw_dir), day)
        tmp = out.with_suffix(".tmp")
        bars.to_csv(tmp, index=False, float_format="%.10g", compression="gzip")
        tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="*", default=list(SYMBOLS))
    parser.add_argument("--days", nargs="*", default=list(ALL_DAYS))
    parser.add_argument("--no-bars", action="store_true")
    args = parser.parse_args(argv)

    records, failures = [], []
    for symbol in args.symbols:
        for day in args.days:
            try:
                record = fetch_day(symbol, day)
                if not args.no_bars:
                    build_bars(symbol, day)
            except Exception as exc:  # report every failing day, then exit non-zero
                failures.append(f"{symbol} {day}: {exc}")
                print(f"FAILED {symbol} {day}: {exc}", flush=True)
                continue
            records.append(record)
            flag = "downloaded" if record["downloaded_now"] else "verified"
            print(f"{symbol} {day} {record['bytes'] / 1e6:7.1f} MB {flag}", flush=True)

    total = sum(int(r["bytes"]) for r in records)  # type: ignore[call-overload]
    print(f"{len(records)} files, {total / 1e6:.1f} MB")
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "binance" / "download_manifest.json").write_text(
        json.dumps({"files": records, "total_bytes": total, "failures": failures}, indent=2)
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
