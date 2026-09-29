import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qexec.market.binance import (
    BAR_COLUMNS,
    day_bars,
    minute_bars,
    opposite_side_gaps,
    parse_checksum,
    read_agg_trades,
    sha256_file,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "binance_aggtrades_sample.csv"
DAY = pd.Timestamp("2026-07-22")


@pytest.fixture
def trades() -> pd.DataFrame:
    return read_agg_trades(FIXTURE)


def test_read_agg_trades_parses_microsecond_times_and_taker_side(trades):
    assert len(trades) == 7
    assert trades["time"].iloc[0] == DAY + pd.Timedelta(seconds=1)
    # is_buyer_maker True -> the taker sold.
    assert trades["is_buyer_maker"].tolist() == [False, True, False, True, False, False, True]


def test_read_agg_trades_accepts_millisecond_times_and_a_header(tmp_path):
    lines = FIXTURE.read_text().splitlines()
    converted = []
    for line in lines:
        cols = line.split(",")
        cols[5] = str(int(cols[5]) // 1000)
        converted.append(",".join(cols))
    path = tmp_path / "ms.csv"
    path.write_text(
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,"
        "is_buyer_maker,is_best_match\n" + "\n".join(converted) + "\n"
    )
    ms = read_agg_trades(path)
    assert ms["time"].tolist() == read_agg_trades(FIXTURE)["time"].tolist()


def test_opposite_side_gaps(trades):
    gaps = opposite_side_gaps(trades, max_gap_ms=100)
    # Minute 0: three alternating pairs 50 ms apart; minute 2: one pair 80 ms apart.
    assert len(gaps) == 4
    assert gaps["gap_bps"].iloc[0] == pytest.approx(0.02 / 99.99 * 1e4)
    assert len(opposite_side_gaps(trades, max_gap_ms=10)) == 0


def test_minute_bars_values(trades):
    bars = minute_bars(trades, DAY, DAY + pd.Timedelta(minutes=4))
    assert list(bars.columns) == list(BAR_COLUMNS)
    assert len(bars) == 4
    m0, m1, m2, m3 = (bars.iloc[i] for i in range(4))

    assert (m0.open, m0.high, m0.low, m0.close) == (100.0, 100.0, 99.98, 99.98)
    assert m0.volume == pytest.approx(5.0)
    assert m0.quote_volume == pytest.approx(100 * 1 + 99.98 * 2 + 100 * 1 + 99.98 * 1)
    assert m0.vwap == pytest.approx(m0.quote_volume / 5.0)
    assert m0.trade_count == 6  # underlying trades: 1 + 2 + 1 + 2
    assert m0.buy_volume == pytest.approx(2.0)
    assert m0.signed_volume == pytest.approx(2.0 - 3.0)
    assert m0.spread_pairs == 3
    assert m0.half_spread_bps == pytest.approx(0.02 / 99.99 * 1e4 / 2)

    assert m1.volume == 0 and m1.trade_count == 0
    assert m1.open == m1.close == m1.vwap == 99.98
    assert np.isnan(m1.half_spread_bps)

    assert m2.signed_volume == pytest.approx(4.0 - 2.0)
    assert m2.spread_pairs == 1
    assert np.isnan(m2.half_spread_bps)  # fewer than 3 pairs
    assert m3.volume == 0 and m3.close == 100.18


def test_minute_bars_min_pairs_parameter(trades):
    bars = minute_bars(trades, DAY, DAY + pd.Timedelta(minutes=3), min_pairs=1)
    assert bars["half_spread_bps"].iloc[2] == pytest.approx(0.02 / 100.19 * 1e4 / 2)


def test_signed_volume_identity(trades):
    bars = minute_bars(trades)
    buy = bars["buy_volume"]
    sell = bars["volume"] - buy
    assert np.allclose(bars["signed_volume"], buy - sell)
    assert bars["volume"].sum() == pytest.approx(trades["quantity"].sum())


def test_day_bars_from_zip_covers_the_whole_day(tmp_path):
    archive = tmp_path / "LINKUSDT-aggTrades-2026-07-22.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(FIXTURE, "LINKUSDT-aggTrades-2026-07-22.csv")
    bars = day_bars(archive, "2026-07-22")
    assert len(bars) == 1440
    assert bars["timestamp"].iloc[0] == DAY
    assert bars["timestamp"].iloc[-1] == DAY + pd.Timedelta(minutes=1439)
    assert bars["volume"].sum() == pytest.approx(11.0)


def test_checksum_helpers(tmp_path):
    path = tmp_path / "f.zip"
    path.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert sha256_file(path) == digest
    assert parse_checksum(f"{digest.upper()}  f.zip\n") == digest
    with pytest.raises(ValueError):
        parse_checksum("not a checksum")
