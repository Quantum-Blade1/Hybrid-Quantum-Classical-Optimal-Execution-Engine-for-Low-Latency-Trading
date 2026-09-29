import pytest

from qexec.market.loader import DataLoader


def test_loader_normalises_aliases_and_sorts_by_time(tmp_path):
    path = tmp_path / "ticks.csv"
    path.write_text(
        " Time ,Close,QTY\n2024-01-02 09:31:00,101.0,300\n2024-01-02 09:30:00,100.0,200\n"
    )
    ticks = DataLoader.load_csv(path, symbol="XYZ")
    assert list(ticks.data.columns) == ["timestamp", "price", "volume"]
    assert ticks.data["price"].tolist() == [100.0, 101.0]
    assert ticks.start_time < ticks.end_time


def test_loader_requires_timestamp_and_price(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("time,volume\n2024-01-02 09:30:00,100\n")
    with pytest.raises(ValueError, match="timestamp' and 'price"):
        DataLoader.load_csv(path)
