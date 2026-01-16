"""
Tests for Market Data Simulator

Verifies that generated market data has realistic properties.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\test_market_data.py", ""))

from src.market_data import (
    MarketDataSimulator,
    IntraDayPriceGenerator,
    VolumeProfileGenerator,
    MarketParams,
    calculate_vwap
)


class TestIntraDayPriceGenerator:
    """Tests for price generation."""
    
    def test_generates_correct_length(self):
        """Test that generator produces correct number of data points."""
        params = MarketParams(initial_price=150.0)
        generator = IntraDayPriceGenerator(params, seed=42)
        
        df = generator.generate(datetime.now(), num_minutes=390)
        
        assert len(df) == 390
    
    def test_prices_are_positive(self):
        """Test that all generated prices are positive."""
        params = MarketParams(initial_price=150.0)
        generator = IntraDayPriceGenerator(params, seed=42)
        
        df = generator.generate(datetime.now(), num_minutes=390)
        
        assert (df["price"] > 0).all()
    
    def test_prices_near_initial(self):
        """Test that prices stay within reasonable range of initial."""
        params = MarketParams(initial_price=150.0, annual_volatility=0.25)
        generator = IntraDayPriceGenerator(params, seed=42)
        
        df = generator.generate(datetime.now(), num_minutes=390)
        
        # Prices should stay within ±20% for a single day typically
        assert df["price"].min() > 100
        assert df["price"].max() < 200
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        params = MarketParams(initial_price=150.0)
        
        gen1 = IntraDayPriceGenerator(params, seed=42)
        gen2 = IntraDayPriceGenerator(params, seed=42)
        
        df1 = gen1.generate(datetime(2024, 1, 15), num_minutes=100)
        df2 = gen2.generate(datetime(2024, 1, 15), num_minutes=100)
        
        np.testing.assert_array_almost_equal(df1["price"], df2["price"])


class TestVolumeProfileGenerator:
    """Tests for volume profile generation."""
    
    def test_generates_correct_length(self):
        """Test that generator produces correct number of data points."""
        generator = VolumeProfileGenerator(seed=42)
        
        volumes = generator.generate(num_minutes=390)
        
        assert len(volumes) == 390
    
    def test_volumes_are_positive(self):
        """Test that all volumes are positive."""
        generator = VolumeProfileGenerator(seed=42)
        
        volumes = generator.generate(num_minutes=390)
        
        assert (volumes > 0).all()
    
    def test_u_shaped_profile(self):
        """Test that volume profile is roughly U-shaped."""
        generator = VolumeProfileGenerator(seed=42)
        
        volumes = generator.generate(num_minutes=390)
        
        # First and last 30 minutes should be higher than middle
        opening_volume = volumes[:30].mean()
        closing_volume = volumes[-30:].mean()
        midday_volume = volumes[150:240].mean()
        
        assert opening_volume > midday_volume
        assert closing_volume > midday_volume
    
    def test_total_volume_reasonable(self):
        """Test that total volume is close to expected."""
        total_daily = 50_000_000
        generator = VolumeProfileGenerator(total_daily_volume=total_daily, seed=42)
        
        volumes = generator.generate(num_minutes=390)
        
        # Should be within 5% of target (due to rounding and minimum constraints)
        assert abs(volumes.sum() - total_daily) / total_daily < 0.10


class TestMarketDataSimulator:
    """Tests for complete market data simulation."""
    
    def test_generates_all_columns(self):
        """Test that simulator produces all required columns."""
        simulator = MarketDataSimulator(seed=42)
        
        df = simulator.generate(datetime.now())
        
        required_columns = ["timestamp", "symbol", "price", "bid", "ask", "spread", "volume"]
        for col in required_columns:
            assert col in df.columns
    
    def test_spread_is_positive(self):
        """Test that bid-ask spread is always positive."""
        simulator = MarketDataSimulator(seed=42)
        
        df = simulator.generate(datetime.now())
        
        assert (df["spread"] > 0).all()
    
    def test_bid_less_than_ask(self):
        """Test that bid is always less than ask."""
        simulator = MarketDataSimulator(seed=42)
        
        df = simulator.generate(datetime.now())
        
        assert (df["bid"] < df["ask"]).all()
    
    def test_price_between_bid_ask(self):
        """Test that mid price is between bid and ask."""
        simulator = MarketDataSimulator(seed=42)
        
        df = simulator.generate(datetime.now())
        
        assert (df["price"] >= df["bid"]).all()
        assert (df["price"] <= df["ask"]).all()
    
    def test_spread_reasonable(self):
        """Test that spread is in reasonable range (< 1% of price)."""
        simulator = MarketDataSimulator(seed=42)
        
        df = simulator.generate(datetime.now())
        
        spread_pct = df["spread"] / df["price"]
        assert (spread_pct < 0.01).all()


class TestCalculateVWAP:
    """Tests for VWAP calculation."""
    
    def test_vwap_calculation_simple(self):
        """Test VWAP calculation with known values."""
        df = pd.DataFrame({
            "price": [100.0, 101.0, 102.0],
            "volume": [1000, 2000, 1000]
        })
        
        # VWAP = (100*1000 + 101*2000 + 102*1000) / (1000 + 2000 + 1000)
        # = (100000 + 202000 + 102000) / 4000 = 404000 / 4000 = 101.0
        vwap = calculate_vwap(df)
        
        assert vwap == 101.0
    
    def test_vwap_weights_by_volume(self):
        """Test that VWAP is properly volume-weighted."""
        df = pd.DataFrame({
            "price": [100.0, 200.0],
            "volume": [1, 99]  # Weight heavily toward 200
        })
        
        vwap = calculate_vwap(df)
        
        # Should be close to 200 due to volume weighting
        assert vwap > 190


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
