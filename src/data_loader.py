"""
Real Data Loader for Tick Data

Handles ingestion of CSV/Parquet files from exchanges (NSE, Binance, NYSE).
Format expected: timestamp, price, volume
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class TickData:
    symbol: str
    data: pd.DataFrame
    start_time: pd.Timestamp
    end_time: pd.Timestamp

class DataLoader:
    @staticmethod
    def load_csv(filepath: str, symbol: str = "UNKNOWN") -> TickData:
        """
        Load tick data from CSV.
        Expected columns: 'timestamp' (or 'time'), 'price', 'volume'
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        df = pd.read_csv(filepath)
        
        # Normalize columns
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Map common names
        col_map = {
            'time': 'timestamp',
            'date': 'timestamp',
            'datetime': 'timestamp',
            'last': 'price',
            'close': 'price', # If OHLC, use close as tick
            'vol': 'volume',
            'qty': 'volume',
            'quantity': 'volume'
        }
        df = df.rename(columns=col_map)
        
        if 'timestamp' not in df.columns or 'price' not in df.columns:
            raise ValueError(f"CSV must contain 'timestamp' and 'price' columns. Found: {df.columns}")
            
        # Parse Dates
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Ensure volume
        if 'volume' not in df.columns:
            df['volume'] = 100 # Default size
            
        # Fill missing
        df = df.ffill().bfill()
        
        return TickData(
            symbol=symbol,
            data=df,
            start_time=df['timestamp'].iloc[0],
            end_time=df['timestamp'].iloc[-1]
        )

    @staticmethod
    def create_dummy_nse_data(filepath: str = "data/sample_nse_ticks.csv"):
        """Create a synthetic NSE-like tick file for demonstration."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        dates = pd.date_range("2024-01-01 09:15:00", "2024-01-01 15:30:00", freq="1s") # NSE Hours
        n = len(dates)
        
        # Realistic NIFTY simulation
        price = 21000.0
        drift = 0.0001
        vol = 0.10
        
        returns = np.random.normal(drift/n, vol/np.sqrt(n), n)
        prices = price * np.cumprod(1 + returns)
        
        volumes = np.random.exponential(50, n).astype(int) + 10
        
        df = pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'volume': volumes,
            'symbol': 'NIFTY50'
        })
        
        df.to_csv(filepath, index=False)
        print(f"Generated sample data at {filepath}")

if __name__ == "__main__":
    # Test generation
    DataLoader.create_dummy_nse_data()
    data = DataLoader.load_csv("data/sample_nse_ticks.csv", "NIFTY")
    print(f"Loaded {len(data.data)} ticks for {data.symbol}")
