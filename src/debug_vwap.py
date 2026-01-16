
import pandas as pd
import numpy as np
from src.market_data import MarketDataSimulator, MarketParams
from src.vwap_strategy import VWAPStrategy
from src.execution_engine import ExecutionEngine

def debug():
    params = MarketParams(initial_price=100.0, annual_volatility=0.30)
    sim = MarketDataSimulator(params=params)
    data = sim.generate(num_minutes=60)
    
    print("Data types:")
    print(data.dtypes)
    print("Volume type:", type(data['volume'].values))
    
    engine = ExecutionEngine(data)
    strategy = VWAPStrategy(participation_rate=0.1, order_book=engine.order_book)
    
    total_shares = 100000
    
    print("Executing strategy...")
    try:
        metrics = strategy.execute(total_shares, 'buy', data)
        print("Success!")
        print(metrics)
    except Exception as e:
        print("Caught exception:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
