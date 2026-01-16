"""
Load Testing Suite

Simulates high-throughput execution environment:
- 100 simultaneous orders
- Mixed sizes and durations
- Measures system throughput and latency
"""

import time
import queue
import logging
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import os

from src.hybrid_async import HybridController

# Configure logging (reduce verbosity for load test)
logging.getLogger("src.hybrid_async").setLevel(logging.WARNING)
logging.getLogger("src.optimizer_resilience").setLevel(logging.WARNING)

@dataclass
class LoadTestConfig:
    num_orders: int = 100
    concurrency: int = 20
    min_shares: int = 100
    max_shares: int = 10000
    min_slices: int = 10
    max_slices: int = 50

@dataclass
class OrderResult:
    order_id: int
    total_shares: int
    executed_shares: int
    num_slices: int
    duration: float
    throughput: float # shares/sec
    optimizations: int
    success: bool
    error: str = ""

def run_single_order(order_id: int, config: LoadTestConfig) -> OrderResult:
    """Execute a single order via HybridController."""
    try:
        # Randomized parameters
        total_shares = random.randint(config.min_shares, config.max_shares)
        num_slices = random.randint(config.min_slices, config.max_slices)
        
        # Instantiate controller for this order
        # In a real system, we might reuse controllers or have a pool
        # Here we simulate independent execution agents
        controller = HybridController(
            optimizer_type='sa',
            optimizer_interval=1.0, # Slower optimization to reduce CPU contention
            engine_tick_interval=0.05 # Fast execution tick
        )
        
        start = time.time()
        
        # Execute
        res = controller.execute_order(
            total_shares=total_shares,
            num_slices=num_slices
        )
        
        duration = time.time() - start
        
        return OrderResult(
            order_id=order_id,
            total_shares=total_shares,
            executed_shares=res['executed_shares'],
            num_slices=num_slices,
            duration=duration,
            throughput=res['executed_shares'] / max(0.001, duration),
            optimizations=res['num_optimizations'],
            success=True
        )
        
    except Exception as e:
        return OrderResult(
            order_id=order_id,
            total_shares=0,
            executed_shares=0,
            num_slices=0,
            duration=0,
            throughput=0,
            optimizations=0,
            success=False,
            error=str(e)
        )

def run_load_test():
    """Run the full load test."""
    print("="*70)
    print(" PERFORMANCE LOAD TEST")
    print("="*70)
    
    config = LoadTestConfig()
    print(f"Configuration:")
    print(f"  Orders: {config.num_orders}")
    print(f"  Concurrency: {config.concurrency}")
    print(f"  Shares: {config.min_shares}-{config.max_shares}")
    
    print("\nStarting execution...")
    
    results: List[OrderResult] = []
    process = psutil.Process(os.getpid())
    
    start_time = time.time()
    initial_cpu = process.cpu_percent()
    initial_mem = process.memory_info().rss / 1024 / 1024
    
    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {
            executor.submit(run_single_order, i, config): i 
            for i in range(config.num_orders)
        }
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 10 == 0:
                print(f"  Progress: {completed}/{config.num_orders} orders completed")
    
    total_duration = time.time() - start_time
    final_mem = process.memory_info().rss / 1024 / 1024
    
    # Analysis
    print("\n" + "="*70)
    print(" RESULTS ANALYSIS")
    print("="*70)
    
    df = pd.DataFrame([vars(r) for r in results])
    
    success_rate = df['success'].mean() * 100
    total_volume = df['executed_shares'].sum()
    avg_latency = df['duration'].mean()
    p95_latency = df['duration'].quantile(0.95)
    p99_latency = df['duration'].quantile(0.99)
    avg_tps = df['throughput'].mean()
    
    # System Throughput (Orders per second)
    ops = config.num_orders / total_duration
    
    print(f"Execution Time: {total_duration:.2f}s")
    print(f"Success Rate:   {success_rate:.1f}%")
    print(f"Total Volume:   {total_volume:,.0f} shares")
    print(f"System TPS:     {ops:.2f} orders/sec")
    print(f"Memory Usage:   {initial_mem:.1f}MB -> {final_mem:.1f}MB")
    print("-" * 30)
    print("Latency Distribution (per order):")
    print(f"  Avg: {avg_latency:.2f}s")
    print(f"  p50: {df['duration'].median():.2f}s")
    print(f"  p95: {p95_latency:.2f}s")
    print(f"  p99: {p99_latency:.2f}s")
    print("-" * 30)
    print("Throughput Distribution (shares/sec):")
    print(f"  Avg: {avg_tps:.1f}")
    print("-" * 30)
    
    if not df[~df['success']].empty:
        print("\nErrors:")
        print(df[~df['success']][['order_id', 'error']])
    
    # Check constraints
    if success_rate < 99:
        print("\n[WARNING] Success rate below 99%")
    if avg_latency > 10.0:
        print("\n[WARNING] Average latency high (>10s)")

if __name__ == "__main__":
    run_load_test()
