from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings

# Property tests stay cheap: few examples, no per-example deadline (first calls JIT-import scipy).
settings.register_profile(
    "qexec", max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
settings.load_profile("qexec")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def small_market() -> pd.DataFrame:
    from qexec.market.simulator import MarketDataSimulator

    return MarketDataSimulator(total_daily_volume=10_000_000, seed=7).generate(
        datetime(2024, 1, 2), num_minutes=30
    )


@pytest.fixture
def small_qubo(rng: np.random.Generator) -> np.ndarray:
    Q = rng.standard_normal((6, 6))
    return (Q + Q.T) / 2


def all_bitstrings(n: int) -> np.ndarray:
    """Every x in {0,1}^n as rows, row i having bit k = (i >> k) & 1."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)) & 1).astype(np.int8)


@pytest.fixture(scope="session")
def bitstrings():
    """`all_bitstrings` as a fixture, so test modules need not import conftest."""
    return all_bitstrings
