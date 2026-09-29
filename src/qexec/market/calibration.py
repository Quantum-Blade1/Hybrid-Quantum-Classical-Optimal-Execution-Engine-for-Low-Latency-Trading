from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from qexec.execution.cost_model import CostModel

MINUTES_PER_DAY = 1440


@dataclass(frozen=True)
class IntradayProfile:
    """Per-minute-of-day expected volume, half spread (bps) and volatility (bps)."""

    volume: NDArray[np.float64]
    half_spread_bps: NDArray[np.float64]
    sigma_bps: NDArray[np.float64]
    bucket_minutes: int

    def window(self, start_minute: int, minutes: int) -> tuple[NDArray[np.float64], ...]:
        """(volume, half spread, sigma) for minutes-of-day start..start+minutes-1 (wrapping)."""
        idx = (start_minute + np.arange(minutes)) % MINUTES_PER_DAY
        return self.volume[idx], self.half_spread_bps[idx], self.sigma_bps[idx]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "minute_of_day": np.arange(MINUTES_PER_DAY),
                "expected_volume": self.volume,
                "half_spread_bps": self.half_spread_bps,
                "sigma_bps": self.sigma_bps,
            }
        )


@dataclass(frozen=True)
class ImpactFit:
    beta_bps: float
    std_error: float
    r_squared: float
    num_bars: int
    beta_with_intercept: float
    intercept_bps: float


@dataclass(frozen=True)
class Calibration:
    symbol: str
    days: tuple[str, ...]
    adv: float
    profile: IntradayProfile
    impact: ImpactFit
    lot_size: float

    def cost_model(
        self,
        start_minute: int,
        minutes: int,
        *,
        impact_multiple: float = 1.0,
        risk_aversion: float = 0.0,
    ) -> CostModel:
        """Cost model of an order window, with volumes in lots."""
        volume, spread, sigma = self.profile.window(start_minute, minutes)
        return CostModel(
            expected_volume=np.maximum(volume / self.lot_size, 1.0),
            half_spread_bps=spread,
            sigma_bps=sigma,
            impact_bps=self.impact.beta_bps * impact_multiple,
            risk_aversion=risk_aversion,
        )

    def summary(self) -> dict[str, float | int | str]:
        return {
            "symbol": self.symbol,
            "num_days": len(self.days),
            "first_day": self.days[0],
            "last_day": self.days[-1],
            "adv_base": self.adv,
            "lot_size": self.lot_size,
            "beta_bps": self.impact.beta_bps,
            "beta_std_error": self.impact.std_error,
            "beta_r_squared": self.impact.r_squared,
            "beta_with_intercept": self.impact.beta_with_intercept,
            "intercept_bps": self.impact.intercept_bps,
            "impact_num_bars": self.impact.num_bars,
            "mean_half_spread_bps": float(np.mean(self.profile.half_spread_bps)),
            "mean_sigma_bps": float(np.mean(self.profile.sigma_bps)),
            "mean_volume_per_minute": float(np.mean(self.profile.volume)),
        }


def _minute_of_day(bars: pd.DataFrame) -> NDArray[np.int_]:
    ts = pd.DatetimeIndex(bars["timestamp"])
    return np.asarray(ts.hour * 60 + ts.minute, dtype=np.int_)


def log_returns_bps(bars: pd.DataFrame) -> NDArray[np.float64]:
    """1-minute close-to-close log returns in bps; NaN at each day's first bar."""
    close = bars["close"].to_numpy(dtype=np.float64)
    r = np.full(close.size, np.nan)
    r[1:] = np.log(close[1:] / close[:-1]) * 1e4
    day = pd.DatetimeIndex(bars["timestamp"]).normalize()
    r[np.r_[True, day[1:] != day[:-1]]] = np.nan
    return r


def _bucketed(
    values: NDArray[np.float64], buckets: NDArray[np.int_], how: str, n: int
) -> NDArray[np.float64]:
    series = pd.Series(values).groupby(buckets)
    agg = {
        "mean": series.mean(),
        "median": series.median(),
        "rms": series.apply(lambda s: float(np.sqrt(np.nanmean(np.square(s))))),
    }[how]
    return np.asarray(agg.reindex(range(n)).to_numpy(), dtype=np.float64)


def intraday_profile(bars: pd.DataFrame, bucket_minutes: int = 15) -> IntradayProfile:
    if MINUTES_PER_DAY % bucket_minutes:
        raise ValueError("bucket_minutes must divide 1440")
    n = MINUTES_PER_DAY // bucket_minutes
    buckets = _minute_of_day(bars) // bucket_minutes
    volume = _bucketed(bars["volume"].to_numpy(dtype=np.float64), buckets, "mean", n)
    spread = _bucketed(bars["half_spread_bps"].to_numpy(dtype=np.float64), buckets, "median", n)
    sigma = _bucketed(log_returns_bps(bars), buckets, "rms", n)
    # A bucket without any spread estimate takes the overall median.
    spread = np.where(np.isnan(spread), np.nanmedian(spread), spread)
    return IntradayProfile(
        volume=np.repeat(volume, bucket_minutes),
        half_spread_bps=np.repeat(spread, bucket_minutes),
        sigma_bps=np.repeat(sigma, bucket_minutes),
        bucket_minutes=bucket_minutes,
    )


def _ols_origin(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    return float(x @ y / (x @ x))


def fit_impact(
    bars: pd.DataFrame, profile: IntradayProfile, *, n_boot: int = 1000, seed: int = 0
) -> ImpactFit:
    """OLS through the origin of r_k = beta SV_k / Vbar_k + e_k (bps), day-block bootstrap SE."""
    r = log_returns_bps(bars)
    expected = profile.volume[_minute_of_day(bars)]
    x = bars["signed_volume"].to_numpy(dtype=np.float64) / expected
    ok = np.isfinite(r) & np.isfinite(x) & (expected > 0)
    x, y = x[ok], r[ok]
    beta = _ols_origin(x, y)
    resid = y - beta * x
    r2 = 1 - float(resid @ resid) / float(y @ y)
    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)

    day = pd.DatetimeIndex(bars["timestamp"]).normalize()[ok]
    codes, uniques = pd.factorize(day)
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        boots.append(_ols_origin(x[pick], y[pick]))
    return ImpactFit(
        beta_bps=beta,
        std_error=float(np.std(boots, ddof=1)),
        r_squared=r2,
        num_bars=int(x.size),
        beta_with_intercept=float(slope),
        intercept_bps=float(intercept),
    )


def lot_size_for(smallest_order: float, min_lots: int) -> float:
    """Largest power of ten such that `smallest_order` is at least `min_lots` lots."""
    if smallest_order <= 0:
        raise ValueError("smallest_order must be positive")
    return float(10.0 ** np.floor(np.log10(smallest_order / min_lots)))


def calibrate(
    symbol: str,
    bars: pd.DataFrame,
    *,
    smallest_order_pct_adv: float,
    min_lots: int,
    bucket_minutes: int = 15,
) -> Calibration:
    """Calibrate on whole UTC days of development bars (ADV = mean daily base volume)."""
    days = tuple(sorted({str(d.date()) for d in pd.DatetimeIndex(bars["timestamp"]).normalize()}))
    adv = float(bars["volume"].sum() / len(days))
    profile = intraday_profile(bars, bucket_minutes)
    return Calibration(
        symbol=symbol,
        days=days,
        adv=adv,
        profile=profile,
        impact=fit_impact(bars, profile),
        lot_size=lot_size_for(adv * smallest_order_pct_adv / 100, min_lots),
    )
