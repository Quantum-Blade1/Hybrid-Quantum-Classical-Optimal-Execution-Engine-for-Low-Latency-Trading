"""Per-tick microstructure state (Kyle's lambda, VPIN, adverse selection) for the HFT QUBO."""

from dataclasses import dataclass

from qexec.microstructure.adverse_selection import AdverseSelectionModel
from qexec.microstructure.kyle import KyleLambdaEstimator
from qexec.microstructure.vpin import TOXICITY_THRESHOLD, VPINEstimator

_WIDE_SPREAD_BPS = 10.0


@dataclass(frozen=True)
class MicrostructureState:
    """Microstructure estimates after one tick."""

    kyle_lambda: float = 0.0
    vpin: float = 0.0
    adverse_selection_cost: float = 0.0
    toxicity_flag: bool = False
    spread_regime: str = "normal"
    effective_spread_bps: float = 0.0
    realized_spread_bps: float = 0.0
    timestamp_ns: int = 0


class MicrostructureAnalyzer:
    """Feeds each tick to the Kyle, VPIN and adverse-selection estimators.

    Signed volume for Kyle's lambda uses the tick rule; the spread regime is
    "toxic" when VPIN > 0.7, "wide" when the quoted spread exceeds 10 bps, else "normal".
    """

    def __init__(
        self,
        kyle_window: int = 100,
        vpin_bucket_size: int = 1000,
        vpin_buckets: int = 50,
    ) -> None:
        self.kyle = KyleLambdaEstimator(kyle_window)
        self.vpin_estimator = VPINEstimator(vpin_bucket_size, vpin_buckets)
        self.adverse_selection = AdverseSelectionModel()
        self._prev_price: float | None = None
        self._prev_mid: float | None = None
        self._state = MicrostructureState()

    def process_tick(
        self,
        price: float,
        volume: int,
        bid: float,
        ask: float,
        side: str = "buy",
        *,
        timestamp_ns: int = 0,
    ) -> MicrostructureState:
        mid = (bid + ask) / 2
        spread = ask - bid

        if self._prev_price is not None:
            signed_vol = volume if price > self._prev_price else -volume
            self.kyle.update(price - self._prev_price, signed_vol)
            self.vpin_estimator.update(price, volume, self._prev_price)

        if self._prev_mid is not None:
            self.adverse_selection.update(price, mid, side)

        eff_spread, real_spread, as_cost = self.adverse_selection.estimate()
        eff_spread_bps = eff_spread / mid * 10_000 if mid > 0 else 0.0
        real_spread_bps = real_spread / mid * 10_000 if mid > 0 else 0.0

        if self.vpin_estimator.vpin > TOXICITY_THRESHOLD:
            regime = "toxic"
        elif spread / mid * 10_000 > _WIDE_SPREAD_BPS:
            regime = "wide"
        else:
            regime = "normal"

        self._state = MicrostructureState(
            kyle_lambda=self.kyle.lambda_value,
            vpin=self.vpin_estimator.vpin,
            adverse_selection_cost=as_cost,
            toxicity_flag=self.vpin_estimator.is_toxic,
            spread_regime=regime,
            effective_spread_bps=eff_spread_bps,
            realized_spread_bps=real_spread_bps,
            timestamp_ns=timestamp_ns,
        )
        self._prev_price = price
        self._prev_mid = mid
        return self._state
