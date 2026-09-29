from dataclasses import dataclass

import numpy as np
import pandas as pd

# Below this value of kappa*T the sinh trajectory is numerically linear (risk-neutral limit).
_LINEAR_LIMIT = 1e-4
_ETA_EPS = 1e-9
_KAPPA_ZERO_ETA = 100.0


@dataclass(frozen=True)
class ACConfig:
    """Model inputs; `sigma` is daily volatility of returns, time is measured in days."""

    total_shares: int
    n_days: float = 1.0
    n_steps: int = 10
    sigma: float = 0.02
    price: float = 100.0
    eta: float = 0.05 / 10_000
    rho: float = 0.01 / 10_000
    risk_aversion: float = 1e-6


class AlmgrenChrissSolver:
    """Almgren & Chriss (2000): mean-variance optimal liquidation of X shares over T."""

    def __init__(self, config: ACConfig) -> None:
        self.config = config

    @property
    def _tau(self) -> float:
        return self.config.n_days / self.config.n_steps

    @property
    def _price_variance(self) -> float:
        return (self.config.sigma * self.config.price) ** 2

    def kappa(self) -> float:
        """Urgency kappa = sqrt(lambda sigma^2 / eta) (continuous-time limit)."""
        if abs(self.config.eta) < _ETA_EPS:
            return _KAPPA_ZERO_ETA
        return float(np.sqrt(self.config.risk_aversion * self._price_variance / self.config.eta))

    def compute_trajectory(self) -> pd.DataFrame:
        """Holdings x(t) = X sinh(kappa(T-t)) / sinh(kappa T), Almgren & Chriss (2000), eq. 18."""
        X = self.config.total_shares
        T = self.config.n_days
        N = self.config.n_steps
        kappa = self.kappa()
        t = np.linspace(0, T, N + 1)

        if kappa * T < _LINEAR_LIMIT:
            x_t = X * (1 - t / T)
        else:
            x_t = X * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)

        return pd.DataFrame(
            {
                "step": range(N),
                "time": t[1:],
                "shares_held_start": x_t[:-1],
                "shares_held_end": x_t[1:],
                "shares_to_trade": -np.diff(x_t),
            }
        )

    def calculate_expected_cost(self, trajectory: pd.DataFrame) -> float:
        """E[C] = gamma X^2 / 2 + (eta_tilde / tau) sum n_k^2, A&C (2000) eq. 20, no fixed cost."""
        X = self.config.total_shares
        n = trajectory["shares_to_trade"].to_numpy()
        # The -gamma tau / 2 term is the permanent impact incurred within an interval.
        eta_tilde = self.config.eta - 0.5 * self.config.rho * self._tau
        permanent = 0.5 * self.config.rho * X**2
        temporary = float(np.sum(eta_tilde * n**2 / self._tau))
        return permanent + temporary

    def calculate_variance(self, trajectory: pd.DataFrame) -> float:
        """V[C] = sigma^2 tau sum_{k=1}^{N} x_k^2, with x_k the holdings after interval k."""
        x = trajectory["shares_held_end"].to_numpy()
        return float(self._price_variance * np.sum(x**2) * self._tau)
