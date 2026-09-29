"""Streamlit monitor for a live HybridController run against a random-walk price feed.

The price feed is a Gaussian random walk for display only; fills are marked at the
feed price of their tick (the runtime has no fill model).

    streamlit run apps/dashboard.py
"""

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from qexec.runtime.controller import HybridController

TICKS_PER_MINUTE = 10
INITIAL_PRICE = 150.0
PRICE_STEP_STD = 0.05
SPEED_DELAYS = {"1x": 1.0, "5x": 0.2, "10x": 0.1, "Max": 0.01}
DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "white"},
}

st.set_page_config(page_title="Hybrid Execution Monitor", layout="wide")
st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button:hover { background-color: #45a049; }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state(seed: int) -> None:
    defaults: dict[str, Any] = {
        "simulation_running": False,
        "controller": None,
        "market_data": [],
        "rng": np.random.default_rng(seed),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Execution Config")
    total_shares = st.sidebar.number_input("Total Shares", 1000, 1_000_000, 10_000, step=1000)
    duration_min = st.sidebar.slider("Duration (minutes)", 10, 390, 60)
    seed = st.sidebar.number_input("Seed", 0, 2**31 - 1, 42)
    sim_speed = st.sidebar.select_slider("Simulation Speed", list(SPEED_DELAYS), value="5x")
    return {
        "total_shares": int(total_shares),
        "duration": int(duration_min),
        "seed": int(seed),
        "sim_speed": sim_speed,
    }


def start_simulation(config: dict[str, Any]) -> None:
    st.session_state.simulation_running = True
    st.session_state.market_data = []
    st.session_state.rng = np.random.default_rng(config["seed"])
    controller = HybridController(
        optimizer_type="sa",
        optimizer_interval=1.0,
        engine_tick_interval=0.1,
        seed=config["seed"],
    )
    controller.optimizer.start(config["total_shares"], config["duration"])
    controller.engine.start(config["duration"] * TICKS_PER_MINUTE)
    st.session_state.controller = controller
    st.toast("Simulation started")


def stop_simulation() -> None:
    if st.session_state.controller:
        st.session_state.controller.optimizer.stop()
        st.session_state.controller.engine.stop()
    st.session_state.simulation_running = False
    st.toast("Simulation stopped")


def price_figure(prices: pd.DataFrame, fills: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=prices["price"], mode="lines", name="Price", line={"color": "#00ff00"})
    )
    if not fills.empty:
        fig.add_trace(
            go.Scatter(
                x=fills["tick"],
                y=fills["price"],
                mode="markers",
                name="Executions",
                marker={"color": "orange", "size": 6},
            )
        )
    fig.update_layout(
        title="Market Price & Executions", height=350, margin={"l": 0, "r": 0, "t": 30, "b": 0}
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


def fill_figure(fills: pd.DataFrame, total: int, duration_ticks: int) -> go.Figure:
    fig = go.Figure()
    if not fills.empty:
        fig.add_trace(
            go.Scatter(
                x=fills["tick"],
                y=fills["cumulative"],
                mode="lines",
                name="Hybrid",
                fill="tozeroy",
                line={"color": "#00ccff"},
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, duration_ticks],
            y=[0, total],
            mode="lines",
            name="TWAP",
            line={"color": "white", "dash": "dash"},
        )
    )
    fig.update_layout(title="Execution Trajectory vs TWAP", height=300)
    fig.update_layout(**DARK_LAYOUT)
    return fig


@dataclass
class Placeholders:
    price: Any
    shares: Any
    twap: Any
    policy: Any
    chart_price: Any
    chart_fill: Any
    chart_cost: Any
    chart_activity: Any
    table_log: Any
    chart_qubo: Any


def build_layout(config: dict[str, Any]) -> Placeholders:
    col1, col2 = st.columns([3, 1])
    col1.title("Hybrid Execution Monitor")
    with col2:
        if not st.session_state.simulation_running:
            if st.button("Start"):
                start_simulation(config)
        elif st.button("Stop"):
            stop_simulation()

    with st.expander("How to read this dashboard"):
        st.markdown(
            """
- **Current Price**: last price of the simulated feed and its change from the previous tick.
- **Executed Shares**: shares filled so far and the fraction of the order.
- **vs TWAP Schedule**: filled shares minus a linear schedule; positive means ahead.
- **Optimizer tab**: the schedule of the policy the fast path is currently executing.
"""
        )

    m1, m2, m3, m4 = st.columns(4)
    tab1, tab2, tab3 = st.tabs(["Monitor", "Fills", "Optimizer"])
    with tab2:
        col_a, col_b = st.columns(2)
    return Placeholders(
        price=m1.empty(),
        shares=m2.empty(),
        twap=m3.empty(),
        policy=m4.empty(),
        chart_price=tab1.empty(),
        chart_fill=tab1.empty(),
        chart_cost=col_a.empty(),
        chart_activity=col_b.empty(),
        table_log=tab2.empty(),
        chart_qubo=tab3.empty(),
    )


def next_price() -> tuple[float, float]:
    """Append one random-walk tick to the feed; return (price, change)."""
    rng: np.random.Generator = st.session_state.rng
    feed = st.session_state.market_data
    last_price = feed[-1]["price"] if feed else INITIAL_PRICE
    change = float(rng.normal(0, PRICE_STEP_STD))
    feed.append(
        {
            "timestamp": datetime.now(),
            "price": last_price + change,
            "volume": int(rng.integers(100, 1000)),
        }
    )
    return last_price + change, change


def fills_with_prices(controller: HybridController, prices: pd.DataFrame) -> pd.DataFrame:
    log = pd.DataFrame(controller.engine.execution_log)
    if log.empty:
        return log
    fills = log[log["tick"] < len(prices)].copy()
    fills["price"] = prices["price"].to_numpy()[fills["tick"].to_numpy()]
    fills["slippage"] = fills["price"] - INITIAL_PRICE
    return fills


def render_fills(fills: pd.DataFrame, ph: Placeholders) -> None:
    if fills.empty:
        return
    fig_cost = px.bar(fills, x="tick", y="slippage", title="Fill Price minus Arrival ($)")
    fig_cost.update_traces(marker_color=np.where(fills["slippage"] < 0, "#4CAF50", "#FF5252"))
    fig_cost.update_layout(**DARK_LAYOUT)
    ph.chart_cost.plotly_chart(fig_cost, use_container_width=True)
    fig_act = px.area(fills, x="tick", y="shares", title="Shares per Tick")
    fig_act.update_layout(**DARK_LAYOUT)
    ph.chart_activity.plotly_chart(fig_act, use_container_width=True)
    ph.table_log.dataframe(fills[["tick", "shares", "cumulative", "price", "slippage"]].tail(10))


def render_tick(config: dict[str, Any], ph: Placeholders) -> None:
    controller: HybridController = st.session_state.controller
    current_tick = len(st.session_state.market_data)
    new_price, price_change = next_price()
    prices = pd.DataFrame(st.session_state.market_data)

    executed = controller.engine.executed_shares
    total = config["total_shares"]
    duration_ticks = config["duration"] * TICKS_PER_MINUTE
    twap_executed = min(total, (current_tick + 1) * total / duration_ticks)
    policy = controller.engine.current_policy

    ph.price.metric("Current Price", f"${new_price:.2f}", f"{price_change:.2f}")
    ph.shares.metric("Executed Shares", f"{executed:,}", f"{min(1.0, executed / total):.1%}")
    ph.twap.metric("vs TWAP Schedule", f"{executed - twap_executed:,.0f} shares")
    ph.policy.metric("Policy", f"#{policy.policy_id}" if policy else "none")

    fills = fills_with_prices(controller, prices)
    ph.chart_price.plotly_chart(price_figure(prices, fills), use_container_width=True)
    ph.chart_fill.plotly_chart(fill_figure(fills, total, duration_ticks), use_container_width=True)
    if policy is not None:
        fig_qubo = px.bar(
            x=range(len(policy.schedule)), y=policy.schedule, title="Current Policy Schedule"
        )
        fig_qubo.update_traces(marker_color="#4CAF50")
        fig_qubo.update_layout(xaxis_title="Time Slice", yaxis_title="Shares", **DARK_LAYOUT)
        ph.chart_qubo.plotly_chart(fig_qubo, use_container_width=True)
    render_fills(fills, ph)

    if executed >= total or len(st.session_state.market_data) > duration_ticks:
        stop_simulation()
        st.success("Execution complete")


def main() -> None:
    config = render_sidebar()
    initialize_session_state(config["seed"])
    placeholders = build_layout(config)
    if st.session_state.simulation_running:
        render_tick(config, placeholders)
        time.sleep(SPEED_DELAYS[config["sim_speed"]])
        st.rerun()


if __name__ == "__main__":
    main()
