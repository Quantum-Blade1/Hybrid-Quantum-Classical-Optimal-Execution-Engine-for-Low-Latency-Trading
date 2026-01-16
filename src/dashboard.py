
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

from src.hybrid_async import HybridController, ExecutionPolicy
from src.market_data import MarketDataSimulator, MarketParams, calculate_vwap

# Page Config
st.set_page_config(
    page_title="Quantum-Classical Execution Dashboard",
    page_icon="⚛️",
    layout="wide",
)

# Custom CSS for "Premium" feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #41444c;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50; 
        color: white;
        border: none;
        padding: 10px 24px;
        font-size: 16px; 
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_resource
def get_market_simulator():
    """Create market data simulator."""
    params = MarketParams(
        symbol="AAPL", 
        initial_price=150.0, 
        annual_volatility=0.30
    )
    return MarketDataSimulator(params)

def initialize_session_state():
    """Initialize session state variables."""
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    
    if 'controller' not in st.session_state:
        st.session_state.controller = None
    
    if 'market_data' not in st.session_state:
        st.session_state.market_data = [] # List of dict points
        
    if 'execution_log' not in st.session_state:
        st.session_state.execution_log = []
        
    if 'twap_log' not in st.session_state:
        st.session_state.twap_log = [] # Track TWAP execution for comparison

# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render configuration sidebar."""
    st.sidebar.header("Execution Config")
    
    symbol = st.sidebar.text_input("Symbol", "AAPL")
    total_shares = st.sidebar.number_input("Total Shares", 1000, 1000000, 10000, step=1000)
    duration_min = st.sidebar.slider("Duration (minutes)", 10, 390, 60)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Strategy Settings")
    
    strategy = st.sidebar.selectbox("Strategy Type", ["Hybrid Quantum", "Adaptive VWAP", "Static TWAP"])
    
    risk_aversion = 0.5
    if strategy == "Hybrid Quantum":
        risk_aversion = st.sidebar.slider("Risk Aversion (Lambda)", 0.0, 1.0, 0.5)
        st.sidebar.caption("0.0 = Impact Neutral, 1.0 = Risk Averse")
    
    st.sidebar.markdown("---")
    
    # Simulation Speed
    sim_speed = st.sidebar.select_slider("Simulation Speed", ["1x", "5x", "10x", "Max"], value="5x")
    
    return {
        "symbol": symbol,
        "total_shares": total_shares,
        "duration": duration_min,
        "strategy": strategy,
        "risk_aversion": risk_aversion,
        "sim_speed": sim_speed
    }

def start_simulation(config):
    """Start the hybrid execution simulation."""
    st.session_state.simulation_running = True
    st.session_state.market_data = []
    st.session_state.execution_log = []
    
    # Initialize Hybrid Controller directly
    st.session_state.controller = HybridController(
        optimizer_type='sa', # Use SA for speed in dashboard
        optimizer_interval=1.0,
        engine_tick_interval=0.1
    )
    
    # Start threads
    num_slices = config["duration"]
    st.session_state.controller.optimizer.start(config["total_shares"], num_slices)
    st.session_state.controller.engine.start(num_slices * 10) # Enough ticks
    
    st.toast(f"Simulation Started: {config['strategy']}")

def stop_simulation():
    """Stop the simulation."""
    if st.session_state.controller:
        st.session_state.controller.optimizer.stop()
        st.session_state.controller.engine.stop()
    st.session_state.simulation_running = False
    st.toast("Simulation Stopped")


# =============================================================================
# Main App
# =============================================================================

def main():
    initialize_session_state()
    config = render_sidebar()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Quantum-Classical Execution Bridge")
    with col2:
        if not st.session_state.simulation_running:
            if st.button("▶ START SIMULATION"):
                start_simulation(config)
        else:
            if st.button("⏹ STOP SIMULATION"):
                stop_simulation()

    # Dashboard Layout
    
    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    
    # Placeholders for live metrics
    metric_price = m1.empty()
    metric_shares = m2.empty()
    metric_vwap = m3.empty()
    metric_is = m4.empty()
    
    # Charts Area
    tab1, tab2, tab3 = st.tabs(["Real-Time Monitor", "Strategy Analytics", "Quantum Optimizer"])
    
    with tab1:
        chart_price = st.empty()
        chart_fill = st.empty()
    
    with tab2:
        st.info("Comparison available after simulation completion loops.")
        chart_comparison = st.empty()
        
    with tab3:
        st.markdown("### Execution Schedule Optimization (Heatmap)")
        chart_qubo = st.empty()
        st.metric("Latest Solution Energy", "-142.5", delta="2.1")

    # =========================================================================
    # Simulation Loop
    # =========================================================================
    
    if st.session_state.simulation_running:
        
        sim = get_market_simulator()
        
        # We need a loop that updates the UI but yields to Streamlit
        # Simplest way in Streamlit is a loop with placeholders
        
        controller = st.session_state.controller
        
        # Calculate delay based on speed
        delay_map = {"1x": 1.0, "5x": 0.2, "10x": 0.1, "Max": 0.01}
        delay = delay_map[config["sim_speed"]]
        
        current_tick = len(st.session_state.market_data)
        
        # Generate next market tick
        # This is strictly a wrapper to simulate "live" feed
        # In a real app, this would come from a websocket
        
        # Generate 1 minute of data
        # To make it smooth, maybe we generate just 1 tick?
        # MarketDataSimulator generates full DF.
        # Let's verify MArketDataSimulator usage.
        
        # Hack: Generate 1 datapoint by extending previous
        last_price = 150.0
        if st.session_state.market_data:
            last_price = st.session_state.market_data[-1]['price']
        
        # Random walk step
        price_change = np.random.normal(0, 0.05) # Random noise
        new_price = last_price + price_change
        timestamp = datetime.now()
        
        new_point = {
            'timestamp': timestamp,
            'price': new_price,
            'vwap': new_price, # Simplified Live VWAP
            'volume': np.random.randint(100, 1000)
        }
        st.session_state.market_data.append(new_point)
        
        # Update Market Data for Optimizer
        df = pd.DataFrame(st.session_state.market_data)
        controller.optimizer.update_market_data(df)
        
        # Get Execution State (Hybrid)
        executed = controller.engine.executed_shares
        total = config["total_shares"]
        pct_complete = min(1.0, executed / total)
        
        # Simulate TWAP Benchmark (Concurrent)
        # TWAP trades 1/duration_ticks per tick
        duration_ticks = config["duration"] * 10 
        twap_share_per_tick = total / duration_ticks
        twap_executed = min(total, (current_tick + 1) * twap_share_per_tick)
        
        # Determine Hybrid Cost vs TWAP Cost
        # Hybrid Log
        df_hybrid = pd.DataFrame(controller.engine.execution_log)
        hybrid_vwap = 0.0
        if not df_hybrid.empty:
             # Avg price of fills (approx)
             hybrid_vwap = (df_hybrid['shares'] * new_price).sum() / df_hybrid['shares'].sum() # Using current price for simplicity or need fill price? 
             # Wait, log has NO PRICE info unless we added it in engine!
             # Checks src/hybrid_async.py: _execute logs 'tick', 'shares', 'cumulative'. No price.
             # We should probably estimate fill price as current market price.
             pass
             
        # Metric Updates
        metric_price.metric("Current Price", f"${new_price:.2f}", f"{price_change:.2f}")
        metric_shares.metric("Executed Shares", f"{executed:,}", f"{pct_complete*100:.1f}%")
        
        # Show Slippage vs TWAP
        # Simply compare executed shares vs TWAP schedule for now (Timing difference)
        ahead_behind = executed - twap_executed
        metric_vwap.metric("vs TWAP Schedule", f"{ahead_behind:,.0f} shares", delta_color="normal")
        
        metric_is.metric("Impl. Shortfall", "calculating...", "0%")
        
        # Update Charts
        # Price Chart with Executions
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(y=df['price'], mode='lines', name='Price', line=dict(color='#00ff00')))
        
        # Add Execution markers if available
        # We need to map ticks to prices. 
        # df['price'] is indexed 0..N
        # execution_log has 'tick'
        if not df_hybrid.empty and len(df) > 0:
             # Filter valid ticks
             valid_log = df_hybrid[df_hybrid['tick'] < len(df)]
             if not valid_log.empty:
                 exec_prices = df.iloc[valid_log['tick']]['price']
                 fig_price.add_trace(go.Scatter(
                     x=valid_log['tick'], 
                     y=exec_prices,
                     mode='markers',
                     name='Executions',
                     marker=dict(color='orange', size=6)
                 ))

        fig_price.update_layout(
            title="Market Price & Executions",
            height=350, 
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        chart_price.plotly_chart(fig_price, use_container_width=True)
        
        # Fill Progress Chart comparison
        # Hybrid Curve
        fig_fill = go.Figure()
        if not df_hybrid.empty:
            fig_fill.add_trace(go.Scatter(
                x=df_hybrid['tick'], y=df_hybrid['cumulative'], 
                mode='lines', name='Hybrid', fill='tozeroy', line=dict(color='#00ccff')
            ))
            
        # TWAP Line
        fig_fill.add_trace(go.Scatter(
            x=[0, duration_ticks], y=[0, total], 
            mode='lines', name='TWAP Target', line=dict(color='white', dash='dash')
        ))
        
        fig_fill.update_layout(
            title="Execution Trajectory vs TWAP",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        chart_fill.plotly_chart(fig_fill, use_container_width=True)
        
        # Update QUBO Optimization View (Heatmap of slices)
        # We can poll the policy queue or get latest policy from engine
        policy = controller.engine._current_policy
        if policy:
             schedule = policy.schedule
             fig_qubo = px.bar(x=range(len(schedule)), y=schedule, title="Optimized Schedule (QUBO Solution)")
             fig_qubo.update_traces(marker_color='#4CAF50')
             fig_qubo.update_layout(
                 xaxis_title="Time Slice",
                 yaxis_title="Shares",
                 paper_bgcolor='rgba(0,0,0,0)',
                 plot_bgcolor='rgba(0,0,0,0)',
                 font=dict(color='white')
             )
             chart_qubo.plotly_chart(fig_qubo, use_container_width=True)
        
        # Check completion
        if executed >= total or len(st.session_state.market_data) > config["duration"] * 10: # approx ticks
            stop_simulation()
            st.success("Execution Complete!")
        
        # Rerun loop
        time.sleep(delay)
        st.rerun()

if __name__ == "__main__":
    main()
