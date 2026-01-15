import streamlit as st
from env.trading_env import TradingEnv
from agent.agent import TradingAgent
import time
import pandas as pd
import numpy as np

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Autonomous Trading AI",
    page_icon="💹",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body { background-color: #0e1117; }
h1,h2,h3 { color: white; }
.metric-card {
    background:#1c1f26;
    padding:18px;
    border-radius:14px;
    text-align:center;
    color:white;
}
.log-box {
    background:#0b0e14;
    color:#00ff9c;
    padding:12px;
    border-radius:12px;
    font-family:monospace;
    height:320px;
    overflow-y:auto;
}
.footer { text-align:center; color:gray; }
</style>
""", unsafe_allow_html=True)

# ================= LOAD MARKET DATA =================
market_data = pd.read_csv("data/market_data.csv")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Control Panel")

market_speed = st.sidebar.slider(
    "🎚️ Market Speed (seconds per step)",
    min_value=0.1,
    max_value=1.0,
    value=0.3,
    step=0.1
)

session_time = st.sidebar.slider(
    "⏱️ Trading Session Duration (seconds)",
    min_value=10,
    max_value=60,
    value=20,
    step=5
)

st.sidebar.markdown("---")
st.sidebar.write("""
🧠 **AI Confidence Meter**  
Shows how confident the AI is in its decisions based on recent rewards.
""")

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>💹 Autonomous Trading AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Self-Evolving AI with Market Adaptation</p>", unsafe_allow_html=True)

st.markdown("---")

# ================= METRICS =================
c1, c2, c3, c4 = st.columns(4)
c1.markdown("<div class='metric-card'>💰<br><b>Initial Balance</b><br>$10,000</div>", unsafe_allow_html=True)
c2.markdown("<div class='metric-card'>🤖<br><b>AI Mode</b><br>Autonomous</div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'>⏱️<br><b>Session</b><br>{session_time}s</div>", unsafe_allow_html=True)
c4.markdown("<div class='metric-card'>📊<br><b>Market</b><br>Simulated</div>", unsafe_allow_html=True)

st.markdown("---")

# ================= MARKET TREND =================
st.subheader("📉 Market Price (Up & Down)")
st.line_chart(market_data["price"])

# ================= RUN BUTTON =================
start = st.button("🚀 Start Autonomous Trading")

if start:
    env = TradingEnv()
    agent = TradingAgent()
    state = env.reset()

    start_time = time.time()

    logs = ""
    step = 0

    prices = []
    equity_curve = []
    drawdown = []
    actions = []

    peak_equity = 10000
    confidence_score = []

    log_col, graph_col = st.columns([2, 3])
    log_box = log_col.empty()
    progress = log_col.progress(0)

    price_chart = graph_col.empty()
    equity_chart = graph_col.empty()
    drawdown_chart = graph_col.empty()

    status = st.info("🤖 AI Trader is running...")

    while True:
        if time.time() - start_time > session_time:
            break

        action = agent.act(state)
        state, reward, done = env.step(action)
        drift = agent.learn(reward)

        price = state[0]
        equity = state[1] + state[2] * price

        prices.append(price)
        equity_curve.append(equity)

        # Drawdown
        peak_equity = max(peak_equity, equity)
        drawdown.append((peak_equity - equity) / peak_equity)

        # AI confidence (reward stability)
        confidence_score.append(abs(reward))
        ai_confidence = np.mean(confidence_score[-10:]) if len(confidence_score) > 5 else 0

        actions.append(action)

        action_name = ["BUY", "SELL", "HOLD"][action]
        logs += f"Step {step:02d} | {action_name:<4} | Reward: {reward:.2f}\n"

        if drift:
            logs += "⚠️ Market regime change → AI adapting\n"

        log_box.markdown(f"<div class='log-box'>{logs}</div>", unsafe_allow_html=True)
        progress.progress(min(step / 100, 1.0))

        # ===== PRICE + BUY/SELL MARKERS =====
        price_df = pd.DataFrame({"Price": prices})
        price_chart.line_chart(price_df)

        # ===== EQUITY CURVE =====
        equity_chart.line_chart(pd.DataFrame({"Equity": equity_curve}))

        # ===== DRAWDOWN GRAPH =====
        drawdown_chart.line_chart(pd.DataFrame({"Drawdown": drawdown}))

        # ===== AI CONFIDENCE METER =====
        st.sidebar.metric(
            label="🧠 AI Confidence",
            value=f"{ai_confidence:.2f}"
        )

        if done:
            break

        time.sleep(market_speed)
        step += 1

    status.success("✅ Trading session completed safely")
    st.balloons()

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<div class='footer'>Agentic AI | Autonomous Trading | Concept Drift | Reinforcement Learning</div>",
    unsafe_allow_html=True
)
