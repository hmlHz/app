import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import time

# 页面配置
st.set_page_config(page_title="AI 交易工作台 V3.2 修复版", layout="wide", page_icon="📱")

# CSS 移动端优化
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stMetric { background-color: #1E222D; padding: 8px; border-radius: 8px; border: 1px solid #2A2E39; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2962FF; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 数据获取与指标计算 ---
@st.cache_data(ttl=5)
def get_gate_futures_klines(contract="BTC_USDT", interval="1h", limit=100):
    try:
        url = f"https://api.gateio.ws/api/v4/futures/usdt/candlesticks?contract={contract}&interval={interval}&limit={limit}"
        response = requests.get(url, timeout=5)
        data = response.json()
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        for col in ['c', 'h', 'l', 'o', 'v']:
            df[col] = df[col].astype(float)
        df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    return df

# --- 2. 控制面板 ---
st.sidebar.title("⚙️ 控制面板")
symbol = st.sidebar.selectbox("币种", ["BTC_USDT", "ETH_USDT", "SOL_USDT"], index=0)
timeframe = st.sidebar.selectbox("周期", ["15m", "1h", "4h"], index=1)
auto_refresh = st.sidebar.checkbox("⏱️ 开启自动刷新 (15s)", value=False)

df = get_gate_futures_klines(contract=symbol, interval=timeframe)

if not df.empty:
    df = calculate_indicators(df)
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change_pct = ((current_price - prev_price) / prev_price) * 100
    
    ma20_val = df['MA20'].iloc[-1]
    rsi_val = df['RSI'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    bb_lower = df['BB_Lower'].iloc[-1]
    bb_upper = df['BB_Upper'].iloc[-1]
    support_val = df['Low'].tail(20).min()
    resistance_val = df['High'].tail(20).max()

    # 顶部卡片
    col1, col2 = st.columns(2)
    col1.metric("当前价格", f"${current_price:,.1f}", f"{change_pct:+.2f}%")
    col2.metric("MA20 强弱线", f"${ma20_val:,.1f}")

    col3, col4 = st.columns(2)
    col3.metric("RSI 指标", f"{rsi_val:.1f}")
    col4.metric("MACD 状态", "🟢看多" if macd_hist>0 else "🔴看空")

    st.markdown("---")

    # --- 🧠 核心：AI 动态评分与方向匹配 ---
    st.subheader("🧠 AI 战局诊断与决策指南")

    score = 50
    reasons = []

    if current_price > ma20_val:
        score += 15
        reasons.append("价格站在 MA20 均线上方，多头占据中短期主导权。")
    else:
        score -= 15
        reasons.append("价格受制于 MA20 均线下方，短期上方存在抛压。")

    if macd_hist > 0:
        score += 15
        reasons.append("MACD 柱状图向上放量，多头冲劲充足。")
    else:
        score -= 15
        reasons.append("MACD 柱状图为负，向下寻找支撑动能较强。")

    if rsi_val < 38:
        score += 20
        reasons.append("RSI 探底接近超卖区间，随时可能引发触底反弹。")
    elif rsi_val > 65:
        score -= 20
        reasons.append("RSI 指标偏向高位超买，谨防追高被套。")

    if current_price <= bb_lower * 1.003:
        score += 20
        reasons.append("价格逼近布林带下轨，属于强力插针反弹区。")

    # --- 关键修复：根据得分动态计算【做多】还是【做空】点位 ---
    if score >= 65:
        trade_dir = "买入做多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        
        st.success(f"### 🟢 AI 决策建议：低吸开多 (得分: {score}/100)")
        st.write(f"👉 **战术指南：** 极佳做多机会！在附近支撑位 **`${entry_p}`** 挂限价多单，带好 0.8% 止损。")

    elif score <= 35:
        trade_dir = "卖出做空 (SHORT)"
        entry_p = round(resistance_val * 0.999, 1)
        sl_p = round(entry_p * 1.008, 1) # 止损在阻力位上方
        tp_p = round(entry_p - (sl_p - entry_p) * 1.8, 1) # 止盈在下方
        
        st.warning(f"### 🔴 AI 决策建议：逢高做空 / 观望 (得分: {score}/100)")
        st.write(f"👉 **战术指南：** 上方压制明显！建议反弹至高位阻力位 **`${entry_p}`** 附近挂限价空单，止损设在 `${sl_p}`。")

    else:
        trade_dir = "挂单接多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        
        st.info(f"### 🟡 AI 决策建议：按兵不动，挂单等待 (得分: {score}/100)")
        st.write(f"👉 **战术指南：** 价格处于震荡区域中间，切勿按市价追单！安心在支撑位 **`${entry_p}`** 提前挂限价多单。")

    with st.expander("🔍 点击查看 AI 详细逻辑分析"):
        for r in reasons:
            st.write(f"• {r}")

    # 手机一键复制点位区域（修复：动态匹配做多/做空方向）
    st.markdown("---")
    st.subheader("📋 手机一键复制参数 (已自动匹配方向)")
    copy_str = f"方向: {trade_dir} | 开仓位: {entry_p} | 止损: {sl_p} | 止盈: {tp_p}"
    st.code(copy_str, language="text")

    # 标签页
    tab1, tab2, tab3 = st.tabs(["📈 K线&MACD", "🧮 10U算仓", "📓 记账复盘"])

    with tab1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K线'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['BB_Lower'], mode='lines', name='布林下轨', line=dict(color='gray', dash='dash')), row=1, col=1)
        
        colors_macd = ['#00E676' if v >= 0 else '#FF5252' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df['Time'], y=df['MACD_Hist'], name='MACD柱', marker_color=colors_macd), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD'], mode='lines', name='DIF', line=dict(color='white', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD_Signal'], mode='lines', name='DEA', line=dict(color='yellow', width=1)), row=2, col=1)

        colors_vol = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='成交量', marker_color=colors_vol), row=3, col=1)

        fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🧮 10U 保命级仓位计算器")
        capital = st.number_input("账户本金 (USDT)", value=10.0)
        risk_pct = st.slider("单笔允许亏损 (%)", 1.0, 5.0, 2.0)
        leverage = st.slider("杠杆倍数", 2, 20, 5)
        
        risk_usdt = capital * (risk_pct / 100.0)
        price_risk_pct = abs(entry_p - sl_p) / entry_p
        
        if price_risk_pct > 0:
            pos_usdt = risk_usdt / price_risk_pct
            margin_usdt = pos_usdt / leverage
            st.success(f"👉 **Gate 下单拉动保证金：`{margin_usdt:.2f} USDT`**")
            st.info(f"若触及止损，最大亏损仅为：`{risk_usdt:.2f} USDT` (安全无痛！)")

    with tab3:
        st.subheader("📓 交易复盘日记")
        with st.form("journal"):
            f_dir = st.selectbox("方向", ["买入做多 (LONG)", "卖出做空 (SHORT)"])
            f_price = st.number_input("成交价", value=current_price)
            f_note = st.text_input("开仓依据", value="AI建议看多 + 踩支撑位")
            if st.form_submit_button("💾 保存记录"):
                j_df = pd.DataFrame([{"时间": datetime.now().strftime("%H:%M:%S"), "币种": symbol, "方向": f_dir, "价格": f_price, "依据": f_note}])
                j_df.to_csv("journal.csv", mode='a', header=not os.path.exists("journal.csv"), index=False, encoding='utf-8-sig')
                st.success("已保存！")

    if auto_refresh:
        time.sleep(15)
        st.rerun()
