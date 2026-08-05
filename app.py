import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time

# 引入深度学习库
try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 页面配置
st.set_page_config(page_title="LSTM 神经网络币圈预测平台", layout="wide", page_icon="🧠")

# CSS 样式优化
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stMetric { background-color: #1E222D; padding: 8px; border-radius: 8px; border: 1px solid #2A2E39; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2962FF; color: white; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 1. 定义 PyTorch LSTM 神经网络模型 ---
if HAS_TORCH:
    class CryptoLSTM(nn.Module):
        def __init__(self, input_size=4, hidden_size=32, num_layers=1, output_size=1):
            super(CryptoLSTM, self).__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

# --- 2. 数据获取与特征工程 ---
@st.cache_data(ttl=5)
def get_gate_futures_klines(contract="BTC_USDT", interval="1h", limit=120):
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

# --- 3. LSTM 在线训练与预测引擎 ---
def train_and_predict_lstm(df, seq_length=10, pred_steps=3):
    if not HAS_TORCH or len(df) < 50:
        return None, 0.0

    # 1. 提取多因子特征：收盘价、成交量、RSI、MA20差值
    feature_df = df[['Close', 'Volume', 'RSI', 'MA20']].copy().dropna()
    feature_df['MA_Diff'] = feature_df['Close'] - feature_df['MA20']
    features = feature_df[['Close', 'Volume', 'RSI', 'MA_Diff']].values

    # 2. 归一化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 0]) # 预测Close

    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 3. 构建并训练模型
    model = CryptoLSTM(input_size=4, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(25): # 在线训练 25 轮
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # 4. 预测未来的步数 (Pred steps)
    model.eval()
    last_seq = scaled_data[-seq_length:].copy()
    future_preds_scaled = []

    for _ in range(pred_steps):
        input_seq = torch.tensor(last_seq[-seq_length:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_val = model(input_seq).item()
        future_preds_scaled.append(pred_val)

        # 构造下一步数据填充
        next_row = last_seq[-1].copy()
        next_row[0] = pred_val
        last_seq = np.vstack([last_seq, next_row])

    # 5. 反归一化还原价格
    dummy = np.zeros((len(future_preds_scaled), 4))
    dummy[:, 0] = future_preds_scaled
    unscaled_preds = scaler.inverse_transform(dummy)[:, 0]

    current_p = df['Close'].iloc[-1]
    predicted_change_pct = ((unscaled_preds[-1] - current_p) / current_p) * 100

    return unscaled_preds, predicted_change_pct

# --- 4. 控制面板 ---
st.sidebar.title("🧠 LSTM 控制中心")
symbol = st.sidebar.selectbox("币种", ["BTC_USDT", "ETH_USDT", "SOL_USDT"], index=0)
timeframe = st.sidebar.selectbox("周期", ["15m", "1h", "4h"], index=1)
auto_refresh = st.sidebar.checkbox("⏱️ 自动刷新 (15s)", value=False)

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
    support_val = df['Low'].tail(20).min()
    resistance_val = df['High'].tail(20).max()

    # 运行 LSTM 神经网络预测
    lstm_preds, lstm_change_pct = train_and_predict_lstm(df)

    # 顶部卡片
    col1, col2 = st.columns(2)
    col1.metric("当前实时价", f"${current_price:,.1f}", f"{change_pct:+.2f}%")
    col2.metric("MA20 强弱线", f"${ma20_val:,.1f}")

    col3, col4 = st.columns(2)
    col3.metric("RSI 指标", f"{rsi_val:.1f}")
    if lstm_preds is not None:
        col4.metric("🧠 LSTM 预测走势", f"{lstm_change_pct:+.2f}%", "看涨" if lstm_change_pct>0 else "看跌")
    else:
        col4.metric("LSTM 状态", "未激活 (需安装 torch)")

    st.markdown("---")

    # --- 🧠 深度学习 + 多因子融合分析决策 ---
    st.subheader("🧠 LSTM 深度学习 + 传统指标融合指南")

    score = 50
    reasons = []

    # LSTM 神经网络贡献度
    if lstm_preds is not None:
        if lstm_change_pct > 0.5:
            score += 25
            reasons.append(f"🤖 LSTM 神经网络预测：未来 3 根 K 线趋势看涨 (+{lstm_change_pct:.2f}%)。")
        elif lstm_change_pct < -0.5:
            score -= 25
            reasons.append(f"🤖 LSTM 神经网络预测：未来 3 根 K 线趋势看跌 ({lstm_change_pct:.2f}%)。")

    if current_price > ma20_val:
        score += 15
        reasons.append("价格站在 MA20 上方，中期多头结构完整。")
    else:
        score -= 15
        reasons.append("价格位于 MA20 下方，受制于均线压制。")

    if macd_hist > 0:
        score += 10
        reasons.append("MACD 动能柱向上放量。")
    else:
        score -= 10
        reasons.append("MACD 动能柱向下延伸。")

    if rsi_val < 38:
        score += 15
        reasons.append("RSI 探底超卖，随时可能迎来反弹。")
    elif rsi_val > 65:
        score -= 15
        reasons.append("RSI 指标偏向超买。")

    # 决策输出
    if score >= 65:
        trade_dir = "买入做多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        st.success(f"### 🟢 LSTM 神经网络共振建议：低吸开多 (综合置信度: {score}/100)")
        st.write(f"👉 **战术指南：** 深度学习模型与技术面高度看多共振！建议在支撑位 **`${entry_p}`** 附近挂限价多单。")

    elif score <= 35:
        trade_dir = "卖出做空 (SHORT)"
        entry_p = round(resistance_val * 0.999, 1)
        sl_p = round(entry_p * 1.008, 1)
        tp_p = round(entry_p - (sl_p - entry_p) * 1.8, 1)
        st.warning(f"### 🔴 LSTM 神经网络共振建议：逢高做空 / 观望 (综合置信度: {score}/100)")
        st.write(f"👉 **战术指南：** 模型预测存在下行压力！建议在高位阻力 **`${entry_p}`** 挂限价空单。")

    else:
        trade_dir = "挂单接多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        st.info(f"### 🟡 AI 建议：多空拉锯，按兵不动 (综合置信度: {score}/100)")
        st.write(f"👉 **战术指南：** 震荡格局，切勿按市价追单！安心在支撑位 **`${entry_p}`** 挂限价多单。")

    with st.expander("🔍 点击查看 LSTM 多因子逻辑细节"):
        for r in reasons:
            st.write(f"• {r}")

    # 手机一键复制
    st.markdown("---")
    st.subheader("📋 手机一键复制参数 (去 Gate 粘贴)")
    copy_str = f"方向: {trade_dir} | 开仓位: {entry_p} | 止损: {sl_p} | 止盈: {tp_p}"
    st.code(copy_str, language="text")

    # 标签页
    tab1, tab2, tab3 = st.tabs(["📈 K线 & LSTM 预测折线", "🧮 10U算仓", "📓 记账复盘"])

    with tab1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.25, 0.2])
        # K线 & 均线
        fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='历史K线'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
        
        # 绘制 LSTM 未来预测曲线 (紫色虚线)
        if lstm_preds is not None:
            last_time = df['Time'].iloc[-1]
            future_times = [last_time + timedelta(hours=i+1) for i in range(len(lstm_preds))]
            pred_times = [last_time] + future_times
            pred_values = [current_price] + list(lstm_preds)
            
            fig.add_trace(go.Scatter(x=pred_times, y=pred_values, mode='lines+markers', name='LSTM 预测未来走势', line=dict(color='#D500F9', width=3, dash='dot')), row=1, col=1)

        # MACD
        colors_macd = ['#00E676' if v >= 0 else '#FF5252' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df['Time'], y=df['MACD_Hist'], name='MACD柱', marker_color=colors_macd), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MACD'], mode='lines', name='DIF', line=dict(color='white', width=1)), row=2, col=1)

        # 成交量
        colors_vol = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='成交量', marker_color=colors_vol), row=3, col=1)

        fig.update_layout(height=550, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02))
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
            f_note = st.text_input("开仓依据", value="LSTM神经网络看涨 + 踩MA20")
            if st.form_submit_button("💾 保存记录"):
                j_df = pd.DataFrame([{"时间": datetime.now().strftime("%H:%M:%S"), "币种": symbol, "方向": f_dir, "价格": f_price, "依据": f_note}])
                j_df.to_csv("journal.csv", mode='a', header=not os.path.exists("journal.csv"), index=False, encoding='utf-8-sig')
                st.success("已保存！")

    if auto_refresh:
        time.sleep(15)
        st.rerun()
