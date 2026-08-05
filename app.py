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
st.set_page_config(page_title="AI 交易工作台 V5 宏观旗舰版", layout="wide", page_icon="🌐")

# CSS 样式优化
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stMetric { background-color: #1E222D; padding: 8px; border-radius: 8px; border: 1px solid #2A2E39; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2962FF; color: white; font-weight: bold; }
    .news-box { background-color: #1E222D; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 3px solid #2962FF; }
    </style>
""", unsafe_allow_html=True)

# --- 1. PyTorch LSTM 模型定义 ---
if HAS_TORCH:
    class CryptoLSTM(nn.Module):
        def __init__(self, input_size=4, hidden_size=32, num_layers=1, output_size=1):
            super(CryptoLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out

# --- 2. 实时宏观政策新闻抓取与 NLP 情绪分析 ---
@st.cache_data(ttl=300) # 缓存 5 分钟
def get_macro_policy_news():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url, timeout=5)
        data = response.json()
        articles = data.get("Data", [])[:5]

        bullish_words = ['cut', 'rate cut', 'approval', 'etf', 'bull', 'stimulus', 'support', 'trump', 'buy', 'surge', 'dovish', 'gain']
        bearish_words = ['ban', 'sec', 'lawsuit', 'inflation', 'war', 'hawkish', 'sell', 'crash', 'drop', 'investigation', 'hike']

        news_list = []
        macro_score = 0

        for a in articles:
            title = a.get("title", "")
            source = a.get("source_info", {}).get("name", "Macro")
            link = a.get("url", "#")
            text = (title + " " + a.get("body", "")).lower()

            item_score = 0
            for w in bullish_words:
                if w in text: item_score += 15
            for w in bearish_words:
                if w in text: item_score -= 15

            macro_score += item_score
            news_list.append({"title": title, "source": source, "url": link, "score": item_score})

        macro_score = max(min(macro_score, 100), -100) # 限制在 [-100, 100]
        return news_list, macro_score
    except Exception as e:
        return [], 0

# --- 3. K线数据与技术指标 ---
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

# --- 4. LSTM 预测 ---
def train_and_predict_lstm(df, seq_length=10, pred_steps=3):
    if not HAS_TORCH or len(df) < 50:
        return None, 0.0

    feature_df = df[['Close', 'Volume', 'RSI', 'MA20']].copy().dropna()
    feature_df['MA_Diff'] = feature_df['Close'] - feature_df['MA20']
    features = feature_df[['Close', 'Volume', 'RSI', 'MA_Diff']].values

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 0])

    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = CryptoLSTM(input_size=4, hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(25):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    last_seq = scaled_data[-seq_length:].copy()
    future_preds_scaled = []

    for _ in range(pred_steps):
        input_seq = torch.tensor(last_seq[-seq_length:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_val = model(input_seq).item()
        future_preds_scaled.append(pred_val)

        next_row = last_seq[-1].copy()
        next_row[0] = pred_val
        last_seq = np.vstack([last_seq, next_row])

    dummy = np.zeros((len(future_preds_scaled), 4))
    dummy[:, 0] = future_preds_scaled
    unscaled_preds = scaler.inverse_transform(dummy)[:, 0]

    current_p = df['Close'].iloc[-1]
    predicted_change_pct = ((unscaled_preds[-1] - current_p) / current_p) * 100

    return unscaled_preds, predicted_change_pct

# --- 5. 侧边栏 ---
st.sidebar.title("🌐 宏观&AI控制中心")
symbol = st.sidebar.selectbox("币种", ["BTC_USDT", "ETH_USDT", "SOL_USDT"], index=0)
timeframe = st.sidebar.selectbox("周期", ["15m", "1h", "4h"], index=1)
auto_refresh = st.sidebar.checkbox("⏱️ 自动刷新 (15s)", value=False)

df = get_gate_futures_klines(contract=symbol, interval=timeframe)
news_list, macro_score = get_macro_policy_news()

if not df.empty:
    df = calculate_indicators(df)
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    change_pct = ((current_price - prev_price) / prev_price) * 100
    
    ma20_val = df['MA20'].iloc[-1]
    rsi_val = df['RSI'].iloc[-1]
    macd_hist = df['MACD_Hist'].iloc[-1]
    support_val = df['Low'].tail(20).min()
    resistance_val = df['High'].tail(20).max()

    lstm_preds, lstm_change_pct = train_and_predict_lstm(df)

    # 顶部分析卡片
    col1, col2 = st.columns(2)
    col1.metric("当前实时价", f"${current_price:,.1f}", f"{change_pct:+.2f}%")
    col2.metric("MA20 强弱线", f"${ma20_val:,.1f}")

    col3, col4 = st.columns(2)
    col3.metric("🧠 LSTM 预测走势", f"{lstm_change_pct:+.2f}%" if lstm_preds is not None else "未计算")
    
    macro_status_str = "🟢偏鸽/利好" if macro_score > 15 else ("🔴偏鹰/利空" if macro_score < -15 else "🟡中性/观望")
    col4.metric("🌐 宏观政策情绪", macro_status_str, f"情绪指数: {macro_score}")

    st.markdown("---")

    # --- 🧠 宏观 + 深度学习 + 技术面“三位一体”总决策 ---
    st.subheader("⚖️ 宏观局势 + 深度学习 + 技术面“三位一体”总决策")

    # 权重划分: 35% 技术面 + 35% LSTM + 30% 宏观政策
    total_score = 50
    reasons = []

    # 1. 宏观政策加分项
    if macro_score > 15:
        total_score += 20
        reasons.append(f"🌐 **全球宏观政策面利好 (+20分)：** 包含降息/ETF支持等鸽派政策信号，情绪指数为 {macro_score}。")
    elif macro_score < -15:
        total_score -= 20
        reasons.append(f"🌐 **全球宏观政策面利空 (-20分)：** 包含监管/通胀/鹰派等压力信号，情绪指数为 {macro_score}。")
    else:
        reasons.append("🌐 **全球宏观政策面中性：** 暂无重大突发政策冲击。")

    # 2. LSTM 深度学习加分项
    if lstm_preds is not None:
        if lstm_change_pct > 0.5:
            total_score += 20
            reasons.append(f"🧠 **LSTM 神经网络模型看涨 (+20分)：** 推演未来 3 根 K 线呈上涨趋势 (+{lstm_change_pct:.2f}%)。")
        elif lstm_change_pct < -0.5:
            total_score -= 20
            reasons.append(f"🧠 **LSTM 神经网络模型看跌 (-20分)：** 推演未来 3 根 K 线呈下跌趋势 ({lstm_change_pct:.2f}%)。")

    # 3. 技术面加分项
    if current_price > ma20_val:
        total_score += 10
        reasons.append("📊 **技术面：** 站稳 MA20 均线上方。")
    else:
        total_score -= 10
        reasons.append("📊 **技术面：** 位于 MA20 均线下方。")

    if rsi_val < 38:
        total_score += 10
        reasons.append("📊 **技术面：** RSI 指标接近超卖。")
    elif rsi_val > 65:
        total_score -= 10
        reasons.append("📊 **技术面：** RSI 指标接近超买。")

    # 决策判定
    if total_score >= 65:
        trade_dir = "买入做多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        st.success(f"### 🟢 三位一体共振决策：低吸开多 (综合置信度得分: {total_score}/100)")
        st.write(f"👉 **战术指南：** 宏观政策 + LSTM 模型 + 技术面高度共振！建议在支撑位 **`${entry_p}`** 挂限价多单。")

    elif total_score <= 35:
        trade_dir = "卖出做空 (SHORT)"
        entry_p = round(resistance_val * 0.999, 1)
        sl_p = round(entry_p * 1.008, 1)
        tp_p = round(entry_p - (sl_p - entry_p) * 1.8, 1)
        st.warning(f"### 🔴 三位一体共振决策：逢高做空 / 防守观望 (综合置信度得分: {total_score}/100)")
        st.write(f"👉 **战术指南：** 宏观政策利空或神经网络下行！建议在阻力位 **`${entry_p}`** 挂限价空单。")

    else:
        trade_dir = "挂单接多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        st.info(f"### 🟡 三位一体共振决策：按兵不动，挂单等待 (综合置信度得分: {total_score}/100)")
        st.write(f"👉 **战术指南：** 宏观与技术面拉锯，切勿市价追单！提前在支撑位 **`${entry_p}`** 挂限价多单。")

    with st.expander("🔍 点击查看宏观与多因子逻辑细节"):
        for r in reasons:
            st.write(f"• {r}")

    # 手机一键复制
    st.markdown("---")
    st.subheader("📋 手机一键复制参数 (去 Gate 粘贴)")
    copy_str = f"方向: {trade_dir} | 开仓位: {entry_p} | 止损: {sl_p} | 止盈: {tp_p}"
    st.code(copy_str, language="text")

    # 标签页划分
    tab1, tab2, tab3, tab4 = st.tabs(["📈 K线&LSTM预测", "🌐 实时宏观政策快讯", "🧮 10U算仓", "📓 记账复盘"])

    with tab1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='历史K线'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
        
        if lstm_preds is not None:
            last_time = df['Time'].iloc[-1]
            future_times = [last_time + timedelta(hours=i+1) for i in range(len(lstm_preds))]
            pred_times = [last_time] + future_times
            pred_values = [current_price] + list(lstm_preds)
            fig.add_trace(go.Scatter(x=pred_times, y=pred_values, mode='lines+markers', name='LSTM 预测走势', line=dict(color='#D500F9', width=3, dash='dot')), row=1, col=1)

        colors_macd = ['#00E676' if v >= 0 else '#FF5252' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df['Time'], y=df['MACD_Hist'], name='MACD柱', marker_color=colors_macd), row=2, col=1)
        colors_vol = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='成交量', marker_color=colors_vol), row=3, col=1)

        fig.update_layout(height=520, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📰 实时全球加密宏观与政策快讯")
        st.caption("后台自动采集全球美联储、SEC监管、大选态度与地缘局势最新大事件：")
        if news_list:
            for item in news_list:
                s_tag = "🟢 偏鸽/利好" if item['score'] > 0 else ("🔴 偏鹰/利空" if item['score'] < 0 else "⚪ 中性")
                st.markdown(f"""
                <div class="news-box">
                    <b>{item['title']}</b><br/>
                    <small>来源: {item['source']} | 情绪判定: {s_tag}</small><br/>
                    <a href="{item['url']}" target="_blank" style="color:#2962FF; font-size:0.8rem;">查看英文原文 ↗</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂未抓取到最新政策新闻或网络请求超时。")

    with tab3:
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

    with tab4:
        st.subheader("📓 交易复盘日记")
        with st.form("journal"):
            f_dir = st.selectbox("方向", ["买入做多 (LONG)", "卖出做空 (SHORT)"])
            f_price = st.number_input("成交价", value=current_price)
            f_note = st.text_input("开仓依据", value="宏观利好+LSTM看涨")
            if st.form_submit_button("💾 保存记录"):
                j_df = pd.DataFrame([{"时间": datetime.now().strftime("%H:%M:%S"), "币种": symbol, "方向": f_dir, "价格": f_price, "依据": f_note}])
                j_df.to_csv("journal.csv", mode='a', header=not os.path.exists("journal.csv"), index=False, encoding='utf-8-sig')
                st.success("已保存！")

    if auto_refresh:
        time.sleep(15)
        st.rerun()
