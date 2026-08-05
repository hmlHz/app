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
st.set_page_config(page_title="AI 交易工作台 V6 分形推演版", layout="wide", page_icon="🔍")

# CSS 样式优化
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; padding-left: 0.5rem; padding-right: 0.5rem; }
    .stMetric { background-color: #1E222D; padding: 8px; border-radius: 8px; border: 1px solid #2A2E39; }
    div[data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2962FF; color: white; font-weight: bold; }
    .scenario-box { background-color: #1A2130; padding: 10px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #2A2E39; }
    </style>
""", unsafe_allow_html=True)

# --- 1. PyTorch LSTM 模型 ---
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

# --- 2. 🔍 K线历史分形匹配与概率推演算法 ---
def pattern_matching_forecast(df, window_size=12, forecast_horizon=4):
    if len(df) < window_size + forecast_horizon + 20:
        return None

    close_prices = df['Close'].values
    target = close_prices[-window_size:]
    target_norm = (target - target[0]) / target[0]

    matches = []
    search_range = len(close_prices) - window_size - forecast_horizon

    for i in range(search_range):
        hist_window = close_prices[i : i + window_size]
        hist_norm = (hist_window - hist_window[0]) / hist_window[0]

        # 计算欧氏距离 (越小说明形态越相似)
        distance = np.sqrt(np.mean((target_norm - hist_norm) ** 2))
        
        future_window = close_prices[i + window_size : i + window_size + forecast_horizon]
        future_return = ((future_window[-1] - hist_window[-1]) / hist_window[-1]) * 100
        future_norm = (future_window - hist_window[-1]) / hist_window[-1]

        matches.append({
            'distance': distance,
            'future_return': future_return,
            'future_norm': future_norm
        })

    # 排序找到 Top 5 最相似历史形态
    matches = sorted(matches, key=lambda x: x['distance'])
    top_matches = matches[:5]

    bull_count = sum(1 for m in top_matches if m['future_return'] > 0.3)
    bear_count = sum(1 for m in top_matches if m['future_return'] < -0.3)
    side_count = len(top_matches) - bull_count - bear_count

    prob_bull = int((bull_count / len(top_matches)) * 100)
    prob_bear = int((bear_count / len(top_matches)) * 100)
    prob_side = 100 - prob_bull - prob_bear

    avg_future_norm = np.mean([m['future_norm'] for m in top_matches], axis=0)
    avg_future_prices = close_prices[-1] * (1 + avg_future_norm)

    return {
        'prob_bull': prob_bull,
        'prob_bear': prob_bear,
        'prob_side': prob_side,
        'avg_future_prices': avg_future_prices,
        'similarity': round((1 - top_matches[0]['distance']) * 100, 1)
    }

# --- 3. 宏观新闻获取 ---
@st.cache_data(ttl=300)
def get_macro_policy_news():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url, timeout=5)
        data = response.json()
        articles = data.get("Data", [])[:5]

        bullish_words = ['cut', 'rate cut', 'approval', 'etf', 'bull', 'stimulus', 'support', 'trump', 'buy', 'surge', 'dovish']
        bearish_words = ['ban', 'sec', 'lawsuit', 'inflation', 'war', 'hawkish', 'sell', 'crash', 'drop', 'investigation']

        news_list, macro_score = [], 0
        for a in articles:
            title = a.get("title", "")
            source = a.get("source_info", {}).get("name", "Macro")
            link = a.get("url", "#")
            text = (title + " " + a.get("body", "")).lower()

            item_score = sum(15 for w in bullish_words if w in text) - sum(15 for w in bearish_words if w in text)
            macro_score += item_score
            news_list.append({"title": title, "source": source, "url": link, "score": item_score})

        return news_list, max(min(macro_score, 100), -100)
    except Exception:
        return [], 0

# --- 4. K线数据获取 ---
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
    except Exception:
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

# --- 5. LSTM 训练 ---
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
    model = CryptoLSTM(input_size=4, hidden_size=32)
    criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        loss = criterion(model(torch.tensor(X, dtype=torch.float32)), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
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
    return unscaled_preds, ((unscaled_preds[-1] - current_p) / current_p) * 100

# --- 6. 界面渲染 ---
st.sidebar.title("🔍 量化控制中心")
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
    support_val = df['Low'].tail(20).min()
    resistance_val = df['High'].tail(20).max()

    lstm_preds, lstm_change_pct = train_and_predict_lstm(df)
    fractal_res = pattern_matching_forecast(df) # 运行分形匹配

    # 顶部分析卡片
    c1, c2 = st.columns(2)
    c1.metric("当前实时价", f"${current_price:,.1f}", f"{change_pct:+.2f}%")
    c2.metric("MA20 强弱线", f"${ma20_val:,.1f}")

    c3, c4 = st.columns(2)
    c3.metric("🧠 LSTM 预测", f"{lstm_change_pct:+.2f}%" if lstm_preds is not None else "未计算")
    c4.metric("🔍 历史分形最高概率", f"{fractal_res['prob_bull']}% 看涨" if fractal_res and fractal_res['prob_bull']>fractal_res['prob_bear'] else f"{fractal_res['prob_bear']}% 看跌" if fractal_res else "计算中")

    st.markdown("---")

    # --- 🔍 关键模块：历史分形比对与三大情景演算 ---
    st.subheader("🔍 K线历史形态比对与三大情景演算")
    
    if fractal_res:
        sc_col1, sc_col2, sc_col3 = st.columns(3)
        sc_col1.markdown(f"🚀 **情景A (反弹突破):** `{fractal_res['prob_bull']}%`")
        sc_col2.markdown(f"📉 **情景B (回调砸盘):** `{fractal_res['prob_bear']}%`")
        sc_col3.markdown(f"↔️ **情景C (震荡横盘):** `{fractal_res['prob_side']}%`")

        # 选出最可能发生的结论
        if fractal_res['prob_bull'] >= 50:
            primary_scenario = "🟢 **最可能发生：冲高反弹**（历史上类似形态后 5 根 K 线大多向上突破）"
        elif fractal_res['prob_bear'] >= 50:
            primary_scenario = "🔴 **最可能发生：回调下破**（历史上类似形态后 5 根 K 线大多向下砸盘）"
        else:
            primary_scenario = "🟡 **最可能发生：区间震荡**（历史上类似形态后无明显方向，处于横盘）"

        st.info(f"**分形匹配结论 (相似度 {fractal_res['similarity']}%):**\n\n{primary_scenario}")

    # --- 综合 Master 四位一体决策 ---
    st.markdown("---")
    st.subheader("⚖️ 综合 Master AI 四位一体决策")

    total_score = 50
    reasons = []

    # 1. 历史分形比对权重
    if fractal_res:
        if fractal_res['prob_bull'] >= 60:
            total_score += 20
            reasons.append(f"🔍 **历史分形共振看涨 (+20分)：** 相似历史形态向上突破概率高达 {fractal_res['prob_bull']}%。")
        elif fractal_res['prob_bear'] >= 60:
            total_score -= 20
            reasons.append(f"🔍 **历史分形共振看跌 (-20分)：** 相似历史形态向下砸盘概率高达 {fractal_res['prob_bear']}%。")

    # 2. LSTM 权重
    if lstm_preds is not None:
        if lstm_change_pct > 0.5:
            total_score += 15
            reasons.append(f"🧠 **LSTM 神经网络看涨 (+15分)：** 预测未来趋势上涨 +{lstm_change_pct:.2f}%。")
        elif lstm_change_pct < -0.5:
            total_score -= 15
            reasons.append(f"🧠 **LSTM 神经网络看跌 (-15分)：** 预测未来趋势下跌 {lstm_change_pct:.2f}%。")

    # 3. 宏观权重
    if macro_score > 15:
        total_score += 15
        reasons.append(f"🌐 **宏观政策面利好 (+15分)：** 政策情绪偏鸽。")
    elif macro_score < -15:
        total_score -= 15
        reasons.append(f"🌐 **宏观政策面利空 (-15分)：** 政策情绪偏鹰。")

    # 4. 技术面权重
    if current_price > ma20_val: total_score += 10
    else: total_score -= 10

    # 决策输出
    if total_score >= 65:
        trade_dir = "买入做多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        st.success(f"### 🟢 Master 最终决策：低吸开多 (综合置信度: {total_score}/100)")
        st.write(f"👉 **战术指南：** 历史分形 + LSTM + 宏观共振做多！建议在支撑位 **`${entry_p}`** 挂限价多单。")

    elif total_score <= 35:
        trade_dir = "卖出做空 (SHORT)"
        entry_p = round(resistance_val * 0.999, 1)
        sl_p = round(entry_p * 1.008, 1)
        tp_p = round(entry_p - (sl_p - entry_p) * 1.8, 1)
        st.warning(f"### 🔴 Master 最终决策：逢高做空 / 防守观望 (综合置信度: {total_score}/100)")
        st.write(f"👉 **战术指南：** 历史分形与模型看空！建议在阻力位 **`${entry_p}`** 挂限价空单。")

    else:
        trade_dir = "挂单接多 (LONG)"
        entry_p = round(support_val * 1.001, 1)
        sl_p = round(entry_p * 0.992, 1)
        tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
        st.info(f"### 🟡 Master 最终决策：按兵不动，挂单等待 (综合置信度: {total_score}/100)")
        st.write(f"👉 **战术指南：** 多空平衡震荡，切勿市价追单！在支撑位 **`${entry_p}`** 提前挂限价多单。")

    with st.expander("🔍 点击查看详细打分逻辑"):
        for r in reasons:
            st.write(f"• {r}")

    # 手机一键复制
    st.markdown("---")
    st.subheader("📋 手机一键复制参数 (去 Gate 粘贴)")
    copy_str = f"方向: {trade_dir} | 开仓位: {entry_p} | 止损: {sl_p} | 止盈: {tp_p}"
    st.code(copy_str, language="text")

    # 标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📈 K线 & 双预测曲线", "🌐 宏观快讯", "🧮 10U算仓", "📓 记账"])

    with tab1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.25, 0.2])
        fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='历史K线'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Time'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
        
        last_time = df['Time'].iloc[-1]
        
        # 1. 绘制 LSTM 未来预测曲线 (紫色虚线)
        if lstm_preds is not None:
            future_times = [last_time + timedelta(hours=i+1) for i in range(len(lstm_preds))]
            fig.add_trace(go.Scatter(x=[last_time]+future_times, y=[current_price]+list(lstm_preds), mode='lines+markers', name='LSTM神经网络推演', line=dict(color='#D500F9', width=2.5, dash='dot')), row=1, col=1)

        # 2. 绘制 历史分形预测曲线 (黄色虚线)
        if fractal_res is not None:
            future_times_f = [last_time + timedelta(hours=i+1) for i in range(len(fractal_res['avg_future_prices']))]
            fig.add_trace(go.Scatter(x=[last_time]+future_times_f, y=[current_price]+list(fractal_res['avg_future_prices']), mode='lines+markers', name='历史分形概率推演', line=dict(color='#FFEA00', width=2.5, dash='dash')), row=1, col=1)

        colors_macd = ['#00E676' if v >= 0 else '#FF5252' for v in df['MACD_Hist']]
        fig.add_trace(go.Bar(x=df['Time'], y=df['MACD_Hist'], name='MACD柱', marker_color=colors_macd), row=2, col=1)
        colors_vol = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='成交量', marker_color=colors_vol), row=3, col=1)

        fig.update_layout(height=540, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📰 实时全球加密宏观与政策快讯")
        if news_list:
            for item in news_list:
                s_tag = "🟢 利好" if item['score'] > 0 else ("🔴 利空" if item['score'] < 0 else "⚪ 中性")
                st.markdown(f"<div class='scenario-box'><b>{item['title']}</b><br/><small>来源: {item['source']} | 情绪: {s_tag}</small></div>", unsafe_allow_html=True)

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

    with tab4:
        st.subheader("📓 交易复盘日记")
        with st.form("journal"):
            f_dir = st.selectbox("方向", ["买入做多 (LONG)", "卖出做空 (SHORT)"])
            f_price = st.number_input("成交价", value=current_price)
            f_note = st.text_input("开仓依据", value="历史分形突破概率高+LSTM看涨")
            if st.form_submit_button("💾 保存记录"):
                j_df = pd.DataFrame([{"时间": datetime.now().strftime("%H:%M:%S"), "币种": symbol, "方向": f_dir, "价格": f_price, "依据": f_note}])
                j_df.to_csv("journal.csv", mode='a', header=not os.path.exists("journal.csv"), index=False, encoding='utf-8-sig')
                st.success("已保存！")

    if auto_refresh:
        time.sleep(15)
        st.rerun()
