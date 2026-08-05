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
st.set_page_config(page_title="AI Crypto Workbench V9.1 Fix", layout="wide", page_icon="⚡")

# CSS 机构级暗黑 UI
st.markdown("""
    <style>
    .stApp { background-color: #0B0E14; }
    section[data-testid="stSidebar"] { background-color: #12161F !important; border-right: 1px solid #1E232F !important; }
    div[data-testid="stMetric"] { background: #161A23 !important; border: 1px solid #222735 !important; border-radius: 10px !important; padding: 12px 16px !important; min-height: 95px !important; }
    div[data-testid="stMetricLabel"] > div { color: #8A93A6 !important; font-size: 0.85rem !important; }
    div[data-testid="stMetricValue"] > div { color: #F0F3FA !important; font-size: 1.4rem !important; font-weight: 700 !important; }
    button[data-baseweb="tab"] { background-color: #161A23 !important; border: 1px solid #222735 !important; border-radius: 8px !important; color: #8A93A6 !important; padding: 10px 20px !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #2962FF !important; color: #FFFFFF !important; border-color: #2962FF !important; }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. 多交易所智能修复 API ====================
@st.cache_data(ttl=5)
def fetch_klines(exchange="Gate.io", symbol="BTC_USDT", interval="1h", limit=120):
    used_exchange = exchange
    try:
        if exchange == "Binance":
            b_symbol = symbol.replace("_", "")
            url = f"https://api.binance.com/api/v3/klines?symbol={b_symbol}&interval={interval}&limit={limit}"
            res = requests.get(url, timeout=3).json()
            if isinstance(res, list) and len(res) > 0:
                df = pd.DataFrame(res, columns=['t', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'num_trades', 'tb', 'tq', 'i'])
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
                df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                return df, used_exchange

        elif exchange == "OKX":
            # 修复：OKX 必须传大写 1H, 4H, 1D
            okx_bar = interval.upper() if interval.endswith(('h', 'd')) else interval
            okx_symbol = symbol.replace("_", "-") + "-SWAP"
            url = f"https://www.okx.com/api/v5/market/candles?instId={okx_symbol}&bar={okx_bar}&limit={limit}"
            res = requests.get(url, timeout=3).json()
            data = res.get("data", [])
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data, columns=['t', 'o', 'h', 'l', 'c', 'v', 'volCcy', 'volCcyQuote', 'confirm'])
                df['t'] = pd.to_datetime(df['t'].astype(float), unit='ms')
                for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
                df = df.iloc[::-1]
                df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                return df, used_exchange
    except Exception:
        pass # 若失败，无缝降级走 Gate 备用通道

    # 智能备用通道 (Gate.io)
    try:
        used_exchange = f"{exchange} (已切至Gate备用)" if exchange != "Gate.io" else "Gate.io"
        url = f"https://api.gateio.ws/api/v4/futures/usdt/candlesticks?contract={symbol}&interval={interval}&limit={limit}"
        res = requests.get(url, timeout=5).json()
        df = pd.DataFrame(res)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        for col in ['c', 'h', 'l', 'o', 'v']: df[col] = df[col].astype(float)
        df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df, used_exchange
    except Exception:
        return pd.DataFrame(), used_exchange

@st.cache_data(ttl=60)
def fetch_funding_rate(symbol="BTC_USDT"):
    try:
        url = f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{symbol}"
        res = requests.get(url, timeout=5).json()
        return float(res.get("funding_rate", 0)) * 100
    except Exception:
        return 0.001

@st.cache_data(ttl=300)
def fetch_macro_news():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        articles = requests.get(url, timeout=5).json().get("Data", [])[:5]
        bullish = ['cut', 'rate cut', 'approval', 'etf', 'bull', 'stimulus', 'support', 'trump', 'buy']
        bearish = ['ban', 'sec', 'lawsuit', 'inflation', 'war', 'hawkish', 'sell', 'crash', 'drop']
        news_list, score = [], 0
        for a in articles:
            text = (a.get("title", "") + " " + a.get("body", "")).lower()
            s = sum(15 for w in bullish if w in text) - sum(15 for w in bearish if w in text)
            score += s
            news_list.append({"title": a.get("title", ""), "source": a.get("source_info", {}).get("name", "News"), "url": a.get("url", "#"), "score": s})
        return news_list, max(min(score, 100), -100)
    except Exception:
        return [], 0

def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id: return False, "请先配置 Telegram Bot Token 和 Chat ID！"
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        res = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}, timeout=5)
        return (True, "推送成功！") if res.status_code == 200 else (False, f"失败: {res.text}")
    except Exception as e:
        return False, f"网络错误: {e}"

# ==================== 2. AI ADVISOR ENGINES ====================
if HAS_TORCH:
    class CryptoLSTM(nn.Module):
        def __init__(self, input_size=4, hidden_size=32, num_layers=1, output_size=1):
            super(CryptoLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

def calculate_indicators(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=min(200, len(df))).mean()
    std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['MA20'] + (std * 2)
    df['BB_Lower'] = df['MA20'] - (std * 2)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def pattern_matching_forecast(df, window_size=12, forecast_horizon=4):
    if len(df) < window_size + forecast_horizon + 20: return None
    close_prices = df['Close'].values
    target = close_prices[-window_size:]
    target_norm = (target - target[0]) / target[0]
    matches = []
    for i in range(len(close_prices) - window_size - forecast_horizon):
        hist_window = close_prices[i : i + window_size]
        hist_norm = (hist_window - hist_window[0]) / hist_window[0]
        dist = np.sqrt(np.mean((target_norm - hist_norm) ** 2))
        future_window = close_prices[i + window_size : i + window_size + forecast_horizon]
        future_return = ((future_window[-1] - hist_window[-1]) / hist_window[-1]) * 100
        matches.append({'distance': dist, 'future_return': future_return, 'future_norm': (future_window - hist_window[-1]) / hist_window[-1]})
    top_matches = sorted(matches, key=lambda x: x['distance'])[:5]
    bull_count = sum(1 for m in top_matches if m['future_return'] > 0.3)
    bear_count = sum(1 for m in top_matches if m['future_return'] < -0.3)
    return {'prob_bull': int((bull_count / len(top_matches)) * 100), 'prob_bear': int((bear_count / len(top_matches)) * 100), 'avg_future_prices': close_prices[-1] * (1 + np.mean([m['future_norm'] for m in top_matches], axis=0))}

def train_and_predict_lstm(df, seq_length=10, pred_steps=3):
    if not HAS_TORCH or len(df) < 50: return None, 0.0
    feature_df = df[['Close', 'Volume', 'RSI', 'MA20']].copy().dropna()
    feature_df['MA_Diff'] = feature_df['Close'] - feature_df['MA20']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_df[['Close', 'Volume', 'RSI', 'MA_Diff']].values)
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length]); y.append(scaled_data[i+seq_length, 0])
    model = CryptoLSTM()
    criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        loss = criterion(model(torch.tensor(np.array(X), dtype=torch.float32)), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1))
        loss.backward(); optimizer.step()
    model.eval()
    last_seq = scaled_data[-seq_length:].copy()
    future_preds_scaled = []
    for _ in range(pred_steps):
        input_seq = torch.tensor(last_seq[-seq_length:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad(): pred_val = model(input_seq).item()
        future_preds_scaled.append(pred_val)
        next_row = last_seq[-1].copy(); next_row[0] = pred_val
        last_seq = np.vstack([last_seq, next_row])
    dummy = np.zeros((len(future_preds_scaled), 4)); dummy[:, 0] = future_preds_scaled
    unscaled_preds = scaler.inverse_transform(dummy)[:, 0]
    return unscaled_preds, ((unscaled_preds[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100

# ==================== 3. 侧边栏与数据抓取 ====================
st.sidebar.title("⚡ AI Advisor V9.1")
exchange = st.sidebar.selectbox("数据源", ["Gate.io", "Binance", "OKX"], index=0)
symbol = st.sidebar.selectbox("标的", ["BTC_USDT", "ETH_USDT", "SOL_USDT"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("📱 **Telegram 机器人配置**")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")
auto_refresh = st.sidebar.checkbox("⏱️ 自动刷新 (15s)", value=False)

# 获取短线 (1H) 与长线 (1D) 数据
df_short, real_ex_short = fetch_klines(exchange=exchange, symbol=symbol, interval="1h", limit=120)
df_long, real_ex_long = fetch_klines(exchange=exchange, symbol=symbol, interval="1d", limit=200)

funding_rate = fetch_funding_rate(symbol=symbol)
news_list, macro_score = fetch_macro_news()

if not df_short.empty:
    df_short = calculate_indicators(df_short)
    current_price = df_short['Close'].iloc[-1]
    ma20_val = df_short['MA20'].iloc[-1]
    rsi_short = df_short['RSI'].iloc[-1]
    support_val = df_short['Low'].tail(20).min()
    resistance_val = df_short['High'].tail(20).max()

    lstm_preds, lstm_change_pct = train_and_predict_lstm(df_short)
    fractal_res = pattern_matching_forecast(df_short)

    # 顶栏仪表盘
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("当前实时价", f"${current_price:,.1f}", f"通道: {real_ex_short}")
    c2.metric("资金费率", f"{funding_rate:+.4f}%", "偏热" if abs(funding_rate)>0.015 else "健康")
    c3.metric("短线预测 (LSTM)", f"{lstm_change_pct:+.2f}%" if lstm_preds is not None else "N/A")
    c4.metric("宏观情绪指数", f"{macro_score}", "偏鸽/利好" if macro_score>10 else "偏鹰/利空")

    st.markdown("<br/>", unsafe_allow_html=True)

    # V9 双轨主导航
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "⚡ 短线 AI Advisor (Short-Term)", 
        "🏛️ 长线 AI Advisor (Long-Term)", 
        "💼 账户与风控 (Account & Risk)", 
        "📓 复盘中心 (Journal & Analytics)"
    ])

    # ==================== TAB 1: 短线策略 AI ADVISOR ====================
    with main_tab1:
        st.subheader("⚡ 短线交易 AI Advisor (1H/4H 高频智能建议)")
        
        short_score = 50
        reasons_short = []
        if current_price > ma20_val: short_score += 15; reasons_short.append("站稳 1H MA20 均线，短线多头偏强。")
        else: short_score -= 15; reasons_short.append("低于 1H MA20 均线，受压制。")

        if lstm_preds is not None and lstm_change_pct > 0.5: short_score += 20; reasons_short.append(f"LSTM 模型预测短期上涨 +{lstm_change_pct:.2f}%。")
        elif lstm_preds is not None and lstm_change_pct < -0.5: short_score -= 20; reasons_short.append(f"LSTM 模型预测短期下跌 {lstm_change_pct:.2f}%。")

        if fractal_res and fractal_res['prob_bull'] >= 60: short_score += 15; reasons_short.append(f"历史分形向上突破概率高达 {fractal_res['prob_bull']}%。")

        if short_score >= 65:
            trade_dir = "买入做多 (LONG)"
            entry_p = round(support_val * 1.001, 1); sl_p = round(entry_p * 0.992, 1); tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
            st.success(f"### 🟢 短线 AI 建议：限价买入做多 (置信度: {short_score}/100)")
        elif short_score <= 35:
            trade_dir = "卖出做空 (SHORT)"
            entry_p = round(resistance_val * 0.999, 1); sl_p = round(entry_p * 1.008, 1); tp_p = round(entry_p - (sl_p - entry_p) * 1.8, 1)
            st.warning(f"### 🔴 短线 AI 建议：限价卖出做空 (置信度: {short_score}/100)")
        else:
            trade_dir = "挂单接多 (LONG)"
            entry_p = round(support_val * 1.001, 1); sl_p = round(entry_p * 0.992, 1); tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
            st.info(f"### 🟡 短线 AI 建议：震荡观望，回调挂多 (置信度: {short_score}/100)")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("建议进场点 (Limit)", f"${entry_p}")
        col_s2.metric("建议止损点 (SL)", f"${sl_p}")
        col_s3.metric("建议止盈点 (TP)", f"${tp_p}")
        col_s4.metric("建议杠杆", "5X 逐仓")

        fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig_s.add_trace(go.Candlestick(x=df_short['Time'], open=df_short['Open'], high=df_short['High'], low=df_short['Low'], close=df_short['Close'], name='1H K线'), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=df_short['Time'], y=df_short['MA20'], mode='lines', name='MA20', line=dict(color='orange')), row=1, col=1)
        if lstm_preds is not None:
            ft = [df_short['Time'].iloc[-1] + timedelta(hours=i+1) for i in range(len(lstm_preds))]
            fig_s.add_trace(go.Scatter(x=[df_short['Time'].iloc[-1]]+ft, y=[current_price]+list(lstm_preds), mode='lines+markers', name='LSTM推演', line=dict(color='#D500F9', dash='dot')), row=1, col=1)
        fig_s.update_layout(height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5))
        st.plotly_chart(fig_s, use_container_width=True)

        report_short = f"⚡ *短线 AI 交易信号* ⚡\n• 标的: {symbol}\n• 方向: {trade_dir}\n• 进场点: ${entry_p}\n• 止损: ${sl_p}\n• 止盈: ${tp_p}"
        if st.button("📱 推送短线信号到 Telegram"):
            ok, msg = send_telegram_alert(tg_token, tg_chat_id, report_short)
            if ok: st.success(msg)
            else: st.error(msg)

    # ==================== TAB 2: 长线配置 AI ADVISOR ====================
    with main_tab2:
        st.subheader("🏛️ 长线配置 AI Advisor (日K/周K 周期估值与定投建议)")

        if not df_long.empty:
            df_long = calculate_indicators(df_long)
            ma200_long = df_long['MA200'].iloc[-1]
            rsi_long = df_long['RSI'].iloc[-1]
            valuation_ratio = current_price / ma200_long if ma200_long > 0 else 1.0

            st.markdown("#### 📊 长线周期估值模型 (Valuation & Cycles)")
            v_col1, v_col2, v_col3 = st.columns(3)
            v_col1.metric("日线 MA200 牛熊生命线", f"${ma200_long:,.1f}")
            v_col2.metric("估值乘数 (Price/MA200)", f"{valuation_ratio:.2f} X")
            v_col3.metric("日线 RSI 相对强弱", f"{rsi_long:.1f}")

            if valuation_ratio < 0.85 or rsi_long < 35:
                long_action = "🟢 **策略：强力加仓 / 加速定投 (Heavy Accumulation)**"
                long_desc = "当前价格处于严重低估区（甚至低于 200 日均线），历史极度罕见的长期黄金底部！建议加大现货定投额度，分批抄底。"
                alloc_cash, alloc_spot = "10%", "90%"
            elif 0.85 <= valuation_ratio <= 1.8:
                long_action = "🟡 **策略：按月定投 / 观望持有 (Steady DCA)**"
                long_desc = "价格处于长线健康上升通道，估值合理。建议维持正常的现货定投节奏，不追高也不恐慌出局。"
                alloc_cash, alloc_spot = "30%", "70%"
            else:
                long_action = "🔴 **策略：分批止盈 / 逐级减仓 (Profit Taking)**"
                long_desc = "当前估值乘数进入泡沫阶段（远高于 MA200 均线），大周期超买！建议停止定投，分批将现货锁定为 USDT 避险。"
                alloc_cash, alloc_spot = "60%", "40%"

            st.info(f"### 长线 AI 核心配置建议：\n{long_action}\n\n**周期评估分析：** {long_desc}")

            c_a, c_b = st.columns(2)
            c_a.metric("建议现货持仓比例", alloc_spot)
            c_b.metric("建议 USDT 避险现金流比例", alloc_cash)

            fig_l = go.Figure()
            fig_l.add_trace(go.Candlestick(x=df_long['Time'], open=df_long['Open'], high=df_long['High'], low=df_long['Low'], close=df_long['Close'], name='日K线'))
            fig_l.add_trace(go.Scatter(x=df_long['Time'], y=df_long['MA200'], mode='lines', name='MA200 牛熊分界线', line=dict(color='purple', width=2.5)))
            fig_l.update_layout(height=450, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5))
            st.plotly_chart(fig_l, use_container_width=True)

    # ==================== TAB 3: ACCOUNT & RISK ====================
    with main_tab3:
        st.subheader("💼 账户与风控中心 (Account & Risk Engine)")
        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            capital = st.number_input("账户总本金 (USDT)", value=10.0, step=1.0)
            risk_pct = st.slider("允许单笔最大亏损 (%)", 1.0, 5.0, 2.0)
            leverage = st.slider("杠杆倍数", 2, 20, 5)

            risk_usdt = capital * (risk_pct / 100.0)
            price_risk_pct = abs(entry_p - sl_p) / entry_p
            if price_risk_pct > 0:
                pos_usdt = risk_usdt / price_risk_pct
                margin_usdt = pos_usdt / leverage
                st.success(f"👉 **Gate 合约下单保证金：`{margin_usdt:.2f} USDT`**")
                st.info(f"触发止损最大亏损：`{risk_usdt:.2f} USDT`")

        with acc_col2:
            st.markdown("#### 📋 手机 Gate.io 一键复制参数")
            copy_str = f"方向: {trade_dir} | 进场: {entry_p} | 止损: {sl_p} | 止盈: {tp_p}"
            st.code(copy_str, language="text")

    # ==================== TAB 4: JOURNAL ====================
    with main_tab4:
        st.subheader("📓 复盘中心与胜率统计 (Journal & Analytics)")
        journal_file = "trade_journal.csv"

        with st.form("journal_form"):
            f_col1, f_col2 = st.columns(2)
            f_dir = f_col1.selectbox("方向", ["买入做多 (LONG)", "卖出做空 (SHORT)"])
            f_price = f_col2.number_input("成交价格", value=current_price)
            f_status = f_col1.selectbox("交易状态", ["持仓中", "止盈盈利 (Win)", "止损亏损 (Loss)"])
            f_pnl = f_col2.number_input("平仓盈亏 (USDT)", value=0.0, step=0.1)
            f_note = st.text_input("开仓依据 / 反思", value="按照短线 AI 研判挂单")
            if st.form_submit_button("💾 保存交易日志"):
                new_row = pd.DataFrame([{"时间": datetime.now().strftime("%Y-%m-%d %H:%M"), "标的": symbol, "方向": f_dir, "价格": f_price, "状态": f_status, "盈亏USDT": f_pnl, "心得": f_note}])
                new_row.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False, encoding='utf-8-sig')
                st.success("成功保存！")

        if os.path.exists(journal_file):
            st.markdown("---")
            j_df = pd.read_csv(journal_file, encoding='utf-8-sig')
            st.dataframe(j_df, use_container_width=True)

    if auto_refresh:
        time.sleep(15)
        st.rerun()
