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

# 页面基本配置
st.set_page_config(page_title="AI Crypto Workbench V8 System", layout="wide", page_icon="⚡")

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
    .risk-box { background: #1A1D24; padding: 12px; border-radius: 8px; border-left: 4px solid #FF5252; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==================== 1. EXTERNAL API LAYER (多交易所适配器) ====================
@st.cache_data(ttl=5)
def fetch_klines(exchange="Gate.io", symbol="BTC_USDT", interval="1h", limit=120):
    try:
        if exchange == "Binance":
            b_symbol = symbol.replace("_", "")
            url = f"https://fapi.binance.com/fapi/v1/klines?symbol={b_symbol}&interval={interval}&limit={limit}"
            res = requests.get(url, timeout=5).json()
            df = pd.DataFrame(res, columns=['t', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'num_trades', 'tb', 'tq', 'i'])
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
            df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            return df
        elif exchange == "OKX":
            okx_symbol = symbol.replace("_", "-") + "-SWAP"
            url = f"https://www.okx.com/api/v5/market/candles?instId={okx_symbol}&bar={interval}&limit={limit}"
            res = requests.get(url, timeout=5).json().get("data", [])
            df = pd.DataFrame(res, columns=['t', 'o', 'h', 'l', 'c', 'v', 'volCcy', 'volCcyQuote', 'confirm'])
            df['t'] = pd.to_datetime(df['t'].astype(float), unit='ms')
            for col in ['o', 'h', 'l', 'c', 'v']: df[col] = df[col].astype(float)
            df = df.iloc[::-1]
            df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            return df
        else: # Gate.io
            url = f"https://api.gateio.ws/api/v4/futures/usdt/candlesticks?contract={symbol}&interval={interval}&limit={limit}"
            res = requests.get(url, timeout=5).json()
            df = pd.DataFrame(res)
            df['t'] = pd.to_datetime(df['t'], unit='s')
            for col in ['c', 'h', 'l', 'o', 'v']: df[col] = df[col].astype(float)
            df.rename(columns={'t': 'Time', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            return df
    except Exception:
        return pd.DataFrame()

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

# Telegram Bot 推送函数
def send_telegram_alert(bot_token, chat_id, message):
    if not bot_token or not chat_id:
        return False, "请先在侧边栏配置 Telegram Bot Token 和 Chat ID！"
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        res = requests.post(url, json=payload, timeout=5)
        return (True, "推送成功！") if res.status_code == 200 else (False, f"失败: {res.text}")
    except Exception as e:
        return False, f"网络错误: {e}"

# ==================== 2. STRATEGY ENGINE & PyTorch LSTM ====================
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
    prob_bull = int((bull_count / len(top_matches)) * 100)
    prob_bear = int((bear_count / len(top_matches)) * 100)
    avg_future_norm = np.mean([m['future_norm'] for m in top_matches], axis=0)
    return {'prob_bull': prob_bull, 'prob_bear': prob_bear, 'avg_future_prices': close_prices[-1] * (1 + avg_future_norm)}

def train_and_predict_lstm(df, seq_length=10, pred_steps=3):
    if not HAS_TORCH or len(df) < 50: return None, 0.0
    feature_df = df[['Close', 'Volume', 'RSI', 'MA20']].copy().dropna()
    feature_df['MA_Diff'] = feature_df['Close'] - feature_df['MA20']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_df[['Close', 'Volume', 'RSI', 'MA_Diff']].values)
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 0])
    model = CryptoLSTM()
    criterion, optimizer = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(20):
        optimizer.zero_grad()
        loss = criterion(model(torch.tensor(np.array(X), dtype=torch.float32)), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1))
        loss.backward()
        optimizer.step()
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

# ==================== 3. USER LAYER & SIDEBAR CONTROL ====================
st.sidebar.title("⚡ V8 量化控制台")
exchange = st.sidebar.selectbox("数据源交易所", ["Gate.io", "Binance", "OKX"], index=0)
symbol = st.sidebar.selectbox("标的交易对", ["BTC_USDT", "ETH_USDT", "SOL_USDT"], index=0)
timeframe = st.sidebar.selectbox("K线周期", ["15m", "1h", "4h"], index=1)

st.sidebar.markdown("---")
st.sidebar.markdown("📱 **Telegram 机器人配置**")
tg_token = st.sidebar.text_input("Bot Token", type="password")
tg_chat_id = st.sidebar.text_input("Chat ID")
auto_refresh = st.sidebar.checkbox("⏱️ 自动刷新 (15s)", value=False)

# 加载核心数据
df = fetch_klines(exchange=exchange, symbol=symbol, interval=timeframe)
funding_rate = fetch_funding_rate(symbol=symbol)
news_list, macro_score = fetch_macro_news()

if not df.empty:
    df = calculate_indicators(df)
    current_price = df['Close'].iloc[-1]
    ma20_val = df['MA20'].iloc[-1]
    ma50_val = df['MA50'].iloc[-1]
    rsi_val = df['RSI'].iloc[-1]
    support_val = df['Low'].tail(20).min()
    resistance_val = df['High'].tail(20).max()

    lstm_preds, lstm_change_pct = train_and_predict_lstm(df)
    fractal_res = pattern_matching_forecast(df)

    # 顶栏仪表盘
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("实时价格", f"${current_price:,.1f}", f"数据源: {exchange}")
    c2.metric("资金费率", f"{funding_rate:+.4f}%", "偏热" if abs(funding_rate)>0.015 else "健康")
    c3.metric("LSTM 预测", f"{lstm_change_pct:+.2f}%" if lstm_preds is not None else "N/A")
    c4.metric("宏观情绪", f"{macro_score}", "利好" if macro_score>10 else "利空")

    st.markdown("<br/>", unsafe_allow_html=True)

    # V8 系统架构主 Tab 导航
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "📈 策略与预测引擎 (Strategy Engine)", 
        "🛡️ 风控与仓位引擎 (Risk Engine)", 
        "🤖 AI Agent 决策与推送 (Decision Center)", 
        "📓 交易与复盘数据库 (Data Layer)"
    ])

    # --- TAB 1: STRATEGY ENGINE ---
    with main_tab1:
        col_c, col_info = st.columns([2.8, 1.2])
        with col_c:
            st.subheader("📊 K线行情与双预测折线")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K线'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Time'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Time'], y=df['MA50'], mode='lines', name='MA50', line=dict(color='cyan', width=1.5)), row=1, col=1)
            
            if lstm_preds is not None:
                ft = [df['Time'].iloc[-1] + timedelta(hours=i+1) for i in range(len(lstm_preds))]
                fig.add_trace(go.Scatter(x=[df['Time'].iloc[-1]]+ft, y=[current_price]+list(lstm_preds), mode='lines+markers', name='LSTM神经网络推演', line=dict(color='#D500F9', width=2, dash='dot')), row=1, col=1)
            if fractal_res:
                ft_f = [df['Time'].iloc[-1] + timedelta(hours=i+1) for i in range(len(fractal_res['avg_future_prices']))]
                fig.add_trace(go.Scatter(x=[df['Time'].iloc[-1]]+ft_f, y=[current_price]+list(fractal_res['avg_future_prices']), mode='lines+markers', name='历史分形预测', line=dict(color='#FFEA00', width=2, dash='dash')), row=1, col=1)

            colors_vol = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='成交量', marker_color=colors_vol), row=2, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.subheader("🔍 分形与周期分析")
            if fractal_res:
                st.markdown(f"• **分形看涨概率：** `{fractal_res['prob_bull']}%`")
                st.markdown(f"• **分形看跌概率：** `{fractal_res['prob_bear']}%`")
            st.markdown("---")
            st.markdown("#### 📰 实时快讯")
            for item in news_list[:3]:
                st.markdown(f"<small>• <a href='{item['url']}' target='_blank'>{item['title']}</a></small>", unsafe_allow_html=True)

    # --- TAB 2: RISK ENGINE ---
    with main_tab2:
        st.subheader("🛡️ 独立风控与仓位计算引擎 (Risk Engine)")
        
        r_col1, r_col2 = st.columns(2)
        with r_col1:
            capital = st.number_input("账户本金 (USDT)", value=10.0, step=1.0)
            risk_pct = st.slider("允许单笔最大亏损 (%)", 1.0, 5.0, 2.0)
            leverage = st.slider("杠杆倍数", 2, 20, 5)

            # 计算风险指数 (0-100)
            risk_score = min(int(abs(funding_rate)*2000 + (leverage*3) + (100-rsi_val if current_price<ma20_val else rsi_val)*0.3), 100)
            
            st.markdown("#### ⚠️ 系统风险综合评分")
            st.progress(risk_score / 100.0)
            st.write(f"当前系统风险得分：**`{risk_score} / 100`** (" + ("🟢 安全" if risk_score<40 else ("🟡 中等" if risk_score<70 else "🔴 极高危险")) + ")")

        with r_col2:
            st.markdown("#### 🧮 动态仓位计算")
            entry_calc = round(support_val * 1.001, 1)
            sl_calc = round(entry_calc * 0.992, 1)
            price_risk_pct = abs(entry_calc - sl_calc) / entry_calc
            
            if price_risk_pct > 0:
                risk_usdt = capital * (risk_pct / 100.0)
                pos_usdt = risk_usdt / price_risk_pct
                margin_usdt = pos_usdt / leverage
                st.success(f"👉 **推荐下单保证金：`{margin_usdt:.2f} USDT`**")
                st.info(f"若触发止损，预估亏损为：`{risk_usdt:.2f} USDT`")

    # --- TAB 3: AI DECISION AGENT ---
    with main_tab3:
        st.subheader("🤖 AI Agent 决策中心与手机推送")

        # 评分计算
        total_score = 50
        reasons = []
        if current_price > ma20_val: total_score += 15; reasons.append("站稳 MA20 均线，多头掌控中短期。")
        else: total_score -= 15; reasons.append("运行于 MA20 均线下方。")

        if lstm_preds is not None and lstm_change_pct > 0.5: total_score += 20; reasons.append(f"LSTM 神经网络预测上涨 (+{lstm_change_pct:.2f}%)。")
        elif lstm_preds is not None and lstm_change_pct < -0.5: total_score -= 20; reasons.append(f"LSTM 神经网络预测下跌 ({lstm_change_pct:.2f}%)。")

        if fractal_res and fractal_res['prob_bull'] >= 60: total_score += 15; reasons.append(f"历史分形向上突破概率高达 {fractal_res['prob_bull']}%。")
        elif fractal_res and fractal_res['prob_bear'] >= 60: total_score -= 15; reasons.append(f"历史分形向下突破概率高达 {fractal_res['prob_bear']}%。")

        if total_score >= 65:
            trade_dir = "买入做多 (LONG)"
            entry_p = round(support_val * 1.001, 1)
            sl_p = round(entry_p * 0.992, 1)
            tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
            st.success(f"### 🟢 AI Agent 研判结论：买入做多 (置信度: {total_score}/100)")
        elif total_score <= 35:
            trade_dir = "卖出做空 (SHORT)"
            entry_p = round(resistance_val * 0.999, 1)
            sl_p = round(entry_p * 1.008, 1)
            tp_p = round(entry_p - (sl_p - entry_p) * 1.8, 1)
            st.warning(f"### 🔴 AI Agent 研判结论：卖出做空 / 观望 (置信度: {total_score}/100)")
        else:
            trade_dir = "挂单接多 (LONG)"
            entry_p = round(support_val * 1.001, 1)
            sl_p = round(entry_p * 0.992, 1)
            tp_p = round(entry_p + (entry_p - sl_p) * 1.8, 1)
            st.info(f"### 🟡 AI Agent 研判结论：震荡观望 (置信度: {total_score}/100)")

        # 生成交易报告
        report_text = f"""⚡ *AI Crypto Agent 交易信号推送* ⚡
----------------------------------
• **交易标的：** {symbol} ({exchange} - {timeframe})
• **研判方向：** {trade_dir}
• **建议进场点：** `${entry_p}`
• **建议止损点：** `${sl_p}`
• **建议止盈点：** `${tp_p}`
• **综合置信度：** {total_score} / 100
----------------------------------
*依据：*
""" + "\n".join([f"- {r}" for r in reasons])

        st.code(report_text, language="markdown")

        # 推送按钮
        if st.button("📱 一键推送信号到 Telegram"):
            ok, msg = send_telegram_alert(tg_token, tg_chat_id, report_text)
            if ok: st.success(msg)
            else: st.error(msg)

    # --- TAB 4: DATA LAYER & JOURNAL ---
    with main_tab4:
        st.subheader("📓 交易与复盘数据库 (Data Layer)")
        journal_file = "trade_journal.csv"

        with st.form("journal_form"):
            f_col1, f_col2 = st.columns(2)
            f_dir = f_col1.selectbox("方向", ["买入做多 (LONG)", "卖出做空 (SHORT)"])
            f_price = f_col2.number_input("成交价格", value=current_price)
            f_status = f_col1.selectbox("交易状态", ["持仓中", "止盈盈利 (Win)", "止损亏损 (Loss)"])
            f_pnl = f_col2.number_input("平仓盈亏 (USDT)", value=0.0, step=0.1)
            f_note = st.text_input("开仓依据 / 心得反思", value="AI Agent 信号")
            
            if st.form_submit_button("💾 写入交易数据库"):
                new_row = pd.DataFrame([{
                    "时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "标的": symbol,
                    "方向": f_dir,
                    "价格": f_price,
                    "状态": f_status,
                    "盈亏USDT": f_pnl,
                    "心得": f_note
                }])
                new_row.to_csv(journal_file, mode='a', header=not os.path.exists(journal_file), index=False, encoding='utf-8-sig')
                st.success("成功写入数据库！")

        if os.path.exists(journal_file):
            st.markdown("---")
            j_df = pd.read_csv(journal_file, encoding='utf-8-sig')
            st.dataframe(j_df, use_container_width=True)

    if auto_refresh:
        time.sleep(15)
        st.rerun()
