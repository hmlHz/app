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
st.set_page_config(page_title="AI Crypto Workbench V7.1", layout="wide", page_icon="⚡")

# --- 深度重构 CSS 样式（修复遮挡 + 美化UI） ---
st.markdown("""
    <style>
    /* 1. 深度暗黑背景 */
    .stApp { background-color: #0B0E14; }

    /* 2. 侧边栏瘦身与美化 */
    section[data-testid="stSidebar"] {
        background-color: #12161F !important;
        border-right: 1px solid #1E232F !important;
    }

    /* 3. 彻底修复顶部 4 张卡片文字遮挡/截断问题 */
    div[data-testid="stMetric"] {
        background: #161A23 !important;
        border: 1px solid #222735 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        min-height: 95px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #8A93A6 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: #F0F3FA !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }

    /* 4. Tab 导航栏“大卡片高亮”升级 */
    div[data-baseweb="tab-list"] {
        gap: 10px !important;
        background-color: transparent !important;
        margin-bottom: 15px !important;
    }
    button[data-baseweb="tab"] {
        background-color: #161A23 !important;
        border: 1px solid #222735 !important;
        border-radius: 8px !important;
        color: #8A93A6 !important;
        padding: 10px 20px !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        border-color: #2962FF !important;
        color: #F0F3FA !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #2962FF !important;
        color: #FFFFFF !important;
        border-color: #2962FF !important;
        box-shadow: 0 4px 12px rgba(41, 98, 255, 0.3) !important;
    }

    /* 5. 新闻卡片美化 */
    .news-card {
        background: #161A23;
        border: 1px solid #222735;
        border-left: 3px solid #2962FF;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 10px;
        transition: transform 0.2s ease;
    }
    .news-card:hover {
        border-left-color: #00E676;
        transform: translateX(2px);
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. PyTorch LSTM 深度学习模型 ---
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

# --- 2. 数据 API 获取 ---
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

@st.cache_data(ttl=60)
def get_gate_contract_info(contract="BTC_USDT"):
    try:
        url = f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{contract}"
        res = requests.get(url, timeout=5).json()
        funding_rate = float(res.get("funding_rate", 0)) * 100
        return funding_rate
    except Exception:
        return 0.001

@st.cache_data(ttl=300)
def get_macro_policy_news():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        response = requests.get(url, timeout=5)
        articles = response.json().get("Data", [])[:5]

        bullish_words = ['cut', 'rate cut', 'approval', 'etf', 'bull', 'stimulus', 'support', 'trump', 'buy', 'dovish']
        bearish_words = ['ban', 'sec', 'lawsuit', 'inflation', 'war', 'hawkish', 'sell', 'crash', 'drop']

        news_list, macro_score = [], 0
        for a in articles:
            title = a.get("title", "")
            text = (title + " " + a.get("body", "")).lower()
            score = sum(15 for w in bullish_words if w in text) - sum(15 for w in bearish_words if w in text)
            macro_score += score
            news_list.append({"title": title, "source": a.get("source_info", {}).get("name", "News"), "url": a.get("url", "#"), "score": score})

        return news_list, max(min(macro_score, 100), -100)
    except Exception:
        return [], 0

# 指标计算
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

# 分形匹配算法
def pattern_matching_forecast(df, window_size=12, forecast_horizon=4):
    if len(df) < window_size + forecast_horizon + 20: return None
    close_prices = df['Close'].values
    target = close_prices[-window_size:]
    target_norm = (target - target[0]) / target[0]

    matches = []
    for i in range(len(close_prices) - window_size - forecast_horizon):
        hist_window = close_prices[i : i + window_size]
        hist_norm = (hist_window - hist_window[0]) / hist_window[0]
        distance = np.sqrt(np.mean((target_norm - hist_norm) ** 2))
        future_window = close_prices[i + window_size : i + window_size + forecast_horizon]
        future_return = ((future_window[-1] - hist_window[-1]) / hist_window[-1]) * 100
        matches.append({'distance': distance, 'future_return': future_return, 'future_norm': (future_window - hist_window[-1]) / hist_window[-1]})

    top_matches = sorted(matches, key=lambda x: x['distance'])[:5]
    bull_count = sum(1 for m in top_matches if m['future_return'] > 0.3)
    bear_count = sum(1 for m in top_matches if m['future_return'] < -0.3)
    
    prob_bull = int((bull_count / len(top_matches)) * 100)
    prob_bear = int((bear_count / len(top_matches)) * 100)
    prob_side = 100 - prob_bull - prob_bear

    avg_future_norm = np.mean([m['future_norm'] for m in top_matches], axis=0)
    return {'prob_bull': prob_bull, 'prob_bear': prob_bear, 'prob_side': prob_side, 'avg_future_prices': close_prices[-1] * (1 + avg_future_norm), 'similarity': round((1 - top_matches[0]['distance']) * 100, 1)}

# LSTM 在线训练
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

    model = CryptoLSTM(input_size=4, hidden_size=32)
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

# --- 3. 侧边栏 ---
st.sidebar.title("⚡ AI Workbench")
symbol = st.sidebar.selectbox("标的交易对", ["BTC_USDT", "ETH_USDT", "SOL_USDT"], index=0)
timeframe = st.sidebar.selectbox("K线周期", ["15m", "1h", "4h"], index=1)
auto_refresh = st.sidebar.checkbox("⏱️ 自动刷新 (15s)", value=False)

# 加载数据
df = get_gate_futures_klines(contract=symbol, interval=timeframe)
funding_rate = get_gate_contract_info(contract=symbol)
news_list, macro_score = get_macro_policy_news()

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

    # 顶栏 Dashboard (优化文字防裁剪)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("实时价格", f"${current_price:,.1f}")
    c2.metric("资金费率", f"{funding_rate:+.4f}%", "费率健康" if abs(funding_rate)<0.015 else "情绪偏热")
    c3.metric("LSTM 预测", f"{lstm_change_pct:+.2f}%" if lstm_preds is not None else "N/A")
    c4.metric("宏观情绪指数", f"{macro_score}", "利好" if macro_score>10 else ("利空" if macro_score<-10 else "中性"))

    st.markdown("<br/>", unsafe_allow_html=True)

    # --- 架构四大主功能 Tab 导航 ---
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs([
        "📊 市场周期 (Cycles)", 
        "🤖 AI 策略中心 (Strategy)", 
        "💼 我的账户 (Account)", 
        "📓 复盘中心 (Journal)"
    ])

    # ==================== TAB 1: 市场周期与牛熊判断 ====================
    with main_tab1:
        cycle_col1, cycle_col2 = st.columns([2.8, 1.2])

        with cycle_col1:
            st.subheader("🌐 市场周期与 K 线走势分析")
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K线'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Time'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['Time'], y=df['MA50'], mode='lines', name='MA50', line=dict(color='cyan', width=1.5)), row=1, col=1)
            
            # 双预测线
            if lstm_preds is not None:
                ft = [df['Time'].iloc[-1] + timedelta(hours=i+1) for i in range(len(lstm_preds))]
                fig.add_trace(go.Scatter(x=[df['Time'].iloc[-1]]+ft, y=[current_price]+list(lstm_preds), mode='lines+markers', name='LSTM神经网络预测', line=dict(color='#D500F9', width=2, dash='dot')), row=1, col=1)
            if fractal_res:
                ft_f = [df['Time'].iloc[-1] + timedelta(hours=i+1) for i in range(len(fractal_res['avg_future_prices']))]
                fig.add_trace(go.Scatter(x=[df['Time'].iloc[-1]]+ft_f, y=[current_price]+list(fractal_res['avg_future_prices']), mode='lines+markers', name='历史分形预测', line=dict(color='#FFEA00', width=2, dash='dash')), row=1, col=1)

            colors_vol = ['#00E676' if c >= o else '#FF5252' for c, o in zip(df['Close'], df['Open'])]
            fig.add_trace(go.Bar(x=df['Time'], y=df['Volume'], name='成交量', marker_color=colors_vol), row=2, col=1)
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        with cycle_col2:
            st.subheader("⚖️ 战局格局判断")
            if current_price > ma50_val:
                st.success("🟢 **大周期：多头趋势 (Bullish)**\n\n价格运行于 MA50 均线上方，属于中线买盘控盘阶段。")
            else:
                st.warning("🔴 **大周期：空头调整 (Bearish)**\n\n价格低于 MA50 均线，属于阶段性洗盘寻底阶段。")

            st.markdown("#### 📰 实时加密宏观新闻")
            if news_list:
                for item in news_list[:4]:
                    tag_color = "#00E676" if item['score'] > 0 else ("#FF5252" if item['score'] < 0 else "#8A93A6")
                    tag_text = "利好" if item['score'] > 0 else ("利空" if item['score'] < 0 else "中性")
                    st.markdown(f"""
                    <div class="news-card">
                        <div style="font-size:0.85rem; font-weight:bold; margin-bottom:4px;">
                            <a href="{item['url']}" target="_blank" style="color:#F0F3FA; text-decoration:none;">{item['title']}</a>
                        </div>
                        <div style="font-size:0.75rem; color:#8A93A6;">
                            来源: {item['source']} | 情绪: <span style="color:{tag_color}; font-weight:bold;">{tag_text}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ==================== TAB 2: AI 策略中心 & AI Agent ====================
    with main_tab2:
        st.subheader("🤖 AI 策略中心与 Agent 智能研判")

        total_score = 50
        reasons = []
        if current_price > ma20_val: total_score += 15; reasons.append("站稳 MA20 均线，中短期走势偏强。")
        else: total_score -= 15; reasons.append("受制于 MA20 均线下方。")

        if lstm_preds is not None and lstm_change_pct > 0.5: total_score += 20; reasons.append(f"LSTM 神经网络预测上涨 (+{lstm_change_pct:.2f}%)。")
        elif lstm_preds is not None and lstm_change_pct < -0.5: total_score -= 20; reasons.append(f"LSTM 神经网络预测下跌 ({lstm_change_pct:.2f}%)。")

        if fractal_res and fractal_res['prob_bull'] >= 60: total_score += 15; reasons.append(f"历史分形向上突破概率高达 {fractal_res['prob_bull']}%。")
        elif fractal_res and fractal_res['prob_bear'] >= 60: total_score -= 15; reasons.append(f"历史分形向下突破概率高达 {fractal_res['prob_bear']}%。")

        # 生成交易计划
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

        st.markdown("#### 🎯 AI 推荐交易计划 (Execution Plan)")
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric("建议进场价 (Limit)", f"${entry_p}")
        p_col2.metric("建议止损价 (SL)", f"${sl_p}")
        p_col3.metric("建议止盈价 (TP)", f"${tp_p}")

        report_text = f"""# 📄 AI Agent 加密货币交易战术报告
- **时间：** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **标的：** {symbol} ({timeframe})
- **综合判定方向：** {trade_dir}
- **计划进场点：** ${entry_p} | **止损：** ${sl_p} | **止盈：** ${tp_p}
---
### 核心依据：
""" + "\n".join([f"- {r}" for r in reasons])

        st.markdown("#### 📄 交易报告预览")
        st.code(report_text, language="markdown")

    # ==================== TAB 3: 我的账户与仓位管理 ====================
    with main_tab3:
        st.subheader("💼 我的账户与 10 USDT 仓位风控 (Position Management)")

        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            capital = st.number_input("账户总资金 (USDT)", value=10.0, step=1.0)
            risk_pct = st.slider("单笔允许最大亏损 (%)", 1.0, 5.0, 2.0)
            leverage = st.slider("打算使用的杠杆 (倍)", 2, 20, 5)

            risk_usdt = capital * (risk_pct / 100.0)
            price_risk_pct = abs(entry_p - sl_p) / entry_p
            if price_risk_pct > 0:
                pos_usdt = risk_usdt / price_risk_pct
                margin_usdt = pos_usdt / leverage
                st.success(f"👉 **Gate 下单时应拉动保证金：`{margin_usdt:.2f} USDT`**")
                st.info(f"哪怕不幸触发止损，最大亏损也仅为：`{risk_usdt:.2f} USDT`")

        with acc_col2:
            st.markdown("#### 📋 手机 Gate.io 一键复制下单参数")
            copy_str = f"方向: {trade_dir} | 进场: {entry_p} | 止损: {sl_p} | 止盈: {tp_p}"
            st.code(copy_str, language="text")
            st.caption("在手机上点击复制上方文本，粘贴去 Gate App 开限价单。")

    # ==================== TAB 4: 复盘中心与胜率统计 ====================
    with main_tab4:
        st.subheader("📓 复盘中心与胜率统计 (Trading Analytics)")

        journal_file = "trade_journal.csv"

        with st.form("journal_form"):
            f_col1, f_col2 = st.columns(2)
            f_dir = f_col1.selectbox("方向", ["买入做多 (LONG)", "卖出做空 (SHORT)"])
            f_price = f_col2.number_input("成交价格", value=current_price)
            f_status = f_col1.selectbox("交易状态", ["持仓中", "止盈盈利 (Win)", "止损亏损 (Loss)"])
            f_pnl = f_col2.number_input("平仓盈亏 (USDT)", value=0.0, step=0.1)
            f_note = st.text_input("开仓依据 / 心得反思", value="按照 AI Agent 研判挂单")
            
            if st.form_submit_button("💾 保存本笔交易"):
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
                st.success("交易日志入库成功！")

        if os.path.exists(journal_file):
            st.markdown("---")
            st.markdown("#### 📈 复盘统计概览")
            j_df = pd.read_csv(journal_file, encoding='utf-8-sig')
            
            closed_trades = j_df[j_df['状态'].isin(["止盈盈利 (Win)", "止损亏损 (Loss)"])]
            total_closed = len(closed_trades)
            
            if total_closed > 0:
                wins = len(closed_trades[closed_trades['状态'] == "止盈盈利 (Win)"])
                win_rate = (wins / total_closed) * 100
                total_pnl = closed_trades['盈亏USDT'].sum()

                s1, s2, s3 = st.columns(3)
                s1.metric("总已结交易", f"{total_closed} 笔")
                s2.metric("实战胜率", f"{win_rate:.1f}%")
                s3.metric("累计盈亏", f"{total_pnl:+.2f} USDT")

            st.dataframe(j_df, use_container_width=True)

    if auto_refresh:
        time.sleep(15)
        st.rerun()
