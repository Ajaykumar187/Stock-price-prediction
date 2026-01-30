import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# ================= HELPER FUNCTIONS =================
def format_inr(value):
    """Format number as Indian Rupees with commas"""
    s = f"{value:,.2f}"
    parts = s.split(".")
    integer = parts[0].replace(",", "")
    if len(integer) > 3:
        integer = integer[:-3][::-1]
        integer = ",".join([integer[i:i+2] for i in range(0, len(integer), 2)])
        integer = integer[::-1] + "," + parts[0][-3:]
    else:
        integer = parts[0]
    return f"₹{integer}.{parts[1]}"

def detect_market(symbol, market):
    """Append correct Yahoo Finance suffix"""
    symbol = symbol.upper().strip()
    if market == "NSE":
        return symbol + ".NS"
    if market == "BSE":
        return symbol + ".BO"
    return symbol

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    color: #000000;
}
header, footer {visibility: hidden;}

.navbar {
    background-color: #0a2540;
    padding: 16px 30px;
    border-radius: 12px;
    margin-bottom: 25px;
}
.navbar-title {
    color: white;
    font-size: 28px;
    font-weight: bold;
}

.card {
    background-color: #f5f7fa;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    text-align: center;
}

section[data-testid="stSidebar"] {
    background-color: #0a2540;
}
section[data-testid="stSidebar"] * {
    color: white;
}

input {
    color: black !important;
    background-color: white !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0a2540, #163f73);
    color: white;
    border-radius: 30px;
    padding: 12px 26px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    box-shadow: 0px 4px 12px rgba(10,37,64,0.4);
}
.stButton > button:hover {
    transform: scale(1.05);
}

div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 15px;
    border-radius: 12px;
}
div[data-testid="metric-container"] * {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ================= NAVBAR =================
st.markdown("""
<div class="navbar">
    <span class="navbar-title">📈 Stock Prediction Dashboard</span>
</div>
""", unsafe_allow_html=True)

# ================= FEATURE CARDS =================
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='card'><h3>🤖 AI Prediction</h3><p>LSTM-based time series forecasting</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='card'><h3>🇮🇳 Indian & 🇺🇸 US Markets</h3><p>Auto market detection (NSE/BSE/US)</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='card'><h3>🎨 Clean UI</h3><p>White theme with deep blue branding</p></div>", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("📌 Stock Controls")

market = st.sidebar.radio(
    "Select Market",
    ["NSE", "BSE", "US"],
    index=0
)

symbol_input = st.sidebar.text_input(
    "Enter Stock Symbol",
    value="RELIANCE" if market != "US" else "AAPL"
)

predict_btn = st.sidebar.button("🚀 Predict Stock Price")

# ================= MAIN LOGIC =================
if predict_btn:
    try:
        stock_symbol = detect_market(symbol_input, market)
        st.info(f"📥 Downloading data for {stock_symbol} ...")

        data = yf.download(stock_symbol, start="2010-01-01")

        if data.empty:
            st.error("❌ Invalid stock symbol")
            st.stop()

        data = data[['Close']]
        data.fillna(method='ffill', inplace=True)

        latest_price = float(data['Close'].iloc[-1])
        prev_price = float(data['Close'].iloc[-2])
        change = latest_price - prev_price
        percent_change = (change / prev_price) * 100

        m1, m2, m3 = st.columns(3)

        if market == "US":
            m1.metric("💰 Current Price", f"${latest_price:,.2f}")
        else:
            m1.metric("💰 Current Price", format_inr(latest_price))

        m2.metric("📉 Change", f"{change:.2f}", f"{percent_change:.2f}%")
        m3.metric("📊 Data Points", len(data))

        # -------- LSTM PREP --------
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values)

        train_len = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_len]

        x_train, y_train = [], []
        timesteps = 60

        for i in range(timesteps, len(train_data)):
            x_train.append(train_data[i-timesteps:i, 0])
            y_train.append(train_data[i, 0])

        x_train = np.array(x_train).reshape(-1, timesteps, 1)
        y_train = np.array(y_train)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(timesteps, 1)),
            LSTM(50),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")

        st.info("🧠 Training LSTM model...")
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)

        test_data = scaled_data[train_len - timesteps:]
        x_test = []

        for i in range(timesteps, len(test_data)):
            x_test.append(test_data[i-timesteps:i, 0])

        x_test = np.array(x_test).reshape(-1, timesteps, 1)

        predictions = scaler.inverse_transform(model.predict(x_test))

        train = data[:train_len]
        valid = data[train_len:]
        valid["Predictions"] = predictions

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train['Close'], label="Training", color="#0a2540")
        ax.plot(valid['Close'], label="Actual", color="#ff9900")
        ax.plot(valid['Predictions'], label="Predicted", color="#2ecc71")
        ax.legend()
        ax.set_title("Stock Price Prediction")

        st.pyplot(fig)
        st.success("✅ Prediction completed successfully!")

    except Exception as e:
        st.error(f"❌ Error: {e}")
