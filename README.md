# 📈 Stock Price Prediction Dashboard using LSTM

## 🔍 Project Overview
This project is a **web-based stock price prediction dashboard** built using **LSTM (Long Short-Term Memory)** neural networks.  
Users can enter a stock ticker symbol and view predicted stock prices based on historical data through an interactive web interface.

The application fetches real-time stock data, trains an LSTM model, and visualizes **actual vs predicted prices**, providing a clean UI inspired by stock trading platforms.

---

## 🚀 Key Features

- ✅ Supports **Indian (NSE, BSE)** and **US stock markets**
- 🔍 **Automatic market detection** based on user selection
- 💰 Currency formatting:
  - **₹ Indian Rupees** with comma style (₹1,23,456.78)
  - **$ USD** for US stocks
- 📊 **Real-time stock data** using Yahoo Finance
- 🤖 **LSTM-based time series prediction**
- 📉 Interactive **Actual vs Predicted** price visualization
- 🎨 Clean **white-themed UI** with deep-blue branding
- 🧭 Sidebar-based controls for better UX
- ⚡ Fast, lightweight, and easy to run

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** TensorFlow, Keras (LSTM)  
- **Data Processing:** Pandas, NumPy  
- **Data Source:** Yahoo Finance (`yfinance`)  
- **Visualization:** Matplotlib  
- **Web Framework:** Streamlit  

---

## 📂 Project Structure

stock-price-prediction-dashboard/
│
├── app.py # Complete Streamlit web application
├── requirements.txt # Python dependencies
└── README.md # Project documentation

🧪 How It Works

1. Select the market (NSE / BSE / US)
2. Enter a stock symbol (e.g., RELIANCE, TCS, AAPL)
3. The system automatically:
     Detects the market
     Applies correct Yahoo Finance ticker
     Formats currency (₹ / $)
4. Historical stock data is fetched in real time
5. Data is preprocessed and scaled
6. An LSTM model is trained on historical prices
7. Future prices are predicted and visualized

📊 Example Inputs

NSE: RELIANCE, INFY, TCS
BSE: 500325, 532540
US: AAPL, MSFT, TSLA

📈 Output

📌 Current stock price with correct currency
📉 Daily price change and percentage change
📊 Interactive graph showing:
     Training data
     Actual prices
     Predicted prices

🧠 Learning Outcomes

1. Practical understanding of LSTM for time-series forecasting
2. Handling real-world financial data
3. Market-specific data handling (NSE/BSE/US)
4. Currency localization and formatting
5. Building production-style ML dashboards
6. Integrating ML models with web applications

🚧 Limitations

1. Model uses limited training epochs for faster execution
2. Predictions are based only on historical closing prices
3. Not intended for real-time trading or investment decisions

🔮 Future Enhancements

📌 Improve model accuracy with extended training
📌 Add evaluation metrics (RMSE, MAE)
📌 Deploy the application publicly (Streamlit Cloud)
📌 Add technical indicators (RSI, Moving Averages)
📌 Enable multi-stock comparison

