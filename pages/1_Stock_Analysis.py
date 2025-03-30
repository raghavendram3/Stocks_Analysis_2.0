import streamlit as st

# Configure page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="StockTrackPro - Stock Analysis",
    page_icon="📈",
    layout="wide"
)

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
# Technical analysis libraries
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Header and navigation
st.title("📈 Stock Analysis")
st.markdown("Get comprehensive financial data, technical indicators, and investment insights")

# Function to get stock data
@st.cache_data
def load_stock_data(ticker, period="1y"):
    """Load stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        return hist, info, None
    except Exception as e:
        return None, None, f"Error fetching data: {e}"

# Function to get financial ratios
def get_financial_ratios(info):
    """Extract relevant financial ratios from stock info"""
    ratios = {}
    
    # Key metrics to extract (with fallbacks for missing data)
    metrics = {
        "P/E Ratio": info.get("trailingPE", info.get("forwardPE", None)),
        "EPS": info.get("trailingEPS", None),
        "Dividend Yield (%)": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None,
        "Market Cap (B)": info.get("marketCap", 0) / 1e9 if info.get("marketCap") else None,
        "52W High": info.get("fiftyTwoWeekHigh", None),
        "52W Low": info.get("fiftyTwoWeekLow", None),
        "Price to Book": info.get("priceToBook", None),
        "Return on Equity": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
        "Debt to Equity": info.get("debtToEquity", None),
        "Quick Ratio": info.get("quickRatio", None),
        "Current Ratio": info.get("currentRatio", None),
        "Profit Margin (%)": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
        "Operating Margin (%)": info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else None,
    }
    
    # Filter out None values and format numbers
    for key, value in metrics.items():
        if value is not None:
            if isinstance(value, float):
                ratios[key] = round(value, 2)
            else:
                ratios[key] = value
                
    return ratios

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate various technical indicators for stock data"""
    # Make a copy of the dataframe to avoid modifying the original
    df_ta = df.copy()
    
    # 1. MACD (Moving Average Convergence Divergence)
    macd = MACD(
        close=df_ta["Close"],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df_ta["macd"] = macd.macd()
    df_ta["macd_signal"] = macd.macd_signal()
    df_ta["macd_histogram"] = macd.macd_diff()
    
    # 2. RSI (Relative Strength Index)
    rsi = RSIIndicator(
        close=df_ta["Close"],
        window=14
    )
    df_ta["rsi"] = rsi.rsi()
    
    # 3. Bollinger Bands
    bollinger = BollingerBands(
        close=df_ta["Close"],
        window=20,
        window_dev=2
    )
    df_ta["bollinger_mavg"] = bollinger.bollinger_mavg()
    df_ta["bollinger_hband"] = bollinger.bollinger_hband()
    df_ta["bollinger_lband"] = bollinger.bollinger_lband()
    
    # 4. Moving Averages (for Fibonacci Retracement base)
    df_ta["sma_50"] = SMAIndicator(close=df_ta["Close"], window=50).sma_indicator()
    df_ta["sma_200"] = SMAIndicator(close=df_ta["Close"], window=200).sma_indicator()
    
    # 5. Stochastic Oscillator
    stoch = StochasticOscillator(
        high=df_ta["High"],
        low=df_ta["Low"],
        close=df_ta["Close"],
        window=14,
        smooth_window=3
    )
    df_ta["stoch"] = stoch.stoch()
    df_ta["stoch_signal"] = stoch.stoch_signal()
    
    # 6. On-Balance Volume (OBV)
    obv = OnBalanceVolumeIndicator(
        close=df_ta["Close"],
        volume=df_ta["Volume"]
    )
    df_ta["obv"] = obv.on_balance_volume()
    
    # 7. Ichimoku Cloud
    if len(df_ta) >= 52:  # Ichimoku needs at least 52 periods
        ichimoku = IchimokuIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            window1=9,
            window2=26,
            window3=52
        )
        df_ta["ichimoku_a"] = ichimoku.ichimoku_a()
        df_ta["ichimoku_b"] = ichimoku.ichimoku_b()
        df_ta["ichimoku_base"] = ichimoku.ichimoku_base_line()
        df_ta["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
    
    # Fibonacci Retracement levels (based on high-low range)
    if len(df_ta) > 0:
        # Find the max and min values within the period for fibonacci calculation
        price_max = df_ta["High"].max()
        price_min = df_ta["Low"].min()
        diff = price_max - price_min
        
        # Calculate Fibonacci levels
        df_ta["fib_0"] = price_min  # 0% level
        df_ta["fib_23.6"] = price_min + 0.236 * diff
        df_ta["fib_38.2"] = price_min + 0.382 * diff
        df_ta["fib_50"] = price_min + 0.5 * diff
        df_ta["fib_61.8"] = price_min + 0.618 * diff
        df_ta["fib_100"] = price_max  # 100% level
    
    return df_ta

# Function to analyze technical indicators
def analyze_technical_indicators(df_ta):
    """Analyze technical indicators and provide signals"""
    signals = []
    
    # Need at least 30 days of data for meaningful technical analysis
    if len(df_ta) < 30:
        return ["Insufficient data for technical analysis. Need more historical data."]
    
    # Get latest values
    latest = df_ta.iloc[-1]
    prev = df_ta.iloc[-2]
    
    # Helper function to streamline indicator status checks
    def get_indicator_status(name, condition_bullish, condition_bearish):
        if condition_bullish:
            return f"🔵 {name}: Bullish signal"
        elif condition_bearish:
            return f"🔴 {name}: Bearish signal"
        return None
    
    # 1. MACD Analysis
    if "macd" in df_ta.columns and "macd_signal" in df_ta.columns:
        # MACD Crossover (MACD line crosses above signal line)
        bullish = prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]
        # MACD Crossunder (MACD line crosses below signal line)
        bearish = prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"]
        
        status = get_indicator_status("MACD", bullish, bearish)
        if status:
            signals.append(status)
        # MACD above/below zero
        elif latest["macd"] > 0 and latest["macd_signal"] > 0:
            signals.append("🔵 MACD: Both MACD and signal lines are above zero, suggesting a strong bullish trend.")
        elif latest["macd"] < 0 and latest["macd_signal"] < 0:
            signals.append("🔴 MACD: Both MACD and signal lines are below zero, suggesting a strong bearish trend.")
    
    # 2. RSI Analysis
    if "rsi" in df_ta.columns:
        if latest["rsi"] > 70:
            signals.append("⚠️ RSI: Overbought condition (RSI > 70) - the stock may be overvalued and due for a correction.")
        elif latest["rsi"] < 30:
            signals.append("🔍 RSI: Oversold condition (RSI < 30) - the stock may be undervalued and due for a rebound.")
        elif 40 <= latest["rsi"] <= 60:
            signals.append("➡️ RSI: Neutral zone (40-60) - the stock shows balanced buying and selling pressure.")
    
    # 3. Bollinger Bands Analysis
    if all(x in df_ta.columns for x in ["bollinger_hband", "bollinger_lband"]):
        close = latest["Close"]
        upper_band = latest["bollinger_hband"]
        lower_band = latest["bollinger_lband"]
        
        # Price near or above upper band
        if close >= upper_band * 0.98:
            signals.append("⚠️ Bollinger Bands: Price near or above upper band - indicates potential overbought condition.")
        # Price near or below lower band
        elif close <= lower_band * 1.02:
            signals.append("🔍 Bollinger Bands: Price near or below lower band - indicates potential oversold condition.")
        # Price in the middle
        else:
            band_width = (upper_band - lower_band) / latest["bollinger_mavg"]
            if band_width < 0.1:  # Narrow bands
                signals.append("🔔 Bollinger Bands: Bands narrowing - suggests a potential breakout or increased volatility soon.")
    
    # 4. Moving Average Analysis
    if "sma_50" in df_ta.columns and "sma_200" in df_ta.columns:
        if not pd.isna(latest["sma_50"]) and not pd.isna(latest["sma_200"]):
            # Golden Cross (50 MA crosses above 200 MA)
            bullish = prev["sma_50"] <= prev["sma_200"] and latest["sma_50"] > latest["sma_200"]
            # Death Cross (50 MA crosses below 200 MA)
            bearish = prev["sma_50"] >= prev["sma_200"] and latest["sma_50"] < latest["sma_200"]
            
            status = get_indicator_status("Moving Averages", bullish, bearish)
            if status:
                signals.append(status + (" - Golden Cross detected" if bullish else " - Death Cross detected"))
            # Price above/below both MAs
            elif latest["Close"] > latest["sma_50"] and latest["Close"] > latest["sma_200"]:
                signals.append("🔵 Moving Averages: Price above both 50-day and 200-day MAs - indicates a strong bullish trend.")
            elif latest["Close"] < latest["sma_50"] and latest["Close"] < latest["sma_200"]:
                signals.append("🔴 Moving Averages: Price below both 50-day and 200-day MAs - indicates a strong bearish trend.")
    
    # 5. Stochastic Oscillator Analysis
    if "stoch" in df_ta.columns and "stoch_signal" in df_ta.columns:
        if latest["stoch"] > 80:
            signals.append("⚠️ Stochastic Oscillator: Above 80 - indicates overbought conditions.")
        elif latest["stoch"] < 20:
            signals.append("🔍 Stochastic Oscillator: Below 20 - indicates oversold conditions.")
        
        # Stochastic Crossover
        bullish = prev["stoch"] < prev["stoch_signal"] and latest["stoch"] > latest["stoch_signal"]
        bearish = prev["stoch"] > prev["stoch_signal"] and latest["stoch"] < latest["stoch_signal"]
        
        status = get_indicator_status("Stochastic Oscillator", bullish, bearish)
        if status:
            signals.append(status + (" - %K line crossed above %D line" if bullish else " - %K line crossed below %D line"))
    
    # 6. OBV (On-Balance Volume) Analysis
    if "obv" in df_ta.columns and len(df_ta) >= 10:
        # Calculate OBV trend over past 10 days
        recent_obv = df_ta["obv"].iloc[-10:].values
        obv_trend = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
        price_trend = np.polyfit(range(10), df_ta["Close"].iloc[-10:].values, 1)[0]
        
        if obv_trend > 0 and price_trend > 0:
            signals.append("🔵 OBV: Rising OBV confirms uptrend - volume supports price movement.")
        elif obv_trend < 0 and price_trend < 0:
            signals.append("🔴 OBV: Falling OBV confirms downtrend - volume supports price movement.")
        elif obv_trend > 0 and price_trend < 0:
            signals.append("⚠️ OBV: OBV rising while price falling - potential bullish divergence.")
        elif obv_trend < 0 and price_trend > 0:
            signals.append("⚠️ OBV: OBV falling while price rising - potential bearish divergence.")
    
    # 7. Ichimoku Cloud Analysis
    cloud_cols = ["ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conversion"]
    if all(x in df_ta.columns for x in cloud_cols):
        if not pd.isna(latest["ichimoku_a"]) and not pd.isna(latest["ichimoku_b"]):
            close = latest["Close"]
            senkou_a = latest["ichimoku_a"]
            senkou_b = latest["ichimoku_b"]
            base = latest["ichimoku_base"]
            conversion = latest["ichimoku_conversion"]
            
            # Price above/below cloud
            if close > max(senkou_a, senkou_b):
                signals.append("🔵 Ichimoku Cloud: Price above the cloud - bullish signal.")
            elif close < min(senkou_a, senkou_b):
                signals.append("🔴 Ichimoku Cloud: Price below the cloud - bearish signal.")
            else:
                signals.append("➡️ Ichimoku Cloud: Price inside the cloud - indicates consolidation or indecision.")
            
            # Tenkan-sen / Kijun-sen cross
            bullish = prev["ichimoku_conversion"] < prev["ichimoku_base"] and conversion > base
            bearish = prev["ichimoku_conversion"] > prev["ichimoku_base"] and conversion < base
            
            status = get_indicator_status("Ichimoku Cloud", bullish, bearish)
            if status:
                signals.append(status + (" - Conversion line crossed above base line" if bullish else " - Conversion line crossed below base line"))
    
    # 8. Fibonacci Retracement Analysis
    if "fib_61.8" in df_ta.columns:
        close = latest["Close"]
        
        # Find the nearest Fibonacci level
        fib_levels = {
            "0%": latest["fib_0"],
            "23.6%": latest["fib_23.6"],
            "38.2%": latest["fib_38.2"],
            "50%": latest["fib_50"],
            "61.8%": latest["fib_61.8"],
            "100%": latest["fib_100"]
        }
        
        # Find closest Fibonacci level
        closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - close))
        
        signals.append(f"🔍 Fibonacci Retracement: Price is closest to the {closest_fib[0]} retracement level (${closest_fib[1]:.2f}).")
    
    # If no signals could be generated
    if not signals:
        signals.append("No clear technical signals at this time. Consider fundamental analysis or longer timeframes.")
    
    return signals

# Function to analyze stock and provide tips
def analyze_stock(hist, info, ratios):
    """Analyze stock data and provide basic investment tips"""
    tips = []
    
    # Check if we have enough data
    if hist is None or hist.empty or info is None:
        return ["Insufficient data to provide analysis."]
    
    # Calculate recent performance
    if not hist.empty and 'Close' in hist.columns:
        current_price = hist['Close'].iloc[-1]
        start_price = hist['Close'].iloc[0]
        price_change_pct = ((current_price - start_price) / start_price) * 100
        
        # Recent performance tip
        if price_change_pct > 15:
            tips.append(f"📈 Strong uptrend: The stock has risen {price_change_pct:.2f}% over the period, showing strong momentum.")
        elif price_change_pct > 5:
            tips.append(f"📈 Moderate uptrend: The stock has risen {price_change_pct:.2f}% over the period.")
        elif price_change_pct > -5:
            tips.append(f"➡️ Sideways movement: The stock has changed by {price_change_pct:.2f}% over the period, showing relative stability.")
        elif price_change_pct > -15:
            tips.append(f"📉 Moderate downtrend: The stock has fallen {abs(price_change_pct):.2f}% over the period.")
        else:
            tips.append(f"📉 Strong downtrend: The stock has fallen {abs(price_change_pct):.2f}% over the period, showing significant weakness.")
        
        # Volatility check
        volatility = hist['Close'].pct_change().std() * 100
        if volatility > 3:
            tips.append(f"⚠️ High volatility: Daily price changes average {volatility:.2f}%, indicating higher risk.")
        elif volatility < 1:
            tips.append(f"🛡️ Low volatility: Daily price changes average {volatility:.2f}%, indicating lower risk.")
    
    # P/E Ratio analysis
    pe_ratio = ratios.get("P/E Ratio")
    if pe_ratio:
        if pe_ratio < 10:
            tips.append(f"💰 Potentially undervalued: P/E ratio of {pe_ratio} is relatively low, which may indicate the stock is undervalued.")
        elif pe_ratio > 30:
            tips.append(f"💸 Potentially overvalued: P/E ratio of {pe_ratio} is relatively high, which may indicate the stock is overvalued.")
    
    # Dividend analysis
    dividend_yield = ratios.get("Dividend Yield (%)")
    if dividend_yield:
        if dividend_yield > 4:
            tips.append(f"💵 High dividend yield: {dividend_yield}% dividend yield may be attractive for income-focused investors.")
        elif dividend_yield > 0:
            tips.append(f"💵 Dividend paying: {dividend_yield}% dividend yield provides some income.")
    elif dividend_yield == 0:
        tips.append("ℹ️ No dividends: This stock doesn't currently pay dividends.")
    
    # Debt analysis
    debt_to_equity = ratios.get("Debt to Equity")
    if debt_to_equity:
        if debt_to_equity > 2:
            tips.append(f"⚠️ High debt: Debt-to-Equity ratio of {debt_to_equity} indicates significant leverage.")
        elif debt_to_equity < 0.5:
            tips.append(f"👍 Low debt: Debt-to-Equity ratio of {debt_to_equity} indicates conservative financial management.")
    
    # Profitability
    profit_margin = ratios.get("Profit Margin (%)")
    if profit_margin:
        if profit_margin > 20:
            tips.append(f"💪 High profitability: Profit margin of {profit_margin}% indicates strong business fundamentals.")
        elif profit_margin < 5:
            tips.append(f"👀 Low profitability: Profit margin of {profit_margin}% may indicate challenges in maintaining earnings.")
    
    # General disclaimer
    tips.append("⚠️ Disclaimer: These are simplified tips based on basic metrics. Always do thorough research and consider consulting a financial advisor before investing.")
    
    return tips

# Sidebar for user input
with st.sidebar:
    st.header("Stock Selection")
    ticker_input = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper()
    
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    selected_period = st.selectbox("Select Time Period", list(period_options.keys()))
    period = period_options[selected_period]
    
    st.caption("Data provided by Yahoo Finance")

# Main content
if ticker_input:
    # Load data
    with st.spinner(f"Loading data for {ticker_input}..."):
        hist, info, error = load_stock_data(ticker_input, period)
    
    if error:
        st.error(error)
    elif hist is None or hist.empty:
        st.error(f"No data found for {ticker_input}. Please check the symbol and try again.")
    else:
        # Display company info
        if info and 'longName' in info:
            st.header(info['longName'])
            
            # Company description in expander
            if 'longBusinessSummary' in info:
                with st.expander("Company Description"):
                    st.write(info['longBusinessSummary'])
            
            # Current stock price and basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = hist['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                if len(hist) > 1:
                    price_change = current_price - hist['Close'].iloc[-2]
                    price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
                    st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            
            with col3:
                if 'marketCap' in info and info['marketCap']:
                    market_cap = info['marketCap']
                    if market_cap >= 1e12:
                        market_cap_str = f"${market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"${market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"${market_cap/1e6:.2f}M"
                    else:
                        market_cap_str = f"${market_cap:.2f}"
                    st.metric("Market Cap", market_cap_str)
            
            with col4:
                if 'sector' in info and info['sector']:
                    st.metric("Sector", info['sector'])
                    
            # Calculate and display financial ratios
            ratios = get_financial_ratios(info)
            
            # Stock price chart
            st.subheader("Stock Price History")
            
            # Create Plotly chart
            fig = go.Figure()
            
            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name="OHLC"
            ))
            
            # Add volume as bar chart on secondary y-axis
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name="Volume",
                marker_color='rgba(0, 0, 255, 0.3)',
                yaxis="y2"
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{ticker_input} Stock Price and Volume",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                xaxis_rangeslider_visible=False,
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Analysis section
            st.subheader("Technical Analysis")
            
            # Calculate technical indicators
            with st.spinner("Calculating technical indicators..."):
                df_ta = calculate_technical_indicators(hist)
                
            # Create technical analysis tabs
            tech_tab1, tech_tab2, tech_tab3 = st.tabs(["Signals", "Indicators", "Charts"])
            
            with tech_tab1:
                # Technical analysis signals
                st.subheader("Technical Signals")
                
                signals = analyze_technical_indicators(df_ta)
                
                for signal in signals:
                    st.markdown(f"- {signal}")
            
            with tech_tab2:
                # Show main technical indicators
                st.subheader("Key Technical Indicators")
                
                indicators_df = pd.DataFrame({
                    'Date': df_ta.index,
                    'Close': df_ta['Close'],
                    'RSI (14)': df_ta['rsi'].round(2),
                    'MACD': df_ta['macd'].round(2),
                    'MACD Signal': df_ta['macd_signal'].round(2),
                    'Stochastic %K': df_ta['stoch'].round(2),
                    'Bollinger Upper': df_ta['bollinger_hband'].round(2),
                    'Bollinger Lower': df_ta['bollinger_lband'].round(2),
                    'SMA 50': df_ta['sma_50'].round(2),
                    'SMA 200': df_ta['sma_200'].round(2),
                }).reset_index(drop=True)
                
                st.dataframe(indicators_df.tail(10), use_container_width=True)
                
                # Download button for indicator data
                st.download_button(
                    label="Download Technical Indicators as CSV",
                    data=indicators_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{ticker_input}_technical_indicators.csv",
                    mime="text/csv"
                )
            
            with tech_tab3:
                # Display technical indicator charts
                chart_option = st.selectbox(
                    "Select Technical Indicator to Display",
                    ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Stochastic Oscillator"]
                )
                
                if chart_option == "RSI":
                    # RSI Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['rsi'], name="RSI"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=[70] * len(df_ta), name="Overbought (70)", line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=[30] * len(df_ta), name="Oversold (30)", line=dict(color='green', dash='dash')))
                    fig.update_layout(title="Relative Strength Index (RSI)", height=400, yaxis_title="RSI Value")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_option == "MACD":
                    # MACD Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd'], name="MACD"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd_signal'], name="Signal Line"))
                    fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['macd_histogram'], name="Histogram"))
                    fig.update_layout(title="MACD", height=400, yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_option == "Bollinger Bands":
                    # Bollinger Bands Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['bollinger_hband'], name="Upper Band", line=dict(dash='dash')))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['bollinger_mavg'], name="MA (20)"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['bollinger_lband'], name="Lower Band", line=dict(dash='dash')))
                    fig.update_layout(title="Bollinger Bands", height=400, yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_option == "Moving Averages":
                    # Moving Averages Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['sma_50'], name="SMA 50"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['sma_200'], name="SMA 200"))
                    fig.update_layout(title="Moving Averages", height=400, yaxis_title="Price")
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif chart_option == "Stochastic Oscillator":
                    # Stochastic Oscillator Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['stoch'], name="%K"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['stoch_signal'], name="%D"))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=[80] * len(df_ta), name="Overbought (80)", line=dict(color='red', dash='dash')))
                    fig.add_trace(go.Scatter(x=df_ta.index, y=[20] * len(df_ta), name="Oversold (20)", line=dict(color='green', dash='dash')))
                    fig.update_layout(title="Stochastic Oscillator", height=400, yaxis_title="Value")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Financial ratios in table
            st.subheader("Key Financial Metrics")
            
            if ratios:
                # Convert to DataFrame for easier display
                ratio_df = pd.DataFrame(list(ratios.items()), columns=['Metric', 'Value'])
                
                # Display metrics in columns
                cols = st.columns(3)
                items_per_col = (len(ratio_df) + 2) // 3
                
                for i, col in enumerate(cols):
                    start_idx = i * items_per_col
                    end_idx = min(start_idx + items_per_col, len(ratio_df))
                    
                    if start_idx < len(ratio_df):
                        for _, row in ratio_df.iloc[start_idx:end_idx].iterrows():
                            col.metric(row['Metric'], row['Value'])
                
                # Download button for financial metrics
                st.download_button(
                    label="Download Financial Metrics as CSV",
                    data=ratio_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{ticker_input}_financial_metrics.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No financial metrics available for this stock.")
            
            # Historical data table
            st.subheader("Historical Stock Data")
            
            # Format the dataframe for display
            display_df = hist.copy()
            display_df = display_df.reset_index()
            display_df['Date'] = display_df['Date'].dt.date
            
            for col in ['Open', 'High', 'Low', 'Close']:
                display_df[col] = display_df[col].round(2)
            
            # Only show the most recent 10 days by default
            st.dataframe(display_df.head(10), use_container_width=True)
            
            # Download button for full historical data
            st.download_button(
                label="Download Complete Historical Data as CSV",
                data=display_df.to_csv(index=False).encode('utf-8'),
                file_name=f"{ticker_input}_historical_data.csv",
                mime="text/csv"
            )
            
            # Investment tips based on analysis
            st.subheader("Investment Analysis")
            
            tips = analyze_stock(hist, info, ratios)
            
            for tip in tips:
                st.markdown(f"- {tip}")
                
        else:
            st.error(f"Cannot retrieve company information for {ticker_input}. Please check the symbol and try again.")
else:
    # Show welcome message when no ticker is entered
    st.info("👈 Enter a stock symbol in the sidebar to get started!")
    
    # Sample tickers as suggestions
    st.subheader("Popular stock symbols:")
    col1, col2, col3, col4 = st.columns(4)
    col1.write("- AAPL (Apple)")
    col1.write("- MSFT (Microsoft)")
    col2.write("- GOOGL (Alphabet)")
    col2.write("- AMZN (Amazon)")
    col3.write("- TSLA (Tesla)")
    col3.write("- META (Meta)")
    col4.write("- NVDA (NVIDIA)")
    col4.write("- JPM (JPMorgan)")