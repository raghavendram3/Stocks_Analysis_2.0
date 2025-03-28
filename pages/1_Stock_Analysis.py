import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
from datetime import datetime, timedelta
# Technical analysis libraries - using only ta library, not pandas_ta
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

# Configure page
st.set_page_config(
    page_title="StockTrackPro - Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# Header and navigation
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 Stock Analysis")
        st.markdown("Get comprehensive financial data, technical indicators, and investment insights")
    
    with col2:
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            st.page_link("home.py", label="Home", icon="🏠")
        with nav_col2:
            st.page_link("pages/1_Stock_Analysis.py", label="Stock Analysis", icon="📈")

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
    
    # 1. MACD Analysis
    if "macd" in df_ta.columns and "macd_signal" in df_ta.columns:
        # MACD Crossover (MACD line crosses above signal line)
        if prev["macd"] < prev["macd_signal"] and latest["macd"] > latest["macd_signal"]:
            signals.append("🔵 MACD: Bullish signal - MACD line crossed above signal line, indicating potential upward momentum.")
        # MACD Crossunder (MACD line crosses below signal line)
        elif prev["macd"] > prev["macd_signal"] and latest["macd"] < latest["macd_signal"]:
            signals.append("🔴 MACD: Bearish signal - MACD line crossed below signal line, indicating potential downward momentum.")
        # MACD above zero
        elif latest["macd"] > 0 and latest["macd_signal"] > 0:
            signals.append("🔵 MACD: Both MACD and signal lines are above zero, suggesting a strong bullish trend.")
        # MACD below zero
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
    if "sma_50" in df_ta.columns and "sma_200" in df_ta.columns and not pd.isna(latest["sma_50"]) and not pd.isna(latest["sma_200"]):
        # Golden Cross (50 MA crosses above 200 MA)
        if prev["sma_50"] <= prev["sma_200"] and latest["sma_50"] > latest["sma_200"]:
            signals.append("🔵 Golden Cross: 50-day MA crossed above 200-day MA - widely regarded as a bullish signal.")
        # Death Cross (50 MA crosses below 200 MA)
        elif prev["sma_50"] >= prev["sma_200"] and latest["sma_50"] < latest["sma_200"]:
            signals.append("🔴 Death Cross: 50-day MA crossed below 200-day MA - widely regarded as a bearish signal.")
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
        if prev["stoch"] < prev["stoch_signal"] and latest["stoch"] > latest["stoch_signal"]:
            signals.append("🔵 Stochastic Oscillator: Bullish crossover - %K line crossed above %D line.")
        elif prev["stoch"] > prev["stoch_signal"] and latest["stoch"] < latest["stoch_signal"]:
            signals.append("🔴 Stochastic Oscillator: Bearish crossover - %K line crossed below %D line.")
    
    # 6. OBV (On-Balance Volume) Analysis
    if "obv" in df_ta.columns:
        # Calculate OBV trend over past 10 days
        if len(df_ta) >= 10:
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
    if all(x in df_ta.columns for x in ["ichimoku_a", "ichimoku_b", "ichimoku_base", "ichimoku_conversion"]):
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
            if prev["ichimoku_conversion"] < prev["ichimoku_base"] and conversion > base:
                signals.append("🔵 Ichimoku Cloud: Conversion line crossed above base line - bullish signal.")
            elif prev["ichimoku_conversion"] > prev["ichimoku_base"] and conversion < base:
                signals.append("🔴 Ichimoku Cloud: Conversion line crossed below base line - bearish signal.")
    
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
    
    # Technical Analysis settings
    st.header("Technical Analysis")
    st.write("Analyze the stock using various technical indicators:")
    
    with st.expander("About Technical Indicators"):
        st.markdown("""
        **Technical indicators** are mathematical calculations based on price, volume, or open interest of a security. They help traders identify trading opportunities and analyze market conditions.
        
        This app includes 7 powerful technical indicators, based on [Investopedia's top technical analysis tools](https://www.investopedia.com/top-7-technical-analysis-tools-4773275):
        
        1. **MACD** - Trend following momentum indicator
        2. **RSI** - Momentum oscillator measuring speed and change of price movements
        3. **Bollinger Bands** - Volatility bands placed above and below a moving average
        4. **Moving Averages** - Smoothed average price data with Golden/Death Cross signals
        5. **Stochastic Oscillator** - Momentum indicator comparing closing price to price range
        6. **On-Balance Volume** - Volume indicator that relates volume to price change
        7. **Ichimoku Cloud** - Collection of indicators showing support, resistance and trend
        8. **Fibonacci Retracement** - Price levels indicating potential support/resistance
        """)
    
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
            
            # Technical Analysis Section
            st.subheader("Technical Analysis")
            
            # Calculate all technical indicators
            with st.spinner("Calculating technical indicators..."):
                df_ta = calculate_technical_indicators(hist)
                
            # Tab sections for different technical indicators
            tabs = st.tabs(["Overview", "MACD", "RSI", "Bollinger Bands", "Moving Averages", "Stochastic", "OBV", "Ichimoku Cloud", "Fibonacci"])
            
            with tabs[0]:  # Overview Tab
                # Technical analysis signals
                st.subheader("Technical Signals")
                signals = analyze_technical_indicators(df_ta)
                for signal in signals:
                    st.markdown(f"- {signal}")
                
                # Summary of indicators with colored status
                st.subheader("Indicators Overview")
                
                # Create indicator status grid
                col1, col2, col3 = st.columns(3)
                
                # Function to get indicator status and color
                def get_indicator_status(name, condition_bullish, condition_bearish):
                    if condition_bullish:
                        return f"**{name}**: 🟢 Bullish"
                    elif condition_bearish:
                        return f"**{name}**: 🔴 Bearish"
                    else:
                        return f"**{name}**: 🟡 Neutral"
                
                latest = df_ta.iloc[-1]
                
                # MACD Status
                if "macd" in df_ta.columns and "macd_signal" in df_ta.columns:
                    macd_bullish = latest["macd"] > latest["macd_signal"]
                    macd_bearish = latest["macd"] < latest["macd_signal"]
                    col1.markdown(get_indicator_status("MACD", macd_bullish, macd_bearish))
                
                # RSI Status
                if "rsi" in df_ta.columns:
                    rsi_bullish = 40 <= latest["rsi"] <= 60 or latest["rsi"] < 30
                    rsi_bearish = latest["rsi"] > 70
                    col1.markdown(get_indicator_status("RSI", rsi_bullish, rsi_bearish))
                
                # Bollinger Bands Status
                if all(x in df_ta.columns for x in ["bollinger_hband", "bollinger_lband"]):
                    bb_bullish = latest["Close"] <= latest["bollinger_lband"] * 1.02
                    bb_bearish = latest["Close"] >= latest["bollinger_hband"] * 0.98
                    col1.markdown(get_indicator_status("Bollinger Bands", bb_bullish, bb_bearish))
                
                # Moving Averages Status
                if "sma_50" in df_ta.columns and "sma_200" in df_ta.columns:
                    ma_bullish = latest["Close"] > latest["sma_50"] and latest["sma_50"] > latest["sma_200"]
                    ma_bearish = latest["Close"] < latest["sma_50"] and latest["sma_50"] < latest["sma_200"]
                    col2.markdown(get_indicator_status("Moving Averages", ma_bullish, ma_bearish))
                
                # Stochastic Status
                if "stoch" in df_ta.columns and "stoch_signal" in df_ta.columns:
                    stoch_bullish = latest["stoch"] < 20 or (latest["stoch"] > latest["stoch_signal"] and latest["stoch"] < 80)
                    stoch_bearish = latest["stoch"] > 80 or (latest["stoch"] < latest["stoch_signal"] and latest["stoch"] > 20)
                    col2.markdown(get_indicator_status("Stochastic", stoch_bullish, stoch_bearish))
                
                # OBV Status
                if "obv" in df_ta.columns and len(df_ta) >= 10:
                    recent_obv = df_ta["obv"].iloc[-10:].values
                    recent_close = df_ta["Close"].iloc[-10:].values
                    obv_trend = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
                    price_trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
                    obv_bullish = obv_trend > 0
                    obv_bearish = obv_trend < 0
                    col2.markdown(get_indicator_status("OBV", obv_bullish, obv_bearish))
                
                # Ichimoku Status
                if all(x in df_ta.columns for x in ["ichimoku_a", "ichimoku_b"]):
                    if not pd.isna(latest["ichimoku_a"]) and not pd.isna(latest["ichimoku_b"]):
                        cloud_bullish = latest["Close"] > max(latest["ichimoku_a"], latest["ichimoku_b"])
                        cloud_bearish = latest["Close"] < min(latest["ichimoku_a"], latest["ichimoku_b"])
                        col3.markdown(get_indicator_status("Ichimoku Cloud", cloud_bullish, cloud_bearish))
                
                # Fibonacci Status - just show nearest level
                if "fib_61.8" in df_ta.columns:
                    fib_levels = {
                        "0%": latest["fib_0"],
                        "23.6%": latest["fib_23.6"],
                        "38.2%": latest["fib_38.2"],
                        "50%": latest["fib_50"],
                        "61.8%": latest["fib_61.8"],
                        "100%": latest["fib_100"]
                    }
                    closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - latest["Close"]))
                    col3.markdown(f"**Fibonacci**: Nearest level {closest_fib[0]} (${closest_fib[1]:.2f})")
            
            # MACD Tab
            with tabs[1]:
                st.subheader("Moving Average Convergence Divergence (MACD)")
                st.write("""
                MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
                - **MACD Line**: Difference between 12-period and 26-period EMAs
                - **Signal Line**: 9-period EMA of the MACD Line
                - **Histogram**: Difference between MACD Line and Signal Line
                """)
                
                # Create MACD Chart
                if "macd" in df_ta.columns:
                    fig_macd = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                               vertical_spacing=0.1, row_heights=[0.7, 0.3])
                    
                    # Add price to top subplot
                    fig_macd.add_trace(go.Candlestick(
                        x=df_ta.index,
                        open=df_ta['Open'],
                        high=df_ta['High'],
                        low=df_ta['Low'],
                        close=df_ta['Close'],
                        name="Price"
                    ), row=1, col=1)
                    
                    # Add MACD to bottom subplot
                    fig_macd.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['macd'],
                        name="MACD",
                        line=dict(color='blue', width=2)
                    ), row=2, col=1)
                    
                    fig_macd.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['macd_signal'],
                        name="Signal",
                        line=dict(color='red', width=1)
                    ), row=2, col=1)
                    
                    # Add histogram as bar chart
                    colors = ['green' if val >= 0 else 'red' for val in df_ta['macd_histogram']]
                    fig_macd.add_trace(go.Bar(
                        x=df_ta.index,
                        y=df_ta['macd_histogram'],
                        name="Histogram",
                        marker_color=colors
                    ), row=2, col=1)
                    
                    # Update layout
                    fig_macd.update_layout(
                        title=f"{ticker_input} Price and MACD",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True)
                    
                    # MACD Interpretation
                    latest_macd = df_ta['macd'].iloc[-1]
                    latest_signal = df_ta['macd_signal'].iloc[-1]
                    
                    st.subheader("Current MACD Status")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("MACD Line", f"{latest_macd:.3f}")
                    col2.metric("Signal Line", f"{latest_signal:.3f}")
                    col3.metric("Histogram", f"{latest_macd - latest_signal:.3f}")
                    
                    # MACD Interpretation
                    if latest_macd > latest_signal:
                        st.success("🔵 Bullish Signal: MACD is above the signal line, indicating upward momentum")
                    else:
                        st.error("🔴 Bearish Signal: MACD is below the signal line, indicating downward momentum")
                else:
                    st.warning("Insufficient data to calculate MACD")
                
            # RSI Tab
            with tabs[2]:
                st.subheader("Relative Strength Index (RSI)")
                st.write("""
                RSI measures the speed and magnitude of a security's price movements to evaluate overvalued or undervalued conditions.
                - **RSI > 70**: Typically indicates overbought conditions (potential sell signal)
                - **RSI < 30**: Typically indicates oversold conditions (potential buy signal)
                - **RSI 40-60**: Indicates neutral market conditions
                """)
                
                if "rsi" in df_ta.columns:
                    # Create RSI Chart
                    fig_rsi = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                              vertical_spacing=0.1, row_heights=[0.7, 0.3])
                    
                    # Add price to top subplot
                    fig_rsi.add_trace(go.Candlestick(
                        x=df_ta.index,
                        open=df_ta['Open'],
                        high=df_ta['High'],
                        low=df_ta['Low'],
                        close=df_ta['Close'],
                        name="Price"
                    ), row=1, col=1)
                    
                    # Add RSI to bottom subplot
                    fig_rsi.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['rsi'],
                        name="RSI",
                        line=dict(color='purple', width=2)
                    ), row=2, col=1)
                    
                    # Add reference lines at 30 and 70
                    fig_rsi.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
                    fig_rsi.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                    
                    # Update layout
                    fig_rsi.update_layout(
                        title=f"{ticker_input} Price and RSI",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True)
                    
                    # RSI Status
                    latest_rsi = df_ta['rsi'].iloc[-1]
                    st.subheader("Current RSI Status")
                    
                    # Display the current RSI value with color indicating its state
                    if latest_rsi > 70:
                        st.error(f"⚠️ Overbought - RSI: {latest_rsi:.2f}")
                    elif latest_rsi < 30:
                        st.success(f"🔍 Oversold - RSI: {latest_rsi:.2f}")
                    else:
                        st.info(f"➡️ Neutral - RSI: {latest_rsi:.2f}")
                else:
                    st.warning("Insufficient data to calculate RSI")
                    
            # Bollinger Bands Tab
            with tabs[3]:
                st.subheader("Bollinger Bands")
                st.write("""
                Bollinger Bands consist of three lines:
                - **Middle Band**: 20-period simple moving average (SMA)
                - **Upper Band**: 20-period SMA + (2 × 20-period standard deviation)
                - **Lower Band**: 20-period SMA - (2 × 20-period standard deviation)
                
                The bands expand and contract based on volatility. Prices reaching the bands can indicate:
                - Price reaching upper band may suggest overbought conditions
                - Price reaching lower band may suggest oversold conditions
                - Narrow bands indicate low volatility, often preceding significant price movements
                """)
                
                if "bollinger_mavg" in df_ta.columns:
                    # Create Bollinger Bands Chart
                    fig_bb = go.Figure()
                    
                    # Add price as candlestick
                    fig_bb.add_trace(go.Candlestick(
                        x=df_ta.index,
                        open=df_ta['Open'],
                        high=df_ta['High'],
                        low=df_ta['Low'],
                        close=df_ta['Close'],
                        name="Price"
                    ))
                    
                    # Add Bollinger Bands
                    fig_bb.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['bollinger_mavg'],
                        name="Middle Band (SMA 20)",
                        line=dict(color='blue', width=1)
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['bollinger_hband'],
                        name="Upper Band",
                        line=dict(color='green', width=1)
                    ))
                    
                    fig_bb.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['bollinger_lband'],
                        name="Lower Band",
                        line=dict(color='red', width=1)
                    ))
                    
                    # Update layout
                    fig_bb.update_layout(
                        title=f"{ticker_input} with Bollinger Bands",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_bb, use_container_width=True)
                    
                    # Bollinger Bands Status
                    latest_close = df_ta['Close'].iloc[-1]
                    latest_upper = df_ta['bollinger_hband'].iloc[-1]
                    latest_middle = df_ta['bollinger_mavg'].iloc[-1]
                    latest_lower = df_ta['bollinger_lband'].iloc[-1]
                    
                    # Calculate band width as percentage
                    band_width = ((latest_upper - latest_lower) / latest_middle) * 100
                    
                    # Current status
                    st.subheader("Current Bollinger Bands Status")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Upper Band", f"${latest_upper:.2f}")
                    col2.metric("Middle Band", f"${latest_middle:.2f}")
                    col3.metric("Lower Band", f"${latest_lower:.2f}")
                    
                    st.metric("Bollinger Band Width", f"{band_width:.2f}%")
                    
                    # Position relative to bands
                    if latest_close >= latest_upper * 0.98:
                        st.error("⚠️ Price is near or above upper band - potential overbought condition")
                    elif latest_close <= latest_lower * 1.02:
                        st.success("🔍 Price is near or below lower band - potential oversold condition")
                    else:
                        position = (latest_close - latest_lower) / (latest_upper - latest_lower) * 100
                        st.info(f"➡️ Price is {position:.1f}% of the way between lower and upper bands")
                    
                    # Band width interpretation
                    if band_width < 10:
                        st.warning("🔔 Bollinger Bands are narrow - this often precedes a significant price move")
                else:
                    st.warning("Insufficient data to calculate Bollinger Bands")
                
            # Moving Averages Tab
            with tabs[4]:
                st.subheader("Moving Averages & Golden/Death Cross")
                st.write("""
                Moving averages smooth price data to form a trend-following indicator:
                - **50-day MA**: Short-term trend indicator 
                - **200-day MA**: Long-term trend indicator
                
                Key signals:
                - **Golden Cross**: 50-day MA crosses above 200-day MA (bullish)
                - **Death Cross**: 50-day MA crosses below 200-day MA (bearish)
                - Price above both MAs suggests bullish momentum
                - Price below both MAs suggests bearish momentum
                """)
                
                if "sma_50" in df_ta.columns and "sma_200" in df_ta.columns:
                    # Create Moving Averages Chart
                    fig_ma = go.Figure()
                    
                    # Add price
                    fig_ma.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['Close'],
                        name="Price",
                        line=dict(color='black', width=1)
                    ))
                    
                    # Add Moving Averages
                    fig_ma.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['sma_50'],
                        name="50-day MA",
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_ma.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['sma_200'],
                        name="200-day MA",
                        line=dict(color='red', width=2)
                    ))
                    
                    # Update layout
                    fig_ma.update_layout(
                        title=f"{ticker_input} with 50-day and 200-day Moving Averages",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_ma, use_container_width=True)
                    
                    # Moving Averages Status
                    latest_close = df_ta['Close'].iloc[-1]
                    latest_sma50 = df_ta['sma_50'].iloc[-1]
                    latest_sma200 = df_ta['sma_200'].iloc[-1]
                    
                    # Golden/Death Cross detection
                    prev_diff = df_ta['sma_50'].iloc[-2] - df_ta['sma_200'].iloc[-2]
                    curr_diff = latest_sma50 - latest_sma200
                    
                    st.subheader("Current Moving Averages Status")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${latest_close:.2f}")
                    col2.metric("50-day MA", f"${latest_sma50:.2f}")
                    col3.metric("200-day MA", f"${latest_sma200:.2f}")
                    
                    # Check for Golden/Death Cross
                    if prev_diff <= 0 and curr_diff > 0:
                        st.success("🔵 Golden Cross detected! 50-day MA crossed above 200-day MA (bullish signal)")
                    elif prev_diff >= 0 and curr_diff < 0:
                        st.error("🔴 Death Cross detected! 50-day MA crossed below 200-day MA (bearish signal)")
                    
                    # Price position relative to MAs
                    if latest_close > latest_sma50 and latest_close > latest_sma200:
                        st.success("🔵 Price is above both 50-day and 200-day MAs - bullish trend")
                    elif latest_close < latest_sma50 and latest_close < latest_sma200:
                        st.error("🔴 Price is below both 50-day and 200-day MAs - bearish trend")
                    elif latest_close > latest_sma50 and latest_close < latest_sma200:
                        st.info("➡️ Price is above 50-day MA but below 200-day MA - potential bullish momentum building")
                    elif latest_close < latest_sma50 and latest_close > latest_sma200:
                        st.warning("⚠️ Price is below 50-day MA but above 200-day MA - potential short-term weakness in bullish trend")
                    
                    # MA relationship
                    if latest_sma50 > latest_sma200:
                        st.success("🔵 50-day MA is above 200-day MA - long-term uptrend")
                    else:
                        st.error("🔴 50-day MA is below 200-day MA - long-term downtrend")
                else:
                    st.warning("Insufficient data to calculate Moving Averages")
            
            # Stochastic Oscillator Tab
            with tabs[5]:
                st.subheader("Stochastic Oscillator")
                st.write("""
                The Stochastic Oscillator compares a security's closing price to its price range over a specific period.
                - **%K Line**: The current value of the stochastic indicator
                - **%D Line**: 3-period moving average of %K
                - **> 80**: Generally indicates overbought conditions
                - **< 20**: Generally indicates oversold conditions
                - Crossovers between %K and %D lines can signal trading opportunities
                """)
                
                if "stoch" in df_ta.columns and "stoch_signal" in df_ta.columns:
                    # Create Stochastic Oscillator Chart
                    fig_stoch = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                                vertical_spacing=0.1, row_heights=[0.7, 0.3])
                    
                    # Add price to top subplot
                    fig_stoch.add_trace(go.Candlestick(
                        x=df_ta.index,
                        open=df_ta['Open'],
                        high=df_ta['High'],
                        low=df_ta['Low'],
                        close=df_ta['Close'],
                        name="Price"
                    ), row=1, col=1)
                    
                    # Add Stochastic to bottom subplot
                    fig_stoch.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['stoch'],
                        name="%K Line",
                        line=dict(color='blue', width=2)
                    ), row=2, col=1)
                    
                    fig_stoch.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['stoch_signal'],
                        name="%D Line",
                        line=dict(color='red', width=1)
                    ), row=2, col=1)
                    
                    # Add reference lines at 20 and 80
                    fig_stoch.add_hline(y=20, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
                    fig_stoch.add_hline(y=80, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
                    
                    # Update layout
                    fig_stoch.update_layout(
                        title=f"{ticker_input} Price and Stochastic Oscillator",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_stoch, use_container_width=True)
                    
                    # Stochastic Status
                    latest_k = df_ta['stoch'].iloc[-1]
                    latest_d = df_ta['stoch_signal'].iloc[-1]
                    
                    st.subheader("Current Stochastic Oscillator Status")
                    col1, col2 = st.columns(2)
                    col1.metric("%K Line", f"{latest_k:.2f}")
                    col2.metric("%D Line", f"{latest_d:.2f}")
                    
                    # Interpretation
                    if latest_k > 80:
                        st.error("⚠️ Overbought - Stochastic %K > 80")
                    elif latest_k < 20:
                        st.success("🔍 Oversold - Stochastic %K < 20")
                    else:
                        st.info("➡️ Neutral zone - Stochastic %K between 20-80")
                    
                    # Crossover detection
                    prev_k = df_ta['stoch'].iloc[-2]
                    prev_d = df_ta['stoch_signal'].iloc[-2]
                    
                    if prev_k < prev_d and latest_k > latest_d:
                        st.success("🔵 Bullish crossover - %K line crossed above %D line")
                    elif prev_k > prev_d and latest_k < latest_d:
                        st.error("🔴 Bearish crossover - %K line crossed below %D line")
                else:
                    st.warning("Insufficient data to calculate Stochastic Oscillator")
            
            # OBV Tab
            with tabs[6]:
                st.subheader("On-Balance Volume (OBV)")
                st.write("""
                On-Balance Volume is a momentum indicator that uses volume flow to predict changes in stock price.
                
                When OBV moves with price:
                - Upward OBV trend with rising price confirms uptrend
                - Downward OBV trend with falling price confirms downtrend
                
                When OBV diverges from price:
                - OBV rises while price falls suggests potential bullish reversal
                - OBV falls while price rises suggests potential bearish reversal
                """)
                
                if "obv" in df_ta.columns:
                    # Create OBV Chart
                    fig_obv = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                              vertical_spacing=0.1, row_heights=[0.7, 0.3])
                    
                    # Add price to top subplot
                    fig_obv.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['Close'],
                        name="Price",
                        line=dict(color='black', width=1)
                    ), row=1, col=1)
                    
                    # Add OBV to bottom subplot
                    fig_obv.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['obv'],
                        name="OBV",
                        line=dict(color='purple', width=2)
                    ), row=2, col=1)
                    
                    # Update layout
                    fig_obv.update_layout(
                        title=f"{ticker_input} Price and On-Balance Volume",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_obv, use_container_width=True)
                    
                    # OBV Analysis
                    if len(df_ta) >= 10:
                        # Calculate trends over past 10 days
                        recent_obv = df_ta["obv"].iloc[-10:].values
                        recent_close = df_ta["Close"].iloc[-10:].values
                        
                        obv_trend = np.polyfit(range(len(recent_obv)), recent_obv, 1)[0]
                        price_trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
                        
                        st.subheader("Recent OBV Trend Analysis")
                        
                        # Display OBV and price trends with arrows
                        col1, col2 = st.columns(2)
                        obv_trend_str = "↗️ Rising" if obv_trend > 0 else "↘️ Falling"
                        price_trend_str = "↗️ Rising" if price_trend > 0 else "↘️ Falling"
                        
                        col1.metric("OBV Trend", obv_trend_str)
                        col2.metric("Price Trend", price_trend_str)
                        
                        # Interpretation based on OBV and price trends
                        if obv_trend > 0 and price_trend > 0:
                            st.success("🔵 Bullish confirmation - OBV and price are both rising, confirming the uptrend")
                        elif obv_trend < 0 and price_trend < 0:
                            st.error("🔴 Bearish confirmation - OBV and price are both falling, confirming the downtrend")
                        elif obv_trend > 0 and price_trend < 0:
                            st.info("⚠️ Bullish divergence - OBV rising while price falling may suggest a potential upward reversal")
                        elif obv_trend < 0 and price_trend > 0:
                            st.warning("⚠️ Bearish divergence - OBV falling while price rising may suggest a potential downward reversal")
                    else:
                        st.warning("Insufficient data for OBV trend analysis")
                else:
                    st.warning("Insufficient data to calculate On-Balance Volume")
            
            # Ichimoku Cloud Tab
            with tabs[7]:
                st.subheader("Ichimoku Cloud")
                st.write("""
                The Ichimoku Cloud is a collection of technical indicators that show support and resistance, momentum, and trend direction. It consists of:
                - **Conversion Line (Tenkan-sen)**: 9-period average of high+low/2
                - **Base Line (Kijun-sen)**: 26-period average of high+low/2
                - **Leading Span A (Senkou Span A)**: Average of Conversion and Base Lines, plotted 26 periods ahead
                - **Leading Span B (Senkou Span B)**: 52-period average of high+low/2, plotted 26 periods ahead
                - **Lagging Span (Chikou Span)**: Current closing price, plotted 26 periods behind
                
                Key signals:
                - Price above the cloud is bullish
                - Price below the cloud is bearish
                - Price inside the cloud indicates consolidation
                - Cloud color (green when A > B, red when B > A) indicates trend bias
                """)
                
                if "ichimoku_a" in df_ta.columns and "ichimoku_b" in df_ta.columns:
                    # Create Ichimoku Cloud Chart
                    fig_ichimoku = go.Figure()
                    
                    # Create cloud shape by filling area between Senkou Span A and B
                    upper = np.maximum(df_ta['ichimoku_a'], df_ta['ichimoku_b'])
                    lower = np.minimum(df_ta['ichimoku_a'], df_ta['ichimoku_b'])
                    
                    # Add candlestick chart
                    fig_ichimoku.add_trace(go.Candlestick(
                        x=df_ta.index,
                        open=df_ta['Open'],
                        high=df_ta['High'],
                        low=df_ta['Low'],
                        close=df_ta['Close'],
                        name="Price"
                    ))
                    
                    # Add cloud fill
                    for i in range(len(df_ta)):
                        if not pd.isna(df_ta['ichimoku_a'].iloc[i]) and not pd.isna(df_ta['ichimoku_b'].iloc[i]):
                            # Green cloud when Senkou Span A >= Senkou Span B (bullish)
                            # Red cloud when Senkou Span A < Senkou Span B (bearish)
                            color = 'rgba(0, 255, 0, 0.1)' if df_ta['ichimoku_a'].iloc[i] >= df_ta['ichimoku_b'].iloc[i] else 'rgba(255, 0, 0, 0.1)'
                            
                            fig_ichimoku.add_trace(go.Scatter(
                                x=[df_ta.index[i], df_ta.index[i]],
                                y=[lower[i], upper[i]],
                                fill=None,
                                mode='lines',
                                line=dict(color=color, width=0),
                                showlegend=False
                            ))
                    
                    # Add Ichimoku lines
                    fig_ichimoku.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['ichimoku_conversion'],
                        name="Conversion Line (Tenkan-sen)",
                        line=dict(color='blue', width=1)
                    ))
                    
                    fig_ichimoku.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['ichimoku_base'],
                        name="Base Line (Kijun-sen)",
                        line=dict(color='red', width=1)
                    ))
                    
                    fig_ichimoku.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['ichimoku_a'],
                        name="Leading Span A (Senkou Span A)",
                        line=dict(color='green', width=1)
                    ))
                    
                    fig_ichimoku.add_trace(go.Scatter(
                        x=df_ta.index,
                        y=df_ta['ichimoku_b'],
                        name="Leading Span B (Senkou Span B)",
                        line=dict(color='purple', width=1)
                    ))
                    
                    # Update layout
                    fig_ichimoku.update_layout(
                        title=f"{ticker_input} with Ichimoku Cloud",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_ichimoku, use_container_width=True)
                    
                    # Ichimoku Cloud Status
                    if not pd.isna(df_ta['ichimoku_a'].iloc[-1]) and not pd.isna(df_ta['ichimoku_b'].iloc[-1]):
                        latest_close = df_ta['Close'].iloc[-1]
                        latest_conversion = df_ta['ichimoku_conversion'].iloc[-1]
                        latest_base = df_ta['ichimoku_base'].iloc[-1]
                        latest_span_a = df_ta['ichimoku_a'].iloc[-1]
                        latest_span_b = df_ta['ichimoku_b'].iloc[-1]
                        
                        st.subheader("Current Ichimoku Cloud Status")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Conversion Line", f"${latest_conversion:.2f}")
                        col2.metric("Base Line", f"${latest_base:.2f}")
                        col3.metric("Current Price", f"${latest_close:.2f}")
                        
                        col1, col2 = st.columns(2)
                        col1.metric("Span A", f"${latest_span_a:.2f}")
                        col2.metric("Span B", f"${latest_span_b:.2f}")
                        
                        # Cloud color
                        cloud_bullish = latest_span_a >= latest_span_b
                        cloud_color = "Green (Bullish)" if cloud_bullish else "Red (Bearish)"
                        st.metric("Cloud Color", cloud_color)
                        
                        # Price position relative to cloud
                        if latest_close > max(latest_span_a, latest_span_b):
                            st.success("🔵 Price is above the cloud - bullish signal")
                        elif latest_close < min(latest_span_a, latest_span_b):
                            st.error("🔴 Price is below the cloud - bearish signal")
                        else:
                            st.warning("➡️ Price is inside the cloud - indicating indecision or consolidation")
                        
                        # Conversion/Base Line cross
                        if latest_conversion > latest_base:
                            st.success("🔵 Conversion Line is above Base Line - bullish signal")
                        else:
                            st.error("🔴 Conversion Line is below Base Line - bearish signal")
                    else:
                        st.warning("Some Ichimoku Cloud components could not be calculated due to insufficient data")
                else:
                    st.warning("Insufficient data to calculate Ichimoku Cloud (requires at least 52 periods)")
            
            # Fibonacci Retracement Tab
            with tabs[8]:
                st.subheader("Fibonacci Retracement")
                st.write("""
                Fibonacci Retracement uses horizontal lines to indicate potential support or resistance levels at which a price could reverse. The levels are derived from the Fibonacci sequence.
                
                Key Fibonacci Levels:
                - 0.0% - Start of the retracement
                - 23.6%, 38.2%, 50%, 61.8% - Key retracement levels
                - 100% - Complete reversal to the original price
                
                These levels can act as:
                - Support levels during an uptrend
                - Resistance levels during a downtrend
                """)
                
                if "fib_0" in df_ta.columns:
                    # Create Fibonacci Chart
                    fig_fib = go.Figure()
                    
                    # Add price
                    fig_fib.add_trace(go.Candlestick(
                        x=df_ta.index,
                        open=df_ta['Open'],
                        high=df_ta['High'],
                        low=df_ta['Low'],
                        close=df_ta['Close'],
                        name="Price"
                    ))
                    
                    # Add Fibonacci levels
                    fib_levels = {
                        "0%": df_ta["fib_0"].iloc[-1],
                        "23.6%": df_ta["fib_23.6"].iloc[-1],
                        "38.2%": df_ta["fib_38.2"].iloc[-1],
                        "50%": df_ta["fib_50"].iloc[-1],
                        "61.8%": df_ta["fib_61.8"].iloc[-1],
                        "100%": df_ta["fib_100"].iloc[-1]
                    }
                    
                    # Colors for different Fibonacci levels
                    colors = ["blue", "purple", "green", "orange", "red", "black"]
                    
                    # Add horizontal lines for each Fibonacci level
                    for i, (level, value) in enumerate(fib_levels.items()):
                        fig_fib.add_trace(go.Scatter(
                            x=[df_ta.index[0], df_ta.index[-1]],
                            y=[value, value],
                            mode="lines",
                            name=f"Fib {level}",
                            line=dict(color=colors[i], width=1, dash='dash')
                        ))
                    
                    # Update layout
                    fig_fib.update_layout(
                        title=f"{ticker_input} with Fibonacci Retracement Levels",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        xaxis_rangeslider_visible=False,
                        height=600
                    )
                    
                    st.plotly_chart(fig_fib, use_container_width=True)
                    
                    # Fibonacci Analysis
                    st.subheader("Fibonacci Retracement Levels")
                    
                    # Calculate price high and low points
                    price_high = df_ta['High'].max()
                    price_low = df_ta['Low'].min()
                    current_price = df_ta['Close'].iloc[-1]
                    
                    # Check which Fibonacci level is closest to current price
                    closest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))
                    
                    # Display all levels
                    cols = st.columns(3)
                    
                    for i, (level, value) in enumerate(fib_levels.items()):
                        col_idx = i % 3
                        # Highlight the closest level
                        if level == closest_fib[0]:
                            cols[col_idx].metric(f"Fib {level}", f"${value:.2f} ⭐")
                        else:
                            cols[col_idx].metric(f"Fib {level}", f"${value:.2f}")
                    
                    # Interpretation based on closest level
                    st.info(f"🔍 Current price (${current_price:.2f}) is closest to the {closest_fib[0]} Fibonacci retracement level (${closest_fib[1]:.2f}).")
                    
                    # Add context about direction
                    if df_ta['Close'].iloc[-1] > df_ta['Close'].iloc[-10]:
                        st.write("In the current uptrend, Fibonacci levels below the price may act as support if price pulls back.")
                    else:
                        st.write("In the current downtrend, Fibonacci levels above the price may act as resistance if price rebounds.")
                else:
                    st.warning("Insufficient data to calculate Fibonacci Retracement")
            
            # Investment tips based on analysis
            st.subheader("Fundamental Analysis")
            
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

# Add navigation back to home
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.page_link("home.py", label="🏠 Home", icon="🏠")
st.sidebar.page_link("pages/1_Stock_Analysis.py", label="📈 Stock Analysis", icon="📈")