import streamlit as st

# Configure page
st.set_page_config(
    page_title="StockTrackPro",
    page_icon="📈",
    layout="wide"
)

# Header and navigation
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 StockTrackPro")
        st.markdown("Your comprehensive stock analysis platform")
    
    with col2:
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            st.page_link("home.py", label="Home", icon="🏠")
        with nav_col2:
            st.page_link("pages/1_Stock_Analysis.py", label="Stock Analysis", icon="📈")
        with nav_col3:
            st.page_link("pages/2_Predictive_Analytics.py", label="Predictions", icon="🔮")

# Hero section
st.markdown("""
<div style="background-color:#f0f8ff; padding:20px; border-radius:10px; margin-bottom:20px">
    <h2 style="text-align:center">Empower Your Investment Decisions</h2>
    <p style="text-align:center">Get real-time stock data, advanced technical indicators, and personalized insights</p>
</div>
""", unsafe_allow_html=True)

# Main content with feature cards
st.header("🔍 Explore Our Features")

# App cards in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="border:1px solid #e0e0e0; border-radius:10px; padding:20px; height:100%">
        <h3>📊 Stock Analysis Tool</h3>
        <p>Get powerful insights with our comprehensive stock analysis:</p>
        <ul>
            <li>Interactive price and volume visualizations</li>
            <li>Real-time data from Yahoo Finance</li>
            <li>Key financial metrics and ratios</li>
            <li>Advanced technical indicators</li>
            <li>Investment insights and risk assessment</li>
            <li>Downloadable data in CSV format</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Call to action
    st.page_link("pages/1_Stock_Analysis.py", label="Start Analyzing Stocks", icon="📈")

with col2:
    st.markdown("""
    <div style="border:1px solid #e0e0e0; border-radius:10px; padding:20px; height:100%">
        <h3>🧠 Technical Analysis</h3>
        <p>Make better trading decisions with Investopedia's top 7 technical analysis tools:</p>
        <ul>
            <li>MACD (Moving Average Convergence Divergence)</li>
            <li>RSI (Relative Strength Index)</li>
            <li>Bollinger Bands</li>
            <li>Fibonacci Retracement</li>
            <li>Ichimoku Cloud</li>
            <li>Stochastic Oscillator</li>
            <li>On-Balance Volume (OBV)</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Technical Analysis details
    with st.expander("Learn About Technical Analysis"):
        st.markdown("""
        **Technical analysis** is a trading discipline that evaluates investments and identifies trading opportunities by analyzing statistical trends gathered from trading activity.
        
        Our platform implements all the major technical indicators recommended by investment professionals:
        
        - **MACD**: Trend-following momentum indicator showing the relationship between two moving averages of a security's price
        - **RSI**: Momentum oscillator measuring the speed and change of price movements
        - **Bollinger Bands**: Volatility indicator placing bands above and below a moving average
        - **Fibonacci Retracement**: Potential support and resistance levels based on the Fibonacci sequence
        - **Ichimoku Cloud**: A comprehensive indicator showing support, resistance, momentum, and trend direction
        - **Stochastic Oscillator**: Momentum indicator comparing a particular closing price to a range of prices over time
        - **On-Balance Volume**: Uses volume flow to predict changes in stock price
        """)

# How to use section
st.header("🚀 Getting Started")
st.markdown("""
1. **Navigate to Stock Analysis**: Click the "Start Analyzing Stocks" button above
2. **Enter a Stock Symbol**: Type in any valid ticker (e.g., AAPL, MSFT, GOOGL)
3. **Select Time Period**: Choose from 1 month to maximum available history
4. **Explore the Data**: View charts, indicators, and personalized analysis
5. **Download Reports**: Save your analysis as CSV files for offline review
""")

# Additional features
st.header("✨ Available Features")
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔮 Predictive Analytics")
    st.markdown("""
    We've implemented advanced machine learning models to help forecast potential stock price movements:
    
    - **LSTM Neural Networks**: Deep learning time-series analysis
    - **Facebook Prophet**: Statistical forecasting with confidence intervals
    - **Model Comparison**: Compare different forecasting approaches
    """)
    st.page_link("pages/2_Predictive_Analytics.py", label="Try Price Predictions", icon="🔮")

with col2:
    st.subheader("🚀 Features Coming Soon")
    st.markdown("""
    We're constantly improving! Watch for these upcoming features:
    
    - Multi-stock comparison tool
    - Portfolio tracking and analysis
    - Custom alerts and notifications 
    - Backtesting capabilities for trading strategies
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center;">
    <div>© 2025 StockTrackPro. All rights reserved.</div>
    <div>
        <a href="https://github.com/raghavendram3/StockTrackPro" target="_blank" style="text-decoration: none; color: inherit; margin-right: 10px;">
            GitHub
        </a>
        |
        <span style="margin-left: 10px;">Powered by Yahoo Finance</span>
    </div>
</div>
""", unsafe_allow_html=True)