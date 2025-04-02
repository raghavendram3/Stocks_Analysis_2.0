import streamlit as st
from utils.analytics import inject_google_tag_manager

# Configure page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="StockTrackPro",
    page_icon="📈",
    layout="wide"
)

# Inject Google Tag Manager
inject_google_tag_manager()

# Custom CSS for modern website look
st.markdown("""
<style>
    .main {
        padding: 0 !important;
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .hero-section {
        background: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)), url('https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80');
        background-size: cover;
        padding: 4rem 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: -1rem -1rem 2rem -1rem;
    }
    .hero-content {
        max-width: 800px;
        margin: 0 auto;
        text-align: center;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .nav-link {
        text-decoration: none;
        color: inherit;
    }
    .nav-link:hover {
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-section">
    <div class="hero-content">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">📈 StockTrackPro</h1>
        <p style="font-size: 1.5rem; color: #666; margin-bottom: 2rem;">Your comprehensive stock analysis platform</p>
        <div style="font-size: 1.2rem; color: #333;">
            Empower your investment decisions with real-time data, advanced analytics, and AI-powered insights
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content with feature cards
st.markdown("<h2 style='text-align: center; margin: 2rem 0;'>🔍 Explore Our Features</h2>", unsafe_allow_html=True)

# App cards in columns
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; font-size: 1.5rem; margin-bottom: 1rem;">📊 Stock Analysis Tool</h3>
        <p style="color: #666; font-size: 1.1rem;">Get powerful insights with our comprehensive stock analysis:</p>
        <ul style="list-style-type: none; padding: 0;">
            <li style="margin: 10px 0; color: #444;">📊 Interactive price and volume visualizations</li>
            <li style="margin: 10px 0; color: #444;">🔄 Real-time data from Yahoo Finance</li>
            <li style="margin: 10px 0; color: #444;">📈 Key financial metrics and ratios</li>
            <li style="margin: 10px 0; color: #444;">📉 Advanced technical indicators</li>
            <li style="margin: 10px 0; color: #444;">💡 Investment insights and risk assessment</li>
            <li style="margin: 10px 0; color: #444;">📥 Downloadable data in CSV format</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to action
    st.page_link("pages/1_Stock_Analysis.py", label="Start Analyzing Stocks 📈", use_container_width=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="color: #1E88E5; font-size: 1.5rem; margin-bottom: 1rem;">🧠 Technical Analysis</h3>
        <p style="color: #666; font-size: 1.1rem;">Make better trading decisions with top technical analysis tools:</p>
        <ul style="list-style-type: none; padding: 0;">
            <li style="margin: 10px 0; color: #444;">📊 MACD (Moving Average Convergence Divergence)</li>
            <li style="margin: 10px 0; color: #444;">📈 RSI (Relative Strength Index)</li>
            <li style="margin: 10px 0; color: #444;">📉 Bollinger Bands</li>
            <li style="margin: 10px 0; color: #444;">🎯 Fibonacci Retracement</li>
            <li style="margin: 10px 0; color: #444;">☁️ Ichimoku Cloud</li>
            <li style="margin: 10px 0; color: #444;">📊 Stochastic Oscillator</li>
            <li style="margin: 10px 0; color: #444;">📈 On-Balance Volume (OBV)</li>
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

# Additional features section
st.markdown("<h2 style='text-align: center; margin: 3rem 0 2rem;'>✨ Available Features</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card" style="height: 100%;">
        <h3 style="color: #1E88E5; font-size: 1.3rem; margin-bottom: 1rem;">🔮 Predictive Analytics</h3>
        <p style="color: #666;">Advanced machine learning models to forecast stock movements:</p>
        <ul style="list-style-type: none; padding: 0;">
            <li style="margin: 10px 0; color: #444;">🌲 Random Forest Prediction</li>
            <li style="margin: 10px 0; color: #444;">🎯 Support Vector Regression</li>
            <li style="margin: 10px 0; color: #444;">📊 Facebook Prophet Forecasting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Predictive_Analytics.py", label="Try Price Predictions 🔮", use_container_width=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="height: 100%;">
        <h3 style="color: #1E88E5; font-size: 1.3rem; margin-bottom: 1rem;">📝 User Feedback</h3>
        <p style="color: #666;">We value your input! Share your thoughts:</p>
        <ul style="list-style-type: none; padding: 0;">
            <li style="margin: 10px 0; color: #444;">💭 Submit anonymous feedback</li>
            <li style="margin: 10px 0; color: #444;">💡 Request new features</li>
            <li style="margin: 10px 0; color: #444;">🐛 Report bugs or issues</li>
            <li style="margin: 10px 0; color: #444;">❓ Ask questions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_User_Feedback.py", label="Share Your Feedback 📝", use_container_width=True)

with col3:
    st.markdown("""
    <div class="feature-card" style="height: 100%;">
        <h3 style="color: #1E88E5; font-size: 1.3rem; margin-bottom: 1rem;">🚀 Coming Soon</h3>
        <p style="color: #666;">Exciting features in development:</p>
        <ul style="list-style-type: none; padding: 0;">
            <li style="margin: 10px 0; color: #444;">📊 Multi-stock comparison</li>
            <li style="margin: 10px 0; color: #444;">💼 Portfolio tracking</li>
            <li style="margin: 10px 0; color: #444;">🔔 Custom alerts</li>
            <li style="margin: 10px 0; color: #444;">⚡ Strategy backtesting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="background: white; margin: 4rem -1rem -1rem -1rem; padding: 2rem; border-radius: 20px 20px 0 0; box-shadow: 0 -4px 6px rgba(0,0,0,0.1);">
    <div style="max-width: 1200px; margin: 0 auto; display: flex; justify-content: space-between; align-items: center;">
        <div style="color: #666;">
            © 2025 StockTrackPro. All rights reserved.
        </div>
        <div style="color: #666;">
            Powered by Yahoo Finance
        </div>
    </div>
</div>
""", unsafe_allow_html=True)