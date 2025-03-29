import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# This is added to handle Replit deployment - redirect to home.py
if __name__ == "__main__":
    # Check if running via the entrypoint mechanism (via app.py)
    if os.path.basename(sys.argv[0]) == "app.py":
        # Re-execute using home.py as the entrypoint
        # This helps with deployment issues where app.py might be expected
        # but home.py is the actual entrypoint
        print("Redirecting to home.py as the entrypoint...")
        os.system(f"{sys.executable} home.py {' '.join(sys.argv[1:])}")
        sys.exit(0)

# Configure page
st.set_page_config(
    page_title="Stock Analysis Tool",
    page_icon="📈",
    layout="wide"
)

# Header section
st.title("📈 Stock Analysis Tool")
st.markdown("Enter a stock symbol to get financial data visualization and basic investment tips.")

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
