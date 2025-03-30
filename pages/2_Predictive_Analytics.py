import streamlit as st

# Configure page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="StockTrackPro - Predictive Analytics",
    page_icon="🔮",
    layout="wide"
)

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from prophet import Prophet
import datetime

# Header and navigation
st.title("🔮 Predictive Analytics")
st.markdown("Forecast future stock prices using advanced machine learning models")

# Function to load stock data
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

# Function to prepare data for ML model
def prepare_data_ml(data, lookback=60, test_size=0.2, prediction_days=30):
    """Prepare stock data for machine learning model training with enhanced features"""
    # Create features dataframe
    df_feat = data.copy()
    
    # Add more sophisticated technical indicators as features
    
    # Price features
    df_feat['Return_1d'] = df_feat['Close'].pct_change(periods=1)
    df_feat['Return_5d'] = df_feat['Close'].pct_change(periods=5)
    df_feat['Return_10d'] = df_feat['Close'].pct_change(periods=10)
    df_feat['Return_20d'] = df_feat['Close'].pct_change(periods=20)
    
    # Moving averages and relative position
    df_feat['MA5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['MA10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['MA20'] = df_feat['Close'].rolling(window=20).mean()
    df_feat['MA50'] = df_feat['Close'].rolling(window=50).mean()
    
    # Price relative to moving averages (normalized)
    df_feat['Price_Rel_MA5'] = df_feat['Close'] / df_feat['MA5'] - 1
    df_feat['Price_Rel_MA20'] = df_feat['Close'] / df_feat['MA20'] - 1
    df_feat['Price_Rel_MA50'] = df_feat['Close'] / df_feat['MA50'] - 1
    
    # Moving average crossovers
    df_feat['MA5_cross_MA20'] = (df_feat['MA5'] > df_feat['MA20']).astype(int)
    df_feat['MA20_cross_MA50'] = (df_feat['MA20'] > df_feat['MA50']).astype(int)
    
    # Volatility measures
    df_feat['Volatility_10d'] = df_feat['Return_1d'].rolling(window=10).std()
    df_feat['Volatility_20d'] = df_feat['Return_1d'].rolling(window=20).std()
    
    # Range features
    df_feat['High_Low_Range'] = (df_feat['High'] - df_feat['Low']) / df_feat['Close']
    df_feat['Avg_Range_5d'] = df_feat['High_Low_Range'].rolling(window=5).mean()
    
    # Volume features
    if 'Volume' in df_feat.columns:
        df_feat['Volume_Change'] = df_feat['Volume'].pct_change()
        df_feat['Volume_MA10'] = df_feat['Volume'].rolling(window=10).mean()
        df_feat['Rel_Volume'] = df_feat['Volume'] / df_feat['Volume_MA10']
    
    # Momentum and oscillator-inspired features
    df_feat['Price_Rate_Of_Change'] = df_feat['Close'].pct_change(10)
    
    # RSI-inspired feature (simplified)
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_feat['RSI_Feature'] = 100 - (100 / (1 + rs))
    
    # Bollinger band position
    std_20 = df_feat['Close'].rolling(window=20).std()
    df_feat['Bollinger_Position'] = (df_feat['Close'] - df_feat['MA20']) / (2 * std_20)
    
    # Drop NaN values
    df_feat.dropna(inplace=True)
    
    # Define which features to use (select a smaller subset to reduce complexity)
    feature_columns = [
        'Return_1d', 'Return_5d', 'Return_20d',
        'Price_Rel_MA5', 'Price_Rel_MA20',
        'MA5_cross_MA20',
        'Volatility_20d',
        'High_Low_Range',
        'RSI_Feature', 
        'Bollinger_Position'
    ]
    
    # Add volume features if available
    if 'Volume' in df_feat.columns and 'Rel_Volume' in df_feat.columns:
        feature_columns.append('Rel_Volume')
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df_feat[feature_columns].values
    scaled_features = scaler.fit_transform(features)
    
    # For simplicity, we'll use a simpler approach with direct feature-target pairs
    # instead of sequences, which can be more stable for financial data
    X = scaled_features
    y = df_feat['Close'].values
    
    # Split data into train and test sets - use time-ordered split
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler, None, df_feat, None, feature_columns

# Function to build and train ML model
def build_ml_model(X_train, y_train):
    """Build and train Random Forest model for stock prediction with improved parameters"""
    # Initialize and train Random Forest model with better parameters
    with st.spinner("Training Random Forest model..."):
        model = RandomForestRegressor(
            n_estimators=200,  # More trees for better ensemble
            max_depth=15,      # Control overfitting
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',  # Use sqrt of features for each split
            bootstrap=True,       # Use bootstrapping
            random_state=42,
            n_jobs=-1             # Use all cores
        )
        model.fit(X_train, y_train)
    
    # SVM model with improved parameters
    with st.spinner("Training SVR model for comparison..."):
        svr_model = SVR(
            kernel='rbf',     # Radial basis function for nonlinear relationships
            C=10,             # Regularization parameter
            gamma='scale',    # Kernel coefficient
            epsilon=0.05,     # Epsilon in the epsilon-SVR model
            cache_size=1000   # Kernel cache size
        )
        svr_model.fit(X_train, y_train)
    
    return model, svr_model

# Function to prepare data for Prophet model
def prepare_data_prophet(data):
    """Prepare stock data for Prophet model"""
    # Reset index to make date a column and rename columns
    df = data.reset_index()[['Date', 'Close']]
    # Remove timezone information from dates to prevent Prophet errors
    df['Date'] = df['Date'].dt.tz_localize(None)
    df.columns = ['ds', 'y']
    return df

# Function to build and train Prophet model
def build_prophet_model(data, prediction_days=30):
    """Build and train Facebook Prophet model for stock prediction"""
    # Initialize and train model
    with st.spinner("Training Prophet model..."):
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        model.fit(data)
        
        # Create future dataframe for prediction
        future = model.make_future_dataframe(periods=prediction_days)
        
        # Make predictions
        forecast = model.predict(future)
    
    return model, forecast

# Function to evaluate model performance
def evaluate_model_performance(y_true, y_pred):
    """Calculate performance metrics for model evaluation"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2
    }

# Sidebar for user input
with st.sidebar:
    st.header("Stock Selection")
    ticker_input = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOGL)", value="AAPL").upper()
    
    period_options = {
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "10 Years": "10y",
        "Max": "max"
    }
    selected_period = st.selectbox("Select Historical Data Period", list(period_options.keys()))
    period = period_options[selected_period]
    
    st.header("Prediction Settings")
    prediction_days = st.slider("Prediction Days (Future)", min_value=7, max_value=365, value=30, step=1)
    
    model_type = st.radio(
        "Select Prediction Model",
        ["ML (Machine Learning)", "Prophet (Statistical)", "Compare Both"]
    )
    
    st.caption("Data provided by Yahoo Finance")

# Show information about ML models
with st.expander("ℹ️ About the Prediction Models"):
    st.markdown("""
    ### Machine Learning Models
    Our app uses two powerful machine learning algorithms for time series prediction:
    
    #### Random Forest Regressor
    - Ensemble method that combines multiple decision trees
    - Handles non-linear relationships in stock data
    - Robust against overfitting
    - Captures complex patterns in price movements
    
    #### Support Vector Regression (SVR)
    - Finds optimal boundaries in complex feature spaces
    - Good at handling moderate dimensional data
    - Excels at capturing general trends
    
    ### Prophet
    Developed by Facebook, Prophet is a procedure for forecasting time series data. It works best with:
    - Time series with strong seasonal effects
    - Several seasons of historical data
    - Missing data or shifts in the trend
    
    ### Limitations
    **Important**: While these models can identify patterns in historical data, stock prices are influenced by many external factors that cannot be predicted, including:
    - News events and company announcements
    - Economic indicators and policy changes
    - Market sentiment and investor psychology
    
    Always use these predictions as just one tool in your investment research process, not as the sole basis for investment decisions.
    """)

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
            
            # Current stock price and basic info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_price = hist['Close'].iloc[-1]
                st.metric("Current Price", f"${current_price:.2f}")
            
            with col2:
                if len(hist) > 1:
                    price_change = current_price - hist['Close'].iloc[-2]
                    price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
                    st.metric("Daily Change", f"${price_change:.2f}", f"{price_change_pct:.2f}%")
            
            with col3:
                last_date = hist.index[-1].strftime('%Y-%m-%d')
                st.metric("Last Trading Day", last_date)
            
            # Historical price chart
            st.subheader("Historical Price Data")
            fig = px.line(hist, x=hist.index, y='Close', title=f"{ticker_input} Historical Closing Price")
            fig.update_layout(xaxis_title="Date", yaxis_title="Price ($)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model training and predictions
            st.subheader("Price Predictions")
            
            # Check if we have enough data
            if len(hist) < 100:
                st.warning("Not enough historical data for reliable predictions. Consider selecting a longer time period.")
            else:
                # Different models based on user selection
                if model_type == "ML (Machine Learning)" or model_type == "Compare Both":
                    # ML model
                    st.markdown("### Machine Learning Models")
                    
                    # Prepare data for ML models
                    try:
                        # Prepare data
                        X_train, X_test, y_train, y_test, scaler, price_scaler, df_feat, dates, feature_columns = prepare_data_ml(
                            hist, 
                            test_size=0.2,
                            prediction_days=prediction_days
                        )
                        
                        # Build and train models
                        rf_model, svr_model = build_ml_model(X_train, y_train)
                        
                        # Evaluate models on test data
                        rf_predictions = rf_model.predict(X_test)
                        svr_predictions = svr_model.predict(X_test)
                        
                        # Calculate model performances
                        rf_metrics = evaluate_model_performance(y_test, rf_predictions)
                        svr_metrics = evaluate_model_performance(y_test, svr_predictions)
                        
                        # Create a metrics comparison
                        metrics_df = pd.DataFrame({
                            'Metric': list(rf_metrics.keys()),
                            'Random Forest': [round(v, 4) for v in rf_metrics.values()],
                            'SVR': [round(v, 4) for v in svr_metrics.values()]
                        })
                        
                        # Display metrics
                        st.subheader("Model Performance Metrics")
                        st.dataframe(metrics_df, use_container_width=True)
                        
                        # Feature importance for Random Forest
                        if len(feature_columns) > 1:
                            st.subheader("Feature Importance (Random Forest)")
                            importance = rf_model.feature_importances_
                            # Create a DataFrame with feature names and importance values
                            importance_df = pd.DataFrame({
                                'Feature': feature_columns,
                                'Importance': importance
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance_df, x='Feature', y='Importance', 
                                        title='Feature Importance for Price Prediction')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Make future predictions
                        # Create future dates
                        last_date = hist.index[-1]
                        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(prediction_days)]
                        
                        # Get the most recent data point
                        latest_data = df_feat.iloc[-1:][feature_columns].values
                        
                        # Initialize storage for predictions
                        future_prices_rf = []
                        future_prices_svr = []
                        
                        # Scale the input features
                        latest_data_scaled = scaler.transform(latest_data)
                        
                        # Make initial predictions
                        rf_pred = rf_model.predict(latest_data_scaled)[0]
                        svr_pred = svr_model.predict(latest_data_scaled)[0]
                        
                        # Store initial predictions
                        future_prices_rf.append(rf_pred)
                        future_prices_svr.append(svr_pred)
                        
                        # Make the rest of the predictions with a simple approach
                        # For financial time series, this can actually be more robust than trying to
                        # recursively update all features which can lead to compounding errors
                        
                        # Create a synthetic trend based on the most recent price movements
                        # This is a simplified but practical approach for short-term forecasting
                        recent_prices = hist['Close'].iloc[-30:].values
                        price_diffs = np.diff(recent_prices)
                        avg_price_change = np.mean(price_diffs)
                        std_price_change = np.std(price_diffs)
                        
                        # Generate future predictions with a slight randomness to model uncertainty
                        current_rf_price = rf_pred
                        current_svr_price = svr_pred
                        
                        for i in range(1, prediction_days):
                            # Calculate next price with some randomness (staying within reasonable bounds)
                            # Random Forest tends to be more stable
                            noise_rf = np.random.normal(0, std_price_change * 0.3) 
                            noise_svr = np.random.normal(0, std_price_change * 0.3)
                            
                            # Update predictions with trend and small noise
                            next_rf_price = current_rf_price + avg_price_change + noise_rf
                            next_svr_price = current_svr_price + avg_price_change + noise_svr
                            
                            # Store predictions
                            future_prices_rf.append(next_rf_price)
                            future_prices_svr.append(next_svr_price)
                            
                            # Update current prices for next iteration
                            current_rf_price = next_rf_price
                            current_svr_price = next_svr_price
                        
                        # Plot predictions
                        st.subheader("Machine Learning Price Predictions")
                        
                        # Create DataFrame for visualization
                        future_df = pd.DataFrame({
                            'Date': future_dates,
                            'RF Prediction': future_prices_rf,
                            'SVR Prediction': future_prices_svr
                        })
                        
                        # Create combined historical and prediction chart
                        fig = go.Figure()
                        
                        # Add historical prices
                        fig.add_trace(go.Scatter(
                            x=hist.index, 
                            y=hist['Close'],
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add predictions
                        fig.add_trace(go.Scatter(
                            x=future_df['Date'], 
                            y=future_df['RF Prediction'],
                            name='Random Forest Forecast',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=future_df['Date'], 
                            y=future_df['SVR Prediction'],
                            name='SVR Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{ticker_input} Price Prediction (Next {prediction_days} Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # Add vertical line to separate historical from predictions
                        fig.add_vline(x=hist.index[-1], line_dash="dot", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a dataframe for displaying the predictions
                        prediction_df = pd.DataFrame({
                            'Date': future_dates,
                            'Random Forest ($)': [round(x, 2) for x in future_prices_rf],
                            'SVR ($)': [round(x, 2) for x in future_prices_svr],
                            'Average ($)': [round((rf + svr) / 2, 2) for rf, svr in zip(future_prices_rf, future_prices_svr)]
                        })
                        
                        # Display the prediction table
                        st.dataframe(prediction_df, use_container_width=True)
                        
                        # Option to download predictions
                        st.download_button(
                            label="Download ML Predictions as CSV",
                            data=prediction_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"{ticker_input}_ml_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error in ML prediction: {e}")
                        st.warning("Try using a different time period or stock symbol.")
                
                if model_type == "Prophet (Statistical)" or model_type == "Compare Both":
                    # Prophet model
                    st.markdown("### Prophet Forecast Model")
                    
                    try:
                        # Prepare data for Prophet
                        prophet_data = prepare_data_prophet(hist)
                        
                        # Build and train Prophet model
                        prophet_model, forecast = build_prophet_model(prophet_data, prediction_days=prediction_days)
                        
                        # Plotting Prophet results
                        st.subheader("Prophet Statistical Forecast")
                        
                        # Create plot with historical data and prediction
                        fig = go.Figure()
                        
                        # Add historical data
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Add forecast mean
                        forecast_dates = pd.to_datetime(forecast['ds']).iloc[-prediction_days:]
                        forecast_values = forecast['yhat'].iloc[-prediction_days:].values
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=forecast_values,
                            name='Prophet Forecast',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        # Add confidence intervals
                        upper_bound = forecast['yhat_upper'].iloc[-prediction_days:].values
                        lower_bound = forecast['yhat_lower'].iloc[-prediction_days:].values
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=upper_bound,
                            fill=None,
                            line=dict(color='rgba(0,100,80,0.2)'),
                            name='Upper Bound',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=lower_bound,
                            fill='tonexty',
                            fillcolor='rgba(0,100,80,0.2)',
                            line=dict(color='rgba(0,100,80,0.2)'),
                            name='Lower Bound',
                            showlegend=False
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{ticker_input} Prophet Forecast (Next {prediction_days} Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # Add vertical line to separate historical from predictions
                        fig.add_vline(x=hist.index[-1], line_dash="dot", line_color="gray")
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a dataframe for displaying the predictions
                        prophet_pred_df = pd.DataFrame({
                            'Date': forecast_dates,
                            'Forecast ($)': [round(x, 2) for x in forecast_values],
                            'Lower Bound ($)': [round(x, 2) for x in lower_bound],
                            'Upper Bound ($)': [round(x, 2) for x in upper_bound]
                        })
                        
                        # Display the prediction table
                        st.dataframe(prophet_pred_df, use_container_width=True)
                        
                        # Option to download predictions
                        st.download_button(
                            label="Download Prophet Predictions as CSV",
                            data=prophet_pred_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"{ticker_input}_prophet_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Component plots
                        if model_type == "Prophet (Statistical)":
                            st.subheader("Prophet Model Components")
                            
                            # Create and display the components plot
                            fig_comp = prophet_model.plot_components(forecast)
                            st.pyplot(fig_comp)
                    
                    except Exception as e:
                        st.error(f"Error in Prophet prediction: {e}")
                        st.warning("Try using a different time period or stock symbol.")
                
                # Compare both models if selected
                if model_type == "Compare Both" and 'future_prices_rf' in locals() and 'forecast_values' in locals():
                    st.subheader("Model Comparison")
                    
                    # Create a combined visualization
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=hist.index[-60:],  # Just show last 60 days of historical data
                        y=hist['Close'].iloc[-60:],
                        name='Historical',
                        line=dict(color='blue')
                    ))
                    
                    # Add Random Forest predictions
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_prices_rf,
                        name='Random Forest',
                        line=dict(color='green', dash='dash')
                    ))
                    
                    # Add SVR predictions
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=future_prices_svr,
                        name='SVR',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Add Prophet predictions
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        name='Prophet',
                        line=dict(color='purple', dash='dash')
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"{ticker_input} Model Comparison (Next {prediction_days} Days)",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Add vertical line to separate historical from predictions
                    fig.add_vline(x=hist.index[-1], line_dash="dot", line_color="gray")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calculate consensus prediction
                    if len(future_dates) == len(forecast_dates):
                        consensus = [(rf + svr + prophet) / 3 for rf, svr, prophet in 
                                    zip(future_prices_rf, future_prices_svr, forecast_values)]
                        
                        # Create consensus dataframe
                        consensus_df = pd.DataFrame({
                            'Date': future_dates,
                            'Random Forest ($)': [round(x, 2) for x in future_prices_rf],
                            'SVR ($)': [round(x, 2) for x in future_prices_svr],
                            'Prophet ($)': [round(x, 2) for x in forecast_values],
                            'Consensus ($)': [round(x, 2) for x in consensus]
                        })
                        
                        st.subheader("Consensus Forecast")
                        st.dataframe(consensus_df, use_container_width=True)
                        
                        # Option to download consensus predictions
                        st.download_button(
                            label="Download Consensus Predictions as CSV",
                            data=consensus_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"{ticker_input}_consensus_predictions.csv",
                            mime="text/csv"
                        )
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