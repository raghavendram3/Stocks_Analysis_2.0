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
    """Prepare stock data for machine learning model training"""
    # Extract close price and convert to numpy array
    prices = data['Close'].values.reshape(-1, 1)
    
    # Create features - we'll use several technical indicators as features
    df_feat = data.copy()
    
    # Add some basic technical indicators as features
    # Moving averages
    df_feat['MA5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['MA20'] = df_feat['Close'].rolling(window=20).mean()
    
    # Price momentum
    df_feat['Price_Momentum'] = df_feat['Close'].pct_change(periods=5)
    
    # Volatility
    df_feat['Volatility'] = df_feat['Close'].rolling(window=20).std()
    
    # Volume features
    if 'Volume' in df_feat.columns:
        df_feat['Volume_Change'] = df_feat['Volume'].pct_change()
    
    # Trading range
    df_feat['High_Low_Range'] = (df_feat['High'] - df_feat['Low']) / df_feat['Close']
    
    # Drop NaN values
    df_feat.dropna(inplace=True)
    
    # Scale the features
    feature_columns = ['Close', 'MA5', 'MA20', 'Price_Momentum', 'Volatility', 'High_Low_Range']
    if 'Volume' in df_feat.columns and 'Volume_Change' in df_feat.columns:
        feature_columns.extend(['Volume', 'Volume_Change'])
    
    # Create normalized features
    scaler = MinMaxScaler(feature_range=(0, 1))
    features = df_feat[feature_columns].values
    scaled_features = scaler.fit_transform(features)
    
    # Prepare data for machine learning
    X = []
    y = []
    
    # For each prediction day, create a target value (the price n days in the future)
    dates = []
    
    for i in range(len(scaled_features) - prediction_days):
        X.append(scaled_features[i])
        y.append(df_feat['Close'].iloc[i + prediction_days])
        dates.append(df_feat.index[i + prediction_days])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    
    return X_train, X_test, y_train, y_test, scaler, df_feat, dates, feature_columns

# Function to build and train ML model
def build_ml_model(X_train, y_train):
    """Build and train Random Forest model for stock prediction"""
    # Initialize and train Random Forest model
    with st.spinner("Training Random Forest model..."):
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
    
    # SVM model as a second option
    with st.spinner("Training SVR model for comparison..."):
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
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
                        X_train, X_test, y_train, y_test, scaler, df_feat, dates, feature_columns = prepare_data_ml(
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
                        # Get the most recent data point as a starting point
                        latest_data = df_feat.iloc[-1:][feature_columns].values
                        
                        # For each future day, predict the price and update the feature set
                        future_prices_rf = []
                        future_prices_svr = []
                        
                        # Create future dates
                        last_date = hist.index[-1]
                        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(prediction_days)]
                        
                        # Simple prediction approach
                        # This is a simplified approach - in real applications you'd want a more sophisticated method
                        current_features = latest_data.copy()
                        
                        for i in range(prediction_days):
                            # Ensure the data is properly scaled
                            current_features_scaled = scaler.transform(current_features)
                            
                            # Make predictions
                            rf_pred = rf_model.predict(current_features_scaled)[0]
                            svr_pred = svr_model.predict(current_features_scaled)[0]
                            
                            # Store predictions
                            future_prices_rf.append(rf_pred)
                            future_prices_svr.append(svr_pred)
                            
                            # Update features for next prediction (simplified approach)
                            # In a real application, you'd need to update all features
                            if 'Close' in feature_columns:
                                close_idx = feature_columns.index('Close')
                                current_features[0, close_idx] = rf_pred
                        
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