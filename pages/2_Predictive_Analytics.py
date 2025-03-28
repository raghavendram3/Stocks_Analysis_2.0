import streamlit as st
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

# Configure page
st.set_page_config(
    page_title="StockTrackPro - Predictive Analytics",
    page_icon="🔮",
    layout="wide"
)

# Header and navigation
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔮 Predictive Analytics")
        st.markdown("Forecast future stock prices using advanced machine learning models")
    
    with col2:
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            st.page_link("home.py", label="Home", icon="🏠")
        with nav_col2:
            st.page_link("pages/1_Stock_Analysis.py", label="Stock Analysis", icon="📈")
        with nav_col3:
            st.page_link("pages/2_Predictive_Analytics.py", label="Predictions", icon="🔮")

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
    
    st.markdown("---")
    st.markdown("### Navigation")
    st.page_link("home.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_Stock_Analysis.py", label="📈 Stock Analysis", icon="📈")
    st.page_link("pages/2_Predictive_Analytics.py", label="🔮 Predictions", icon="🔮")
    
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
                        latest_price = df_feat['Close'].iloc[-1]
                        
                        # For each future day, predict the price and update the feature set
                        future_prices_rf = []
                        future_prices_svr = []
                        
                        # Create future dates
                        last_date = hist.index[-1]
                        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(prediction_days)]
                        
                        # Simple prediction approach (predict next 1 day at a time and use that for future)
                        # This is a simplified approach - in real applications you'd want a more sophisticated method
                        current_features = latest_data.copy()
                        
                        for i in range(prediction_days):
                            # Predict the next day's price
                            # Ensure the data is properly scaled and shaped
                            current_features_scaled = scaler.transform(current_features)
                            
                            # Make predictions
                            rf_pred = rf_model.predict(current_features_scaled)[0]
                            svr_pred = svr_model.predict(current_features_scaled)[0]
                            
                            # Store predictions
                            future_prices_rf.append(rf_pred)
                            future_prices_svr.append(svr_pred)
                            
                            # Update features for next prediction (simplified)
                            # In a real application, you'd need to update all time-dependent features
                            # Here we're just updating the price features as a demo
                            if 'Close' in feature_columns:
                                close_idx = feature_columns.index('Close')
                                current_features[0, close_idx] = rf_pred  # Using RF prediction to update
                            
                            # Update MA features if they exist
                            if 'MA5' in feature_columns and 'Close' in feature_columns:
                                ma5_idx = feature_columns.index('MA5')
                                # Simple approximation for updating MA5
                                if i >= 4:  # Have enough predicted prices to calculate a new MA5
                                    current_features[0, ma5_idx] = np.mean(future_prices_rf[-5:])
                            
                            if 'MA20' in feature_columns and 'Close' in feature_columns:
                                ma20_idx = feature_columns.index('MA20')
                                # Simple approximation for updating MA20
                                if i >= 19:  # Have enough predicted prices to calculate a new MA20
                                    current_features[0, ma20_idx] = np.mean(future_prices_rf[-20:])
                        
                        # Convert predictions to numpy arrays
                        future_predictions_rf = np.array(future_prices_rf)
                        future_predictions_svr = np.array(future_prices_svr)
                        
                        # Plot predictions
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=hist.index[-90:],  # Last 90 days of historical data
                            y=hist['Close'].iloc[-90:],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Random Forest predictions
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions_rf,
                            mode='lines',
                            name='Random Forest Predictions',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # SVR predictions
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions_svr,
                            mode='lines',
                            name='SVR Predictions',
                            line=dict(color='green', dash='dot')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Machine Learning Models: {ticker_input} Stock Price Prediction (Next {prediction_days} Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction summary
                        st.markdown("#### Random Forest Prediction Summary")
                        
                        # Calculate expected price changes (for Random Forest)
                        start_price = hist['Close'].iloc[-1]
                        end_price_rf = future_predictions_rf[-1]
                        price_change_rf = end_price_rf - start_price
                        price_change_pct_rf = (price_change_rf / start_price) * 100
                        
                        # Display RF predictions in columns
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        with pred_col1:
                            st.metric(
                                f"Price in {prediction_days} days", 
                                f"${end_price_rf:.2f}", 
                                f"{price_change_pct_rf:.2f}%"
                            )
                        
                        with pred_col2:
                            max_price_rf = np.max(future_predictions_rf)
                            max_day_rf = np.argmax(future_predictions_rf) + 1
                            st.metric(
                                "Predicted Maximum Price", 
                                f"${max_price_rf:.2f}",
                                f"Day {max_day_rf}"
                            )
                        
                        with pred_col3:
                            min_price_rf = np.min(future_predictions_rf)
                            min_day_rf = np.argmin(future_predictions_rf) + 1
                            st.metric(
                                "Predicted Minimum Price", 
                                f"${min_price_rf:.2f}",
                                f"Day {min_day_rf}"
                            )
                        
                        # Show SVR prediction summary
                        st.markdown("#### SVR Prediction Summary")
                        end_price_svr = future_predictions_svr[-1]
                        price_change_svr = end_price_svr - start_price
                        price_change_pct_svr = (price_change_svr / start_price) * 100
                        
                        # Display SVR predictions
                        svr_col1, svr_col2, svr_col3 = st.columns(3)
                        
                        with svr_col1:
                            st.metric(
                                f"Price in {prediction_days} days", 
                                f"${end_price_svr:.2f}", 
                                f"{price_change_pct_svr:.2f}%"
                            )
                        
                        with svr_col2:
                            max_price_svr = np.max(future_predictions_svr)
                            max_day_svr = np.argmax(future_predictions_svr) + 1
                            st.metric(
                                "Predicted Maximum Price", 
                                f"${max_price_svr:.2f}",
                                f"Day {max_day_svr}"
                            )
                        
                        with svr_col3:
                            min_price_svr = np.min(future_predictions_svr)
                            min_day_svr = np.argmin(future_predictions_svr) + 1
                            st.metric(
                                "Predicted Minimum Price", 
                                f"${min_price_svr:.2f}",
                                f"Day {min_day_svr}"
                            )
                        
                        # Show detailed predictions in an expander
                        with st.expander("View Detailed ML Predictions"):
                            # Create a dataframe with future dates and predictions
                            future_df = pd.DataFrame({
                                'Date': [d.date() for d in future_dates],
                                'Random Forest Prediction': future_predictions_rf.round(2),
                                'SVR Prediction': future_predictions_svr.round(2)
                            })
                            
                            # Display table
                            st.dataframe(future_df, use_container_width=True)
                            
                            # Download option
                            csv = future_df.to_csv(index=False)
                            st.download_button(
                                label="Download ML Predictions as CSV",
                                data=csv,
                                file_name=f"{ticker_input}_ml_predictions.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error in Machine Learning model training or prediction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                if model_type == "Prophet (Statistical)" or model_type == "Compare Both":
                    # Prophet model
                    st.markdown("### Prophet Statistical Model")
                    
                    try:
                        # Prepare data for Prophet
                        prophet_data = prepare_data_prophet(hist)
                        
                        # Build and train model
                        prophet_model, forecast = build_prophet_model(prophet_data, prediction_days=prediction_days)
                        
                        # Plot forecast
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=prophet_data['ds'],
                            y=prophet_data['y'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'][-prediction_days:],
                            y=forecast['yhat'][-prediction_days:],
                            mode='lines',
                            name='Prophet Forecast',
                            line=dict(color='green', dash='dash')
                        ))
                        
                        # Uncertainty intervals
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'][-prediction_days:],
                            y=forecast['yhat_upper'][-prediction_days:],
                            mode='lines',
                            fill=None,
                            line=dict(color='rgba(0,100,80,0.2)'),
                            name='Upper Bound'
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast['ds'][-prediction_days:],
                            y=forecast['yhat_lower'][-prediction_days:],
                            mode='lines',
                            fill='tonexty',
                            line=dict(color='rgba(0,100,80,0.2)'),
                            name='Lower Bound'
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Prophet Model: {ticker_input} Stock Price Prediction (Next {prediction_days} Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction summary
                        st.markdown("#### Prophet Prediction Summary")
                        
                        # Calculate expected price changes
                        start_price = prophet_data['y'].iloc[-1]
                        end_price = forecast['yhat'].iloc[-1]
                        price_change = end_price - start_price
                        price_change_pct = (price_change / start_price) * 100
                        
                        # Display predictions in columns
                        pred_col1, pred_col2, pred_col3 = st.columns(3)
                        
                        with pred_col1:
                            st.metric(
                                f"Price in {prediction_days} days", 
                                f"${end_price:.2f}", 
                                f"{price_change_pct:.2f}%"
                            )
                        
                        with pred_col2:
                            future_forecast = forecast.iloc[-prediction_days:]
                            max_price = future_forecast['yhat'].max()
                            max_day = (future_forecast[future_forecast['yhat'] == max_price]['ds'].iloc[0].date() - 
                                      forecast['ds'].iloc[-prediction_days].date()).days + 1
                            st.metric(
                                "Predicted Maximum Price", 
                                f"${max_price:.2f}",
                                f"Day {max_day}"
                            )
                        
                        with pred_col3:
                            min_price = future_forecast['yhat'].min()
                            min_day = (future_forecast[future_forecast['yhat'] == min_price]['ds'].iloc[0].date() - 
                                      forecast['ds'].iloc[-prediction_days].date()).days + 1
                            st.metric(
                                "Predicted Minimum Price", 
                                f"${min_price:.2f}",
                                f"Day {min_day}"
                            )
                        
                        # Show model components
                        with st.expander("View Prophet Model Components"):
                            fig2 = prophet_model.plot_components(forecast)
                            st.pyplot(fig2)
                        
                        # Show detailed predictions in an expander
                        with st.expander("View Detailed Prophet Predictions"):
                            # Create a dataframe with future dates and predictions
                            future_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-prediction_days:].copy()
                            future_df.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
                            future_df['Date'] = future_df['Date'].dt.date
                            
                            for col in ['Predicted Price', 'Lower Bound', 'Upper Bound']:
                                future_df[col] = future_df[col].round(2)
                            
                            # Display table
                            st.dataframe(future_df, use_container_width=True)
                            
                            # Download option
                            csv = future_df.to_csv(index=False)
                            st.download_button(
                                label="Download Prophet Predictions as CSV",
                                data=csv,
                                file_name=f"{ticker_input}_prophet_predictions.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error in Prophet model training or prediction: {e}")
                        import traceback
                        st.code(traceback.format_exc())
                
                # Add model comparison if both models were run
                if model_type == "Compare Both":
                    # Check if models were successfully run
                    ml_success = 'future_predictions_rf' in locals() and 'future_dates' in locals()
                    prophet_success = 'forecast' in locals()
                    
                    if ml_success or prophet_success:
                        st.markdown("### Model Comparison")
                        
                        # Create combined plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=hist.index[-90:],  # Last 90 days of historical data
                            y=hist['Close'].iloc[-90:],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Add ML predictions if available
                        if ml_success:
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_predictions_rf,
                                mode='lines',
                                name='Random Forest Predictions',
                                line=dict(color='red', dash='dash')
                            ))
                        
                        # Add Prophet predictions if available
                        if prophet_success:
                            fig.add_trace(go.Scatter(
                                x=forecast['ds'][-prediction_days:],
                                y=forecast['yhat'][-prediction_days:],
                                mode='lines',
                                name='Prophet Predictions',
                                line=dict(color='green', dash='dash')
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Model Comparison: {ticker_input} Stock Price Prediction (Next {prediction_days} Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("""
                        **Important Note**: Discrepancies between models are normal and highlight the uncertainty in stock price prediction. 
                        Different models use different approaches:
                        
                        - **Random Forest**: Uses ensemble learning to capture complex patterns in technical indicators
                        - **Prophet**: Uses time series decomposition focusing on seasonal patterns
                        
                        Consider using these predictions as just one of many factors in your investment decisions.
                        """)
                
                # Disclaimer
                st.markdown("---")
                st.caption("""
                **Disclaimer**: The predictions shown are based on historical data and mathematical models. 
                They do not account for future news events, company announcements, or other external factors that may impact stock prices. 
                These predictions should not be used as the sole basis for investment decisions. 
                Always conduct thorough research and consider consulting a financial advisor.
                """)
else:
    # Show welcome message when no ticker is entered
    st.info("👈 Enter a stock symbol in the sidebar to get started with predictive analytics!")
    
    # Feature explanation
    st.markdown("""
    ## Predictive Analytics Features
    
    This tool uses machine learning and statistical modeling to forecast potential future stock price movements based on historical patterns.
    
    ### Available Models:
    
    #### 🧠 Machine Learning Models
    - **Random Forest**: Ensemble method that captures market patterns
    - **SVR**: Support Vector Regression for price trend prediction
    - Both models use technical indicators as features
    
    #### 📊 Prophet
    - Facebook's statistical forecasting model
    - Handles seasonality and trend changes effectively
    - Includes confidence intervals for predictions
    
    #### 🔍 Model Comparison
    - Compare predictions from different models side-by-side
    - Understand the range of possible outcomes
    - See where predictions agree or differ
    
    ### How to Get Started:
    1. Enter a stock symbol in the sidebar
    2. Select your preferred historical data timeframe
    3. Choose the number of days to predict into the future
    4. Select which model(s) to use for prediction
    
    ### Sample Stocks to Try:
    - AAPL (Apple)
    - MSFT (Microsoft)
    - GOOGL (Alphabet)
    - AMZN (Amazon)
    - TSLA (Tesla)
    """)