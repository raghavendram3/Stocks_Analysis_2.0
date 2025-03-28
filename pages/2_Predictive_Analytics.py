import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
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

# Function to prepare data for LSTM model
def prepare_data_lstm(data, lookback=60, test_size=0.2, prediction_days=30):
    """Prepare stock data for LSTM model training"""
    # Extract close price and convert to numpy array
    prices = data['Close'].values.reshape(-1, 1)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    
    # Prepare training data
    x_data = []
    y_data = []
    
    for i in range(lookback, len(scaled_data) - prediction_days):
        x_data.append(scaled_data[i - lookback:i, 0])
        y_data.append(scaled_data[i:i + prediction_days, 0])
    
    x_data, y_data = np.array(x_data), np.array(y_data)
    x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
    
    # Split into train and test
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=False)
    
    return x_train, x_test, y_train, y_test, scaler, prices

# Function to build and train LSTM model
def build_lstm_model(x_train, y_train, epochs=50, batch_size=32):
    """Build and train LSTM model for stock prediction"""
    # Initialize model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=y_train.shape[1]))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model
    with st.spinner("Training LSTM model (this may take a few minutes)..."):
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
    
    return model, history

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
        ["LSTM (Deep Learning)", "Prophet (Statistical)", "Compare Both"]
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
    ### LSTM (Long Short-Term Memory)
    LSTM is a type of recurrent neural network architecture designed to recognize patterns in sequences. It's particularly effective for time series data like stock prices because it can:
    - Capture long-term dependencies in the data
    - Handle the non-linear dynamics of stock markets
    - Learn complex patterns that traditional models might miss
    
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
                if model_type == "LSTM (Deep Learning)" or model_type == "Compare Both":
                    # LSTM model
                    st.markdown("### LSTM Deep Learning Model")
                    
                    # Prepare data for LSTM
                    try:
                        lookback_period = min(60, len(hist) // 4)  # Use smaller lookback if not enough data
                        x_train, x_test, y_train, y_test, scaler, prices = prepare_data_lstm(
                            hist, 
                            lookback=lookback_period, 
                            test_size=0.2,
                            prediction_days=prediction_days
                        )
                        
                        # Build and train model
                        lstm_model, history = build_lstm_model(
                            x_train, 
                            y_train,
                            epochs=50,
                            batch_size=32
                        )
                        
                        # Plot training history
                        hist_fig = px.line(
                            x=range(len(history.history['loss'])),
                            y=history.history['loss'],
                            title='LSTM Training Loss',
                            labels={'x': 'Epoch', 'y': 'Loss'}
                        )
                        hist_fig.add_scatter(x=range(len(history.history['val_loss'])), y=history.history['val_loss'], name='Validation Loss')
                        st.plotly_chart(hist_fig, use_container_width=True)
                        
                        # Make predictions on test data
                        test_predictions = lstm_model.predict(x_test)
                        
                        # Prepare data for future predictions
                        latest_data = prices[-lookback_period:].flatten()
                        # Scale the data using the same scaler
                        scaled_data = scaler.transform(prices)
                        x_future = np.array([scaled_data[-lookback_period:, 0]])
                        x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
                        
                        # Predict future values
                        future_predictions = lstm_model.predict(x_future)
                        
                        # Inverse transform predictions
                        test_predictions_orig = np.zeros((test_predictions.shape[0], test_predictions.shape[1], 1))
                        for i in range(test_predictions.shape[0]):
                            for j in range(test_predictions.shape[1]):
                                test_predictions_orig[i, j, 0] = test_predictions[i, j]
                        
                        test_predictions_orig = scaler.inverse_transform(test_predictions_orig.reshape(-1, 1))
                        
                        future_predictions_orig = scaler.inverse_transform(future_predictions.reshape(-1, 1))
                        
                        # Create future dates
                        last_date = hist.index[-1]
                        future_dates = [last_date + datetime.timedelta(days=i+1) for i in range(prediction_days)]
                        
                        # Plot predictions
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Future predictions
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_predictions_orig.flatten(),
                            mode='lines',
                            name='LSTM Predictions',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"LSTM Model: {ticker_input} Stock Price Prediction (Next {prediction_days} Days)",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=500,
                            hovermode="x unified"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction summary
                        st.markdown("#### LSTM Prediction Summary")
                        
                        # Calculate expected price changes
                        start_price = hist['Close'].iloc[-1]
                        end_price = future_predictions_orig[-1][0]
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
                            max_price = np.max(future_predictions_orig)
                            max_day = np.argmax(future_predictions_orig) + 1
                            st.metric(
                                "Predicted Maximum Price", 
                                f"${max_price:.2f}",
                                f"Day {max_day}"
                            )
                        
                        with pred_col3:
                            min_price = np.min(future_predictions_orig)
                            min_day = np.argmin(future_predictions_orig) + 1
                            st.metric(
                                "Predicted Minimum Price", 
                                f"${min_price:.2f}",
                                f"Day {min_day}"
                            )
                        
                        # Show detailed predictions in an expander
                        with st.expander("View Detailed LSTM Predictions"):
                            # Create a dataframe with future dates and predictions
                            future_df = pd.DataFrame({
                                'Date': future_dates,
                                'Predicted Price': future_predictions_orig.flatten()
                            })
                            future_df['Date'] = future_df['Date'].dt.date
                            future_df['Predicted Price'] = future_df['Predicted Price'].round(2)
                            
                            # Display table
                            st.dataframe(future_df, use_container_width=True)
                            
                            # Download option
                            csv = future_df.to_csv(index=False)
                            st.download_button(
                                label="Download LSTM Predictions as CSV",
                                data=csv,
                                file_name=f"{ticker_input}_lstm_predictions.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error in LSTM model training or prediction: {e}")
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
                    # Check if both models were successfully run
                    lstm_success = 'future_predictions_orig' in locals() and 'future_dates' in locals()
                    prophet_success = 'forecast' in locals()
                    
                    if lstm_success or prophet_success:
                        st.markdown("### Model Comparison")
                        
                        # Create combined plot
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'],
                            mode='lines',
                            name='Historical Data',
                            line=dict(color='blue')
                        ))
                        
                        # Add LSTM predictions if available
                        if lstm_success:
                            fig.add_trace(go.Scatter(
                                x=future_dates,
                                y=future_predictions_orig.flatten(),
                                mode='lines',
                                name='LSTM Predictions',
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
                        
                        - **LSTM**: Uses deep learning patterns from the entire price history
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
    
    #### 🧠 LSTM (Long Short-Term Memory)
    - Deep learning neural network specialized for time series
    - Captures complex patterns and relationships in historical prices
    - Provides detailed price forecasts
    
    #### 📊 Prophet
    - Facebook's statistical forecasting model
    - Handles seasonality and trend changes effectively
    - Includes confidence intervals for predictions
    
    #### 🔍 Model Comparison
    - Compare predictions from both models side-by-side
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