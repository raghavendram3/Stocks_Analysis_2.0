import streamlit as st
from utils.analytics import inject_google_tag_manager

# Configure page - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="StockTrackPro - Stock Analysis",
    page_icon="📈",
    layout="wide"
)

# Inject Google Tag Manager
inject_google_tag_manager()

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
# Technical analysis libraries
from ta.trend import MACD, SMAIndicator, IchimokuIndicator, ADXIndicator, AroonIndicator, DPOIndicator, MassIndex, PSARIndicator, VortexIndicator, KSTIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, AwesomeOscillatorIndicator, KAMAIndicator, ROCIndicator, TSIIndicator, WilliamsRIndicator, UltimateOscillator
from ta.volatility import BollingerBands, AverageTrueRange, UlcerIndex, DonchianChannel
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator, MFIIndicator, ForceIndexIndicator, EaseOfMovementIndicator, VolumePriceTrendIndicator

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
    
    # 8. Advanced Volume Indicators
    if "Volume" in df_ta.columns:
        # Accumulation/Distribution Line
        adl = AccDistIndexIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            volume=df_ta["Volume"]
        )
        df_ta["adl"] = adl.acc_dist_index()
        
        # Chaikin Money Flow (CMF)
        cmf = ChaikinMoneyFlowIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            volume=df_ta["Volume"],
            window=20
        )
        df_ta["cmf"] = cmf.chaikin_money_flow()
        
        # Money Flow Index (MFI)
        mfi = MFIIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            volume=df_ta["Volume"],
            window=14
        )
        df_ta["mfi"] = mfi.money_flow_index()
        
        # Force Index
        fi = ForceIndexIndicator(
            close=df_ta["Close"],
            volume=df_ta["Volume"],
            window=13
        )
        df_ta["force_index"] = fi.force_index()
        
        # Ease of Movement
        eom = EaseOfMovementIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            volume=df_ta["Volume"],
            window=14
        )
        df_ta["eom"] = eom.ease_of_movement()
        
        # Volume Price Trend
        vpt = VolumePriceTrendIndicator(
            close=df_ta["Close"],
            volume=df_ta["Volume"]
        )
        df_ta["vpt"] = vpt.volume_price_trend()
    
    # 9. Advanced Trend Indicators
    
    # ADX (Average Directional Index) - helps determine trend strength
    # ADX needs at least 2 * window periods of data
    if len(df_ta) >= 30:  # Minimum data needed for ADX with window=14
        adx = ADXIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            window=14
        )
        df_ta["adx"] = adx.adx()
        df_ta["adx_pos"] = adx.adx_pos()  # +DI
        df_ta["adx_neg"] = adx.adx_neg()  # -DI
    
    # Aroon Indicator - helps identify when trends are likely to change
    aroon = AroonIndicator(
        high=df_ta["High"],
        low=df_ta["Low"],
        window=25
    )
    df_ta["aroon_up"] = aroon.aroon_up()
    df_ta["aroon_down"] = aroon.aroon_down()
    df_ta["aroon_indicator"] = df_ta["aroon_up"] - df_ta["aroon_down"]
    
    # DPO (Detrended Price Oscillator) - eliminates long-term trends
    dpo = DPOIndicator(
        close=df_ta["Close"],
        window=20
    )
    df_ta["dpo"] = dpo.dpo()
    
    # Mass Index - identifies trend reversals by analyzing range expansions
    if len(df_ta) >= 25:  # Mass Index needs sufficient data
        mass = MassIndex(
            high=df_ta["High"],
            low=df_ta["Low"],
            window_fast=9,
            window_slow=25
        )
        df_ta["mass_index"] = mass.mass_index()
    
    # Parabolic SAR - trend following indicator
    if len(df_ta) >= 25:
        psar = PSARIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            step=0.02,
            max_step=0.2
        )
        df_ta["psar"] = psar.psar()
        df_ta["psar_up"] = psar.psar_up()
        df_ta["psar_down"] = psar.psar_down()
        df_ta["psar_up_indicator"] = psar.psar_up_indicator()
        df_ta["psar_down_indicator"] = psar.psar_down_indicator()
    
    # Vortex Indicator - identifies the start of a trend and its direction
    if len(df_ta) >= 14:
        vortex = VortexIndicator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            window=14
        )
        df_ta["vortex_pos"] = vortex.vortex_indicator_pos()
        df_ta["vortex_neg"] = vortex.vortex_indicator_neg()
        df_ta["vortex_diff"] = df_ta["vortex_pos"] - df_ta["vortex_neg"]
    
    # KST Oscillator (Know Sure Thing) - long-term momentum indicator
    if len(df_ta) >= 55:  # KST needs substantial history
        kst = KSTIndicator(
            close=df_ta["Close"],
            roc1=10, roc2=15, roc3=20, roc4=30,
            window1=10, window2=10, window3=10, window4=15,
            nsig=9
        )
        df_ta["kst"] = kst.kst()
        df_ta["kst_sig"] = kst.kst_sig()
        df_ta["kst_diff"] = df_ta["kst"] - df_ta["kst_sig"]
    
    # 10. Advanced Momentum Indicators
    
    # Awesome Oscillator for momentum detection
    ao = AwesomeOscillatorIndicator(
        high=df_ta["High"],
        low=df_ta["Low"],
        window1=5,
        window2=34
    )
    df_ta["awesome_oscillator"] = ao.awesome_oscillator()
    
    # Kaufman's Adaptive Moving Average (KAMA)
    kama = KAMAIndicator(
        close=df_ta["Close"],
        window=10,
        pow1=2,
        pow2=30
    )
    df_ta["kama"] = kama.kama()
    
    # Rate of Change (ROC)
    roc = ROCIndicator(
        close=df_ta["Close"],
        window=12
    )
    df_ta["roc"] = roc.roc()
    
    # TSI (True Strength Index)
    tsi = TSIIndicator(
        close=df_ta["Close"],
        window_slow=25,
        window_fast=13,
        fillna=True
    )
    df_ta["tsi"] = tsi.tsi()
    
    # Williams %R - momentum indicator that measures overbought/oversold levels
    wr = WilliamsRIndicator(
        high=df_ta["High"],
        low=df_ta["Low"],
        close=df_ta["Close"],
        lbp=14
    )
    df_ta["williams_r"] = wr.williams_r()
    
    # Ultimate Oscillator - multi-timeframe momentum indicator
    if len(df_ta) >= 28:
        uo = UltimateOscillator(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            window1=7,
            window2=14,
            window3=28,
            weight1=4.0,
            weight2=2.0,
            weight3=1.0
        )
        df_ta["ultimate_oscillator"] = uo.ultimate_oscillator()
    
    # 11. Advanced Volatility Indicators
    
    # ATR (Average True Range) - market volatility
    atr = AverageTrueRange(
        high=df_ta["High"],
        low=df_ta["Low"],
        close=df_ta["Close"],
        window=14
    )
    df_ta["atr"] = atr.average_true_range()
    
    # Ulcer Index - downside risk
    if len(df_ta) >= 14:
        ui = UlcerIndex(
            close=df_ta["Close"],
            window=14
        )
        df_ta["ulcer_index"] = ui.ulcer_index()
    
    # Donchian Channel - volatility indicator
    if len(df_ta) >= 20:
        dc = DonchianChannel(
            high=df_ta["High"],
            low=df_ta["Low"],
            close=df_ta["Close"],
            window=20
        )
        df_ta["donchian_high"] = dc.donchian_channel_hband()
        df_ta["donchian_mid"] = dc.donchian_channel_mband()
        df_ta["donchian_low"] = dc.donchian_channel_lband()
        df_ta["donchian_width"] = dc.donchian_channel_wband()
    
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
    
    # 9. ADX Analysis
    if "adx" in df_ta.columns and "adx_pos" in df_ta.columns and "adx_neg" in df_ta.columns:
        adx_value = latest["adx"]
        di_plus = latest["adx_pos"]
        di_minus = latest["adx_neg"]
        
        # Check ADX strength and direction
        if adx_value >= 25:
            if di_plus > di_minus:
                signals.append(f"🔵 ADX: Strong trend detected (ADX: {adx_value:.2f}) with +DI > -DI, indicating a bullish trend.")
            elif di_minus > di_plus:
                signals.append(f"🔴 ADX: Strong trend detected (ADX: {adx_value:.2f}) with -DI > +DI, indicating a bearish trend.")
            else:
                signals.append(f"➡️ ADX: Strong trend, but direction unclear (ADX: {adx_value:.2f}).")
        elif adx_value < 20:
            signals.append(f"➡️ ADX: Weak trend detected (ADX: {adx_value:.2f}) - may indicate a ranging or consolidating market.")
    
    # 10. Parabolic SAR Analysis
    if "psar" in df_ta.columns and "psar_up_indicator" in df_ta.columns and "psar_down_indicator" in df_ta.columns:
        # Check if PSAR indicator changed recently (from down to up or vice versa)
        bullish = prev["psar_up_indicator"] == 0 and latest["psar_up_indicator"] == 1
        bearish = prev["psar_down_indicator"] == 0 and latest["psar_down_indicator"] == 1
        
        status = get_indicator_status("Parabolic SAR", bullish, bearish)
        if status:
            signals.append(status + (" - Trend reversed to bullish" if bullish else " - Trend reversed to bearish"))
        elif latest["psar_up_indicator"] == 1:
            signals.append("🔵 Parabolic SAR: In uptrend - price is above the SAR dots.")
        elif latest["psar_down_indicator"] == 1:
            signals.append("🔴 Parabolic SAR: In downtrend - price is below the SAR dots.")
    
    # 11. Money Flow Index (MFI) Analysis
    if "mfi" in df_ta.columns:
        mfi_value = latest["mfi"]
        
        if mfi_value > 80:
            signals.append(f"⚠️ Money Flow Index: Overbought condition (MFI: {mfi_value:.2f}) - potential reversal or correction ahead.")
        elif mfi_value < 20:
            signals.append(f"🔍 Money Flow Index: Oversold condition (MFI: {mfi_value:.2f}) - potential buying opportunity.")
        
        # Check for divergence between MFI and price
        if "rsi" in df_ta.columns:
            mfi_trend = np.polyfit(range(5), df_ta["mfi"].iloc[-5:].values, 1)[0]
            rsi_trend = np.polyfit(range(5), df_ta["rsi"].iloc[-5:].values, 1)[0]
            
            if mfi_trend < 0 and rsi_trend > 0:
                signals.append("⚠️ MFI Divergence: MFI declining while RSI rising - potential hidden bearish signal.")
            elif mfi_trend > 0 and rsi_trend < 0:
                signals.append("🔍 MFI Divergence: MFI rising while RSI declining - potential hidden bullish signal.")
    
    # 12. Chaikin Money Flow (CMF) Analysis
    if "cmf" in df_ta.columns:
        cmf_value = latest["cmf"]
        
        if cmf_value > 0.1:
            signals.append(f"🔵 Chaikin Money Flow: Strong positive value ({cmf_value:.2f}) - indicates accumulation (buying pressure).")
        elif cmf_value < -0.1:
            signals.append(f"🔴 Chaikin Money Flow: Strong negative value ({cmf_value:.2f}) - indicates distribution (selling pressure).")
        elif -0.05 <= cmf_value <= 0.05:
            signals.append(f"➡️ Chaikin Money Flow: Near zero ({cmf_value:.2f}) - indicates a balance between buying and selling.")
    
    # 13. Williams %R Analysis
    if "williams_r" in df_ta.columns:
        wr_value = latest["williams_r"]
        
        if wr_value > -20:  # Williams %R uses a negative scale
            signals.append("⚠️ Williams %R: Above -20 - indicates overbought conditions.")
        elif wr_value < -80:
            signals.append("🔍 Williams %R: Below -80 - indicates oversold conditions.")
    
    # 14. Rate of Change (ROC) Analysis
    if "roc" in df_ta.columns:
        roc_value = latest["roc"]
        
        if roc_value > 5:
            signals.append(f"🔵 Rate of Change: Strong positive value ({roc_value:.2f}%) - indicates rapid price increase.")
        elif roc_value < -5:
            signals.append(f"🔴 Rate of Change: Strong negative value ({roc_value:.2f}%) - indicates rapid price decrease.")
    
    # 15. Ultimate Oscillator Analysis
    if "ultimate_oscillator" in df_ta.columns:
        uo_value = latest["ultimate_oscillator"]
        
        if uo_value > 70:
            signals.append(f"⚠️ Ultimate Oscillator: Above 70 ({uo_value:.2f}) - indicates overbought conditions.")
        elif uo_value < 30:
            signals.append(f"🔍 Ultimate Oscillator: Below 30 ({uo_value:.2f}) - indicates oversold conditions.")
    
    # 16. Awesome Oscillator Analysis
    if "awesome_oscillator" in df_ta.columns and len(df_ta) >= 5:
        ao_values = df_ta["awesome_oscillator"].iloc[-5:].values
        
        # Zero-line crossover
        if ao_values[-2] < 0 and ao_values[-1] > 0:
            signals.append("🔵 Awesome Oscillator: Crossed above zero line - bullish signal.")
        elif ao_values[-2] > 0 and ao_values[-1] < 0:
            signals.append("🔴 Awesome Oscillator: Crossed below zero line - bearish signal.")
        
        # Twin peaks setup (look for two consecutive peaks/valleys)
        if len(ao_values) >= 5:
            # Bullish twin peaks: two valleys below zero, second higher than first
            if (ao_values[0] < 0 and ao_values[1] < ao_values[0] and 
                ao_values[2] > ao_values[1] and ao_values[3] < ao_values[2] and 
                ao_values[3] > ao_values[1] and ao_values[4] > ao_values[3]):
                signals.append("🔵 Awesome Oscillator: Bullish twin peaks pattern detected.")
            
            # Bearish twin peaks: two peaks above zero, second lower than first
            elif (ao_values[0] > 0 and ao_values[1] > ao_values[0] and 
                  ao_values[2] < ao_values[1] and ao_values[3] > ao_values[2] and 
                  ao_values[3] < ao_values[1] and ao_values[4] < ao_values[3]):
                signals.append("🔴 Awesome Oscillator: Bearish twin peaks pattern detected.")
    
    # 17. Aroon Indicator Analysis
    if "aroon_up" in df_ta.columns and "aroon_down" in df_ta.columns:
        aroon_up = latest["aroon_up"]
        aroon_down = latest["aroon_down"]
        
        # Aroon Up/Down crossover
        bullish = prev["aroon_up"] <= prev["aroon_down"] and aroon_up > aroon_down
        bearish = prev["aroon_up"] >= prev["aroon_down"] and aroon_up < aroon_down
        
        status = get_indicator_status("Aroon Indicator", bullish, bearish)
        if status:
            signals.append(status)
        # Extreme values
        elif aroon_up > 70 and aroon_down < 30:
            signals.append(f"🔵 Aroon Indicator: Aroon Up ({aroon_up:.0f}) much higher than Aroon Down ({aroon_down:.0f}) - strong uptrend.")
        elif aroon_down > 70 and aroon_up < 30:
            signals.append(f"🔴 Aroon Indicator: Aroon Down ({aroon_down:.0f}) much higher than Aroon Up ({aroon_up:.0f}) - strong downtrend.")
        elif aroon_up < 30 and aroon_down < 30:
            signals.append("➡️ Aroon Indicator: Both indicators below 30 - consolidation or trend weakness.")
    
    # 18. Average True Range (ATR) Analysis - Volatility
    if "atr" in df_ta.columns and len(df_ta) >= 20:
        atr_value = latest["atr"]
        atr_percent = (atr_value / latest["Close"]) * 100
        
        # Compare current ATR to recent average
        recent_atr_avg = df_ta["atr"].iloc[-20:-1].mean()
        atr_change = ((atr_value - recent_atr_avg) / recent_atr_avg) * 100
        
        if atr_change > 30:
            signals.append(f"⚠️ ATR: Significant increase in volatility ({atr_change:.1f}% above average) - potential for large price swings.")
        elif atr_change < -30:
            signals.append(f"🔍 ATR: Significant decrease in volatility ({abs(atr_change):.1f}% below average) - market calming, may precede a breakout.")
        
        # Absolute ATR percentage
        if atr_percent > 3:
            signals.append(f"⚠️ ATR: High volatility detected - daily price movement averaging {atr_percent:.2f}% of stock price.")
        elif atr_percent < 1:
            signals.append(f"🔍 ATR: Low volatility detected - daily price movement averaging only {atr_percent:.2f}% of stock price.")
    
    # 19. Donchian Channel Analysis
    if all(x in df_ta.columns for x in ["donchian_high", "donchian_low", "donchian_mid"]):
        close = latest["Close"]
        upper = latest["donchian_high"]
        lower = latest["donchian_low"]
        
        # Price near upper or lower band
        if close >= upper * 0.98:
            signals.append("🔵 Donchian Channel: Price near the upper band - breakout potential or strong uptrend.")
        elif close <= lower * 1.02:
            signals.append("🔴 Donchian Channel: Price near the lower band - breakdown potential or strong downtrend.")
        
        # Breakout detection
        if len(df_ta) >= 5:
            prev_range = max(df_ta["High"].iloc[-6:-1]) - min(df_ta["Low"].iloc[-6:-1])
            curr_range = upper - lower
            
            # Channel narrowing
            if curr_range < prev_range * 0.7:
                signals.append("🔔 Donchian Channel: Channel narrowing - potential for a volatility breakout soon.")
            # Channel widening significantly
            elif curr_range > prev_range * 1.5:
                signals.append("⚠️ Donchian Channel: Channel expanding rapidly - increased volatility detected.")
    
    # 20. KST (Know Sure Thing) Analysis
    if "kst" in df_ta.columns and "kst_sig" in df_ta.columns:
        kst = latest["kst"]
        kst_sig = latest["kst_sig"]
        
        # KST crossover
        bullish = prev["kst"] < prev["kst_sig"] and kst > kst_sig
        bearish = prev["kst"] > prev["kst_sig"] and kst < kst_sig
        
        status = get_indicator_status("KST Oscillator", bullish, bearish)
        if status:
            signals.append(status)
        # KST above/below zero
        elif kst > 0 and kst_sig > 0:
            signals.append("🔵 KST Oscillator: Both KST and signal line above zero - bullish momentum.")
        elif kst < 0 and kst_sig < 0:
            signals.append("🔴 KST Oscillator: Both KST and signal line below zero - bearish momentum.")
    
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
                
                # Create tabs for different categories of indicators
                indicator_tab1, indicator_tab2, indicator_tab3, indicator_tab4, indicator_tab5 = st.tabs([
                    "Core Indicators", "Momentum", "Trend", "Volume", "Volatility"
                ])
                
                with indicator_tab1:
                    # Core/Primary indicators
                    core_indicators = pd.DataFrame({
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
                    
                    st.dataframe(core_indicators.tail(10), use_container_width=True)
                
                with indicator_tab2:
                    # Momentum indicators
                    momentum_cols = {
                        'Date': df_ta.index,
                        'RSI (14)': df_ta['rsi'].round(2),
                        'Stochastic %K': df_ta['stoch'].round(2),
                        'Stochastic %D': df_ta['stoch_signal'].round(2)
                    }
                    
                    # Add advanced momentum indicators if they exist
                    if 'awesome_oscillator' in df_ta.columns:
                        momentum_cols['Awesome Oscillator'] = df_ta['awesome_oscillator'].round(2)
                    if 'roc' in df_ta.columns:
                        momentum_cols['Rate of Change'] = df_ta['roc'].round(2)
                    if 'williams_r' in df_ta.columns:
                        momentum_cols['Williams %R'] = df_ta['williams_r'].round(2)
                    if 'tsi' in df_ta.columns:
                        momentum_cols['True Strength Index'] = df_ta['tsi'].round(2)
                    if 'ultimate_oscillator' in df_ta.columns:
                        momentum_cols['Ultimate Oscillator'] = df_ta['ultimate_oscillator'].round(2)
                    
                    momentum_df = pd.DataFrame(momentum_cols).reset_index(drop=True)
                    st.dataframe(momentum_df.tail(10), use_container_width=True)
                
                with indicator_tab3:
                    # Trend indicators
                    trend_cols = {
                        'Date': df_ta.index,
                        'SMA 50': df_ta['sma_50'].round(2),
                        'SMA 200': df_ta['sma_200'].round(2),
                        'MACD': df_ta['macd'].round(2),
                        'MACD Signal': df_ta['macd_signal'].round(2)
                    }
                    
                    # Add advanced trend indicators if they exist
                    if 'adx' in df_ta.columns:
                        trend_cols['ADX'] = df_ta['adx'].round(2)
                        trend_cols['+DI'] = df_ta['adx_pos'].round(2)
                        trend_cols['-DI'] = df_ta['adx_neg'].round(2)
                    if 'aroon_up' in df_ta.columns:
                        trend_cols['Aroon Up'] = df_ta['aroon_up'].round(2)
                        trend_cols['Aroon Down'] = df_ta['aroon_down'].round(2)
                    if 'dpo' in df_ta.columns:
                        trend_cols['Detrended Price Osc'] = df_ta['dpo'].round(2)
                    if 'kst' in df_ta.columns:
                        trend_cols['KST'] = df_ta['kst'].round(2)
                        trend_cols['KST Signal'] = df_ta['kst_sig'].round(2)
                    if 'psar' in df_ta.columns:
                        trend_cols['PSAR'] = df_ta['psar'].round(2)
                    if 'vortex_pos' in df_ta.columns:
                        trend_cols['Vortex +VI'] = df_ta['vortex_pos'].round(2)
                        trend_cols['Vortex -VI'] = df_ta['vortex_neg'].round(2)
                    
                    trend_df = pd.DataFrame(trend_cols).reset_index(drop=True)
                    st.dataframe(trend_df.tail(10), use_container_width=True)
                
                with indicator_tab4:
                    # Volume indicators
                    volume_cols = {
                        'Date': df_ta.index,
                        'Volume': df_ta['Volume'],
                        'On-Balance Volume': df_ta['obv'].round(2)
                    }
                    
                    # Add advanced volume indicators if they exist
                    if 'adl' in df_ta.columns:
                        volume_cols['A/D Line'] = df_ta['adl'].round(2)
                    if 'cmf' in df_ta.columns:
                        volume_cols['Chaikin Money Flow'] = df_ta['cmf'].round(2)
                    if 'mfi' in df_ta.columns:
                        volume_cols['Money Flow Index'] = df_ta['mfi'].round(2)
                    if 'force_index' in df_ta.columns:
                        volume_cols['Force Index'] = df_ta['force_index'].round(2)
                    if 'eom' in df_ta.columns:
                        volume_cols['Ease of Movement'] = df_ta['eom'].round(2)
                    if 'vpt' in df_ta.columns:
                        volume_cols['Volume Price Trend'] = df_ta['vpt'].round(2)
                    
                    volume_df = pd.DataFrame(volume_cols).reset_index(drop=True)
                    st.dataframe(volume_df.tail(10), use_container_width=True)
                
                with indicator_tab5:
                    # Volatility indicators
                    volatility_cols = {
                        'Date': df_ta.index,
                        'Close': df_ta['Close'],
                        'Bollinger Upper': df_ta['bollinger_hband'].round(2),
                        'Bollinger Middle': df_ta['bollinger_mavg'].round(2),
                        'Bollinger Lower': df_ta['bollinger_lband'].round(2)
                    }
                    
                    # Add advanced volatility indicators if they exist
                    if 'atr' in df_ta.columns:
                        volatility_cols['ATR'] = df_ta['atr'].round(2)
                    if 'ulcer_index' in df_ta.columns:
                        volatility_cols['Ulcer Index'] = df_ta['ulcer_index'].round(2)
                    if 'donchian_high' in df_ta.columns:
                        volatility_cols['Donchian High'] = df_ta['donchian_high'].round(2)
                        volatility_cols['Donchian Middle'] = df_ta['donchian_mid'].round(2)
                        volatility_cols['Donchian Low'] = df_ta['donchian_low'].round(2)
                        volatility_cols['Donchian Width'] = df_ta['donchian_width'].round(2)
                    
                    volatility_df = pd.DataFrame(volatility_cols).reset_index(drop=True)
                    st.dataframe(volatility_df.tail(10), use_container_width=True)
                
                # Combine all indicators for download
                all_indicators_cols = {
                    'Date': df_ta.index,
                    'Open': df_ta['Open'],
                    'High': df_ta['High'],
                    'Low': df_ta['Low'],
                    'Close': df_ta['Close'],
                    'Volume': df_ta['Volume']
                }
                
                # Add all technical indicators that exist in the dataframe
                for col in df_ta.columns:
                    if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date'] and not pd.isna(df_ta[col]).all():
                        if isinstance(df_ta[col].iloc[0], (int, float)):
                            all_indicators_cols[col] = df_ta[col].round(2)
                
                all_indicators_df = pd.DataFrame(all_indicators_cols).reset_index(drop=True)
                
                # Download button for indicator data
                st.download_button(
                    label="Download All Technical Indicators as CSV",
                    data=all_indicators_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{ticker_input}_all_technical_indicators.csv",
                    mime="text/csv"
                )
            
            with tech_tab3:
                # Create categories for chart selection with many options
                chart_category = st.selectbox(
                    "Select Indicator Category",
                    ["Core Indicators", "Momentum Indicators", "Trend Indicators", "Volume Indicators", "Volatility Indicators", "Advanced Indicators"]
                )
                
                if chart_category == "Core Indicators":
                    chart_option = st.selectbox(
                        "Select Core Indicator to Display",
                        ["RSI", "MACD", "Bollinger Bands", "Moving Averages", "Stochastic Oscillator", "OBV"]
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
                        if 'kama' in df_ta.columns:
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['kama'], name="KAMA", line=dict(dash='dot')))
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
                        
                    elif chart_option == "OBV":
                        # On-Balance Volume Chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['obv'], name="OBV"))
                        
                        # Add trend line
                        if len(df_ta) >= 20:
                            # Calculate trend line using last 20 periods
                            x_trend = list(range(20))
                            y_trend = df_ta['obv'].iloc[-20:].values
                            slope, intercept = np.polyfit(x_trend, y_trend, 1)
                            trend_line = [slope * i + intercept for i in x_trend]
                            
                            trend_dates = df_ta.index[-20:]
                            fig.add_trace(go.Scatter(x=trend_dates, y=trend_line, name="OBV Trend", 
                                                 line=dict(dash='dash', color='orange')))
                        
                        fig.update_layout(title="On-Balance Volume (OBV)", height=400, yaxis_title="OBV Value")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_category == "Momentum Indicators":
                    # Check which momentum indicators are available
                    available_indicators = []
                    if 'rsi' in df_ta.columns:
                        available_indicators.append("RSI")
                    if 'stoch' in df_ta.columns:
                        available_indicators.append("Stochastic Oscillator")
                    if 'awesome_oscillator' in df_ta.columns:
                        available_indicators.append("Awesome Oscillator")
                    if 'roc' in df_ta.columns:
                        available_indicators.append("Rate of Change (ROC)")
                    if 'williams_r' in df_ta.columns:
                        available_indicators.append("Williams %R")
                    if 'tsi' in df_ta.columns:
                        available_indicators.append("True Strength Index (TSI)")
                    if 'ultimate_oscillator' in df_ta.columns:
                        available_indicators.append("Ultimate Oscillator")
                    
                    if not available_indicators:
                        st.warning("No momentum indicators available.")
                    else:
                        chart_option = st.selectbox("Select Momentum Indicator", available_indicators)
                        
                        if chart_option == "RSI":
                            # RSI Chart (already defined above)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['rsi'], name="RSI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[70] * len(df_ta), name="Overbought (70)", line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[30] * len(df_ta), name="Oversold (30)", line=dict(color='green', dash='dash')))
                            fig.update_layout(title="Relative Strength Index (RSI)", height=400, yaxis_title="RSI Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Stochastic Oscillator":
                            # Stochastic Oscillator Chart (already defined above)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['stoch'], name="%K"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['stoch_signal'], name="%D"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[80] * len(df_ta), name="Overbought (80)", line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[20] * len(df_ta), name="Oversold (20)", line=dict(color='green', dash='dash')))
                            fig.update_layout(title="Stochastic Oscillator", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Awesome Oscillator":
                            # Awesome Oscillator Chart
                            fig = go.Figure()
                            
                            # Create a list of colors for the bars (green if value > prev, red otherwise)
                            colors = ['green' if val > df_ta['awesome_oscillator'].iloc[i-1] 
                                     else 'red' for i, val in enumerate(df_ta['awesome_oscillator']) 
                                     if i > 0]
                            colors.insert(0, 'green')  # Add color for the first bar
                            
                            fig.add_trace(go.Bar(
                                x=df_ta.index,
                                y=df_ta['awesome_oscillator'],
                                name="Awesome Oscillator",
                                marker_color=colors
                            ))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Awesome Oscillator", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Rate of Change (ROC)":
                            # ROC Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['roc'], name="ROC"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Rate of Change (ROC)", height=400, yaxis_title="ROC Value (%)")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Williams %R":
                            # Williams %R Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['williams_r'], name="Williams %R"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[-20] * len(df_ta), name="Overbought (-20)", 
                                                    line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[-80] * len(df_ta), name="Oversold (-80)", 
                                                    line=dict(color='green', dash='dash')))
                            fig.update_layout(title="Williams %R", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "True Strength Index (TSI)":
                            # TSI Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['tsi'], name="TSI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[25] * len(df_ta), name="Overbought (25)", 
                                                    line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[-25] * len(df_ta), name="Oversold (-25)", 
                                                    line=dict(color='green', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="True Strength Index (TSI)", height=400, yaxis_title="TSI Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Ultimate Oscillator":
                            # Ultimate Oscillator Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['ultimate_oscillator'], name="Ultimate Oscillator"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[70] * len(df_ta), name="Overbought (70)", 
                                                    line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[30] * len(df_ta), name="Oversold (30)", 
                                                    line=dict(color='green', dash='dash')))
                            fig.update_layout(title="Ultimate Oscillator", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif chart_category == "Trend Indicators":
                    # Check which trend indicators are available
                    available_indicators = []
                    if 'macd' in df_ta.columns:
                        available_indicators.append("MACD")
                    if 'sma_50' in df_ta.columns and 'sma_200' in df_ta.columns:
                        available_indicators.append("Moving Averages")
                    if 'adx' in df_ta.columns:
                        available_indicators.append("ADX (Average Directional Index)")
                    if 'aroon_up' in df_ta.columns:
                        available_indicators.append("Aroon Oscillator")
                    if 'psar' in df_ta.columns:
                        available_indicators.append("Parabolic SAR")
                    if 'vortex_pos' in df_ta.columns:
                        available_indicators.append("Vortex Indicator")
                    if 'kst' in df_ta.columns:
                        available_indicators.append("KST Oscillator")
                    
                    if not available_indicators:
                        st.warning("No trend indicators available.")
                    else:
                        chart_option = st.selectbox("Select Trend Indicator", available_indicators)
                        
                        if chart_option == "MACD":
                            # MACD Chart (already defined above)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd'], name="MACD"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd_signal'], name="Signal Line"))
                            fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['macd_histogram'], name="Histogram"))
                            fig.update_layout(title="MACD", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Moving Averages":
                            # Moving Averages Chart (already defined above)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['sma_50'], name="SMA 50"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['sma_200'], name="SMA 200"))
                            fig.update_layout(title="Moving Averages", height=400, yaxis_title="Price")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "ADX (Average Directional Index)":
                            # ADX Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['adx'], name="ADX"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['adx_pos'], name="+DI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['adx_neg'], name="-DI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[25] * len(df_ta), name="Trend Strength Threshold", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Average Directional Index (ADX)", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Aroon Oscillator":
                            # Aroon Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['aroon_up'], name="Aroon Up"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['aroon_down'], name="Aroon Down"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['aroon_indicator'], name="Aroon Oscillator"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[70] * len(df_ta), name="Strong Trend Threshold", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Aroon Indicator", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Parabolic SAR":
                            # Parabolic SAR Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                            
                            # Extract PSAR dots for uptrend and downtrend separately
                            uptrend_idx = df_ta['psar_up_indicator'] == 1
                            downtrend_idx = df_ta['psar_down_indicator'] == 1
                            
                            if uptrend_idx.any():
                                fig.add_trace(go.Scatter(
                                    x=df_ta.index[uptrend_idx],
                                    y=df_ta['psar'][uptrend_idx],
                                    mode='markers',
                                    name="PSAR (Uptrend)",
                                    marker=dict(color='green', size=5)
                                ))
                            
                            if downtrend_idx.any():
                                fig.add_trace(go.Scatter(
                                    x=df_ta.index[downtrend_idx],
                                    y=df_ta['psar'][downtrend_idx],
                                    mode='markers',
                                    name="PSAR (Downtrend)",
                                    marker=dict(color='red', size=5)
                                ))
                            
                            fig.update_layout(title="Parabolic SAR", height=400, yaxis_title="Price")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Vortex Indicator":
                            # Vortex Indicator Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['vortex_pos'], name="+VI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['vortex_neg'], name="-VI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[1] * len(df_ta), name="Signal Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Vortex Indicator", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "KST Oscillator":
                            # KST Oscillator Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['kst'], name="KST"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['kst_sig'], name="Signal Line"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="KST Oscillator", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif chart_category == "Volume Indicators":
                    # Check which volume indicators are available
                    available_indicators = []
                    if 'obv' in df_ta.columns:
                        available_indicators.append("On-Balance Volume (OBV)")
                    if 'adl' in df_ta.columns:
                        available_indicators.append("Accumulation/Distribution Line")
                    if 'cmf' in df_ta.columns:
                        available_indicators.append("Chaikin Money Flow")
                    if 'mfi' in df_ta.columns:
                        available_indicators.append("Money Flow Index")
                    if 'force_index' in df_ta.columns:
                        available_indicators.append("Force Index")
                    if 'eom' in df_ta.columns:
                        available_indicators.append("Ease of Movement")
                    if 'vpt' in df_ta.columns:
                        available_indicators.append("Volume Price Trend")
                    
                    if not available_indicators:
                        st.warning("No volume indicators available.")
                    else:
                        chart_option = st.selectbox("Select Volume Indicator", available_indicators)
                        
                        if chart_option == "On-Balance Volume (OBV)":
                            # OBV Chart (already defined above)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['obv'], name="OBV"))
                            fig.update_layout(title="On-Balance Volume (OBV)", height=400, yaxis_title="OBV Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Accumulation/Distribution Line":
                            # A/D Line Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['adl'], name="A/D Line"))
                            fig.update_layout(title="Accumulation/Distribution Line", height=400, yaxis_title="A/D Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Chaikin Money Flow":
                            # CMF Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['cmf'], name="CMF"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0.1] * len(df_ta), name="Bullish Line", 
                                                    line=dict(color='green', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[-0.1] * len(df_ta), name="Bearish Line", 
                                                    line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Chaikin Money Flow (CMF)", height=400, yaxis_title="CMF Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Money Flow Index":
                            # MFI Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['mfi'], name="MFI"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[80] * len(df_ta), name="Overbought (80)", 
                                                    line=dict(color='red', dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[20] * len(df_ta), name="Oversold (20)", 
                                                    line=dict(color='green', dash='dash')))
                            fig.update_layout(title="Money Flow Index (MFI)", height=400, yaxis_title="MFI Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Force Index":
                            # Force Index Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['force_index'], name="Force Index"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Force Index", height=400, yaxis_title="Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Ease of Movement":
                            # EOM Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['eom'], name="Ease of Movement"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                    line=dict(color='black', dash='dash')))
                            fig.update_layout(title="Ease of Movement (EOM)", height=400, yaxis_title="EOM Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Volume Price Trend":
                            # VPT Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['vpt'], name="Volume Price Trend"))
                            fig.update_layout(title="Volume Price Trend (VPT)", height=400, yaxis_title="VPT Value")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif chart_category == "Volatility Indicators":
                    # Check which volatility indicators are available
                    available_indicators = []
                    if 'bollinger_hband' in df_ta.columns:
                        available_indicators.append("Bollinger Bands")
                    if 'atr' in df_ta.columns:
                        available_indicators.append("Average True Range (ATR)")
                    if 'ulcer_index' in df_ta.columns:
                        available_indicators.append("Ulcer Index")
                    if 'donchian_high' in df_ta.columns:
                        available_indicators.append("Donchian Channel")
                    
                    if not available_indicators:
                        st.warning("No volatility indicators available.")
                    else:
                        chart_option = st.selectbox("Select Volatility Indicator", available_indicators)
                        
                        if chart_option == "Bollinger Bands":
                            # Bollinger Bands Chart (already defined above)
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['bollinger_hband'], name="Upper Band", line=dict(dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['bollinger_mavg'], name="MA (20)"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['bollinger_lband'], name="Lower Band", line=dict(dash='dash')))
                            fig.update_layout(title="Bollinger Bands", height=400, yaxis_title="Price")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Average True Range (ATR)":
                            # ATR Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['atr'], name="ATR"))
                            
                            # Calculate and add the ATR percentage (relative to price)
                            atr_pct = (df_ta['atr'] / df_ta['Close']) * 100
                            fig.add_trace(go.Scatter(x=df_ta.index, y=atr_pct, name="ATR % of Price", 
                                                    line=dict(dash='dot', color='orange')))
                            
                            fig.update_layout(title="Average True Range (ATR)", height=400, 
                                             yaxis_title="ATR Value", yaxis2=dict(title="ATR % of Price", 
                                                                               overlaying='y', side='right'))
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Ulcer Index":
                            # Ulcer Index Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['ulcer_index'], name="Ulcer Index"))
                            fig.update_layout(title="Ulcer Index", height=400, yaxis_title="Ulcer Index Value")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Donchian Channel":
                            # Donchian Channel Chart
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['donchian_high'], name="Upper Band", line=dict(dash='dash')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['donchian_mid'], name="Middle Band"))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['donchian_low'], name="Lower Band", line=dict(dash='dash')))
                            fig.update_layout(title="Donchian Channel", height=400, yaxis_title="Price")
                            st.plotly_chart(fig, use_container_width=True)
                
                elif chart_category == "Advanced Indicators":
                    # Check which advanced indicators are available
                    available_indicators = []
                    
                    if "ichimoku_a" in df_ta.columns and "ichimoku_b" in df_ta.columns:
                        available_indicators.append("Ichimoku Cloud")
                    if "fib_0" in df_ta.columns and "fib_100" in df_ta.columns:
                        available_indicators.append("Fibonacci Retracement")
                    
                    # Create custom combo charts
                    available_indicators.append("RSI + Stochastic Combo")
                    available_indicators.append("MACD + OBV Combo")
                    available_indicators.append("Triple Screen System")
                    
                    if not available_indicators:
                        st.warning("No advanced indicators available.")
                    else:
                        chart_option = st.selectbox("Select Advanced Indicator", available_indicators)
                        
                        if chart_option == "Ichimoku Cloud":
                            # Ichimoku Cloud Chart
                            fig = go.Figure()
                            
                            # Fill the cloud
                            dates = df_ta.index
                            
                            # Create the filled area for the cloud
                            # Green when Senkou Span A > Senkou Span B, red otherwise
                            for i in range(len(dates)-1):
                                if df_ta['ichimoku_a'].iloc[i] >= df_ta['ichimoku_b'].iloc[i]:
                                    # Green cloud
                                    fig.add_trace(go.Scatter(
                                        x=[dates[i], dates[i+1]],
                                        y=[df_ta['ichimoku_a'].iloc[i], df_ta['ichimoku_a'].iloc[i+1]],
                                        fill=None,
                                        mode='lines',
                                        line=dict(color='rgba(0,0,0,0)'),
                                        showlegend=False
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=[dates[i], dates[i+1]],
                                        y=[df_ta['ichimoku_b'].iloc[i], df_ta['ichimoku_b'].iloc[i+1]],
                                        fill='tonexty',
                                        mode='lines',
                                        line=dict(color='rgba(0,0,0,0)'),
                                        fillcolor='rgba(0,255,0,0.1)',
                                        showlegend=False
                                    ))
                                else:
                                    # Red cloud
                                    fig.add_trace(go.Scatter(
                                        x=[dates[i], dates[i+1]],
                                        y=[df_ta['ichimoku_a'].iloc[i], df_ta['ichimoku_a'].iloc[i+1]],
                                        fill=None,
                                        mode='lines',
                                        line=dict(color='rgba(0,0,0,0)'),
                                        showlegend=False
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=[dates[i], dates[i+1]],
                                        y=[df_ta['ichimoku_b'].iloc[i], df_ta['ichimoku_b'].iloc[i+1]],
                                        fill='tonexty',
                                        mode='lines',
                                        line=dict(color='rgba(0,0,0,0)'),
                                        fillcolor='rgba(255,0,0,0.1)',
                                        showlegend=False
                                    ))
                            
                            # Add price and indicator lines
                            fig.add_trace(go.Scatter(x=dates, y=df_ta['Close'], name="Price", line=dict(color='blue')))
                            fig.add_trace(go.Scatter(x=dates, y=df_ta['ichimoku_conversion'], name="Conversion Line (Tenkan-sen)", line=dict(color='orange')))
                            fig.add_trace(go.Scatter(x=dates, y=df_ta['ichimoku_base'], name="Base Line (Kijun-sen)", line=dict(color='red')))
                            fig.add_trace(go.Scatter(x=dates, y=df_ta['ichimoku_a'], name="Leading Span A (Senkou Span A)", line=dict(color='green', dash='dash')))
                            fig.add_trace(go.Scatter(x=dates, y=df_ta['ichimoku_b'], name="Leading Span B (Senkou Span B)", line=dict(color='purple', dash='dash')))
                            
                            fig.update_layout(title="Ichimoku Cloud", height=500, yaxis_title="Price")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Fibonacci Retracement":
                            # Fibonacci Retracement Chart
                            fig = go.Figure()
                            
                            # Price line
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Close Price"))
                            
                            # Fibonacci levels
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['fib_0'], name="0% (Low)", line=dict(dash='dash', color='green')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['fib_23.6'], name="23.6%", line=dict(dash='dash', color='gray')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['fib_38.2'], name="38.2%", line=dict(dash='dash', color='orange')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['fib_50'], name="50%", line=dict(dash='dash', color='black')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['fib_61.8'], name="61.8%", line=dict(dash='dash', color='purple')))
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['fib_100'], name="100% (High)", line=dict(dash='dash', color='red')))
                            
                            fig.update_layout(title="Fibonacci Retracement Levels", height=500, yaxis_title="Price")
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "RSI + Stochastic Combo":
                            # RSI + Stochastic Combo Chart for divergence analysis
                            fig = make_subplots(rows=3, cols=1, 
                                              shared_xaxes=True, 
                                              vertical_spacing=0.05,
                                              row_heights=[0.5, 0.25, 0.25])
                            
                            # Price Chart
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Price"), row=1, col=1)
                            
                            # RSI Chart
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['rsi'], name="RSI"), row=2, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[70] * len(df_ta), name="Overbought (70)", 
                                                  line=dict(color='red', dash='dash')), row=2, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[30] * len(df_ta), name="Oversold (30)", 
                                                  line=dict(color='green', dash='dash')), row=2, col=1)
                            
                            # Stochastic Chart
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['stoch'], name="%K"), row=3, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['stoch_signal'], name="%D"), row=3, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[80] * len(df_ta), name="Overbought (80)", 
                                                  line=dict(color='red', dash='dash')), row=3, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[20] * len(df_ta), name="Oversold (20)", 
                                                  line=dict(color='green', dash='dash')), row=3, col=1)
                            
                            fig.update_layout(height=700, title_text="RSI + Stochastic (Divergence Analysis)")
                            fig.update_yaxes(title_text="Price", row=1, col=1)
                            fig.update_yaxes(title_text="RSI", row=2, col=1)
                            fig.update_yaxes(title_text="Stochastic", row=3, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "MACD + OBV Combo":
                            # MACD + OBV Combo Chart for trend confirmation
                            fig = make_subplots(rows=3, cols=1, 
                                              shared_xaxes=True, 
                                              vertical_spacing=0.05,
                                              row_heights=[0.5, 0.25, 0.25])
                            
                            # Price Chart
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['Close'], name="Price"), row=1, col=1)
                            
                            # MACD Chart
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd'], name="MACD"), row=2, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd_signal'], name="Signal"), row=2, col=1)
                            fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['macd_histogram'], name="Histogram"), row=2, col=1)
                            
                            # OBV Chart - Normalize for visibility
                            min_obv = df_ta['obv'].min()
                            max_obv = df_ta['obv'].max()
                            norm_obv = (df_ta['obv'] - min_obv) / (max_obv - min_obv) * 100
                            
                            fig.add_trace(go.Scatter(x=df_ta.index, y=norm_obv, name="OBV (Normalized)"), row=3, col=1)
                            
                            fig.update_layout(height=700, title_text="MACD + OBV (Trend Confirmation)")
                            fig.update_yaxes(title_text="Price", row=1, col=1)
                            fig.update_yaxes(title_text="MACD", row=2, col=1)
                            fig.update_yaxes(title_text="OBV", row=3, col=1)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        elif chart_option == "Triple Screen System":
                            # Elder's Triple Screen System
                            fig = make_subplots(rows=3, cols=1, 
                                              shared_xaxes=True, 
                                              vertical_spacing=0.05,
                                              row_heights=[0.4, 0.3, 0.3])
                            
                            # First Screen: MACD for trend
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd'], name="MACD"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['macd_signal'], name="Signal"), row=1, col=1)
                            fig.add_trace(go.Bar(x=df_ta.index, y=df_ta['macd_histogram'], name="Histogram"), row=1, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                  line=dict(color='black', dash='dash')), row=1, col=1)
                            
                            # Second Screen: RSI for overbought/oversold
                            fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['rsi'], name="RSI"), row=2, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[70] * len(df_ta), name="Overbought (70)", 
                                                  line=dict(color='red', dash='dash')), row=2, col=1)
                            fig.add_trace(go.Scatter(x=df_ta.index, y=[30] * len(df_ta), name="Oversold (30)", 
                                                  line=dict(color='green', dash='dash')), row=2, col=1)
                            
                            # Third Screen: Force Index for confirmation
                            if 'force_index' in df_ta.columns:
                                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['force_index'], name="Force Index"), row=3, col=1)
                                fig.add_trace(go.Scatter(x=df_ta.index, y=[0] * len(df_ta), name="Zero Line", 
                                                      line=dict(color='black', dash='dash')), row=3, col=1)
                            else:
                                # Use OBV as alternative if Force Index is not available
                                fig.add_trace(go.Scatter(x=df_ta.index, y=df_ta['obv'], name="OBV"), row=3, col=1)
                            
                            fig.update_layout(height=800, title_text="Elder's Triple Screen Trading System")
                            fig.update_yaxes(title_text="MACD (Trend)", row=1, col=1)
                            fig.update_yaxes(title_text="RSI (Timing)", row=2, col=1)
                            fig.update_yaxes(title_text="Force Index (Confirmation)" if 'force_index' in df_ta.columns else "OBV (Confirmation)", row=3, col=1)
                            
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