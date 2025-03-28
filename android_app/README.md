# StockTrackPro Android App

## Overview
StockTrackPro Android application is a native mobile version of the web-based stock analysis tool. It provides comprehensive stock analysis functionality directly on Android devices.

## Features
- Real-time stock data from Yahoo Finance
- Interactive price history charts
- Technical analysis with multiple indicators:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Moving Averages (SMA, EMA)
  - Stochastic Oscillator
  - OBV (On-Balance Volume)
  - Ichimoku Cloud
  - Fibonacci Retracement Levels
- Investment insights based on technical and fundamental analysis
- User-friendly interface with intuitive navigation

## Technical Details
- Written in Kotlin
- MVVM architecture with LiveData and ViewModel
- Retrofit for API communication
- MPAndroidChart for data visualization
- Coroutines for asynchronous operations

## Building the Project
1. Clone the repository
2. Open the project in Android Studio
3. Sync Gradle files
4. Build and run the application on an emulator or physical device

## Requirements
- Android API level 24 (Android 7.0) or higher
- Internet connection for fetching stock data

## Libraries Used
- AndroidX Core, AppCompat, and Material Components
- ConstraintLayout for responsive UI
- MPAndroidChart for stock price visualization
- Retrofit & OkHttp for networking
- Kotlin Coroutines for asynchronous programming
- Lifecycle components for architecture

## Development Roadmap
- [ ] Add authentication system
- [ ] Implement portfolio tracking
- [ ] Add push notifications for price alerts
- [ ] Integrate watchlist functionality
- [ ] Support for additional financial instruments (ETFs, Mutual Funds)