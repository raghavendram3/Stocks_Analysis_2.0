package com.stocktrackpro.models

data class StockData(
    val symbol: String,
    val companyName: String,
    val currentPrice: Double,
    val previousClose: Double,
    val open: Double,
    val high: Double,
    val low: Double,
    val volume: Long,
    val marketCap: Long,
    val peRatio: Double?,
    val dividendYield: Double?,
    val historicalData: List<HistoricalDataPoint>,
    val technicalIndicators: TechnicalIndicators
)

data class HistoricalDataPoint(
    val date: String,
    val open: Double,
    val high: Double,
    val low: Double,
    val close: Double,
    val volume: Long
)

data class TechnicalIndicators(
    val macd: MACD,
    val rsi: Double,
    val bollingerBands: BollingerBands,
    val movingAverages: MovingAverages,
    val stochasticOscillator: StochasticOscillator,
    val obv: Double,
    val ichimokuCloud: IchimokuCloud,
    val fibonacciLevels: FibonacciLevels
)

data class MACD(
    val value: Double,
    val signal: Double,
    val histogram: Double
)

data class BollingerBands(
    val upper: Double,
    val middle: Double,
    val lower: Double
)

data class MovingAverages(
    val sma20: Double,
    val sma50: Double,
    val sma200: Double,
    val ema12: Double,
    val ema26: Double
)

data class StochasticOscillator(
    val k: Double,
    val d: Double
)

data class IchimokuCloud(
    val conversionLine: Double,
    val baseLine: Double,
    val leadingSpanA: Double,
    val leadingSpanB: Double,
    val laggingSpan: Double
)

data class FibonacciLevels(
    val level0: Double,
    val level23_6: Double,
    val level38_2: Double,
    val level50: Double,
    val level61_8: Double,
    val level100: Double
)