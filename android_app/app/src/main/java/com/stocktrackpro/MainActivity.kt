package com.stocktrackpro

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.URL
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : AppCompatActivity() {
    
    private lateinit var stockSymbolInput: EditText
    private lateinit var searchButton: Button
    private lateinit var stockPriceChart: LineChart
    private lateinit var companyNameText: TextView
    private lateinit var currentPriceText: TextView
    private lateinit var indicatorsList: RecyclerView
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize UI components
        stockSymbolInput = findViewById(R.id.stockSymbolInput)
        searchButton = findViewById(R.id.searchButton)
        stockPriceChart = findViewById(R.id.stockPriceChart)
        companyNameText = findViewById(R.id.companyNameText)
        currentPriceText = findViewById(R.id.currentPriceText)
        indicatorsList = findViewById(R.id.indicatorsList)
        
        // Set up RecyclerView
        indicatorsList.layoutManager = LinearLayoutManager(this)
        
        // Set up button click listener
        searchButton.setOnClickListener {
            val symbol = stockSymbolInput.text.toString().trim().uppercase()
            if (symbol.isNotEmpty()) {
                fetchStockData(symbol)
            }
        }
    }
    
    private fun fetchStockData(symbol: String) {
        // Show loading indicator
        // ...
        
        CoroutineScope(Dispatchers.IO).launch {
            try {
                // This would typically use a proper API with authentication
                // For demonstration purposes only
                val apiUrl = "https://api.example.com/stocks/$symbol"
                
                // In a real app, you would use a proper HTTP client like Retrofit or OkHttp
                // This is simplified for demonstration
                val response = URL(apiUrl).readText()
                val jsonObject = JSONObject(response)
                
                withContext(Dispatchers.Main) {
                    // Update UI with stock data
                    updateStockInfo(jsonObject)
                    updateStockChart(jsonObject)
                    updateTechnicalIndicators(jsonObject)
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    // Handle error and update UI
                    showError("Error fetching stock data: ${e.message}")
                }
            }
        }
    }
    
    private fun updateStockInfo(data: JSONObject) {
        // Extract and display company information
        val companyName = data.getString("companyName")
        val currentPrice = data.getDouble("currentPrice")
        val priceChange = data.getDouble("priceChange")
        val percentChange = data.getDouble("percentChange")
        
        companyNameText.text = companyName
        currentPriceText.text = String.format("$%.2f", currentPrice)
        
        // Update price change indicator
        // ...
    }
    
    private fun updateStockChart(data: JSONObject) {
        // Parse historical price data from JSON
        val historicalData = data.getJSONArray("historicalData")
        val entries = ArrayList<Entry>()
        
        for (i in 0 until historicalData.length()) {
            val dataPoint = historicalData.getJSONObject(i)
            val timestamp = dataPoint.getLong("timestamp")
            val closePrice = dataPoint.getDouble("close").toFloat()
            entries.add(Entry(timestamp.toFloat(), closePrice))
        }
        
        // Create dataset and update chart
        val dataSet = LineDataSet(entries, "Price History")
        dataSet.color = getColor(R.color.colorPrimary)
        dataSet.setDrawCircles(false)
        dataSet.lineWidth = 2f
        
        val lineData = LineData(dataSet)
        stockPriceChart.data = lineData
        stockPriceChart.invalidate()
    }
    
    private fun updateTechnicalIndicators(data: JSONObject) {
        // Extract technical indicator data
        val indicators = data.getJSONObject("technicalIndicators")
        
        // Create list of indicators to display
        val indicatorsList = ArrayList<IndicatorItem>()
        
        // MACD
        val macd = indicators.getJSONObject("macd")
        val macdValue = macd.getDouble("value")
        val macdSignal = macd.getDouble("signal")
        val macdHistogram = macd.getDouble("histogram")
        val macdBullish = macdHistogram > 0
        
        indicatorsList.add(
            IndicatorItem(
                "MACD",
                String.format("%.2f, Signal: %.2f, Hist: %.2f", macdValue, macdSignal, macdHistogram),
                macdBullish
            )
        )
        
        // RSI
        val rsi = indicators.getDouble("rsi")
        val rsiBullish = rsi < 70 && rsi > 30
        indicatorsList.add(
            IndicatorItem(
                "RSI",
                String.format("%.2f", rsi),
                rsiBullish
            )
        )
        
        // Bollinger Bands
        val bollinger = indicators.getJSONObject("bollingerBands")
        val upper = bollinger.getDouble("upper")
        val middle = bollinger.getDouble("middle")
        val lower = bollinger.getDouble("lower")
        val currentPrice = data.getDouble("currentPrice")
        val bollingerBullish = currentPrice < middle
        
        indicatorsList.add(
            IndicatorItem(
                "Bollinger Bands",
                String.format("U: %.2f, M: %.2f, L: %.2f", upper, middle, lower),
                bollingerBullish
            )
        )
        
        // ... Add more indicators
        
        // Update RecyclerView with indicators
        this.indicatorsList.adapter = IndicatorAdapter(indicatorsList)
    }
    
    private fun showError(message: String) {
        // Show error message
        // ...
    }
}

// Data class for technical indicators
data class IndicatorItem(
    val name: String,
    val value: String,
    val isBullish: Boolean
)

// Adapter for RecyclerView
class IndicatorAdapter(private val indicators: List<IndicatorItem>) : 
    RecyclerView.Adapter<IndicatorAdapter.ViewHolder>() {
    
    class ViewHolder(view: android.view.View) : RecyclerView.ViewHolder(view) {
        val nameText: TextView = view.findViewById(R.id.indicatorName)
        val valueText: TextView = view.findViewById(R.id.indicatorValue)
        val signalIcon: android.widget.ImageView = view.findViewById(R.id.indicatorSignal)
    }
    
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_indicator, parent, false)
        return ViewHolder(view)
    }
    
    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val indicator = indicators[position]
        holder.nameText.text = indicator.name
        holder.valueText.text = indicator.value
        
        // Set icon based on bullish/bearish signal
        holder.signalIcon.setImageResource(
            if (indicator.isBullish) R.drawable.ic_arrow_up
            else R.drawable.ic_arrow_down
        )
    }
    
    override fun getItemCount() = indicators.size
}