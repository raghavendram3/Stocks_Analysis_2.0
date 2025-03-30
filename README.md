# StockTrackPro 📈

A powerful web application for stock analysis and visualization built with Streamlit.

## Features

- 📊 **Real-time Stock Data**: Fetch and display current market data from Yahoo Finance
- 📉 **Interactive Charts**: View stock price history with candlestick patterns and volume data
- 📈 **Technical Analysis**: Access over 20+ technical indicators organized in categories
- 🧠 **Predictive Analytics**: ML and statistical forecasting for stock price predictions
- 💰 **Financial Metrics**: Access key financial ratios and metrics for investment analysis
- 📋 **Data Downloads**: Export historical data and financial metrics as CSV files
- 💡 **Investment Tips**: Get basic analysis and investment insights based on financial data

## Live Demo

[Access StockTrackPro here](https://stocktrackpro.streamlit.app) *(Update this link once deployed)*

## Installation

### Local Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/StockTrackPro.git
   cd StockTrackPro
   ```

2. Install dependencies:
   ```
   pip install -r requirements-deploy.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Enter a stock symbol (e.g., AAPL, MSFT, GOOGL) in the sidebar
2. Select a time period for historical data
3. View the stock charts, metrics, and analysis
4. Explore technical indicators and predictive models
5. Download data as needed

## Embedding on Your Website

You can add StockTrackPro as a page on your website by embedding it as an iframe. Here's how:

1. Deploy your StockTrackPro application to a hosting platform (like Streamlit Cloud, Heroku, etc.)
2. Use the provided `embed.html` file as a template
3. Replace `YOUR_APP_URL` in the template with the actual URL of your deployed application
4. Copy the HTML code into your website where you want StockTrackPro to appear

Example embed code:
```html
<div class="responsive-wrapper">
    <div class="streamlit-container">
        <iframe 
            src="YOUR_APP_URL" 
            class="streamlit-embed" 
            allowfullscreen
            title="StockTrackPro"
        ></iframe>
    </div>
</div>
```

You can customize the width, height, and styling of the embedded app by modifying the CSS in the template.

## Technologies Used

- [Streamlit](https://streamlit.io/) - The web application framework
- [YFinance](https://pypi.org/project/yfinance/) - Yahoo Finance API wrapper
- [Plotly](https://plotly.com/) - Interactive data visualization
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [TA Library](https://technical-analysis-library-in-python.readthedocs.io/) - Technical analysis indicators
- [Scikit-learn](https://scikit-learn.org/) - Machine learning models
- [Prophet](https://facebook.github.io/prophet/) - Time series forecasting

## Deployment

This application can be deployed to:

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Heroku](https://heroku.com)
- [Replit](https://replit.com)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stock data provided by Yahoo Finance
- Technical indicators based on Investopedia recommendations
- Built with ❤️ by [Your Name]
