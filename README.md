# Volatility Forecasting Web App

## Description
The **Volatility Forecasting Web App** is an interactive platform that allows users to analyze and predict the volatility of financial assets. Leveraging advanced techniques like GARCH (Generalized Autoregressive Conditional Heteroskedasticity) and LSTM (Long Short-Term Memory) neural networks, the app enables users to import historical financial data directly from Yahoo Finance using `yfinance`, making it easier to generate accurate volatility forecasts.

---

## README

### Overview
The **Volatility Forecasting Web App** is designed for traders and analysts seeking to make informed decisions based on robust volatility predictions. By combining statistical and machine learning methods, users can interactively explore their data and visualize forecasting results.

### Key Features
- **User-Friendly Interface**: Intuitive design for easy data upload and visualization.
- **Data Import**: Users can fetch historical financial data directly from Yahoo Finance.
- **ACF and PACF Analysis**: Automatically generates plots for time series analysis.
- **GARCH Model Fitting**: Estimates conditional volatility using GARCH models.
- **LSTM Model Training**: Trains LSTM networks for future volatility forecasting.
- **Performance Metrics**: Visualizes training and validation metrics for both models.
- **Interactive Visualization**: Compare predicted and actual volatility in an engaging way.

### Technologies Used
- **Frontend**: Streamlit, Plotly for interactive visualizations.
- **Backend**: Python with Streamlit framework.
- **Data Processing**: Pandas, NumPy.
- **Modeling**: GARCH from the `arch` library, LSTM from `keras`.
- **Data Import**: `yfinance` for fetching historical data.
- **Visualization**: Plotly for interactive graphs.

### Installation
To set up the Volatility Forecasting Web App locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/wass1m-k/volatility-forecasting-web-app.git
   cd volatility-forecasting-web-app
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and go to `http://localhost:8501` to access the app.

### Usage
1. **Import Data**: Use the app interface to fetch historical financial data directly from Yahoo Finance using a stock ticker (e.g., `AAPL` for Apple).
   
2. **Run Analysis**: Choose GARCH or LSTM modeling options and initiate the analysis. The app will process the data, fit the selected models, and generate plots.

3. **View Results**: 
   - ACF and PACF plots.
   - Training and validation metrics visualization.
   - Interactive comparison of forecasted values with actual volatility.

### Example Code Summary
- **Data Import**: Fetches financial data from Yahoo Finance using `yfinance`.
- **Data Preprocessing**: Processes financial data for volatility calculation.
- **ACF and PACF Plots**: Generates autocorrelation plots for time series data.
- **GARCH Model**: Fits a GARCH model to forecast future volatility.
- **LSTM Model**: Constructs and trains an LSTM model for predictions.
- **Forecasting**: Predicts future volatility and visualizes results interactively.

### Contributing
Contributions are welcome! If you encounter issues or have suggestions, please create an issue or submit a pull request.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements
- Utilizes the `arch` library for GARCH modeling.
- Implements LSTM with Keras and TensorFlow.
- Uses `yfinance` for data import.
- Thanks to all contributors and libraries that made this project possible.

---
