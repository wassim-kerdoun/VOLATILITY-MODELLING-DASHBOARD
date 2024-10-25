import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def fetch_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches data from Yahoo Finance for a given ticker symbol and date range.

    Args:
        ticker (str): Ticker symbol to fetch data for.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the fetched data.
    """
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError(f"Please enter a valid ticker.")
    
    data = data[['Adj Close']]

    return data


# ### Revised Statement:
# "We work with log returns because they effectively scale data, 
# reducing variability and ensuring that values are more stable over time. 
# **Log returns are preferred in finance due to their time-additive property**, 
# meaning the total return over a period is the sum of the log returns for sub-periods, 
# which simplifies analysis. When we plot the distribution of log returns and compare it 
# against the normal distribution, we aim to identify discrepancies such as **fat tails** 
# (extreme values that occur more frequently than the normal distribution predicts). 

# Log returns are often assumed to be normally distributed in theoretical models, but **empirically**,
# they tend to exhibit fat tails and skewness. This comparison helps highlight those deviations, 
# showing that the real market distribution has **heavier tails** and **higher kurtosis** than the normal distribution. 
# This is why many volatility models, such as GARCH or stochastic volatility models,
# account for these discrepancies by adapting to the observed market behavior."

# ### Key Points for Accuracy:
# 1. **Log returns for additivity**: One reason log returns are used is that they are additive over time 
# (i.e., the log return over multiple periods is the sum of the log returns for each period). 
# This simplifies both calculation and interpretation, which is crucial in finance.

# 2. **Log returns aren’t perfectly normally distributed**: 
# While it's convenient to assume log returns are normally distributed in models, in reality,
# financial return distributions tend to exhibit **fat tails** and **skewness**. 
# The normal distribution assumption is often a first approximation, 
# but empirical distributions (like from actual market data) show more extreme values than a normal distribution would predict.

# 3. **Comparison of log return distributions to normal**: 
# The comparison between market-implied log return distributions and the normal distribution 
# allows us to see how much the actual data deviates from theoretical expectations. 
# This deviation is critical for understanding **risk** 
# (since fat tails imply greater risk of extreme moves than the normal distribution would suggest).

def LogReturns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data preprocessing steps on the fetched data.

    Args:
        data (pd.DataFrame): The fetched data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    log_returns_df = pd.DataFrame(log_returns, index=data.index)
    
    log_returns_df.dropna(inplace=True)
    
    return log_returns_df

def PCTChange(data: pd.DataFrame) -> pd.DataFrame:
    """
    Performs data preprocessing steps on the fetched data.

    Args:
        data (pd.DataFrame): The fetched data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    pct_change = data['Adj Close'].pct_change()
    
    pct_change_df = pd.DataFrame(pct_change*100, index=data.index)
    
    pct_change_df.dropna(inplace=True)
    
    return pct_change_df


# Realized volatility is a measure of the historical volatility of log returns over a given time window (e.g., 7 days). 
# It captures the actual volatility experienced in the market during that period. 
# With each data point, we compute the volatility based on the log returns from the preceding window (e.g., 7 days).
# This sliding window approach gives us a time series of realized volatility.

# When we compare realized volatility against a log-normal distribution,
# we do so because both exhibit right skewness—meaning that large values 
# (e.g., sudden spikes in volatility) are more frequent than would be predicted 
# by a symmetric distribution like the normal distribution.

# The key insight comes from recognizing the differences between the empirical realized volatility distribution a
# nd the assumed log-normal distribution. Realized volatility, especially in financial markets,
# often exhibits fat tails—indicating that extreme events (like market shocks or reactions to major news) 
# are more common than what would be predicted by a log-normal distribution. These fat tails represent sudden,
# large changes in volatility that log-normal models may fail to fully capture.

# By comparing the realized volatility distribution to the log-normal distribution, 
# we can identify discrepancies between observed market behavior and the assumptions of more traditional models. 
# In particular, the log-normal distribution typically underestimates the likelihood of extreme market movements, 
# while the realized volatility data (in practice) reflects these jumps and shocks,
# especially in periods of high market stress.

def RealizedVolatility(log_returns: pd.DataFrame, window: int, n_future: int) -> pd.DataFrame:
    """
    Calculates the realized volatility of the given log returns over a specified window.

    Args:
        log_returns (pd.DataFrame): Log returns of the data.
        window (int): The window size.
        n_future (int): The number of futures.

    Returns:
        pd.DataFrame: Realized volatility.
    """
    realized_volatility = log_returns.rolling(window=window).apply(lambda x: np.sqrt((x**2).sum() / (window-1)),
                                                                   raw=False)
    
    future_volatility = log_returns.shift(-n_future).rolling(window=window).apply(lambda x: np.sqrt((x**2).sum() / (window-1)),
                                                                       raw=False)
    
    realized_volatility.columns = ['Realized Volatility']
    future_volatility.columns = ['Future Volatility']
    
    realized_volatility = pd.concat([realized_volatility, future_volatility], axis=1)
    
    return realized_volatility

def indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates technical indicators for the preprocessed data.

    Args:
        data (pd.DataFrame): The preprocessed data.

    Returns:
        pd.DataFrame: Technical indicators.
    """
    indicators = pd.DataFrame(index=data.index)
    indicators['MA_21'] = data['Adj Close'].rolling(window=21).mean()
    indicators['MA_50'] = data['Adj Close'].rolling(window=50).mean()
    indicators['SMA_21'] = data['Adj Close'].rolling(window=21).mean()
    indicators['EMA_21'] = data['Adj Close'].ewm(span=21, adjust=False).mean()
    indicators['RSI'] = calculate_rsi(data['Adj Close'], window=14)
    
    data = data.join(indicators)
    
    return data

def plot_indicators(indicator: pd.DataFrame):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.8, 0.2])

    for indicator_name in indicator.columns:
        if indicator_name != 'RSI':
            fig.add_trace(go.Scatter(x=indicator.index, y=indicator[indicator_name], name=indicator_name),
                          row=1, col=1)

    fig.add_trace(go.Scatter(x=indicator.index, y=indicator['RSI'], name='RSI', line=dict(color='orange')),
                  row=2, col=1)

    fig.update_layout(height=600, width=1000, showlegend=True, 
                      title_text="Technical Indicators",
                      xaxis_title="Date",
                      yaxis_title="Indicator Value")

    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    return fig
    
def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a given time series.

    Args:
        close (pd.Series): The time series of closing prices.
        window (int, optional): The number of periods to consider for RSI calculation. Defaults to 14.

    Returns:
        pd.Series: The calculated RSI values.
    """
    
    delta = close.diff()
    
    gain = (delta.where(delta > 0, 0.0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=window).mean()
    
    rs = gain/loss
    rsi = 100 - (100 / (1+rs))
    
    return rsi

def qqplot(data:pd.DataFrame, dist='norm'):
    
    """
    Plots a Q-Q plot for the given data against a specified distribution.
    
    Parameters:
        data (pd.Series or np.array): The data to be plotted.
        dist (str): The distribution to compare against. Default is 'norm' for normal distribution.
        df (int): degree of freedom in case you are working with t-distribution
    """ 
    
    data = data.dropna()
    
    if dist == 'norm':
        (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = stats.probplot(data, dist=dist)
        
    elif dist == 't':
        (theoretical_quantiles, sample_quantiles), (slope, intercept, _) = stats.probplot(data, dist=dist, sparams=(10,))
 
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=theoretical_quantiles,
                             y=sample_quantiles,mode='markers',
                             name='Sample Data',marker=dict(color='blue',size=6)))
    
    best_fit_line = slope * np.array(theoretical_quantiles) + intercept
    fig.add_trace(go.Scatter(x=theoretical_quantiles,
                             y=best_fit_line,mode='lines',
                             name='Theoritical Line',
                             line=dict(color='red',width=2)))
    
    if dist == 'norm':
        fig.update_layout(title=f'Q-Q Plot of Normal Distribution',
                        xaxis_title='Theoritical Quantiles',
                        yaxis_title='Sample Quantiles',
                        showlegend=True,
                        template='plotly_white')
        
    elif dist == 't':
        fig.update_layout(title=f'Q-Q Plot of Student-t Distribution',
                        xaxis_title='Theoritical Quantiles',
                        yaxis_title='Sample Quantiles',
                        showlegend=True,
                        template='plotly_white')
    
    return fig

def data_splitting(realized_volatility: pd.DataFrame) -> tuple:
    
    """
    Preprocesses the given realized volatility data into training, validation and testing
    sets for a GARCH model.

    Args:
        realized_volatility (pd.DataFrame): The data to preprocess.

    Returns:
        tuple: A tuple containing the training, validation and testing sets for the GARCH model.
    """
    test_size = 30
    val_size = 365
    
    split_time1 = len(realized_volatility) - (test_size + val_size)
    split_time2 = len(realized_volatility) - test_size
    
    train_idx = realized_volatility.index[:split_time1]
    val_idx = realized_volatility.index[split_time1:split_time2]
    test_idx = realized_volatility.index[split_time2:]
    
    y_train = realized_volatility['Future Volatility'][train_idx].values.reshape(-1, 1)
    y_val = realized_volatility['Future Volatility'][val_idx].values.reshape(-1, 1)
    y_test = realized_volatility['Future Volatility'][test_idx].values.reshape(-1, 1)
    
    X_train = realized_volatility['Realized Volatility'][train_idx].values 
    X_val = realized_volatility['Realized Volatility'][val_idx].values     
    X_test = realized_volatility['Realized Volatility'][test_idx].values   
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def transform_volatility_to_scaler(scaler, tf_series):
    
    """
    Scales a given time series of volatility values using a given scaler.

    Args:
        scaler (MinMaxScaler): The scaler to use for scaling.
        tf_series (pd.Series or np.ndarray): The time series of volatility values to scale.

    Returns:
        pd.Series or np.ndarray: The scaled time series of volatility values.
    """
    
    if isinstance(tf_series, pd.Series):
        idx = tf_series.index
        scaled_values = scaler.transform(tf_series.values.reshape(-1, 1))[:, 0]
        output = pd.Series(scaled_values, index=idx)
    else:
        output = scaler.transform(tf_series.reshape(-1, 1))[:, 0]
    return output

def windowed_dataset(X, y, n_past):
    """
    Prepares a windowed dataset for time series modeling.

    Args:
        X (pd.Series): The input feature time series data.
        y (pd.Series): The target time series data.
        n_past (int): The number of past observations to use as input features.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - dataX: The array of input feature sequences.
            - dataY: The array of target values corresponding to each input sequence.
    """
    if len(X) != len(y):
        raise ValueError("Input feature series and target series must have the same length.")
    
    X_seq = []
    y_seq = []
    for i in range(len(X) - n_past):
        X_seq.append(X[i:i + n_past].values)
        y_seq.append(y[i + n_past])
        
    return np.array(X_seq), np.array(y_seq)


# Predicting realized volatility provides valuable insights, 
# but it does not directly indicate the direction of future price movements (upward or downward). 
# Here’s a more detailed explanation:

# ### Understanding Realized Volatility

# 1. **Definition:**
#    - Realized volatility measures the variability of returns over a specified period,
# usually based on historical data. 
# It captures how much the price of an asset has moved over time but does not specify whether the movement is positive or negative.

# 2. **Direction of Movement:**
#    - While realized volatility can signal that large price movements are expected 
# (i.e., it can be high during turbulent market conditions), 
# it does not inherently provide information about the **direction** of those movements. For example:
#      - **High Realized Volatility:** Indicates that there have been significant fluctuations in price,
# but the price could have moved both up and down. 
# It does not tell you whether the next movement will be upward or downward.
#      - **Low Realized Volatility:** Suggests that prices have been stable, 
# but again, it doesn't indicate whether the next movement will be up or down.

# 3. **Usefulness in Trading:**
#    - Traders and analysts often use realized volatility to assess risk and make decisions about options pricing, 
# hedging strategies, and other risk management techniques. 
#    - For instance, if realized volatility is high, traders might anticipate more significant swings in price, 
# leading to strategic decisions regarding buying or selling options to capitalize on that expected movement.

# ### Relationship with Log Returns

# - **Log Returns:** 
#   - Log returns are used to calculate returns in a way that accounts for compounding.
# They can be normally distributed (under certain assumptions) 
# and allow analysts to compare relative changes in price over time.
  
# - **Interplay with Realized Volatility:**
#   - While realized volatility helps gauge how much the price has varied, 
# it is **not predictive of the sign of future log returns.** Therefore:
#     - **Realized Volatility** might signal that you should prepare for significant movements, 
# but you’ll still need to rely on other analyses, models, or indicators to predict whether those movements will be upward or downward.
#     - Models like GARCH or LSTM can be helpful in forecasting future returns based on historical data, 
# but realized volatility itself does not indicate the direction of those future log returns.

# ### Conclusion

# In summary, while realized volatility is a crucial metric for understanding market dynamics and risk, 
# it should be complemented with other tools and analyses to make informed predictions 
# about the direction of future price movements. Predicting realized volatility 
# gives you a sense of how much price might move,
# but understanding the **why** behind those movements requires a deeper analysis of market conditions, 
# sentiment, and other factors influencing prices.