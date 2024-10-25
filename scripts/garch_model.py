import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.preprocessing import MinMaxScaler
from scripts.data_preprocessing import transform_volatility_to_scaler

def acf_pacf_plots(ticker: str, realized_volatility: pd.DataFrame) -> None:
    """
    Plots the ACF and PACF plots for the given data.

    Args:
        ticker (str): The ticker symbol.
        realized_volatility (pd.DataFrame): The data to plot.
    
    Returns:
        plt.Figure: A Matplotlib figure containing the ACF and PACF plots.
    """
    
    fig, ax  = plt.subplots(2,2,figsize=(12,6),sharey=True,sharex=True)

    ax = ax.flatten()

    plot_acf(realized_volatility,ax[0],lags=35)
    ax[0].set_title(f'ACF of {ticker} Realized Volatility')

    plot_pacf(realized_volatility,ax[1],lags=35)
    ax[1].set_title(f'PACF of {ticker} Realized Volatility')

    plot_acf((realized_volatility)**2,ax[2],lags=35)
    ax[2].set_title(f'ACF of {ticker} Squared Realized Volatility')

    plot_pacf((realized_volatility)**2,ax[3],lags=35)
    ax[3].set_title(f'PACF of {ticker} Squared Realized Volatility')

    plt.tight_layout()
    
    return fig

def garch_data_splitting(returns: pd.DataFrame)-> pd.DataFrame:
    
    """
    Splits the given returns data into training, validation, and testing sets.

    Args:
        returns (pd.DataFrame): A pandas DataFrame containing the returns data.

    Returns:
        tuple: A tuple containing the training, validation, and testing sets.
    """
    test_size = 30
    val_size = 365
    
    split_time1 = len(returns) - (val_size + test_size)
    split_time2 = len(returns) - test_size
    
    train_idx = returns.index[:split_time1]
    val_idx = returns.index[split_time1:split_time2]
    test_idx = returns.index[split_time2:]
    
    r_train = returns['Returns'][train_idx]
    r_val = returns['Returns'][val_idx]     
    r_test = returns['Returns'][test_idx]
    
    return r_train, r_val, r_test 

def fit_garch(data: pd.DataFrame, p: int, q: int, dist: str):
    """
    Fits a GARCH(p,q) model to the given data.

    Args:
        data (pd.DataFrame): The data to fit the model to.
        p (int): The order of the GARCH(p,q) model.
        q (int): The order of the GARCH(p,q) model.
        dist (str): The distribution of the GARCH(p,q) model.

    Returns:
        arch_model.ARCH: The fitted GARCH(p,q) model.
    """
    
    dists = ['normal', 'skewt']
    
    if dist not in dists:
        raise ValueError(f"Invalid distribution. Expected one of: {', '.join(dists)}")
    
    if dist == 'normal':
        model = arch_model(data, p=p, q=q, dist=dist,
                        vol='GARCH', mean='AR', lags=2)
    else:
        model = arch_model(data, p=1, o=1, q=1, power=1.0,dist='skewt')
    
    model_fit = model.fit(disp='off')
    
    return model_fit

def scale_tf_cond_vol(model_fit):
    '''
    Scale & Transform Conditional Volatility
    Estimated by GARCH Models
    '''
    
    cond_vol = model_fit.conditional_volatility

    scaler = MinMaxScaler()

    scaler = scaler.fit(cond_vol.values.reshape(-1,1))

    scaled_cond_vol = transform_volatility_to_scaler(scaler, cond_vol)
    return scaler, scaled_cond_vol

def forecast_volatility(returns, n_future, dist, p, q) -> pd.DataFrame:
    """
    Forecasts the future volatility using a GARCH model.

    Args:
        realized_volatility (pd.DataFrame): Data containing realized volatility.
        n_future (int): Number of future periods to forecast.
        dist (str): Distribution assumption for the GARCH model.

    Returns:
        pd.DataFrame: Forecasted volatility values.
    """
    test_size = 30
    val_size = 365
    
    split_time1 = len(returns) - (test_size + val_size)
    split_time2 = len(returns) - test_size
    
    val_idx = returns.index[split_time1:split_time2]
    
    rolling_forecast = []
    
    for i in range(len(val_idx)):
        idx = val_idx[i]
        train = returns['Returns'][:idx]
        
        model = arch_model(train, dist=dist, vol='GARCH', mean='AR', lags=2,
                           p=p, q=q)
        model_fit = model.fit(disp='off')
        
        var = model_fit.forecast(horizon=n_future).variance.values
        pred = np.sqrt(np.mean(var))
        rolling_forecast.append(pred)
        
    garch_forecast = pd.Series(rolling_forecast, index=val_idx)
    
    return garch_forecast

def test_volatility(returns, n_future, dist, p, q) -> pd.DataFrame:
    """
    Forecasts the future volatility using a GARCH model on the test set.

    Args:
        realized_volatility (pd.DataFrame): Data containing realized volatility.
        n_future (int): Number of future periods to forecast.
        dist (str): Distribution assumption for the GARCH model.
        p (int): The order of the autoregressive term.
        q (int): The order of the moving average term.

    Returns:
        pd.DataFrame: Forecasted volatility values on the test set.
    """
    
    test_size = 30
    split_time2 = len(returns) - test_size
    
    test_idx = returns.index[split_time2:]
    
    rolling_forecast = []
    
    for i in range(len(test_idx)):
        idx = test_idx[i]
        test = returns['Returns'][:idx]
        
        model = arch_model(test, dist=dist, vol='GARCH', mean='AR', lags=2,
                           p=p, q=q)
        model_fit = model.fit(disp='off')
        
        var = model_fit.forecast(horizon=n_future).variance.values
        pred = np.sqrt(np.mean(var))
        rolling_forecast.append(pred)
        
    garch_forecast = pd.Series(rolling_forecast, index=test_idx)
    
    return garch_forecast

