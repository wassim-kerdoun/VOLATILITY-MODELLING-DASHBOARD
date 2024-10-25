import pandas as pd
import numpy as np
from scripts.data_preprocessing import windowed_dataset
import keras
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
from typing import Literal

def build_lstm(realized_volatility: pd.DataFrame, n_past: int) -> tuple:
    """
    Prepares the LSTM model for training based on the realized volatility data and past observations.

    Args:
        realized_volatility (pd.DataFrame): The realized volatility data.
        n_past (int): The number of past days to use for forecasting.

    Returns:
        tuple: A tuple containing the training input features, target values, LSTM model, and EarlyStopping callback.
    """
    
    realized_volatility.dropna(inplace=True)
    n_past = n_past
    test_size = 30
    val_size = 365
    split_time1 = len(realized_volatility) - (test_size + val_size)
    train_idx = realized_volatility.index[:split_time1]
    y_train = realized_volatility['Future Volatility'][train_idx]
    X_train = realized_volatility['Realized Volatility'][train_idx]
    X_seq, _ = windowed_dataset(X_train, y_train, n_past)
    X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
    
    lstm = keras.models.Sequential([
        keras.layers.LSTM(128, activation='relu', return_sequences=True, input_shape=(X_seq.shape[1], 1)),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.3),
        keras.layers.LSTM(32, return_sequences=False),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    lstm.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                 loss='mse', metrics=['mae'])
    
    EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  mode='min',
                                                  restore_best_weights=True,
                                                  patience=30)
    
    return X_train, y_train, lstm, EarlyStopping

def fit_lstm(model, X_train, y_train, n_past, EarlyStopping,
             epochs, batch_size) -> np.ndarray:
    
    """
    Fits the LSTM model to the given training data using the provided callbacks and hyperparameters.

    Args:
        model (keras.Model): The LSTM model to fit.
        X_train (np.ndarray): The input sequences for training.
        y_train (np.ndarray): The target values for training.
        EarlyStopping (keras.callbacks.EarlyStopping): The EarlyStopping callback to prevent overfitting.
        epochs (int): The number of epochs to fit the model.
        batch_size (int): The batch size for training.

    Returns:
        np.ndarray: The history object containing the training and validation metrics.
    """
    
    X_train_mat, y_train_mat = windowed_dataset(X_train, y_train, n_past)
    
    lstm_fit = model.fit(X_train_mat, y_train_mat,
                         callbacks=[EarlyStopping],
                         validation_split=0.2, shuffle=True,
                         verbose=1, epochs=epochs, batch_size=batch_size)
    
    return lstm_fit

def plot_model_metrics(model):
    
    """
    Plots the training and validation metrics for the given model using subplots.

    Args:
        model (keras.Model): The model to plot the metrics for.

    Returns:
        go.Figure: The figure containing the training and validation metrics subplots.
    """
    
    history = pd.DataFrame(model.history)
    
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=("Training and Validation Loss", 
                                       "Training and Validation MAE"))

    figure.add_trace(go.Scatter(
        x=history.index + 1,
        y=history['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    ), row=1, col=1)

    figure.add_trace(go.Scatter(
        x=history.index + 1,
        y=history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='green')
    ), row=1, col=1)

    figure.add_trace(go.Scatter(
        x=history.index + 1,
        y=history['mae'],
        mode='lines',
        name='Training MAE',
        line=dict(color='red')
    ), row=2, col=1)

    figure.add_trace(go.Scatter(
        x=history.index + 1,
        y=history['val_mae'],
        mode='lines',
        name='Validation MAE',
        line=dict(color='green')
    ), row=2, col=1)

    figure.update_layout(
        title='Model Training and Validation Metrics',
        xaxis_title='Epoch',
        height=900,
        width=1000,
        template='plotly_white'
    )

    return figure

def lstm_forecast(model, realized_volatility, n_past, index: Literal['train_idx', 'val_idx', 'test_idx']) -> pd.Series:
    """
    Makes a forecast of future volatility using the given LSTM model.

    Args:
        model (keras.Model): The LSTM model to use for forecasting.
        realized_volatility (pd.DataFrame): The realized volatility data.
        n_past (int): The number of past days to use for forecasting.

    Returns:
        pd.Series: The forecasted future volatility values.
    """
    
    test_size = 30
    val_size = 365
    split_time1 = len(realized_volatility) - (test_size + val_size)
    split_time2 = len(realized_volatility) - test_size
    
    if index == 'train_idx':
        idxx = realized_volatility.index[:split_time1]
    elif index == 'val_idx':
        idxx = realized_volatility.index[split_time1:split_time2]
    elif index == 'test_idx':
        idxx = realized_volatility.index[split_time2:]
    else:
        raise ValueError("index must be either 'train_idx' or 'val_idx'.")

    start_idx = idxx[0] - timedelta(days=n_past - 1)
    end_idx = idxx[-1]

    realized_volatility_slice = realized_volatility.loc[start_idx:end_idx]
    if realized_volatility_slice.shape[0] < n_past:
        raise ValueError("Not enough data for the specified number of past observations.")
    
    mat_X, y_future = windowed_dataset(
        realized_volatility_slice['Realized Volatility'],
        realized_volatility_slice['Future Volatility'],
        n_past
    )

    if mat_X.shape[0] != y_future.shape[0]:
        raise ValueError("Input feature series and target series must have the same length.")
    
    preds = model.predict(mat_X)

    start_pred_index = realized_volatility_slice.index[n_past:]
    preds_series_index = start_pred_index[:len(preds)] 

    preds_series = pd.Series(preds[:, 0], index=preds_series_index)

    return preds_series