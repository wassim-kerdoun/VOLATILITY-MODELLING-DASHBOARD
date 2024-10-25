import streamlit as st
from scripts.lstm_model import *
from scripts.data_preprocessing import transform_volatility_to_scaler
import plotly.graph_objects as go
import io
from contextlib import redirect_stdout

def show_page3():
    
    st.title("LSTM MODELLING DASHBOARD")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        epochs = st.number_input("Enter the number of epochs", value=100, min_value=1, max_value=150)
    with col2:
        batch_size = st.number_input("Enter the batch size", value=64, min_value=1, max_value=512)
    with col3:
        n_past = st.number_input("Enter the number of past time steps", value=14, min_value=1, max_value=100)
        
    submit = st.button("Submit")
    
    if submit:
        realized_volatility = st.session_state.realized_volatility
        st.session_state.epochs = epochs
        st.session_state.batch_size = batch_size
        st.session_state.n_past = n_past
        scaler_vol = st.session_state.scaler_vol
        X_val_scaled = st.session_state.X_val_scaled
        y_val_scaled = st.session_state.y_val_scaled
        window = st.session_state.window
        X_train_scaled = st.session_state.X_train_scaled
        y_train_scaled = st.session_state.y_train_scaled
        X_test_scaled = st.session_state.X_test_scaled
        y_test_scaled = st.session_state.y_test_scaled
        
        X_train, y_train, lstm, EarlyStopping = build_lstm(realized_volatility, n_past)
        summary_string = io.StringIO()
        with redirect_stdout(summary_string):
            lstm.summary()
        summary_output = summary_string.getvalue()
        st.subheader("LSTM Model Summary")
        st.text(summary_output)
        
        st.subheader("LSTM Model Metrics")
        lstm_history = fit_lstm(lstm, X_train, y_train, 
                                n_past,EarlyStopping, epochs, batch_size)
        st.dataframe(pd.DataFrame(lstm_history.history))
        
        figure = plot_model_metrics(lstm_history)
        st.plotly_chart(figure, use_container_width=True)

        lstm_predictions_train = lstm_forecast(lstm, realized_volatility, n_past, 'train_idx')
        lstm_predictions_train_scaled = transform_volatility_to_scaler(scaler_vol, lstm_predictions_train)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_train_scaled.index, 
            y=X_train_scaled.values,
            name='Realized Volatility',
            mode='lines',
            line=dict(color='gray',dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=y_train_scaled.index,
            y=y_train_scaled,
            name=f'Scaled {window}-Day Interval Daily Realized Volatility',
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=lstm_predictions_train_scaled.index, 
            y=lstm_predictions_train_scaled.values,
            name='Predicted Volatility using LSTM on training data',
            mode='lines',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'Scaled {window}-Day Interval Daily Realized Volatility vs. Predicted Volatility using LSTM on Training Data',
            xaxis_title='Date',
            yaxis_title='Realized Volatility',
            legend_title='Legend'
        )
        st.plotly_chart(fig, use_container_width=True)

        lstm_predictions_val = lstm_forecast(lstm, realized_volatility, n_past, 'val_idx')
        lstm_predictions_val_scaled = transform_volatility_to_scaler(scaler_vol, lstm_predictions_val)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_val_scaled.index, 
            y=X_val_scaled.values,
            name='Realized Volatility',
            mode='lines',
            line=dict(color='gray',dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=y_val_scaled.index,
            y=y_val_scaled,
            name=f'Scaled {window}-Day Interval Daily Realized Volatility on Validation Data',
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=lstm_predictions_val_scaled.index, 
            y=lstm_predictions_val_scaled.values,
            name='Predicted Volatility using LSTM',
            mode='lines',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'Scaled {window}-Day Interval Daily Realized Volatility vs. Predicted Volatility using LSTM on Validation Data',
            xaxis_title='Date',
            yaxis_title='Realized Volatility',
            legend_title='Legend'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        lstm_predictions_test = lstm_forecast(lstm, realized_volatility, n_past, 'test_idx')
        lstm_predictions_test_scaled = transform_volatility_to_scaler(scaler_vol, lstm_predictions_test)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=X_test_scaled.index, 
            y=X_test_scaled.values,
            name='Realized Volatility',
            mode='lines',
            line=dict(color='gray',dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=y_test_scaled.index,
            y=y_test_scaled,
            name=f'Scaled {window}-Day Interval Daily Realized Volatility on Validation Data',
            mode='lines',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=lstm_predictions_test_scaled.index, 
            y=lstm_predictions_test_scaled.values,
            name='Predicted Volatility using LSTM',
            mode='lines',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'Scaled {window}-Day Interval Daily Realized Volatility vs. Predicted Volatility using LSTM on Test Data',
            xaxis_title='Date',
            yaxis_title='Realized Volatility',
            legend_title='Legend'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        final_result2 = pd.concat([lstm_predictions_train_scaled,
                                  lstm_predictions_val_scaled,
                                  lstm_predictions_test_scaled], axis=0)
        volatility_final = pd.concat([y_train_scaled,
                                      y_val_scaled,
                                      y_test_scaled], axis=0)
        
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=final_result2.index, 
            y=final_result2.values,
            name='Predicted Volatility using LSTM',
            mode='lines',
            line=dict(color='red')
        ))
        figure.add_trace(go.Scatter(
            x=volatility_final.index, 
            y=volatility_final.values,
            name='Realized Volatility',
            mode='lines',
            line=dict(color='blue')
        ))
        figure.update_layout(
            title=f'Scaled {window}-Day Interval Daily Realized Volatility vs. Predicted Volatility using LSTM on Training, Validation and Test Data',
            xaxis_title='Date',
            yaxis_title='Realized Volatility',
            legend_title='Legend'
        )
        st.plotly_chart(figure, use_container_width=True)
        
        st.session_state.final_result2 = final_result2