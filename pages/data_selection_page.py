import streamlit as st
import pandas as pd
import numpy as np
import datetime
from scripts.data_preprocessing import *
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import norm, lognorm
from sklearn.preprocessing import MinMaxScaler


def show_page1():
    st.title("DASHBOARD FOR VOLATILITY MODELLING OF FINANCIAL ASSETS")

    ticker = st.text_input("Enter Ticker", "EURUSD=X")

    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.date_input("End Date", value=datetime.datetime.now())
    
    window = st.number_input("Enter the window size for Realized Volatility", value=30, min_value=1)
    
    n_future = st.number_input("Enter the number of periods to forecast", value=7, min_value=1)

    submit = st.button("Submit")

    if submit:
        st.session_state.ticker = ticker
        st.session_state.window = window
        st.session_state.n_future = n_future

        data = fetch_data(ticker, start_date, end_date)
        
        st.success(f"Fetched data successfully for {ticker} from {start_date} to {end_date}")

        log_return = LogReturns(data)
        
        returns = PCTChange(data)

        log_return.rename(columns={'Adj Close': 'Log Returns'}, inplace=True)
        returns.rename(columns={'Adj Close': 'Returns'}, inplace=True)
        
        st.session_state.log_return = log_return
        st.session_state.returns = returns
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.write("Adjacent Close")
            st.dataframe(data)

        with col2:
            st.write("Log Returns")
            st.dataframe(log_return)
            
        with col3:
            st.write("Returns")
            st.dataframe(returns)

        indicator = indicators(data)

        with col4:
            st.write("Technical Indicators")
            st.dataframe(indicator)

        st.plotly_chart(plot_indicators(indicator), use_container_width=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=log_return.index,
            y=log_return['Log Returns'],
            mode='lines',
            name='Log Returns',
            line=dict(color='blue')
        ))

        fig.update_layout(
            title='Logarithmic Returns',
            xaxis_title='Date',
            yaxis_title='Log Returns',
            legend_title='Legend',
            template='plotly_white'
        )

        st.plotly_chart(fig, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=returns.index,    
            y=returns['Returns'],
            mode='lines',
            name='Returns',
            line=dict(color='blue')
        ))
        fig.update_layout(
            title='Returns',
            xaxis_title='Date', 
            yaxis_title='Returns',
            legend_title='Legend',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        realized_volatility = RealizedVolatility(log_return, window, n_future)
        st.session_state.realized_volatility = realized_volatility
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=realized_volatility.index,
            y=realized_volatility['Realized Volatility'],
            mode='lines',
            name='Realized Volatility (Current)',
            line=dict(color='gray',dash='dot')
        ))
        fig.update_layout(
            title=f'Future vs Current Daily Volatility of {ticker} Using {window}-Day Rolling Window and {n_future}-Day Forecasting',
            xaxis_title='Date',
            yaxis_title='Realized Volatility',
            legend_title='Legend',
            template='plotly_white'
        )
        fig.add_trace(go.Scatter(
            x=realized_volatility.index,
            y=realized_volatility['Future Volatility'],
            mode='lines',
            name='Future Realized Volatility (Target)',
            line=dict(color='orange')
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        X_train, X_val, X_test, y_train, y_val, y_test = data_splitting(realized_volatility)

        y_train = pd.Series(y_train.flatten(), index=realized_volatility.index[:len(y_train)])
        y_val = pd.Series(y_val.flatten(), index=realized_volatility.index[len(y_train):len(y_train)+len(y_val)])
        y_test = pd.Series(y_test.flatten(), index=realized_volatility.index[len(y_train)+len(y_val):])
        
        X_train = pd.Series(X_train.flatten(), index=realized_volatility.index[:len(X_train)])
        X_val = pd.Series(X_val.flatten(), index=realized_volatility.index[len(X_train):len(X_train)+len(X_val)])
        X_test = pd.Series(X_test.flatten(), index=realized_volatility.index[len(X_train)+len(X_val):])
        
        scaler_vol = MinMaxScaler()
        scaler_vol = scaler_vol.fit(X_train.values.reshape(-1,1))
        
        # TRANSFORM TRAINING CURRENT & FUTURE VOLATILITIES
        X_train_scaled = transform_volatility_to_scaler(scaler_vol, X_train)
        y_train_scaled = transform_volatility_to_scaler(scaler_vol, y_train)

        # TRANSFORMING VALIDATION CURRENT & FUTURE VOLATILITIES
        X_val_scaled = transform_volatility_to_scaler(scaler_vol, X_val)
        y_val_scaled = transform_volatility_to_scaler(scaler_vol, y_val)

        # TRANSFORMING TEST CURRENT & FUTURE VOLATILITIES
        X_test_scaled = transform_volatility_to_scaler(scaler_vol, X_test)
        y_test_scaled = transform_volatility_to_scaler(scaler_vol, y_test)
        
        st.session_state.scaler_vol = scaler_vol
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_val_scaled = X_val_scaled
        st.session_state.y_val = y_val
        st.session_state.y_val_scaled = y_val_scaled
        st.session_state.y_test_scaled = y_test_scaled
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.y_train_scaled = y_train_scaled

        figure = go.Figure()

        figure.add_trace(go.Scatter(
            x=log_return.index,
            y=log_return['Log Returns'],
            mode='lines',
            name='Log Returns',
            opacity=0.4,
            line=dict(color='gray')
        ))

        figure.add_trace(go.Scatter(
            x=y_train_scaled.index,
            y=y_train_scaled,
            mode='lines',
            name='Training Data',
            line=dict(color='blue', width=2)
        ))

        figure.add_trace(go.Scatter(
            x=y_val_scaled.index,
            y=y_val_scaled,
            mode='lines',
            name='Validation Data',
            line=dict(color='red', width=2)
        ))

        figure.add_trace(go.Scatter(
            x=y_test_scaled.index,
            y=y_test_scaled,
            mode='lines',
            name='Test Data',
            line=dict(color='green', width=2)
        ))
        
        figure.add_trace(go.Scatter(
            x=realized_volatility.index,
            y=transform_volatility_to_scaler(scaler_vol,
                                             realized_volatility['Realized Volatility']),
            mode='lines',
            name='Realized Volatility',
            line=dict(color='gray',dash='dot')
        ))

        figure.update_layout(
            title='Logarithmic Returns and Data Splits',
            xaxis_title='Date',
            yaxis_title='Log Returns / Volatility',
            legend_title='Data Splits',
            template='plotly_white'
        )
        

        st.plotly_chart(figure, use_container_width=True)


        mu = np.mean(log_return['Log Returns'])
        sigma = np.std(log_return['Log Returns'])

        hist_data = [log_return['Log Returns'].values]
        group_labels = ['Log Returns']

        fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=0.001, show_hist=True, show_rug=False)

        x = np.linspace(log_return['Log Returns'].min(), log_return['Log Returns'].max(), 100)
        y = norm.pdf(x, mu, sigma)

        fig_hist.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig_hist.update_traces(marker=dict(line=dict(color='black', width=1)))

        fig_hist.update_layout(
            title='Distribution of Log Returns with Normal Distribution Overlay',
            xaxis_title='Log Returns',
            yaxis_title='Density',
            template='plotly_white'
        )

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_hist, use_container_width=True)

        
        mu = np.mean(np.log(realized_volatility['Realized Volatility']))
        sigma = np.std(np.log(realized_volatility['Realized Volatility']))

        hist_data = [realized_volatility['Realized Volatility'].dropna().values]
        group_labels = ['Realized Volatility']

        fig_hist = ff.create_distplot(hist_data, group_labels, bin_size=0.0001, show_hist=True, show_rug=False)
        
        x = np.linspace(realized_volatility['Realized Volatility'].dropna().min(),
                        realized_volatility['Realized Volatility'].dropna().max(), 100)
        y = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

        fig_hist.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Log Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig_hist.update_traces(marker=dict(line=dict(color='black', width=1)))

        fig_hist.update_layout(
            title='Distribution of Realized Volatility with Log Normal Distribution Overlay',
            xaxis_title='Realized Volatility',
            yaxis_title='Frequency',
            template='plotly_white'
        )

        with col2:
            st.plotly_chart(fig_hist, use_container_width=True)


        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(qqplot(log_return['Log Returns']))

        with col2:
            st.plotly_chart(qqplot(log_return['Log Returns'], dist='t'))