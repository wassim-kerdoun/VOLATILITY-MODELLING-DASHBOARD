import streamlit as st
from scripts.garch_model import *
from scripts.data_preprocessing import *

def show_page2():
    
    st.title("GARCH MODELLING DASHBOARD")
    
    ticker = st.session_state.ticker
    realized_volatility = st.session_state.realized_volatility
    
    p = st.number_input("Enter the order of Autoregression (p)", value=1, min_value=1, max_value=20)
    q = st.number_input("Enter the order of Moving Average (q)", value=1, min_value=0, max_value=20)
    dist = st.selectbox("Select Distribution", options=['normal', 'skewt'])
    
    horizon = st.number_input("Enter the number of periods to forecast", value=7, min_value=1)
    
    returns = st.session_state.returns
    figure = acf_pacf_plots(ticker,returns)
    
    st.pyplot(figure)
    
    submit = st.button("Submit")
    
    if submit:
        st.session_state.p = p
        st.session_state.q = q
        st.session_state.dist = dist
        st.session_state.horizon = horizon
        y_val_scaled = st.session_state.y_val_scaled
        X_train_scaled = st.session_state.X_train_scaled
        window = st.session_state.window
        returns = st.session_state.returns
        y_test_scaled = st.session_state.y_test_scaled
        y_train_scaled = st.session_state.y_train_scaled
        X_test_scaled = st.session_state.X_test_scaled
        X_val_scaled = st.session_state.X_val_scaled
        
        r_train, _, _ = garch_data_splitting(returns)
        
        model_fit = fit_garch(r_train, p, q, dist)
        
        st.success(f"Model fitted successfully for {ticker} with p={p}, q={q} and dist={dist}")
        
        scaler_garch, scaled_cond_vol = scale_tf_cond_vol(model_fit)
        
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=X_train_scaled.index,
            y=X_train_scaled.values,
            mode='lines',
            name=f'Scaled {st.session_state.window}-Day Interval Daily Realized Volatility',
            line=dict(color='blue')
        ))
        figure.add_trace(go.Scatter(
            x=scaled_cond_vol.index,
            y=scaled_cond_vol,
            mode='lines',
            name=f'Scaled GARCH({p},{q}) Estimated Conditional Volatility',
            line=dict(color='orange')
        ))
        figure.update_layout(
            title=f'Scaled {window}-Day Interval Daily Realized Volatility vs. Scaled GARCH({p},{q}) Estimated Conditional Volatility',
            xaxis_title='Date',
            yaxis_title='Realized Volatility',
            legend_title='Legend',
        )
        st.plotly_chart(figure, use_container_width=True)
        
        garch_forecast = forecast_volatility(returns, horizon, dist,
                                             p,q)
        
        garch_forecast_scaled = transform_volatility_to_scaler(scaler_garch, garch_forecast)
        st.session_state.garch_forecast_scaled = garch_forecast_scaled
        st.session_state.garch_forecast = garch_forecast
        
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=garch_forecast_scaled.index,
            y=garch_forecast_scaled.values,
            mode='lines',
            name=f'GARCH({p},{q}) Forecasted Conditional Volatility',
            line=dict(color='red')
        ))
        
        figure.add_trace(go.Scatter(
            x=y_val_scaled.index,
            y=y_val_scaled,
            mode='lines',
            name=f'Scaled {window}-Day Interval Daily Realized Volatility',
            line=dict(color='blue')
        ))
        
        figure.add_trace(go.Scatter(
            x=X_val_scaled.index,
            y=X_val_scaled.values,
            mode='lines',
            name=f'Realized Volatility',
            line=dict(color='gray', dash='dot')
        ))
        
        figure.update_layout(
            title=f'GARCH({p},{q}) Forecasted Conditional Volatility vs. Scaled {window}-Day Interval Daily Realized Volatility',
            xaxis_title='Date',
            yaxis_title='Conditional Volatility',
            legend_title='Legend',
        )
        
        st.plotly_chart(figure, use_container_width=True)
        
        garch_tested_volatility = test_volatility(returns, horizon, dist, p, q)
        garch_tested_volatility_scaled = transform_volatility_to_scaler(scaler_garch, garch_tested_volatility)
        
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=garch_tested_volatility_scaled.index,
            y=garch_tested_volatility_scaled.values,
            mode='lines',
            name=f'GARCH({p},{q}) Forecasted Conditional Volatility',
            line=dict(color='red')
        ))
        
        figure.add_trace(go.Scatter(
            x=y_test_scaled.index,
            y=y_test_scaled,
            mode='lines',
            name=f'Scaled {window}-Day Interval Daily Realized Volatility',
            line=dict(color='blue')
        ))
        
        figure.add_trace(go.Scatter(
            x=X_test_scaled.index,
            y=X_test_scaled.values,
            mode='lines',
            name=f'Realized Volatility',
            line=dict(color='gray', dash='dot')
        ))
        
        figure.update_layout(
            title=f'GARCH({p},{q}) Forecasted Conditional Volatility on test set vs. Scaled {window}-Day Interval Daily Realized Volatility',
            xaxis_title='Date',
            yaxis_title='Conditional Volatility',
            legend_title='Legend',
        )
        
        st.plotly_chart(figure, use_container_width=True)
        
        final_result1 = pd.concat([scaled_cond_vol, garch_forecast_scaled, garch_tested_volatility_scaled], axis=0)
        return_final = pd.concat([y_train_scaled, y_val_scaled, y_test_scaled], axis=0)
        
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=final_result1.index,
            y=final_result1.values,
            mode='lines',
            name=f'GARCH({p},{q}) Forecasted Conditional Volatility',
            line=dict(color='red')
        ))
        figure.add_trace(go.Scatter(
            x=return_final.index,
            y=return_final.values,
            mode='lines',
            name=f'Realized Volatility',
            line=dict(color='blue')
        ))
        figure.update_layout(
            title=f'GARCH({p},{q}) Forecasted Conditional Volatility on test set vs. Scaled {window}-Day Interval Daily Realized Volatility',
            xaxis_title='Date',
            yaxis_title='Conditional Volatility',
            legend_title='Legend',
        )
        
        st.plotly_chart(figure, use_container_width=True)
        
        st.write(model_fit.summary())
        
        st.session_state.return_final = return_final
        st.session_state.final_result1 = final_result1
        