import streamlit as st
import plotly.graph_objects as go

def show_page4():

    st.title("COMPARING GARCH AND LSTM RESULTS")
    
    realized_volatility = st.session_state.return_final
    lstm = st.session_state.final_result2
    garch = st.session_state.final_result1
    p = st.session_state.p
    q = st.session_state.q
    
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=realized_volatility.index,
        y=realized_volatility.values,
        mode="lines",
        name="Realized Volatility",
        line=dict(color='blue', width=2)
    ))
    figure.add_trace(go.Scatter(
        x=garch.index,
        y=garch.values,
        mode="lines",
        name=f'GARCH({p},{q}) Forecasted Conditional Volatility',
        line=dict(color='orange', width=2)
    ))
    figure.add_trace(go.Scatter(
        x=lstm.index,
        y=lstm.values,
        mode="lines",
        name="LSTM Forecasted Volatility",
        line=dict(color='red', width=2)
    ))
    figure.update_layout(
        xaxis_title="Date",
        yaxis_title="Realized Volatility",
        title=f"Realized Volatility vs. Forecasted Conditional Volatility for GARCH({p},{q}) and LSTM models",
    )
    st.plotly_chart(figure, use_container_width=True)