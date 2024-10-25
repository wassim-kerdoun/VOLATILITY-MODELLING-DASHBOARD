import streamlit as st
from streamlit_navigation_bar import st_navbar
import pages.data_selection_page
import pages.garch_page
import pages.lstm_page
import pages.results_comparison


st.set_page_config(
    page_title="Volatility Modelling Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

styles = {
    "nav": {
        "background-color": "rgb(123, 209, 146)",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "rgb(49, 51, 63)",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

nav_bar = st_navbar(["DATA SELECTION", "GARCH", 'LSTM', 'COMPARISON'], styles=styles)

if nav_bar == 'DATA SELECTION':
    pages.data_selection_page.show_page1()

elif nav_bar == 'GARCH':
    pages.garch_page.show_page2()

elif nav_bar == 'LSTM':
    pages.lstm_page.show_page3()

elif nav_bar == 'COMPARISON':
    pages.results_comparison.show_page4()
