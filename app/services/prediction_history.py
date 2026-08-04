import pandas as pd
from datetime import datetime
import streamlit as st


class PredictionHistory:

    @staticmethod
    def add(prediction, confidence=None):

        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []

        st.session_state["prediction_history"].append(
            {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Prediction": prediction,
                "Confidence": confidence,
            }
        )

    @staticmethod
    def get():

        if "prediction_history" not in st.session_state:
            return pd.DataFrame()

        return pd.DataFrame(
            st.session_state["prediction_history"]
        )

    @staticmethod
    def clear():

        st.session_state["prediction_history"] = []