import streamlit as st


def render_sidebar(profile=None):

    st.sidebar.title("🤖 AI Data Analyst")

    st.sidebar.markdown("---")

    if profile:

        st.sidebar.success("Dataset Loaded")

        st.sidebar.metric(
            "Rows",
            profile["number_of_rows"]
        )

        st.sidebar.metric(
            "Columns",
            profile["number_of_columns"]
        )

        st.sidebar.metric(
            "Numeric",
            len(profile["numeric_columns"])
        )

        st.sidebar.metric(
            "Categorical",
            len(profile["categorical_columns"])
        )

    else:

        st.sidebar.info(
            "Upload a dataset to begin."
        )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Enterprise AI Data Analyst Copilot v2.0"
    )