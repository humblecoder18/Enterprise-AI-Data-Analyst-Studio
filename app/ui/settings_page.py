import streamlit as st


def render_settings_page():

    st.title("⚙️ Settings")

    st.subheader("Appearance")

    theme = st.selectbox(
        "Theme",
        [
            "System",
            "Light",
            "Dark"
        ]
    )

    st.success(f"Selected Theme: {theme}")

    st.divider()

    st.subheader("Application")

    st.checkbox(
        "Enable Notifications",
        value=True
    )

    st.checkbox(
        "Save Chat History",
        value=True
    )

    st.checkbox(
        "Save Prediction History",
        value=True
    )

    st.checkbox(
        "Enable Activity Logs",
        value=True
    )

    st.divider()

    st.subheader("AI Settings")

    model = st.selectbox(

        "Groq Model",

        [

            "llama-3.3-70b-versatile",

            "mixtral-8x7b-32768",

            "llama3-8b-8192",

            "gemma2-9b-it"

        ]

    )

    temperature = st.slider(

        "Temperature",

        0.0,

        1.0,

        0.3

    )

    max_tokens = st.slider(

        "Max Tokens",

        256,

        4096,

        1024

    )

    st.divider()

    if st.button(

        "💾 Save Settings",

        use_container_width=True

    ):

        st.success(
            "Settings saved successfully."
        )