import streamlit as st
import pandas as pd

from app.services.dataset_chat import DatasetChatService


def render_dataset_chat_tab(
    cleaned_df: pd.DataFrame
):

    st.title("💬 Chat with Dataset")

    st.caption(
        "Ask questions about your uploaded dataset using AI."
    )

    st.divider()

    # ==========================================
    # INITIALIZE CHAT SERVICE
    # ==========================================

    if "dataset_chat" not in st.session_state:

        st.session_state.dataset_chat = (
            DatasetChatService(
                cleaned_df
            )
        )

    chat = st.session_state.dataset_chat

    # ==========================================
    # CHAT HISTORY
    # ==========================================

    if "chat_messages" not in st.session_state:

        st.session_state.chat_messages = []

    # ==========================================
    # SIDEBAR
    # ==========================================

    with st.sidebar:

        st.subheader("💡 Suggested Questions")

        for question in chat.suggested_questions():

            if st.button(
                question,
                use_container_width=True
            ):

                st.session_state.selected_question = (
                    question
                )

        st.divider()

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):

            chat.reset_chat()

            st.session_state.chat_messages = []

            st.rerun()

        st.divider()

        st.caption(
            chat.version()
        )

    # ==========================================
    # DISPLAY CHAT
    # ==========================================

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )
        # ==========================================
    # SUGGESTED QUESTION
    # ==========================================

    default_question = ""

    if "selected_question" in st.session_state:

        default_question = (
            st.session_state.selected_question
        )

        del st.session_state.selected_question

    # ==========================================
    # CHAT INPUT
    # ==========================================

    user_prompt = st.chat_input(
        "Ask anything about your dataset..."
    )

    if default_question:

        user_prompt = default_question

    # ==========================================
    # PROCESS QUESTION
    # ==========================================

    if user_prompt:

        # -----------------------------
        # USER MESSAGE
        # -----------------------------

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        with st.chat_message("user"):

            st.markdown(user_prompt)

        # -----------------------------
        # AI RESPONSE
        # -----------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                response = chat.ask(
                    user_prompt
                )

                st.markdown(
                    response
                )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
        st.divider()

    # ==========================================================
    # DATASET OVERVIEW
    # ==========================================================

    with st.expander("📊 Dataset Overview", expanded=False):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Dataset Shape")

            rows, cols = chat.shape()

            st.metric("Rows", rows)

            st.metric("Columns", cols)

            st.metric(
                "Duplicate Rows",
                chat.duplicates()
            )

        with col2:

            st.subheader("Missing Values")

            st.dataframe(
                chat.missing_values(),
                use_container_width=True
            )

    # ==========================================================
    # DATASET PREVIEW
    # ==========================================================

    with st.expander("📄 Dataset Preview"):

        st.dataframe(
            chat.sample(10),
            use_container_width=True,
            hide_index=True
        )

    # ==========================================================
    # COLUMN INFORMATION
    # ==========================================================

    with st.expander("📑 Column Information"):

        st.dataframe(
            chat.column_information(),
            use_container_width=True,
            hide_index=True
        )

    # ==========================================================
    # STATISTICAL SUMMARY
    # ==========================================================

    with st.expander("📈 Statistical Summary"):

        stats = chat.statistics()

        if not stats.empty:

            st.dataframe(
                stats,
                use_container_width=True
            )

        else:

            st.info(
                "No statistics available."
            )

    # ==========================================================
    # CORRELATION MATRIX
    # ==========================================================

    with st.expander("📊 Correlation Matrix"):

        corr = chat.correlation_matrix()

        if not corr.empty:

            st.dataframe(
                corr,
                use_container_width=True
            )

        else:

            st.info(
                "No correlation matrix available."
            )

    # ==========================================================
    # AI DATASET SUMMARY
    # ==========================================================

    with st.expander("🧠 AI Dataset Summary"):

        if st.button(
            "Generate AI Summary",
            use_container_width=True
        ):

            with st.spinner(
                "Generating summary..."
            ):

                summary = chat.generate_summary()

                st.success(summary)

    # ==========================================================
    # AI INSIGHTS
    # ==========================================================

    with st.expander("💡 AI Business Insights"):

        if st.button(
            "Generate AI Insights",
            use_container_width=True
        ):

            with st.spinner(
                "Generating insights..."
            ):

                insights = chat.generate_insights()

                st.success(insights)

    # ==========================================================
    # EXPORT CHAT
    # ==========================================================

    st.divider()

    st.subheader("📥 Export Chat")

    export_df = chat.export_chat()

    st.download_button(

        label="⬇️ Download Chat History",

        data=export_df.to_csv(index=False),

        file_name="chat_history.csv",

        mime="text/csv",

        use_container_width=True

    )

    st.divider()

    st.caption(
        "Enterprise AI Data Analyst Copilot • Dataset Chat Module"
    )