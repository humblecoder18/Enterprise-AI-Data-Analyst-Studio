import streamlit as st
import pandas as pd

from app.services.ai_insight_agent import AIInsightAgent
from app.services.ml_recommendation_agent import MLRecommendationAgent
from app.services.chat_agent import ChatAgent


def render_ai_tab(
    cleaned_df: pd.DataFrame,
    profile: dict,
    eda_results: dict
):
    """
    Render the AI Assistant interface.

    Features:
    - AI-generated dataset insights
    - ML recommendations
    - Chat with dataset
    """

    # =========================================================
    # HEADER
    # =========================================================

    st.subheader("🤖 AI Data Assistant")

    st.caption(
        "Generate AI-powered insights, receive machine learning "
        "recommendations, and ask questions about your dataset."
    )

    if cleaned_df is None or cleaned_df.empty:
        st.warning("No dataset available.")
        return

    # Safe defaults
    if profile is None:
        profile = {}

    if eda_results is None:
        eda_results = {}

    # =========================================================
    # AI ASSISTANT SECTIONS
    # =========================================================

    insight_tab, ml_tab, chat_tab = st.tabs(
        [
            "📊 AI Insights",
            "🧠 ML Recommendations",
            "💬 Chat with Data"
        ]
    )

    # =========================================================
    # AI INSIGHTS
    # =========================================================

    with insight_tab:

        st.markdown("### 📊 AI-Generated Insights")

        st.write(
            "Analyze the exploratory data analysis results "
            "using the local AI model."
        )

        if st.button(
            "✨ Generate AI Insights",
            type="primary",
            key="generate_ai_insights"
        ):

            try:

                with st.spinner(
                    "Analyzing dataset with AI..."
                ):

                    agent = AIInsightAgent()

                    insights = agent.generate_insights(
                        eda_results
                    )

                st.session_state[
                    "ai_insights"
                ] = insights

                st.success(
                    "AI insights generated successfully."
                )

            except Exception as e:

                st.error(
                    f"❌ AI insight generation failed: {e}"
                )

        # -----------------------------------------------------
        # Display saved AI insights
        # -----------------------------------------------------

        ai_insights = st.session_state.get(
            "ai_insights"
        )

        if ai_insights:

            st.divider()

            st.markdown(
                "### 💡 Analysis Results"
            )

            st.markdown(
                str(ai_insights)
            )

    # =========================================================
    # ML RECOMMENDATIONS
    # =========================================================

    with ml_tab:

        st.markdown(
            "### 🧠 Machine Learning Recommendations"
        )

        st.write(
            "Analyze the dataset profile and exploratory analysis "
            "to recommend suitable machine learning approaches."
        )

        # -----------------------------------------------------
        # Dataset information
        # -----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Dataset Rows",
                len(cleaned_df)
            )

        with col2:

            st.metric(
                "Dataset Columns",
                len(cleaned_df.columns)
            )

        st.divider()

        # -----------------------------------------------------
        # Generate recommendations
        # -----------------------------------------------------

        if st.button(
            "🚀 Generate ML Recommendations",
            type="primary",
            key="generate_ml_recommendations"
        ):

            try:

                with st.spinner(
                    "Analyzing dataset and generating "
                    "machine learning recommendations..."
                ):

                    agent = MLRecommendationAgent()

                    # IMPORTANT:
                    # recommend() requires BOTH profile
                    # and eda_results.

                    recommendations = agent.recommend(
                        profile=profile,
                        eda_results=eda_results
                    )

                st.session_state[
                    "ml_recommendations"
                ] = recommendations

                st.success(
                    "ML recommendations generated successfully."
                )

            except Exception as e:

                st.error(
                    f"❌ ML recommendation generation failed: {e}"
                )

        # -----------------------------------------------------
        # Display recommendations
        # -----------------------------------------------------

        ml_recommendations = st.session_state.get(
            "ml_recommendations"
        )

        if ml_recommendations:

            st.divider()

            st.markdown(
                "### 🎯 Recommended ML Strategy"
            )

            st.markdown(
                str(ml_recommendations)
            )

    # =========================================================
    # CHAT WITH DATA
    # =========================================================

    with chat_tab:

        st.markdown(
            "### 💬 Chat with Your Dataset"
        )

        st.write(
            "Ask natural-language questions about "
            "the uploaded dataset."
        )

        user_question = st.text_input(
            "Ask a question",
            placeholder=(
                "Example: Which department has "
                "the highest average salary?"
            ),
            key="dataset_chat_question"
        )

        # -----------------------------------------------------
        # Ask AI
        # -----------------------------------------------------

        if st.button(
            "💬 Ask AI",
            type="primary",
            key="ask_dataset_ai"
        ):

            if not user_question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                try:

                    with st.spinner(
                        "Analyzing your dataset..."
                    ):

                        chat_agent = ChatAgent()

                        answer = chat_agent.ask(
                            cleaned_df,
                            user_question
                        )

                    st.session_state[
                        "latest_chat_question"
                    ] = user_question

                    st.session_state[
                        "latest_chat_answer"
                    ] = answer

                except Exception as e:

                    st.error(
                        f"❌ Chat failed: {e}"
                    )

        # -----------------------------------------------------
        # Display previous answer
        # -----------------------------------------------------

        latest_answer = st.session_state.get(
            "latest_chat_answer"
        )

        latest_question = st.session_state.get(
            "latest_chat_question"
        )

        if latest_answer:

            st.divider()

            if latest_question:

                st.markdown(
                    f"**Question:** {latest_question}"
                )

            st.markdown(
                "### 🤖 AI Response"
            )

            st.markdown(
                str(latest_answer)
            )