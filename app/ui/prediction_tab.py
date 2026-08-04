import streamlit as st
import pandas as pd
import plotly.express as px

from app.services.prediction_engine import PredictionEngine
from app.services.prediction_explainer import PredictionExplainer
from app.services.prediction_ai_explainer import PredictionAIExplainer
from app.services.prediction_validator import PredictionValidator
from app.services.error_handler import ErrorHandler
from app.services.prediction_history import PredictionHistory

# NEW
from app.auth.auth_guard import AuthGuard
from app.database.database_service import DatabaseService


def render_prediction_tab():
    """
    Render the AI Prediction interface.

    Features:
    - Dynamic prediction form
    - Classification and regression prediction
    - Prediction confidence
    - Class probabilities
    - AI business explanation using Groq
    - Global model importance
    - Prediction input audit
    """

    # =========================================================
    # HEADER
    # =========================================================

    st.subheader("🔮 AI Prediction")

    st.write(
        "Use the best model trained by AutoML to make predictions "
        "for new data and understand why the model made its decision."
    )

    st.divider()

    # =========================================================
    # GET AUTOML RESULTS
    # =========================================================

    automl_results = st.session_state.get("automl_results")

    if automl_results is None:

        st.warning("⚠️ No trained AutoML model found.")

        st.info(
            "Go to the AutoML tab, select a target column, "
            "and train the models first."
        )

        return

    # =========================================================
    # MODEL INFORMATION
    # =========================================================

    trained_model = automl_results.get("trained_model")
    feature_schema = automl_results.get("feature_schema")

    feature_columns = automl_results.get(
        "feature_columns",
        []
    )

    target_column = automl_results.get(
        "target_column"
    )

    problem_type = automl_results.get(
        "problem_type"
    )

    best_model = automl_results.get(
        "best_model"
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    if trained_model is None:

        st.error("❌ Trained model is not available.")

        st.info(
            "Return to the AutoML tab and retrain the models."
        )

        return

    if not feature_schema:

        st.error("❌ Feature schema is not available.")

        st.info(
            "Return to the AutoML tab and retrain the models."
        )

        return

    if not feature_columns:

        feature_columns = list(
            feature_schema.keys()
        )

    # =========================================================
    # MODEL SUMMARY
    # =========================================================

    st.markdown("### 🧠 Prediction Model")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🎯 Target",
            str(target_column)
        )

    with col2:

        st.metric(
            "🧠 Model",
            str(best_model)
        )

    with col3:

        st.metric(
            "📊 Problem",
            str(problem_type).title()
        )

    with col4:

        st.metric(
            "🔢 Features",
            len(feature_columns)
        )

    st.success(
        f"✅ **{best_model}** is ready to predict "
        f"**{target_column}**."
    )

    st.divider()

    # =========================================================
    # INPUT FORM
    # =========================================================

    st.markdown("## 📝 Enter Data for Prediction")

    st.caption(
        "Enter feature values below. Default values are based "
        "on the uploaded dataset."
    )

    input_data = {}

    with st.form("prediction_input_form"):

        form_col1, form_col2 = st.columns(2)

        for index, column in enumerate(
            feature_columns
        ):

            schema = feature_schema.get(
                column,
                {}
            )

            feature_type = schema.get(
                "type"
            )

            container = (
                form_col1
                if index % 2 == 0
                else form_col2
            )

            # =================================================
            # NUMERIC
            # =================================================

            if feature_type == "numeric":

                minimum = schema.get(
                    "min",
                    0
                )

                maximum = schema.get(
                    "max",
                    100
                )

                median = schema.get(
                    "median",
                    0
                )

                is_integer = schema.get(
                    "is_integer",
                    False
                )

                if pd.isna(minimum):
                    minimum = 0

                if pd.isna(maximum):
                    maximum = 100

                if pd.isna(median):
                    median = minimum

                if minimum > maximum:

                    minimum, maximum = (
                        maximum,
                        minimum
                    )

                median = min(
                    max(
                        median,
                        minimum
                    ),
                    maximum
                )

                if is_integer:

                    value = container.number_input(
                        column,
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(median),
                        step=1,
                        key=f"prediction_{column}"
                    )

                    input_data[column] = int(
                        value
                    )

                else:

                    value = container.number_input(
                        column,
                        min_value=float(minimum),
                        max_value=float(maximum),
                        value=float(median),
                        key=f"prediction_{column}"
                    )

                    input_data[column] = float(
                        value
                    )

            # =================================================
            # CATEGORICAL
            # =================================================

            elif feature_type == "categorical":

                categories = schema.get(
                    "categories",
                    []
                )

                default_value = schema.get(
                    "default",
                    ""
                )

                if categories:

                    categories = list(
                        categories
                    )

                    default_index = 0

                    if default_value in categories:

                        default_index = (
                            categories.index(
                                default_value
                            )
                        )

                    value = container.selectbox(
                        column,
                        categories,
                        index=default_index,
                        key=f"prediction_{column}"
                    )

                    input_data[column] = value

                else:

                    value = container.text_input(
                        column,
                        value=str(
                            default_value
                        ),
                        key=f"prediction_{column}"
                    )

                    input_data[column] = value

            # =================================================
            # FALLBACK
            # =================================================

            else:

                value = container.text_input(
                    column,
                    key=f"prediction_{column}"
                )

                input_data[column] = value

        st.divider()

        predict_button = (
            st.form_submit_button(
                "🔮 Make Prediction",
                type="primary",
                use_container_width=True
            )
        )

    # =========================================================
    # RUN PREDICTION
    # =========================================================

    if predict_button:

        # Clear previous prediction
        st.session_state.pop(
            "latest_prediction",
            None
        )

        st.session_state.pop(
            "latest_prediction_input",
            None
        )

        st.session_state.pop(
            "latest_prediction_explanation",
            None
        )

        st.session_state.pop(
            "latest_local_explanation",
            None
        )

        # Important:
        # remove AI explanation belonging to previous prediction
        st.session_state.pop(
            "prediction_ai_business_explanation",
            None
        )

        # =====================================================
        # VALIDATE PREDICTION INPUT
        # =====================================================

        validation = PredictionValidator.validate_input(
            input_data,
            feature_schema
        )

        if not validation["valid"]:

            st.error("Prediction validation failed.")

            for error in validation["errors"]:
                st.error(error)

            return

        for warning in validation["warnings"]:
            st.warning(warning)

        try:

            with st.spinner(
                "Running prediction..."
            ):

                prediction_engine = (
                    PredictionEngine(
                        model=trained_model,
                        problem_type=problem_type,
                        target_column=target_column
                    )
                )

                prediction_result = (
                    prediction_engine.predict(
                        input_data
                    )
                )

            # =================================================
            # STORE PREDICTION
            # =================================================

            st.session_state[
                "latest_prediction"
            ] = prediction_result

            st.session_state[
                "latest_prediction_input"
            ] = input_data

            PredictionHistory.add(
                prediction=prediction_result.get("prediction"),
                confidence=prediction_result.get("confidence")
            )

            # =================================================
            # EXPLAINABILITY
            # =================================================

            try:

                explainer = PredictionExplainer(
                    trained_model
                )

                # ---------------------------------------------
                # GLOBAL EXPLANATION
                # ---------------------------------------------

                global_explanation = (
                    explainer.explain(
                        input_data=input_data,
                        feature_columns=feature_columns,
                        top_n=10
                    )
                )

                st.session_state[
                    "latest_prediction_explanation"
                ] = global_explanation

                # ---------------------------------------------
                # LOCAL EXPLANATION
                # ---------------------------------------------

                predicted_value = (
                    prediction_result.get(
                        "prediction"
                    )
                )

                local_explanation = (
                    explainer.explain_prediction(
                        input_data=input_data,
                        feature_columns=feature_columns,
                        predicted_class=predicted_value,
                        top_n=10
                    )
                )

                st.session_state[
                    "latest_local_explanation"
                ] = local_explanation

            except Exception as explanation_error:

                error_result = {
                    "available": False,
                    "message": str(
                        explanation_error
                    )
                }

                st.session_state[
                    "latest_prediction_explanation"
                ] = error_result

                st.session_state[
                    "latest_local_explanation"
                ] = error_result

        except Exception as e:

            ErrorHandler.handle_error(
                e,
                "Prediction failed"
            )

            return

    # =========================================================
    # GET PREDICTION
    # =========================================================

    prediction_result = (
        st.session_state.get(
            "latest_prediction"
        )
    )

    if prediction_result is None:
        return

    prediction = prediction_result.get(
        "prediction"
    )

    # =========================================================
    # PREDICTION RESULT
    # =========================================================

    st.divider()

    st.markdown(
        "## 🎯 Prediction Result"
    )

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    if (
        str(problem_type).lower()
        == "classification"
    ):

        confidence = (
            prediction_result.get(
                "confidence"
            )
        )

        class_probabilities = (
            prediction_result.get(
                "class_probabilities"
            )
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        with result_col1:

            st.metric(
                f"Predicted {target_column}",
                str(prediction)
            )

        with result_col2:

            if confidence is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

            else:

                st.metric(
                    "Prediction Confidence",
                    "N/A"
                )

        st.success(
            f"🔮 The model predicts "
            f"**{target_column} = {prediction}**."
        )

        # =====================================================
        # CLASS PROBABILITIES
        # =====================================================

        if class_probabilities:

            st.divider()

            st.markdown(
                "### 📊 Class Probabilities"
            )

            probability_df = pd.DataFrame(
                {
                    "Class": list(
                        class_probabilities.keys()
                    ),

                    "Probability": list(
                        class_probabilities.values()
                    )
                }
            )

            probability_df[
                "Probability (%)"
            ] = (
                probability_df[
                    "Probability"
                ]
                * 100
            )

            probability_columns = (
                st.columns(
                    len(
                        probability_df
                    )
                )
            )

            for index, row in (
                probability_df.iterrows()
            ):

                with probability_columns[
                    index
                ]:

                    st.metric(
                        str(
                            row["Class"]
                        ),
                        (
                            f"{row['Probability (%)']:.2f}%"
                        )
                    )

            # =================================================
            # PROBABILITY CHART
            # =================================================

            probability_fig = px.bar(
                probability_df,
                x="Class",
                y="Probability (%)",
                text="Probability (%)",
                title=(
                    f"{target_column} "
                    f"Prediction Probabilities"
                )
            )

            probability_fig.update_traces(
                texttemplate="%{text:.2f}%"
            )

            probability_fig.update_layout(
                yaxis_title=(
                    "Probability (%)"
                ),
                xaxis_title=None,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                )
            )

            probability_fig.update_yaxes(
                range=[0, 100]
            )

            st.plotly_chart(
                probability_fig,
                use_container_width=True,
                key=(
                    "prediction_probability_chart"
                )
            )

        # =====================================================
        # CONFIDENCE INTERPRETATION
        # =====================================================

        if confidence is not None:

            st.markdown(
                "### 🧠 Confidence Interpretation"
            )

            confidence_percent = (
                confidence * 100
            )

            if confidence_percent >= 80:

                st.success(
                    "High-confidence prediction. "
                    "The model strongly favors "
                    "the predicted class."
                )

            elif confidence_percent >= 60:

                st.info(
                    "Moderate-confidence prediction. "
                    "The model favors this class, "
                    "but there is some uncertainty."
                )

            else:

                st.warning(
                    "Low-confidence prediction. "
                    "The model is uncertain between "
                    "the available classes."
                )

    # =========================================================
    # REGRESSION
    # =========================================================

    else:

        confidence = None

        st.metric(
            f"Predicted {target_column}",
            prediction
        )

        st.success(
            f"🔮 Predicted **{target_column}**: "
            f"**{prediction}**"
        )

        st.caption(
            "Regression predictions estimate "
            "a continuous numerical value."
        )

    # =========================================================
    # LOG PREDICTION
    # =========================================================

    user = AuthGuard.current_user()

    if user:

        DatabaseService.save_prediction(
            user.email,
            prediction
        )

        DatabaseService.log_activity(
            user.email,
            "Generated Prediction"
        )

    # =========================================================
    # LOCAL PREDICTION EXPLANATION
    # =========================================================

    local_explanation = (
        st.session_state.get(
            "latest_local_explanation"
        )
    )

    if local_explanation:

        st.divider()

        st.markdown(
            "## 👤 Why This Prediction?"
        )

        st.caption(
            "This section explains which factors "
            "supported or opposed this specific prediction."
        )

        local_available = (
            local_explanation.get(
                "available",
                False
            )
        )

        if local_available:

            explained_class = (
                local_explanation.get(
                    "predicted_class",
                    prediction
                )
            )

            explanation_type = (
                local_explanation.get(
                    "explanation_type",
                    "Local Explanation"
                )
            )

            st.info(
                f"Explaining prediction "
                f"**{target_column} = {explained_class}** "
                f"using **{explanation_type}**."
            )

            increasing_factors = (
                local_explanation.get(
                    "increasing_factors",
                    []
                )
            )

            decreasing_factors = (
                local_explanation.get(
                    "decreasing_factors",
                    []
                )
            )

            top_contributions = (
                local_explanation.get(
                    "top_contributions",
                    []
                )
            )

            # =================================================
            # SUPPORTING / OPPOSING FACTORS
            # =================================================

            support_col, oppose_col = (
                st.columns(2)
            )

            with support_col:

                st.markdown(
                    "### ⬆️ Supports Prediction"
                )

                if increasing_factors:

                    for factor in (
                        increasing_factors[:5]
                    ):

                        feature = factor.get(
                            "Feature",
                            "Unknown"
                        )

                        contribution = float(
                            factor.get(
                                "Contribution",
                                0
                            )
                        )

                        st.write(
                            f"**{feature}** "
                            f"`+{contribution:.4f}`"
                        )

                else:

                    st.info(
                        "No strong supporting "
                        "factors were identified."
                    )

            with oppose_col:

                st.markdown(
                    "### ⬇️ Opposes Prediction"
                )

                if decreasing_factors:

                    for factor in (
                        decreasing_factors[:5]
                    ):

                        feature = factor.get(
                            "Feature",
                            "Unknown"
                        )

                        contribution = float(
                            factor.get(
                                "Contribution",
                                0
                            )
                        )

                        st.write(
                            f"**{feature}** "
                            f"`{contribution:.4f}`"
                        )

                else:

                    st.info(
                        "No strong opposing "
                        "factors were identified."
                    )

            # =================================================
            # LOCAL CONTRIBUTION TABLE
            # =================================================

            if top_contributions:

                local_df = pd.DataFrame(
                    top_contributions
                )

                st.markdown(
                    "### 🧩 Local Feature Contributions"
                )

                local_display_columns = [
                    column
                    for column in [
                        "Feature",
                        "Contribution",
                        "Direction",
                        "Relative Contribution (%)"
                    ]
                    if column in local_df.columns
                ]

                st.dataframe(
                    local_df[
                        local_display_columns
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # =============================================
                # LOCAL CONTRIBUTION CHART
                # =============================================

                if (
                    "Feature"
                    in local_df.columns
                    and
                    "Contribution"
                    in local_df.columns
                ):

                    local_chart_df = (
                        local_df
                        .sort_values(
                            "Contribution",
                            ascending=True
                        )
                    )

                    local_fig = px.bar(
                        local_chart_df,
                        x="Contribution",
                        y="Feature",
                        orientation="h",
                        text="Contribution",
                        title=(
                            "Factors Supporting and "
                            "Opposing This Prediction"
                        )
                    )

                    local_fig.update_traces(
                        texttemplate="%{text:.3f}"
                    )

                    local_fig.update_layout(
                        xaxis_title=(
                            "Contribution to "
                            "Predicted Class"
                        ),
                        yaxis_title=None
                    )

                    st.plotly_chart(
                        local_fig,
                        use_container_width=True,
                        key=(
                            "local_prediction_"
                            "explanation_chart"
                        )
                    )

                # =============================================
                # HUMAN-READABLE SUMMARY
                # =============================================

                st.markdown(
                    "### 💡 Why the Model Made This Prediction"
                )

                supporting_names = [
                    str(
                        factor.get(
                            "Feature"
                        )
                    )
                    for factor
                    in increasing_factors[:3]
                ]

                opposing_names = [
                    str(
                        factor.get(
                            "Feature"
                        )
                    )
                    for factor
                    in decreasing_factors[:3]
                ]

                if supporting_names:

                    st.write(
                        f"The strongest factors supporting "
                        f"**{target_column} = "
                        f"{explained_class}** are "
                        f"**{', '.join(supporting_names)}**."
                    )

                if opposing_names:

                    st.write(
                        "The main factors pushing the "
                        "model in the opposite direction are "
                        f"**{', '.join(opposing_names)}**."
                    )

                st.caption(
                    "For coefficient-based models, these "
                    "contributions are calculated from the "
                    "preprocessed feature values and the "
                    "coefficients learned by the model."
                )

            # =================================================
            # AI BUSINESS EXPLANATION
            # =================================================

            st.divider()

            st.markdown(
                "### 🤖 AI Business Explanation"
            )

            st.caption(
                "Convert the technical model explanation "
                "into a business-friendly summary using "
                "the Groq AI service."
            )

            # -------------------------------------------------
            # GENERATE BUTTON
            # -------------------------------------------------

            if st.button(
                "✨ Generate AI Business Explanation",
                type="primary",
                key="generate_prediction_ai_explanation"
            ):

                try:

                    with st.spinner(
                        "Groq is analyzing the prediction..."
                    ):

                        ai_explainer = (
                            PredictionAIExplainer()
                        )

                        current_confidence = (
                            prediction_result.get(
                                "confidence"
                            )
                        )

                        ai_business_explanation = (
                            ai_explainer.explain(
                                target_column=target_column,
                                prediction=explained_class,
                                confidence=current_confidence,
                                supporting_factors=(
                                    increasing_factors
                                ),
                                opposing_factors=(
                                    decreasing_factors
                                ),
                                model_name=best_model
                            )
                        )

                        st.session_state[
                            "prediction_ai_business_explanation"
                        ] = ai_business_explanation

                except Exception as e:

                    st.error(
                        "❌ AI explanation generation "
                        f"failed: {e}"
                    )

            # -------------------------------------------------
            # DISPLAY AI EXPLANATION
            # -------------------------------------------------

            ai_business_explanation = (
                st.session_state.get(
                    "prediction_ai_business_explanation"
                )
            )

            if ai_business_explanation:

                st.success(
                    "✅ AI explanation generated successfully."
                )

                st.markdown(
                    ai_business_explanation
                )

                st.caption(
                    "Generated using Groq. "
                    "The explanation summarizes the model's "
                    "prediction and contribution values; it does "
                    "not establish real-world causation."
                )

        else:

            local_message = (
                local_explanation.get(
                    "message",
                    (
                        "Local explanation is not "
                        "available for this model."
                    )
                )
            )

            st.info(
                f"ℹ️ {local_message}"
            )

    # =========================================================
    # GLOBAL MODEL EXPLANATION
    # =========================================================

    explanation_result = (
        st.session_state.get(
            "latest_prediction_explanation"
        )
    )

    if explanation_result:

        st.divider()

        st.markdown(
            "## 🌍 Overall Model Importance"
        )

        st.caption(
            "This section shows which features the "
            "trained model generally relies on most."
        )

        explanation_available = (
            explanation_result.get(
                "available",
                False
            )
        )

        if explanation_available:

            explanation_type = (
                explanation_result.get(
                    "explanation_type",
                    "Feature Importance"
                )
            )

            top_features = (
                explanation_result.get(
                    "top_features",
                    []
                )
            )

            st.info(
                f"Global explanation method: "
                f"**{explanation_type}**"
            )

            if top_features:

                explanation_df = (
                    pd.DataFrame(
                        top_features
                    )
                )

                # =============================================
                # GLOBAL TABLE
                # =============================================

                st.markdown(
                    "### 🧩 Top Model Features"
                )

                display_columns = [
                    column
                    for column in [
                        "Feature",
                        "Importance",
                        "Relative Importance (%)"
                    ]
                    if column
                    in explanation_df.columns
                ]

                st.dataframe(
                    explanation_df[
                        display_columns
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # =============================================
                # GLOBAL CHART
                # =============================================

                if (
                    "Feature"
                    in explanation_df.columns
                    and
                    "Relative Importance (%)"
                    in explanation_df.columns
                ):

                    global_chart_df = (
                        explanation_df
                        .sort_values(
                            "Relative Importance (%)",
                            ascending=True
                        )
                    )

                    global_fig = px.bar(
                        global_chart_df,
                        x=(
                            "Relative Importance (%)"
                        ),
                        y="Feature",
                        orientation="h",
                        text=(
                            "Relative Importance (%)"
                        ),
                        title=(
                            "Overall Model "
                            "Feature Importance"
                        )
                    )

                    global_fig.update_traces(
                        texttemplate="%{text:.1f}%"
                    )

                    global_fig.update_layout(
                        xaxis_title=(
                            "Relative Importance (%)"
                        ),
                        yaxis_title=None
                    )

                    st.plotly_chart(
                        global_fig,
                        use_container_width=True,
                        key=(
                            "global_prediction_"
                            "explanation_chart"
                        )
                    )

                # =============================================
                # GLOBAL SUMMARY
                # =============================================

                top_feature_names = (
                    explanation_df[
                        "Feature"
                    ]
                    .head(5)
                    .astype(str)
                    .tolist()
                )

                if top_feature_names:

                    st.markdown(
                        "### 💡 Overall Model Summary"
                    )

                    st.write(
                        f"The trained **{best_model}** model "
                        f"generally relies most on "
                        f"**{', '.join(top_feature_names)}**."
                    )

                st.caption(
                    "Global importance describes the model "
                    "overall. It should not be interpreted "
                    "as the reason for one individual prediction."
                )

            else:

                st.info(
                    "No global feature-importance "
                    "information was returned."
                )

        else:

            global_message = (
                explanation_result.get(
                    "message",
                    (
                        "Global model explanation "
                        "is not available."
                    )
                )
            )

            st.info(
                f"ℹ️ {global_message}"
            )

    # =========================================================
    # PREDICTION INPUT AUDIT
    # =========================================================

    latest_input = (
        st.session_state.get(
            "latest_prediction_input"
        )
    )

    if latest_input:

        st.divider()

        with st.expander(
            "📋 View Prediction Input"
        ):

            input_df = pd.DataFrame(
                [latest_input]
            )

            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )


    # =========================================================
    # PREDICTION HISTORY
    # =========================================================

    st.divider()
    st.markdown("## 📜 Prediction History")

    history_df = PredictionHistory.get()

    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇️ Download History",
                history_df.to_csv(index=False),
                "prediction_history.csv",
                "text/csv",
                use_container_width=True
            )

        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                PredictionHistory.clear()
                st.rerun()
    else:
        st.info("No predictions have been made yet.")

    # =========================================================
    # DISCLAIMER
    # =========================================================

    st.divider()

    st.caption(
        "⚠️ Predictions and explanations are generated from "
        "the machine learning model trained on the uploaded "
        "dataset. They should be validated before being used "
        "for real-world decisions."
    )