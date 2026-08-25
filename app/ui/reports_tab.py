import streamlit as st
import tempfile
import os

from app.services.pdf_report_generator import PDFReportGenerator

# NEW
from app.auth.auth_guard import AuthGuard
from app.database.database_service import DatabaseService


def render_reports_tab(
    profile: dict,
    cleaning_summary: dict,
    eda_results: dict
):
    """
    Render the automated PDF report generation interface.

    The report can include:
    - Dataset profile
    - Data cleaning summary
    - Exploratory data analysis
    - AI-generated insights
    - ML recommendations
    - AutoML results
    - Prediction results
    - Prediction input
    - Local prediction explanation
    - Global model importance
    - AI business explanation
    """

    # =========================================================
    # HEADER
    # =========================================================

    st.subheader("📄 Automated Reports")

    st.caption(
        "Generate and download a professional analytical PDF "
        "containing dataset analysis, AI insights, AutoML results, "
        "predictions, and model explainability."
    )

    # =========================================================
    # GET DATA FROM SESSION STATE
    # =========================================================

    ai_insights = st.session_state.get(
        "ai_insights",
        ""
    )

    ml_recommendations = st.session_state.get(
        "ml_recommendations",
        ""
    )

    automl_results = st.session_state.get(
        "automl_results"
    )

    prediction_result = st.session_state.get(
        "latest_prediction"
    )

    prediction_input = st.session_state.get(
        "latest_prediction_input"
    )

    local_explanation = st.session_state.get(
        "latest_local_explanation"
    )

    global_explanation = st.session_state.get(
        "latest_prediction_explanation"
    )

    ai_business_explanation = st.session_state.get(
        "prediction_ai_business_explanation"
    )

    # =========================================================
    # REPORT COMPONENT STATUS
    # =========================================================

    st.markdown("### 📋 Report Components")

    row1_col1, row1_col2, row1_col3, row1_col4 = (
        st.columns(4)
    )

    with row1_col1:

        st.metric(
            "Dataset Profile",
            "Ready" if profile else "Missing"
        )

    with row1_col2:

        st.metric(
            "EDA",
            "Ready" if eda_results else "Missing"
        )

    with row1_col3:

        st.metric(
            "AI Insights",
            "Ready" if ai_insights else "Not Generated"
        )

    with row1_col4:

        st.metric(
            "ML Recommendations",
            (
                "Ready"
                if ml_recommendations
                else "Not Generated"
            )
        )

    row2_col1, row2_col2, row2_col3, row2_col4 = (
        st.columns(4)
    )

    with row2_col1:

        st.metric(
            "AutoML",
            (
                "Ready"
                if automl_results
                else "Not Trained"
            )
        )

    with row2_col2:

        st.metric(
            "Prediction",
            (
                "Ready"
                if prediction_result
                else "Not Generated"
            )
        )

    with row2_col3:

        local_ready = (
            isinstance(
                local_explanation,
                dict
            )
            and local_explanation.get(
                "available",
                False
            )
        )

        st.metric(
            "Explainability",
            (
                "Ready"
                if local_ready
                else "Not Generated"
            )
        )

    with row2_col4:

        st.metric(
            "AI Explanation",
            (
                "Ready"
                if ai_business_explanation
                else "Not Generated"
            )
        )

    st.divider()

    # =========================================================
    # REPORT CONTENTS
    # =========================================================

    st.markdown("### 📑 Report Contents")

    st.write(
        """
The generated PDF can contain:

1. Executive Summary
2. Dataset Overview
3. Data Quality & Cleaning
4. Exploratory Data Analysis
5. AI-Generated Insights
6. Machine Learning Recommendations
7. AutoML Model Analysis
8. Prediction Analysis
9. Prediction Explainability
10. AI Business Explanation
11. Conclusion & Model Disclaimer
        """
    )

    # =========================================================
    # READINESS INFORMATION
    # =========================================================

    st.markdown("### 🔎 Report Readiness")

    if not ai_insights:

        st.info(
            "💡 AI Insights have not been generated. "
            "The PDF can still be generated, but the AI Insights "
            "section will state that they are unavailable."
        )

    if not ml_recommendations:

        st.info(
            "💡 ML Recommendations have not been generated. "
            "Generate them from the AI Assistant if you want "
            "them included in the final report."
        )

    if not automl_results:

        st.info(
            "🧠 AutoML has not been trained. "
            "Train a model if you want AutoML analysis included."
        )

    if automl_results and not prediction_result:

        st.info(
            "🔮 No prediction has been generated yet. "
            "Go to the Prediction tab and make a prediction "
            "if you want prediction analysis in the report."
        )

    if prediction_result and not local_ready:

        st.info(
            "🧩 A local prediction explanation is not available. "
            "The remaining report sections can still be generated."
        )

    if prediction_result and not ai_business_explanation:

        st.info(
            "🤖 The AI Business Explanation has not been generated. "
            "Use the Prediction tab's AI explanation button if you "
            "want that section included in the PDF."
        )

    # =========================================================
    # AUTOML SUMMARY
    # =========================================================

    if automl_results:

        st.divider()

        st.markdown("### 🧠 AutoML Summary")

        best_model = automl_results.get(
            "best_model",
            "N/A"
        )

        target_column = automl_results.get(
            "target_column",
            "N/A"
        )

        problem_type = automl_results.get(
            "problem_type",
            "N/A"
        )

        selection_metric = automl_results.get(
            "selection_metric",
            "N/A"
        )

        selection_score = automl_results.get(
            "selection_score"
        )

        auto1, auto2, auto3, auto4 = (
            st.columns(4)
        )

        with auto1:

            st.metric(
                "Target",
                str(target_column)
            )

        with auto2:

            st.metric(
                "Best Model",
                str(best_model)
            )

        with auto3:

            st.metric(
                "Problem",
                str(problem_type).title()
            )

        with auto4:

            st.metric(
                "Selection Metric",
                str(selection_metric)
            )

        if selection_score is not None:

            try:

                st.caption(
                    "Selection Score: "
                    f"{float(selection_score):.4f}"
                )

            except Exception:

                st.caption(
                    f"Selection Score: {selection_score}"
                )

    # =========================================================
    # PREDICTION SUMMARY
    # =========================================================

    if prediction_result:

        st.divider()

        st.markdown("### 🔮 Latest Prediction Summary")

        predicted_value = prediction_result.get(
            "prediction",
            "N/A"
        )

        confidence = prediction_result.get(
            "confidence"
        )

        pred_col1, pred_col2 = (
            st.columns(2)
        )

        with pred_col1:

            st.metric(
                "Prediction",
                str(predicted_value)
            )

        with pred_col2:

            if confidence is not None:

                try:

                    st.metric(
                        "Confidence",
                        f"{float(confidence) * 100:.2f}%"
                    )

                except Exception:

                    st.metric(
                        "Confidence",
                        str(confidence)
                    )

            else:

                st.metric(
                    "Confidence",
                    "N/A"
                )

        class_probabilities = prediction_result.get(
            "class_probabilities"
        )

        if class_probabilities:

            probability_text = []

            for class_name, probability in (
                class_probabilities.items()
            ):

                try:

                    probability_text.append(
                        f"{class_name}: "
                        f"{float(probability) * 100:.2f}%"
                    )

                except Exception:

                    probability_text.append(
                        f"{class_name}: {probability}"
                    )

            st.caption(
                "Class Probabilities — "
                + " | ".join(
                    probability_text
                )
            )

    # =========================================================
    # EXPLAINABILITY SUMMARY
    # =========================================================

    if local_ready:

        st.divider()

        st.markdown(
            "### 🧩 Prediction Explainability Summary"
        )

        supporting_factors = (
            local_explanation.get(
                "increasing_factors",
                []
            )
        )

        opposing_factors = (
            local_explanation.get(
                "decreasing_factors",
                []
            )
        )

        explain_col1, explain_col2 = (
            st.columns(2)
        )

        with explain_col1:

            st.markdown(
                "**⬆️ Supporting Factors**"
            )

            if supporting_factors:

                for factor in (
                    supporting_factors[:5]
                ):

                    feature = factor.get(
                        "Feature",
                        "Unknown"
                    )

                    contribution = factor.get(
                        "Contribution",
                        0
                    )

                    try:

                        contribution = (
                            f"{float(contribution):.4f}"
                        )

                    except Exception:

                        contribution = str(
                            contribution
                        )

                    st.write(
                        f"• {feature}: "
                        f"+{contribution}"
                    )

            else:

                st.caption(
                    "No supporting factors available."
                )

        with explain_col2:

            st.markdown(
                "**⬇️ Opposing Factors**"
            )

            if opposing_factors:

                for factor in (
                    opposing_factors[:5]
                ):

                    feature = factor.get(
                        "Feature",
                        "Unknown"
                    )

                    contribution = factor.get(
                        "Contribution",
                        0
                    )

                    try:

                        contribution = (
                            f"{float(contribution):.4f}"
                        )

                    except Exception:

                        contribution = str(
                            contribution
                        )

                    st.write(
                        f"• {feature}: "
                        f"{contribution}"
                    )

            else:

                st.caption(
                    "No opposing factors available."
                )

    # =========================================================
    # AI BUSINESS EXPLANATION PREVIEW
    # =========================================================

    if ai_business_explanation:

        st.divider()

        st.markdown(
            "### 🤖 AI Business Explanation"
        )

        with st.expander(
            "Preview AI Business Explanation",
            expanded=False
        ):

            st.markdown(
                ai_business_explanation
            )

    # =========================================================
    # GENERATE REPORT
    # =========================================================

    st.divider()

    st.markdown(
        "### 📥 Generate Final PDF Report"
    )

    st.caption(
        "The PDF uses the latest analysis, AutoML, prediction, "
        "and explainability results currently stored in this session."
    )

    if st.button(
        "📄 Generate Final PDF Report",
        type="primary",
        key="generate_pdf_report",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Building your professional analytical report..."
            ):

                # =============================================
                # TEMPORARY PDF
                # =============================================

                temp_file = (
                    tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    )
                )

                output_path = (
                    temp_file.name
                )

                temp_file.close()

                # =============================================
                # FALLBACK AI TEXT
                # =============================================

                report_ai_insights = (
                    str(ai_insights)
                    if ai_insights
                    else (
                        "AI insights were not generated "
                        "for this analysis."
                    )
                )

                report_ml_recommendations = (
                    str(ml_recommendations)
                    if ml_recommendations
                    else (
                        "Machine learning recommendations "
                        "were not generated for this analysis."
                    )
                )

                # =============================================
                # GENERATE PDF
                # =============================================

                PDFReportGenerator.generate(
                    output_path=output_path,

                    profile=(
                        profile or {}
                    ),

                    cleaning_summary=(
                        cleaning_summary or {}
                    ),

                    eda_results=(
                        eda_results or {}
                    ),

                    ai_insights=(
                        report_ai_insights
                    ),

                    ml_recommendations=(
                        report_ml_recommendations
                    ),

                    automl_results=(
                        automl_results
                    ),

                    prediction_result=(
                        prediction_result
                    ),

                    prediction_input=(
                        prediction_input
                    ),

                    local_explanation=(
                        local_explanation
                    ),

                    global_explanation=(
                        global_explanation
                    ),

                    ai_business_explanation=(
                        ai_business_explanation
                    ),
                )

                # =============================================
                # READ PDF
                # =============================================

                with open(
                    output_path,
                    "rb"
                ) as pdf_file:

                    pdf_data = (
                        pdf_file.read()
                    )

                # =============================================
                # STORE PDF
                # =============================================

                st.session_state[
                    "generated_pdf"
                ] = pdf_data

                # =============================================
                # DELETE TEMP FILE
                # =============================================

                try:

                    os.remove(
                        output_path
                    )

                except Exception:
                    pass

                # =============================================
                # LOG REPORT GENERATION
                # =============================================

                user = AuthGuard.current_user()

                if user:

                    DatabaseService.save_report(
                        user.email,
                        "Enterprise AI Report"
                    )

                    DatabaseService.log_activity(
                        user.email,
                        "Generated Report"
                    )

            st.success(
                "✅ Final PDF report generated successfully!"
            )

            st.info(
                "The report now includes all available "
                "dataset, AI, AutoML, prediction, and "
                "explainability results."
            )

        except Exception as e:

            st.error(
                f"❌ PDF report generation failed: {e}"
            )

    # =========================================================
    # DOWNLOAD REPORT
    # =========================================================

    generated_pdf = (
        st.session_state.get(
            "generated_pdf"
        )
    )

    if generated_pdf:

        st.divider()

        st.markdown(
            "### 📥 Download Final Report"
        )

        st.download_button(
            label="⬇️ Download Enterprise AI Report",
            data=generated_pdf,
            file_name=(
                "Enterprise_AI_Data_Analyst_Report.pdf"
            ),
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf_report"
        )

    # =========================================================
    # FINAL NOTE
    # =========================================================

    st.divider()

    st.caption(
        "The generated report reflects the latest results "
        "available in the current Streamlit session. "
        "Predictions and AI-generated interpretations should "
        "be validated before real-world decision-making."
    )