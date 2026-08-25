import streamlit as st
import pandas as pd

from app.services.drift_detector import DriftDetector


def render_drift_detection_tab(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame
):

    st.title("📊 Data Drift Detection")

    st.caption(
        "Compare the original and cleaned datasets to identify distribution changes and potential data drift."
    )

    st.divider()

    detector = DriftDetector(
        original_df,
        cleaned_df
    )

    summary = detector.summary()

    # =====================================================
    # EXECUTIVE METRICS
    # =====================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📊 Drift Score",
            f"{summary['Overall Drift Score']}%"
        )

    with col2:

        st.metric(
            "⚠ Risk Level",
            summary["Risk"]
        )

    with col3:

        st.metric(
            "🔢 Numeric Columns",
            summary["Numeric Columns"]
        )

    with col4:

        st.metric(
            "📝 Categorical Columns",
            summary["Categorical Columns"]
        )

    st.divider()
        # =====================================================
    # NUMERIC DRIFT
    # =====================================================

    st.subheader("📈 Numeric Drift")

    numeric = detector.numeric_drift()

    if not numeric.empty:

        st.dataframe(
            numeric,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No numeric columns found."
        )

    st.divider()

    # =====================================================
    # CATEGORICAL DRIFT
    # =====================================================

    st.subheader("📝 Categorical Drift")

    categorical = detector.categorical_drift()

    if not categorical.empty:

        st.dataframe(
            categorical,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No categorical columns found."
        )

    st.divider()

    # =====================================================
    # MISSING VALUE DRIFT
    # =====================================================

    st.subheader("🔍 Missing Value Drift")

    missing = detector.missing_value_drift()

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
        # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.subheader("📥 Export Drift Report")

    report = detector.full_report()

    st.download_button(
        label="⬇️ Download Numeric Drift Report",
        data=report["numeric"].to_csv(index=False),
        file_name="numeric_drift.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Categorical Drift Report",
        data=report["categorical"].to_csv(index=False),
        file_name="categorical_drift.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.download_button(
        label="⬇️ Download Missing Value Report",
        data=report["missing"].to_csv(index=False),
        file_name="missing_value_drift.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success(
        "✅ Drift detection completed successfully."
    )

    st.caption(
        "Enterprise AI Data Analyst Copilot • Drift Detection Module"
    )