import streamlit as st
import pandas as pd

from app.services.dataset_comparison import DatasetComparison


def render_dashboard_tab(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    profile: dict,
    cleaning_summary: dict
):
    st.title("📊 Enterprise AI Data Analyst Copilot")
    st.caption(
        "AI-powered platform for profiling, cleaning, EDA, AutoML, prediction, explainability and reports."
    )
    st.divider()

    # ==========================================================
    # PLATFORM STATUS
    # ==========================================================

    h1, h2, h3, h4 = st.columns(4)

    h1.metric("🤖 AI Engine", "Qwen 2.5")
    h2.metric("🧠 AutoML", "Ready")
    h3.metric("📄 Reports", "PDF + JSON")
    h4.metric("⚡ Status", "Ready")

    st.divider()

    # ==========================================================
    # EXECUTIVE SUMMARY
    # ==========================================================

    st.subheader("📈 Executive Summary")

    rows = profile.get("number_of_rows", len(cleaned_df))
    cols = profile.get("number_of_columns", len(cleaned_df.columns))
    num = len(profile.get("numeric_columns", []))
    cat = len(profile.get("categorical_columns", []))

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📄 Rows", rows)
    c2.metric("📋 Columns", cols)
    c3.metric("🔢 Numeric", num)
    c4.metric("📝 Categorical", cat)

    st.divider()

    # ==========================================================
    # DATASET HEALTH
    # ==========================================================

    st.subheader("🛡️ Dataset Health")

    missing = int(cleaned_df.isnull().sum().sum())
    dup = int(cleaned_df.duplicated().sum())

    total = cleaned_df.shape[0] * cleaned_df.shape[1]

    completeness = (
        ((total - missing) / total) * 100
        if total else 0
    )

    score = max(
        0,
        round(completeness - dup * 0.5, 2)
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Missing", missing)
    q2.metric("Duplicates", dup)
    q3.metric("Completeness", f"{completeness:.2f}%")
    q4.metric("Health", f"{score:.2f}%")

    if score >= 95:
        st.success("🟢 Dataset ready for ML.")

    elif score >= 80:
        st.warning("🟡 Dataset is usable but can be improved.")

    else:
        st.error("🔴 Dataset needs additional cleaning.")

    st.divider()

    # ==========================================================
    # BEFORE VS AFTER CLEANING
    # ==========================================================

    st.subheader("📊 Before vs After Cleaning")

    comparison_df = DatasetComparison.compare(
        original_df,
        cleaned_df
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    before_quality = comparison_df.loc[
        comparison_df["Metric"] == "Dataset Quality (%)",
        "Before Cleaning"
    ].values[0]

    after_quality = comparison_df.loc[
        comparison_df["Metric"] == "Dataset Quality (%)",
        "After Cleaning"
    ].values[0]

    improvement = round(
        after_quality - before_quality,
        2
    )

    if improvement > 0:
        st.success(
            f"🟢 Dataset quality improved by {improvement}%"
        )
    else:
        st.info(
            "No quality improvement detected."
        )

    st.divider()

    # ==========================================================
    # DATASET PREVIEW
    # ==========================================================

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        cleaned_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================================
    # COLUMN INFORMATION
    # ==========================================================

    st.subheader("📑 Column Information")

    left, right = st.columns(2)

    with left:

        st.markdown("#### 🔢 Numeric Columns")

        nums = profile.get(
            "numeric_columns",
            []
        )

        if nums:

            for col in nums:
                st.write(f"• {col}")

        else:
            st.info("No numeric columns.")

    with right:

        st.markdown("#### 📝 Categorical Columns")

        cats = profile.get(
            "categorical_columns",
            []
        )

        if cats:

            for col in cats:
                st.write(f"• {col}")

        else:
            st.info("No categorical columns.")

    st.divider()

    # ==========================================================
    # CLEANING SUMMARY
    # ==========================================================

    st.subheader("🧹 Cleaning Summary")

    if cleaning_summary:

        a, b, c = st.columns(3)

        a.metric(
            "Duplicates Removed",
            cleaning_summary.get(
                "duplicates_removed",
                0
            )
        )

        b.metric(
            "Empty Rows Removed",
            cleaning_summary.get(
                "empty_rows_removed",
                0
            )
        )

        c.metric(
            "Columns Renamed",
            cleaning_summary.get(
                "columns_renamed",
                0
            )
        )

        with st.expander(
            "📄 Full Cleaning Summary"
        ):
            st.json(cleaning_summary)

    else:

        st.info(
            "No cleaning summary available."
        )

    st.divider()

    # ==========================================================
    # STATISTICS
    # ==========================================================

    st.subheader("📈 Statistical Summary")

    try:

        st.dataframe(
            cleaned_df.describe(
                include="all"
            ),
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Unable to generate statistics: {e}"
        )

    st.divider()

    # ==========================================================
    # COLUMN DETAILS
    # ==========================================================

    with st.expander("🔧 Column Details"):

        details = pd.DataFrame({
            "Column": cleaned_df.columns,
            "Data Type": [
                str(x)
                for x in cleaned_df.dtypes
            ],
            "Missing Values": [
                int(cleaned_df[c].isnull().sum())
                for c in cleaned_df.columns
            ],
            "Unique Values": [
                int(cleaned_df[c].nunique(dropna=True))
                for c in cleaned_df.columns
            ]
        })

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    st.caption(
        "Enterprise AI Data Analyst Copilot • Dashboard Module"
    )