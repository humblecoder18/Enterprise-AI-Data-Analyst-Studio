import streamlit as st
import pandas as pd

from app.services.chart_generator import ChartGenerator


def render_analytics_tab(
    cleaned_df: pd.DataFrame,
    eda_results: dict
):
    """
    Render the interactive analytics and EDA interface.
    """

    # =========================================================
    # HEADER
    # =========================================================

    st.subheader("📈 Interactive Analytics")

    st.caption(
        "Explore distributions, relationships, correlations, "
        "and data-quality patterns in your dataset."
    )

    if cleaned_df is None or cleaned_df.empty:
        st.warning("No dataset available for analysis.")
        return

    # =========================================================
    # DETECT COLUMN TYPES
    # =========================================================

    numeric_columns = (
        cleaned_df
        .select_dtypes(include="number")
        .columns
        .tolist()
    )

    categorical_columns = (
        cleaned_df
        .select_dtypes(
            include=["object", "category", "bool"]
        )
        .columns
        .tolist()
    )

    st.divider()

    # =========================================================
    # CHART TYPE
    # =========================================================

    st.markdown("### 📊 Chart Builder")

    chart_type = st.selectbox(
        "Select Chart Type",
        [
            "Histogram",
            "Box Plot",
            "Pie Chart",
            "Bar Chart",
            "Scatter Plot",
            "Line Chart",
            "Correlation Heatmap",
            "Missing Values"
        ],
        key="analytics_chart_type"
    )

    # =========================================================
    # HISTOGRAM
    # =========================================================

    if chart_type == "Histogram":

        if numeric_columns:

            column = st.selectbox(
                "Choose Numeric Column",
                numeric_columns,
                key="analytics_histogram_column"
            )

            try:

                fig = ChartGenerator.histogram(
                    cleaned_df,
                    column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_histogram_chart"
                )

            except Exception as e:

                st.error(
                    f"Unable to generate histogram: {e}"
                )

        else:

            st.warning(
                "No numeric columns are available."
            )

    # =========================================================
    # BOX PLOT
    # =========================================================

    elif chart_type == "Box Plot":

        if numeric_columns:

            column = st.selectbox(
                "Choose Numeric Column",
                numeric_columns,
                key="analytics_box_column"
            )

            try:

                fig = ChartGenerator.box_plot(
                    cleaned_df,
                    column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_box_chart"
                )

            except Exception as e:

                st.error(
                    f"Unable to generate box plot: {e}"
                )

        else:

            st.warning(
                "No numeric columns are available."
            )

    # =========================================================
    # PIE CHART
    # =========================================================

    elif chart_type == "Pie Chart":

        if categorical_columns:

            column = st.selectbox(
                "Choose Categorical Column",
                categorical_columns,
                key="analytics_pie_column"
            )

            try:

                fig = ChartGenerator.pie_chart(
                    cleaned_df,
                    column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_pie_chart"
                )

            except Exception as e:

                st.error(
                    f"Unable to generate pie chart: {e}"
                )

        else:

            st.warning(
                "No categorical columns are available."
            )

    # =========================================================
    # BAR CHART
    # =========================================================

    elif chart_type == "Bar Chart":

        if categorical_columns:

            column = st.selectbox(
                "Choose Categorical Column",
                categorical_columns,
                key="analytics_bar_column"
            )

            try:

                fig = ChartGenerator.bar_chart(
                    cleaned_df,
                    column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_bar_chart"
                )

            except Exception as e:

                st.error(
                    f"Unable to generate bar chart: {e}"
                )

        else:

            st.warning(
                "No categorical columns are available."
            )

    # =========================================================
    # SCATTER PLOT
    # =========================================================

    elif chart_type == "Scatter Plot":

        if len(numeric_columns) >= 2:

            col1, col2 = st.columns(2)

            with col1:

                x_column = st.selectbox(
                    "X Axis",
                    numeric_columns,
                    key="analytics_scatter_x"
                )

            with col2:

                default_y = (
                    1
                    if len(numeric_columns) > 1
                    else 0
                )

                y_column = st.selectbox(
                    "Y Axis",
                    numeric_columns,
                    index=default_y,
                    key="analytics_scatter_y"
                )

            try:

                fig = ChartGenerator.scatter_plot(
                    cleaned_df,
                    x_column,
                    y_column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_scatter_chart"
                )

            except Exception as e:

                st.error(
                    f"Unable to generate scatter plot: {e}"
                )

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    # =========================================================
    # LINE CHART
    # =========================================================

    elif chart_type == "Line Chart":

        if numeric_columns:

            col1, col2 = st.columns(2)

            with col1:

                x_column = st.selectbox(
                    "X Axis",
                    cleaned_df.columns.tolist(),
                    key="analytics_line_x"
                )

            with col2:

                y_column = st.selectbox(
                    "Y Axis",
                    numeric_columns,
                    key="analytics_line_y"
                )

            try:

                fig = ChartGenerator.line_chart(
                    cleaned_df,
                    x_column,
                    y_column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_line_chart"
                )

            except Exception as e:

                st.error(
                    f"Unable to generate line chart: {e}"
                )

        else:

            st.warning(
                "No numeric columns are available."
            )

    # =========================================================
    # CORRELATION HEATMAP
    # =========================================================

    elif chart_type == "Correlation Heatmap":

        if len(numeric_columns) >= 2:

            try:

                fig = (
                    ChartGenerator
                    .correlation_heatmap(
                        cleaned_df
                    )
                )

                if fig is not None:

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key="analytics_correlation_chart"
                    )

                else:

                    st.warning(
                        "Unable to calculate correlations."
                    )

            except Exception as e:

                st.error(
                    f"Unable to generate correlation heatmap: {e}"
                )

        else:

            st.warning(
                "At least two numeric columns are required."
            )

    # =========================================================
    # MISSING VALUES
    # =========================================================

    elif chart_type == "Missing Values":

        try:

            fig = (
                ChartGenerator
                .missing_values_chart(
                    cleaned_df
                )
            )

            if fig is not None:

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="analytics_missing_values_chart"
                )

            else:

                st.success(
                    "✅ No missing values found in the dataset."
                )

        except Exception as e:

            st.error(
                f"Unable to generate missing-value chart: {e}"
            )

    # =========================================================
    # QUICK ANALYTICS
    # =========================================================

    st.divider()

    st.markdown("### 📋 Quick Analytics")

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    with metric1:

        st.metric(
            "Numeric Features",
            len(numeric_columns)
        )

    with metric2:

        st.metric(
            "Categorical Features",
            len(categorical_columns)
        )

    with metric3:

        st.metric(
            "Missing Values",
            int(
                cleaned_df
                .isnull()
                .sum()
                .sum()
            )
        )

    with metric4:

        st.metric(
            "Duplicate Rows",
            int(
                cleaned_df
                .duplicated()
                .sum()
            )
        )

    # =========================================================
    # NUMERICAL STATISTICS
    # =========================================================

    if numeric_columns:

        st.divider()

        st.markdown(
            "### 🔢 Numerical Statistics"
        )

        try:

            numeric_summary = (
                cleaned_df[
                    numeric_columns
                ]
                .describe()
                .T
            )

            st.dataframe(
                numeric_summary,
                use_container_width=True
            )

        except Exception as e:

            st.warning(
                f"Unable to generate numerical statistics: {e}"
            )

    # =========================================================
    # CATEGORICAL STATISTICS
    # =========================================================

    if categorical_columns:

        st.divider()

        st.markdown(
            "### 📝 Categorical Statistics"
        )

        categorical_summary = []

        for column in categorical_columns:

            mode = (
                cleaned_df[column]
                .mode()
            )

            most_common = (
                str(mode.iloc[0])
                if not mode.empty
                else "N/A"
            )

            categorical_summary.append(
                {
                    "Column": column,
                    "Unique Values": int(
                        cleaned_df[column]
                        .nunique(
                            dropna=True
                        )
                    ),
                    "Most Common": most_common,
                    "Missing Values": int(
                        cleaned_df[column]
                        .isnull()
                        .sum()
                    )
                }
            )

        st.dataframe(
            pd.DataFrame(
                categorical_summary
            ),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================
    # EDA RESULTS
    # =========================================================

    st.divider()

    st.markdown(
        "### 📑 Exploratory Data Analysis"
    )

    if eda_results:

        with st.expander(
            "View Complete EDA Results"
        ):

            st.json(
                eda_results
            )

    else:

        st.info(
            "No EDA results are available."
        )