import pandas as pd
import plotly.express as px


class ChartGenerator:
    """Generate interactive Plotly charts."""

    @staticmethod
    def histogram(df: pd.DataFrame, column: str):
        return px.histogram(
            df,
            x=column,
            title=f"{column} Distribution"
        )

    @staticmethod
    def box_plot(df: pd.DataFrame, column: str):
        return px.box(
            df,
            y=column,
            title=f"{column} Box Plot"
        )

    @staticmethod
    def pie_chart(df: pd.DataFrame, column: str):
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "Count"]

        return px.pie(
            counts,
            names=column,
            values="Count",
            title=f"{column} Distribution"
        )

    @staticmethod
    def bar_chart(df: pd.DataFrame, column: str):
        counts = df[column].value_counts().reset_index()
        counts.columns = [column, "Count"]

        return px.bar(
            counts,
            x=column,
            y="Count",
            title=f"{column} Distribution",
            labels={column: column, "Count": "Count"}
        )

    @staticmethod
    def scatter_plot(df: pd.DataFrame, x_column: str, y_column: str):
        return px.scatter(
            df,
            x=x_column,
            y=y_column,
            title=f"{x_column} vs {y_column}"
        )

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame):
        numeric_df = df.select_dtypes(include="number")

        if numeric_df.empty:
            return None

        corr = numeric_df.corr()

        return px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap"
        )

    @staticmethod
    def missing_values_chart(df: pd.DataFrame):
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            return None

        return px.bar(
            x=missing.index,
            y=missing.values,
            labels={"x": "Columns", "y": "Missing Values"},
            title="Missing Values"
        )

    @staticmethod
    def line_chart(df: pd.DataFrame, x_column: str, y_column: str):
        return px.line(
            df,
            x=x_column,
            y=y_column,
            title=f"{y_column} over {x_column}"
        )