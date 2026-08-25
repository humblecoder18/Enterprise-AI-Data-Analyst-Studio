import pandas as pd
import numpy as np


class EDAAgent:
    """Performs Exploratory Data Analysis on a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        self.df = df.copy()

    def analyze(self) -> dict:
        results = {}

        # Dataset shape
        results["shape"] = self.df.shape

        # Missing values
        results["missing_values"] = self.df.isnull().sum().to_dict()

        # Numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            results["summary_statistics"] = {
                "mean": numeric_df.mean().to_dict(),
                "median": numeric_df.median().to_dict(),
                "std": numeric_df.std().to_dict(),
                "min": numeric_df.min().to_dict(),
                "max": numeric_df.max().to_dict(),
            }

            results["correlation_matrix"] = numeric_df.corr().to_dict()

        # Categorical columns
        categorical = {}

        for column in self.df.select_dtypes(include=["object"]).columns:
            categorical[column] = {
                "unique_count": int(self.df[column].nunique()),
                "top_5_values": self.df[column].value_counts().head(5).to_dict(),
            }

        results["categorical_analysis"] = categorical

        return results