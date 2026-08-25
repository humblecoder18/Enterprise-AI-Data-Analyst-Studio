import os
import matplotlib.pyplot as plt
import pandas as pd


class VisualizationAgent:
    """Creates basic charts from a dataset."""

    @staticmethod
    def generate_charts(df: pd.DataFrame, output_folder="reports/charts"):
        os.makedirs(output_folder, exist_ok=True)

        numeric_columns = df.select_dtypes(include="number").columns

        for column in numeric_columns:
            plt.figure(figsize=(6, 4))
            df[column].hist(bins=20)
            plt.title(column)
            plt.xlabel(column)
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.savefig(f"{output_folder}/{column}.png")
            plt.close()