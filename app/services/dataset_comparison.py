import pandas as pd


class DatasetComparison:

    @staticmethod
    def compare(original_df, cleaned_df):

        original_missing = int(original_df.isnull().sum().sum())
        cleaned_missing = int(cleaned_df.isnull().sum().sum())

        original_duplicates = int(original_df.duplicated().sum())
        cleaned_duplicates = int(cleaned_df.duplicated().sum())

        original_memory = (
            original_df.memory_usage(deep=True).sum()
            / 1024 / 1024
        )

        cleaned_memory = (
            cleaned_df.memory_usage(deep=True).sum()
            / 1024 / 1024
        )

        original_quality = (
            (
                1
                -
                (
                    original_missing
                    + original_duplicates
                )
                /
                max(len(original_df), 1)
            )
            * 100
        )

        cleaned_quality = (
            (
                1
                -
                (
                    cleaned_missing
                    + cleaned_duplicates
                )
                /
                max(len(cleaned_df), 1)
            )
            * 100
        )

        return pd.DataFrame(
            {
                "Metric": [
                    "Rows",
                    "Columns",
                    "Missing Values",
                    "Duplicate Rows",
                    "Memory Usage (MB)",
                    "Dataset Quality (%)"
                ],
                "Before Cleaning": [
                    len(original_df),
                    len(original_df.columns),
                    original_missing,
                    original_duplicates,
                    round(original_memory, 2),
                    round(original_quality, 2)
                ],
                "After Cleaning": [
                    len(cleaned_df),
                    len(cleaned_df.columns),
                    cleaned_missing,
                    cleaned_duplicates,
                    round(cleaned_memory, 2),
                    round(cleaned_quality, 2)
                ]
            }
        )